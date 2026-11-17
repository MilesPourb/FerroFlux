"""Four-Stage Mechanistic Curriculum Pretraining Engine for FerroFlux-FM.

Supports:
- Pre-flight zero-leakage quarantine & <500-file inode cap audit (`audit_parquet_directory`)
- Dual weight-release tracks (`--license-track open|academic`)
- Model variants (`--model-size compact|base|xl`)
- Mixed precision (`--precision bf16|fp32|auto`) on both AVX512/EPYC CPU and NVIDIA H100/A100 CUDA
- Multi-GPU DistributedDataParallel via `torchrun` (`WORLD_SIZE > 1`)
- Concurrent multi-seed CPU worker execution (`--parallel-seeds`) for 128-core highmem nodes
- Automatic preemption checkpointing & resume (`--resume-auto` + `SIGUSR1` SLURM handler)
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import yaml

from ferroflux.models.ferroflux_fm import FerroFluxFM
from ferroflux.models.losses import FerroFluxCurriculumLoss
from ferroflux.security.quarantine import audit_parquet_directory

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
DEFAULT_CONFIG = ROOT_DIR / "configs" / "train_config.yaml"
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_OUT_DIR = ROOT_DIR / "checkpoints" / "open"

_SIGUSR1_FLAG = False


def _handle_sigusr1(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _SIGUSR1_FLAG
    _SIGUSR1_FLAG = True


class SyntheticCurriculumDataset(Dataset):
    """Deterministic synthetic dataset adhering to the #40 FerroFluxMultimodalDataset contract."""

    def __init__(self, n_samples: int = 128, n_genes: int = 256, seed: int = 42) -> None:
        super().__init__()
        self.n_samples = int(n_samples)
        self.n_genes = int(n_genes)
        g = torch.Generator().manual_seed(seed)

        self.gene_expr = torch.randn(n_samples, n_genes, generator=g)
        self.gene_mask = (torch.rand(n_samples, n_genes, generator=g) > 0.05).float()
        self.cis_reg_epigen = torch.randn(n_samples, 64, 8, generator=g) * 0.5
        self.cis_reg_mask = torch.ones(n_samples, 64)
        self.is_3d_organoid = (torch.rand(n_samples, generator=g) > 0.75).float()
        self.is_normal_organ_or_til = (torch.rand(n_samples, generator=g) > 0.80).float()

        self.chem_fp_a = (torch.rand(n_samples, 2048, generator=g) > 0.96).float()
        self.chem_fp_b = (torch.rand(n_samples, 2048, generator=g) > 0.99).float()
        self.chem_desc_a = torch.zeros(n_samples, 12)
        self.chem_desc_a[:, :6] = torch.rand(n_samples, 6, generator=g)
        self.chem_desc_a[:, 6] = 1.0  # covalent GPX4/FIN motif active in synthetic viability challenge
        self.chem_desc_b = torch.zeros(n_samples, 12)
        self.dose_a = torch.linspace(-2.5, 1.5, n_samples)
        self.dose_b = torch.full((n_samples,), -3.0)
        self.gene_ko_mult = torch.ones(n_samples, n_genes)
        self.rta_conc = torch.zeros(n_samples)
        self.rta_conc[1::5] = 4.0  # Liproxstatin-1 rescue arm
        self.dhcr7_mult = torch.ones(n_samples)
        self.dhcr7_mult[2::7] = 0.08
        self.ebp_mult = torch.ones(n_samples)
        self.ebp_mult[3::7] = 0.08
        self.media_ext = torch.ones(n_samples, 10)
        self.media_ext[:, 9] = 0.0
        self.np_params = torch.zeros(n_samples, 5)
        self.np_params[::4] = torch.tensor([6.0, -5.0, 1.0, 1.0, 6.0])
        self.media_ext[::4, 0] = 0.05  # cystine starvation with C' dot
        self.delta_t = torch.full((n_samples,), 24.0)

        # Mechanistically coherent synthetic targets across Stages 1-4
        self.prot_anchor_target = self.gene_expr[:, :28] * 0.6 + 0.2 * torch.randn(n_samples, 28, generator=g)
        self.prot_anchor_mask = torch.ones(n_samples, 28)
        self.metab_anchor_target = torch.randn(n_samples, 10, generator=g) * 0.3
        self.metab_anchor_mask = torch.ones(n_samples, 10)

        self.chronos_target = -0.4 * torch.sigmoid(self.dose_a) + 0.1 * torch.randn(n_samples, generator=g)
        self.chronos_mask = torch.ones(n_samples)
        self.delta_expr_target = -0.15 * self.gene_expr + 0.1 * torch.randn(n_samples, n_genes, generator=g)
        self.delta_expr_mask = torch.ones(n_samples, n_genes)
        self.survivorship_weight = torch.ones(n_samples) * 1.2
        self.sec46_readthrough_target = torch.sigmoid(self.gene_expr[:, 7])
        self.sec46_mask = torch.ones(n_samples)

        raw_drive = (
            1.2 * torch.sigmoid(self.dose_a)
            + 0.9 * (self.np_params[:, 0] > 0).float()
            - 0.8 * (self.dhcr7_mult < 0.5).float()
            + 0.7 * (self.ebp_mult < 0.5).float()
            - 1.6 * (self.rta_conc > 0).float()
            - 0.25 * self.gene_expr[:, 0]
        )
        stress = torch.sigmoid(raw_drive)
        self.viability_target = (1.0 - 0.8 * stress).clamp(0.02, 0.99)
        self.viability_mask = torch.ones(n_samples)
        self.hazard_target = torch.zeros(n_samples, 5)
        self.hazard_target[:, 0] = -torch.log(self.viability_target)
        self.hazard_mask = torch.ones(n_samples, 5)
        self.rescue_delta_target = torch.tanh(0.8 * self.hazard_target[:, 0])
        self.rescue_mask = torch.ones(n_samples)
        self.synergy_target = torch.zeros(n_samples)
        self.synergy_mask = torch.ones(n_samples)

        self.oxylipid_target = torch.stack(
            [stress, torch.relu(self.gene_expr[:, 37 % n_genes]) * 0.3, 0.2 + 0.5 * stress, stress * 0.7],
            dim=-1,
        )
        self.oxylipid_mask = torch.ones(n_samples, 4)
        self.icb_response_target = (stress > 0.55).float()
        self.icb_mask = torch.ones(n_samples)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "gene_expr": self.gene_expr[idx],
            "gene_mask": self.gene_mask[idx],
            "cis_reg_epigen": self.cis_reg_epigen[idx],
            "cis_reg_mask": self.cis_reg_mask[idx],
            "is_3d_organoid": self.is_3d_organoid[idx],
            "is_normal_organ_or_til": self.is_normal_organ_or_til[idx],
            "chem_fp_a": self.chem_fp_a[idx],
            "chem_fp_b": self.chem_fp_b[idx],
            "chem_desc_a": self.chem_desc_a[idx],
            "chem_desc_b": self.chem_desc_b[idx],
            "dose_a": self.dose_a[idx],
            "dose_b": self.dose_b[idx],
            "rta_conc": self.rta_conc[idx],
            "dhcr7_mult": self.dhcr7_mult[idx],
            "ebp_mult": self.ebp_mult[idx],
            "gene_ko_mult": self.gene_ko_mult[idx],
            "media_ext": self.media_ext[idx],
            "np_params": self.np_params[idx],
            "delta_t": self.delta_t[idx],
            "prot_anchor_target": self.prot_anchor_target[idx],
            "prot_anchor_mask": self.prot_anchor_mask[idx],
            "metab_anchor_target": self.metab_anchor_target[idx],
            "metab_anchor_mask": self.metab_anchor_mask[idx],
            "chronos_target": self.chronos_target[idx],
            "chronos_mask": self.chronos_mask[idx],
            "delta_expr_target": self.delta_expr_target[idx],
            "delta_expr_mask": self.delta_expr_mask[idx],
            "survivorship_weight": self.survivorship_weight[idx],
            "sec46_readthrough_target": self.sec46_readthrough_target[idx],
            "sec46_mask": self.sec46_mask[idx],
            "viability_target": self.viability_target[idx],
            "viability_mask": self.viability_mask[idx],
            "hazard_target": self.hazard_target[idx],
            "hazard_mask": self.hazard_mask[idx],
            "rescue_delta_target": self.rescue_delta_target[idx],
            "rescue_mask": self.rescue_mask[idx],
            "synergy_target": self.synergy_target[idx],
            "synergy_mask": self.synergy_mask[idx],
            "oxylipid_target": self.oxylipid_target[idx],
            "oxylipid_mask": self.oxylipid_mask[idx],
            "icb_response_target": self.icb_response_target[idx],
            "icb_mask": self.icb_mask[idx],
        }


def load_training_dataset(
    data_dir: Path,
    license_track: str,
    smoke_test: bool = False,
) -> tuple[Dataset, Any, int]:
    """Instantiate `FerroFluxMultimodalDataset` (or synthetic dataset when `smoke_test=True`)."""
    if smoke_test:
        ds = SyntheticCurriculumDataset(n_samples=128, n_genes=256)
        return ds, None, 256

    from ferroflux.data import dataset as ds_mod

    ds_cls = getattr(ds_mod, "FerroFluxMultimodalDataset")
    sig = inspect.signature(ds_cls.__init__)
    kw: dict[str, Any] = {}
    if "data_dir" in sig.parameters:
        kw["data_dir"] = str(data_dir)
    elif "processed_dir" in sig.parameters:
        kw["processed_dir"] = str(data_dir)
    if "license_track" in sig.parameters:
        kw["license_track"] = license_track
    elif "track" in sig.parameters:
        kw["track"] = license_track

    ds_obj = ds_cls(**kw)
    collate = getattr(ds_obj, "collate_fn", None)
    sample0 = ds_obj[0]
    n_genes = int(sample0["gene_expr"].shape[-1])
    return ds_obj, collate, n_genes


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out


def _resolve_device(device_arg: str, local_rank: int = 0) -> torch.device:
    req = device_arg.strip().lower()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        print("[WARN] --device cuda requested on host without visible GPU; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    if req == "auto":
        return torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cpu")


def pretrain_stage05_chem_sar(
    model: FerroFluxFM,
    *,
    data_dir: Path,
    license_track: str,
    device: torch.device,
    forbid_smiles: set[str] | None = None,
    forbid_ik14: set[str] | None = None,
    forbid_scaffolds: set[str] | None = None,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 1.5e-3,
    seed: int = 101,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Stage-0.5 Leak-Free Biochemical & Phenotypic SAR + CysDB Covalent Prior Pretraining.

    Strictly excludes EVERY held-out `SMILES`, stereochemistry-invariant `InChIKey-14`, and
    Bemis-Murcko scaffold passed in `forbid_*` before touching any weight of `perturbation_op`.
    """
    if smoke_test or model.ablate_chem_prior:
        return {"skipped": True, "n_molecules": 0, "cysdb_loaded": False}

    from ferroflux.data.dataset import (
        FerroChemSARPretrainDataset,
        compute_cysdb_target_ligandability_prior,
    )

    cys_prior = compute_cysdb_target_ligandability_prior(data_dir=data_dir, license_track=license_track)
    with torch.no_grad():
        model.perturbation_op.cysdb_target_prior.copy_(
            torch.from_numpy(cys_prior).to(
                device=device, dtype=model.perturbation_op.cysdb_target_prior.dtype
            )
        )

    if model.ablate_sar_pretrain or epochs <= 0:
        return {"skipped": True, "n_molecules": 0, "cysdb_loaded": True}

    sar_ds = FerroChemSARPretrainDataset(
        data_dir=data_dir,
        license_track=license_track,
        forbid_smiles=forbid_smiles,
        forbid_ik14=forbid_ik14,
        forbid_scaffolds=forbid_scaffolds,
        seed=seed,
    )
    n_mol = len(sar_ds)
    if n_mol < 16:
        return {"skipped": True, "n_molecules": n_mol, "cysdb_loaded": True}

    fp_t = torch.from_numpy(sar_ds.fp_u8)
    desc_t = torch.from_numpy(sar_ds.desc_f32)
    pot_t = torch.from_numpy(sar_ds.potency_target)
    pot_m_t = torch.from_numpy(sar_ds.potency_mask)
    act_t = torch.from_numpy(sar_ds.active_target)
    act_m_t = torch.from_numpy(sar_ds.active_mask)

    sar_modules = [
        model.perturbation_op.chem_in_proj,
        model.perturbation_op.chem_blocks,
        model.perturbation_op.chem_out_norm,
        model.perturbation_op.chem_mech_head,
        model.perturbation_op.sar_pretrain_head,
    ]
    sar_params = [p for mod in sar_modules for p in mod.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(sar_params, lr=lr, weight_decay=1e-4)
    steps_per_ep = max(1, (n_mol + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps_per_ep), eta_min=lr * 0.05)

    rng = np.random.default_rng(seed)
    model.perturbation_op.train()
    last_loss = 0.0

    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(n_mol))
        ep_losses: list[float] = []
        for s in range(0, n_mol, batch_size):
            idx = perm[s : s + batch_size]
            if idx.numel() < 4:
                continue
            b_fp = fp_t[idx].to(device=device, dtype=torch.float32)
            b_desc = desc_t[idx].to(device=device, dtype=torch.float32)
            b_pot = pot_t[idx].to(device=device, dtype=torch.float32)
            b_pm = pot_m_t[idx].to(device=device, dtype=torch.float32) > 0.5
            b_act = act_t[idx].to(device=device, dtype=torch.float32)
            b_am = act_m_t[idx].to(device=device, dtype=torch.float32) > 0.5

            opt.zero_grad(set_to_none=True)
            sar_out = model.perturbation_op.predict_sar_targets(b_fp, b_desc)
            pred_pot = sar_out["pred_potency"]
            act_logits = sar_out["active_logits"]

            loss_terms: list[torch.Tensor] = []
            if bool(b_pm.any().item()):
                loss_terms.append(1.4 * F.huber_loss(pred_pot[b_pm], b_pot[b_pm], delta=0.75))
                # Column-wise Pearson correlation on quantitative targets with >=6 measured potencies
                col_cnt = b_pm.sum(dim=0)
                valid_cols = torch.where(col_cnt >= 6)[0]
                if valid_cols.numel() > 0:
                    corr_acc: list[torch.Tensor] = []
                    for ci in valid_cols.tolist():
                        mc = b_pm[:, ci]
                        px = pred_pot[mc, ci]
                        py = b_pot[mc, ci]
                        pyc = py - py.mean()
                        if float(pyc.pow(2).sum().item()) > 1e-4:
                            pxc = px - px.mean()
                            r_c = (pxc * pyc).sum() / ((pxc.pow(2).sum() + 1e-6).sqrt() * (pyc.pow(2).sum() + 1e-6).sqrt())
                            corr_acc.append(1.0 - r_c)
                    if corr_acc:
                        loss_terms.append(0.35 * torch.stack(corr_acc).mean())
            if bool(b_am.any().item()):
                loss_terms.append(1.0 * F.binary_cross_entropy_with_logits(act_logits[b_am], b_act[b_am]))

            if loss_terms:
                loss = torch.stack(loss_terms).sum()
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(sar_params, 1.0)
                    opt.step()
                    ep_losses.append(float(loss.detach().item()))
            sched.step()
        if ep_losses:
            last_loss = float(np.mean(ep_losses))

    return {
        "skipped": False,
        "n_molecules": n_mol,
        "cysdb_loaded": True,
        "final_sar_loss": round(last_loss, 5),
    }


def _stage_schedule_from_config(
    cfg: dict[str, Any],
    epochs_override: list[int] | None = None,
    batch_size_override: int | None = None,
) -> list[tuple[int, str, int, int, float]]:
    curr = cfg.get("curriculum", {})
    stage_defs = [
        (1, "stage1_multiomics_stoichiometry", 5, 64, 3.0e-4),
        (2, "stage2_crispr_survivorship_deconvolution", 5, 64, 2.0e-4),
        (3, "stage3_chemical_perturbation_and_competing_risks", 8, 64, 1.5e-4),
        (4, "stage4_physical_anchors_3d_and_spatial_waves", 4, 32, 5.0e-5),
    ]
    schedule: list[tuple[int, str, int, int, float]] = []
    for idx, (st_num, key, def_ep, def_bs, def_lr) in enumerate(stage_defs):
        sc = curr.get(key, {})
        ep = int(sc.get("epochs", def_ep))
        if epochs_override is not None:
            ep = int(epochs_override[idx] if len(epochs_override) >= 4 else epochs_override[0])
        bs = int(batch_size_override if batch_size_override is not None else sc.get("batch_size", def_bs))
        lr = float(sc.get("lr", def_lr))
        schedule.append((st_num, key, max(1, ep), max(2, bs), lr))
    return schedule


def train_single_seed(
    *,
    seed: int,
    cfg: dict[str, Any],
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    license_track: str,
    model_size: str,
    precision: str,
    device_str: str,
    resume_auto: bool = False,
    epochs_override: list[int] | None = None,
    max_steps_per_stage: int | None = None,
    batch_size_override: int | None = None,
    smoke_test: bool = False,
    ablation_overrides: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Train one `FerroFluxFM` checkpoint across Stages 1..4 and return summary dict."""
    global _SIGUSR1_FLAG

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1 and dist.is_initialized()

    device = _resolve_device(device_str, local_rank=local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    dataset, collate_fn, n_genes = load_training_dataset(
        data_dir=data_dir,
        license_track=license_track,
        smoke_test=smoke_test,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = output_dir / f"ferroflux_{model_size}_{license_track}_seed{seed}.pt"
    resume_ckpt_path = output_dir / f".resume_state_{model_size}_{license_track}_seed{seed}.ckpt"

    if resume_auto and final_ckpt_path.is_file() and not resume_ckpt_path.is_file():
        if rank == 0:
            print(f"[Seed {seed}] Final checkpoint already exists at {final_ckpt_path}; skipping.")
        return {"seed": seed, "checkpoint": str(final_ckpt_path), "status": "already_completed", "history": []}

    ab_kw = dict(ablation_overrides or {})
    ms_low = model_size.strip().lower()
    if ms_low == "compact":
        model = FerroFluxFM(n_genes=n_genes, d_model=192, n_layers=3, n_heads=4, **ab_kw)
    elif ms_low == "v2_login":
        model = FerroFluxFM(n_genes=n_genes, d_model=256, n_layers=4, n_heads=4, **ab_kw)
    else:
        model = FerroFluxFM.from_config(
            config_path=config_path,
            variant=model_size,
            n_genes=n_genes,
            **ab_kw,
        )
    model = model.to(device)

    raw_model: FerroFluxFM = model
    ddp_or_raw: torch.nn.Module = model
    if use_ddp:
        ddp_or_raw = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    loss_fn = FerroFluxCurriculumLoss().to(device)
    grad_clip = float(cfg.get("curriculum", {}).get("grad_clip_norm", 1.0))
    weight_decay = float(cfg.get("curriculum", {}).get("weight_decay", 0.01))

    prec = precision.strip().lower()
    use_bf16 = prec == "bf16" or (prec == "auto" and device.type == "cuda")

    schedule = _stage_schedule_from_config(
        cfg,
        epochs_override=epochs_override,
        batch_size_override=batch_size_override,
    )

    start_stage = 1
    start_epoch = 0
    history: list[dict[str, Any]] = []
    saved_opt_state: dict[str, Any] | None = None

    if resume_auto and resume_ckpt_path.is_file():
        saved = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(saved["model_state_dict"])
        start_stage = int(saved.get("stage", 1))
        start_epoch = int(saved.get("epoch", 0))
        history = list(saved.get("history", []))
        saved_opt_state = saved.get("optimizer_state_dict")
        if rank == 0:
            print(f"[Seed {seed}] Resumed from {resume_ckpt_path} at stage={start_stage}, epoch={start_epoch}")

    t_seed_start = time.perf_counter()

    # Warm-start gene_essentiality_prior & Stage-0.5 SAR chemical representation when starting fresh
    if start_stage == 1 and start_epoch == 0 and not smoke_test:
        if hasattr(dataset, "records") and isinstance(dataset.records, list):
            sum_c = np.zeros(n_genes, dtype=np.float64)
            cnt_c = np.zeros(n_genes, dtype=np.float64)
            for rc in dataset.records:
                if rc.get("kind") == "crispr" and int(rc.get("is_crispr_fin_epistasis", 0)) == 0:
                    gi = int(rc.get("gene_idx", -1))
                    if 0 <= gi < n_genes:
                        sum_c[gi] += float(rc.get("chronos_score", 0.0))
                        cnt_c[gi] += 1.0
            eb_prior = (sum_c / (cnt_c + 3.0)).astype(np.float32)
            with torch.no_grad():
                raw_model.gene_essentiality_prior.copy_(torch.from_numpy(eb_prior).to(device=device))

        sar_info = pretrain_stage05_chem_sar(
            raw_model,
            data_dir=data_dir,
            license_track=license_track,
            device=device,
            epochs=8,
            batch_size=256,
            seed=seed,
            smoke_test=smoke_test,
        )
        if rank == 0 and not sar_info.get("skipped", True):
            print(
                f"[Seed {seed}] Stage-0.5 SAR pretraining done on {sar_info['n_molecules']:,} molecules "
                f"(loss={sar_info['final_sar_loss']})",
                flush=True,
            )

    for stage_num, stage_name, n_epochs, batch_size, stage_lr in schedule:
        if stage_num < start_stage:
            continue

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed) if use_ddp else None
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, max(1, len(dataset))),
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=False,
        )

        dense_p = [p for n, p in ddp_or_raw.named_parameters() if not n.endswith("gene_essentiality_prior")]
        prior_p = [p for n, p in ddp_or_raw.named_parameters() if n.endswith("gene_essentiality_prior")]
        param_groups = [{"params": dense_p, "lr": stage_lr, "weight_decay": weight_decay}]
        if prior_p:
            param_groups.append({"params": prior_p, "lr": max(stage_lr * 30.0, 1.5e-2), "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(param_groups)
        if stage_num == start_stage and saved_opt_state is not None:
            try:
                optimizer.load_state_dict(saved_opt_state)
            except ValueError:
                pass
            saved_opt_state = None

        steps_per_ep = min(len(loader), max_steps_per_stage) if max_steps_per_stage is not None else len(loader)
        total_steps = max(1, n_epochs * max(1, steps_per_ep))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=stage_lr * 0.1)

        ep_begin = start_epoch if stage_num == start_stage else 0
        for ep in range(ep_begin, n_epochs):
            if sampler is not None:
                sampler.set_epoch(ep)

            ddp_or_raw.train()
            ep_losses: list[float] = []
            t_ep = time.perf_counter()

            for step_idx, raw_batch in enumerate(loader):
                if max_steps_per_stage is not None and step_idx >= max_steps_per_stage:
                    break

                batch = _move_batch_to_device(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    outputs = ddp_or_raw(batch)
                    loss_dict = loss_fn(outputs, batch, stage=stage_num)
                    loss = loss_dict["loss"]

                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ddp_or_raw.parameters(), grad_clip)
                    optimizer.step()
                    ep_losses.append(float(loss.detach().item()))
                scheduler.step()

                if _SIGUSR1_FLAG:
                    if rank == 0:
                        tmp_res = resume_ckpt_path.with_suffix(".tmp")
                        torch.save(
                            {
                                "model_state_dict": raw_model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "stage": stage_num,
                                "epoch": ep,
                                "history": history,
                                "seed": seed,
                            },
                            tmp_res,
                        )
                        tmp_res.replace(resume_ckpt_path)
                        print(f"[Seed {seed}] Flushed SIGUSR1 preemption state to {resume_ckpt_path}", flush=True)
                    if use_ddp:
                        dist.barrier()
                    sys.exit(0)

            mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
            rec = {
                "seed": seed,
                "license_track": license_track,
                "model_size": model_size,
                "stage": stage_num,
                "stage_name": stage_name,
                "epoch": ep + 1,
                "mean_loss": round(mean_loss, 6),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
                "epoch_sec": round(time.perf_counter() - t_ep, 3),
            }
            history.append(rec)

            if rank == 0 and resume_auto:
                tmp_res = resume_ckpt_path.with_suffix(".tmp")
                torch.save(
                    {
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "stage": stage_num if (ep + 1) < n_epochs else stage_num + 1,
                        "epoch": (ep + 1) if (ep + 1) < n_epochs else 0,
                        "history": history,
                        "seed": seed,
                    },
                    tmp_res,
                )
                tmp_res.replace(resume_ckpt_path)

    total_sec = time.perf_counter() - t_seed_start

    if rank == 0:
        ckpt_payload = {
            "model_state_dict": raw_model.state_dict(),
            "model_init_kwargs": {
                "n_genes": raw_model.n_genes,
                "d_model": raw_model.d_model,
                "n_layers": raw_model.n_layers,
                "n_heads": raw_model.n_heads,
                "n_modules": raw_model.n_modules,
                "chem_fp_dim": raw_model.chem_fp_dim,
                "chem_desc_dim": raw_model.chem_desc_dim,
                "epigenomic_dim": raw_model.epigenomic_dim,
                "ablate_diff_ferro_gem": raw_model.ablate_diff_ferro_gem,
                "ablate_sec46_gate": raw_model.ablate_sec46_gate,
                "ablate_survivorship": raw_model.ablate_survivorship,
                "ablate_chem_prior": raw_model.ablate_chem_prior,
                "ablate_physical_anchors": raw_model.ablate_physical_anchors,
                "ablate_cis_reg_slot": raw_model.ablate_cis_reg,
                "ablate_aphynity_residual": raw_model.ablate_aphynity_residual,
                "ablate_sar_pretrain": raw_model.ablate_sar_pretrain,
            },
            "model_size": model_size,
            "license_track": license_track,
            "seed": seed,
            "n_genes": n_genes,
            "total_train_sec": round(total_sec, 3),
            "final_loss": history[-1]["mean_loss"] if history else None,
        }
        tmp_final = final_ckpt_path.with_suffix(".tmp_pt")
        torch.save(ckpt_payload, tmp_final)
        tmp_final.replace(final_ckpt_path)
        if resume_ckpt_path.is_file():
            resume_ckpt_path.unlink()

    if use_ddp:
        dist.barrier()

    return {
        "seed": seed,
        "checkpoint": str(final_ckpt_path),
        "status": "ok",
        "total_train_sec": round(total_sec, 3),
        "final_loss": history[-1]["mean_loss"] if history else None,
        "history": history,
    }


def _parallel_seed_worker(kwargs: dict[str, Any]) -> dict[str, Any]:
    return train_single_seed(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FerroFlux-FM 4-Stage Mechanistic Curriculum Trainer")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML training config path")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Processed Parquet directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output checkpoint directory")
    parser.add_argument("--license-track", type=str, default="open", choices=["open", "academic"])
    parser.add_argument("--model-size", type=str, default="base", choices=["compact", "v2_login", "v3", "base", "xl"])
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "bf16"])
    parser.add_argument("--resume-auto", action="store_true", help="Auto-resume from preemption state")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs-per-stage", type=int, nargs="+", default=None)
    parser.add_argument("--max-steps-per-stage", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--parallel-seeds", type=int, default=1, help="Concurrent CPU seed workers")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test without Parquet shards")
    args = parser.parse_args(argv)

    signal.signal(signal.SIGUSR1, _handle_sigusr1)
    if "OMP_NUM_THREADS" not in os.environ and (os.cpu_count() or 1) > 16:
        torch.set_num_threads(16)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if (torch.cuda.is_available() and args.device != "cpu") else "gloo"
        dist.init_process_group(backend=backend)

    # Mandatory pre-flight zero-leakage quarantine & inode cap check on real data
    if not args.smoke_test:
        if rank == 0:
            audit_parquet_directory(args.data_dir)
        if world_size > 1:
            dist.barrier()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.is_file() else {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    worker_jobs = [
        {
            "seed": int(s),
            "cfg": cfg,
            "config_path": args.config,
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "license_track": args.license_track,
            "model_size": args.model_size,
            "precision": args.precision,
            "device_str": args.device,
            "resume_auto": bool(args.resume_auto),
            "epochs_override": args.epochs_per_stage,
            "max_steps_per_stage": args.max_steps_per_stage,
            "batch_size_override": args.batch_size,
            "smoke_test": bool(args.smoke_test),
        }
        for s in args.seeds
    ]

    can_parallel = (
        args.parallel_seeds > 1
        and world_size == 1
        and _resolve_device(args.device).type == "cpu"
        and len(worker_jobs) > 1
    )
    if can_parallel:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(args.parallel_seeds, len(worker_jobs))) as pool:
            results = pool.map(_parallel_seed_worker, worker_jobs)
    else:
        results = [train_single_seed(**job) for job in worker_jobs]

    if rank == 0:
        all_hist = [row for r in results for row in r.get("history", [])]
        if all_hist:
            curves_csv = args.output_dir / f"training_curves_{args.model_size}_{args.license_track}.csv"
            pd.DataFrame(all_hist).to_csv(curves_csv, index=False)

        summary_json = args.output_dir / f"training_summary_{args.model_size}_{args.license_track}.json"
        summary_payload = {
            "status": "OK",
            "model_size": args.model_size,
            "license_track": args.license_track,
            "seeds": [r["seed"] for r in results],
            "checkpoints": [r["checkpoint"] for r in results],
            "final_losses": {str(r["seed"]): r.get("final_loss") for r in results},
        }
        summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(json.dumps(summary_payload, indent=2))

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
