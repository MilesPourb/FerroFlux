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
import copy
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
    max_samples_per_shard: int | None = 200000,
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
    if "max_samples_per_shard" in sig.parameters and max_samples_per_shard is not None:
        kw["max_samples_per_shard"] = int(max_samples_per_shard)

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
    train_dataset: Any = None,
    train_indices: list[int] | None = None,
    forbid_test_set: set[int] | None = None,
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
    pot_t = torch.from_numpy(sar_ds.potency_target.copy())
    pot_m_t = torch.from_numpy(sar_ds.potency_mask.copy())
    act_t = torch.from_numpy(sar_ds.active_target.copy())
    act_m_t = torch.from_numpy(sar_ds.active_mask.copy())
    rng = np.random.default_rng(seed)

    # On the open track (where CC BY-NC FerrDb labels in cols 19-20 are absent/sparse), preserve quantitative
    # ChEMBL/PubChem potency (`pot_t[:, 19]`) untouched and supervise binary `act_t[:, 19:21]` strictly from
    # open-licensed PubChem qHTS assays (`HT1080`, `ALOX5/12/15`, `NFE2L2`) + disjoint in-split `train_indices`
    # ferroptosis modulators (`0` held-out SMILES, `ik14`, Murcko scaffold, or multi-drug `smiles_b` overlap).
    if act_m_t.shape[1] > 20 and float(act_m_t[:, 20].sum().item()) < 5.0:
        killer_cols = [0, 1, 2, 3, 7, 11, 17, 18, 19]
        is_killer = (act_t[:, killer_cols] * act_m_t[:, killer_cols]).max(dim=1).values > 0.5
        is_alox_nrf2 = ((act_t[:, 4:7] * act_m_t[:, 4:7]).max(dim=1).values > 0.5) | (
            (act_t[:, 14] * act_m_t[:, 14]) > 0.5
        )
        is_rta_chel_safe = ((desc_t[:, 7] > 0.0) | (desc_t[:, 8] > 0.0)) & (act_m_t[:, 17] > 0.5) & (act_t[:, 17] < 0.5)
        pos_idx_20 = np.where(((is_alox_nrf2 | is_rta_chel_safe) & (~is_killer)).numpy())[0]
        neg_idx_20 = np.where(is_killer.numpy())[0]
        if len(pos_idx_20) > 450:
            pos_idx_20 = np.sort(rng.choice(pos_idx_20, size=450, replace=False))
        if len(neg_idx_20) > 480:
            neg_idx_20 = np.sort(rng.choice(neg_idx_20, size=480, replace=False))
        if len(pos_idx_20) > 0:
            act_t[pos_idx_20, 20] = 1.0
            act_m_t[pos_idx_20, 20] = 1.0
        if len(neg_idx_20) > 0:
            act_t[neg_idx_20, 20] = 0.0
            act_m_t[neg_idx_20, 20] = 1.0

        ht_pos = np.where((((act_t[:, 17] * act_m_t[:, 17]) > 0.5) | ((act_t[:, 18] * act_m_t[:, 18]) > 0.5)).numpy())[0]
        ht_neg = np.where(((act_m_t[:, 17] > 0.5) & (act_t[:, 17] < 0.5) & (~is_killer) & (act_m_t[:, 19] < 0.5)).numpy())[0]
        if len(ht_pos) > 480:
            ht_pos = np.sort(rng.choice(ht_pos, size=480, replace=False))
        if len(ht_neg) > 360:
            ht_neg = np.sort(rng.choice(ht_neg, size=360, replace=False))
        if len(ht_pos) > 0:
            only_unset = (act_m_t[ht_pos, 19] < 0.5).numpy()
            if bool(only_unset.any()):
                act_t[ht_pos[only_unset], 19] = 1.0
                act_m_t[ht_pos[only_unset], 19] = 1.0
        if len(ht_neg) > 0:
            act_t[ht_neg, 19] = 0.0
            act_m_t[ht_neg, 19] = 1.0

        if train_dataset is not None and train_indices is not None and hasattr(train_dataset, "records"):
            from ferroflux.data.dataset import get_fast_indexer

            if forbid_test_set:
                assert forbid_test_set.isdisjoint(set(train_indices)), "FATAL: test index in Stage-0.5 train_indices!"
            idxr_tr = get_fast_indexer(train_dataset)
            recs_tr = train_dataset.records
            cmpd_stats: dict[int, list[Any]] = {}
            for ti in train_indices:
                ti_i = int(ti)
                rec = recs_tr[ti_i]
                if rec.get("kind") != "pharmaco" or int(rec.get("is_ferroptosis_compound", 1)) == 0:
                    continue
                if _record_hits_forbidden_chemistry(
                    rec, sar_ds.forbid_smiles, sar_ds.forbid_ik14, sar_ds.forbid_scaffolds
                ):
                    continue
                cid = int(idxr_tr.row_chem_a_idx[ti_i])
                if cid <= 0:
                    continue
                rcd_m = str(rec.get("rcd_modality") or "").lower()
                yv = float(idxr_tr.viability_target[ti_i])
                if cid not in cmpd_stats:
                    cmpd_stats[cid] = [rcd_m, [yv]]
                else:
                    cmpd_stats[cid][1].append(yv)
            if cmpd_stats:
                ex_fps, ex_descs, ex_pots, ex_pms, ex_acts, ex_ams = [], [], [], [], [], []
                n_cols = int(act_t.shape[1])
                for cid, (rcd_m, _y_list) in sorted(cmpd_stats.items()):
                    is_resc = "rescuer" in rcd_m
                    fp_row = (idxr_tr.chem_fp_table[cid, : fp_t.shape[1]] > 0.5).astype(np.uint8)
                    ds_row = idxr_tr.chem_desc_table[cid, : desc_t.shape[1]].astype(np.float32)
                    p_row = np.zeros(n_cols, dtype=np.float32)
                    pm_row = np.zeros(n_cols, dtype=np.float32)
                    a_row = np.zeros(n_cols, dtype=np.float32)
                    am_row = np.zeros(n_cols, dtype=np.float32)
                    a_row[19], am_row[19] = (0.0 if is_resc else 1.0), 1.0
                    a_row[20], am_row[20] = (1.0 if is_resc else 0.0), 1.0
                    for _ in range(12):
                        ex_fps.append(fp_row)
                        ex_descs.append(ds_row)
                        ex_pots.append(p_row)
                        ex_pms.append(pm_row)
                        ex_acts.append(a_row)
                        ex_ams.append(am_row)
                fp_t = torch.cat([fp_t, torch.from_numpy(np.stack(ex_fps, axis=0))], dim=0)
                desc_t = torch.cat([desc_t, torch.from_numpy(np.stack(ex_descs, axis=0))], dim=0)
                pot_t = torch.cat([pot_t, torch.from_numpy(np.stack(ex_pots, axis=0))], dim=0)
                pot_m_t = torch.cat([pot_m_t, torch.from_numpy(np.stack(ex_pms, axis=0))], dim=0)
                act_t = torch.cat([act_t, torch.from_numpy(np.stack(ex_acts, axis=0))], dim=0)
                act_m_t = torch.cat([act_m_t, torch.from_numpy(np.stack(ex_ams, axis=0))], dim=0)
                n_mol = int(fp_t.shape[0])

    sar_modules = [
        model.perturbation_op.chem_fp_proj,
        model.perturbation_op.chem_desc_proj,
        model.perturbation_op.chem_in_proj,
        model.perturbation_op.chem_blocks,
        model.perturbation_op.chem_out_norm,
        model.perturbation_op.chem_mech_head,
        model.perturbation_op.sar_pretrain_head,
    ]
    sar_params = [p for mod in sar_modules for p in mod.parameters() if p.requires_grad]
    if model.perturbation_op.raw_sar_cell_offset.requires_grad:
        sar_params.append(model.perturbation_op.raw_sar_cell_offset)
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

    with torch.no_grad():
        n_sar_cols = int(pot_m_t.shape[1])
        pot_obs_cnt = (pot_m_t > 0.5).sum(dim=0)
        act_obs_cnt = (act_m_t > 0.5).sum(dim=0)
        for j_col in range(min(n_sar_cols, model.perturbation_op.n_sar_targets)):
            if int(pot_obs_cnt[j_col].item()) < 5:
                model.perturbation_op.sar_pretrain_head.weight[j_col].zero_()
                model.perturbation_op.sar_pretrain_head.bias[j_col].zero_()
            act_row = model.perturbation_op.n_sar_targets + j_col
            if act_row < model.perturbation_op.sar_pretrain_head.out_features and int(act_obs_cnt[j_col].item()) < 5:
                model.perturbation_op.sar_pretrain_head.weight[act_row].zero_()
                model.perturbation_op.sar_pretrain_head.bias[act_row].zero_()

    model.perturbation_op.snapshot_stage05_sar_anchor()
    return {
        "skipped": False,
        "n_molecules": n_mol,
        "cysdb_loaded": True,
        "final_sar_loss": round(last_loss, 5),
        "sar_replay_tensors": (fp_t, desc_t, pot_t, pot_m_t, act_t, act_m_t),
    }


def compute_sar_replay_loss(
    model: FerroFluxFM,
    sar_replay_tensors: tuple[torch.Tensor, ...] | None,
    device: torch.device,
    rng: np.random.Generator,
    batch_size: int = 128,
) -> torch.Tensor | None:
    """Compute leak-free Stage-0.5 SAR replay loss so cellular fine-tuning never forgets unseen chemotypes."""
    if sar_replay_tensors is None or model.ablate_sar_pretrain or model.ablate_chem_prior:
        return None
    fp_t, desc_t, pot_t, pot_m_t, act_t, act_m_t = sar_replay_tensors
    n_mol = int(fp_t.shape[0])
    if n_mol < 16:
        return None
    idx = torch.from_numpy(rng.integers(0, n_mol, size=min(batch_size, n_mol)))
    b_fp = fp_t[idx].to(device=device, dtype=torch.float32)
    b_desc = desc_t[idx].to(device=device, dtype=torch.float32)
    b_pot = pot_t[idx].to(device=device, dtype=torch.float32)
    b_pm = pot_m_t[idx].to(device=device, dtype=torch.float32) > 0.5
    b_act = act_t[idx].to(device=device, dtype=torch.float32)
    b_am = act_m_t[idx].to(device=device, dtype=torch.float32) > 0.5

    sar_out = model.perturbation_op.predict_sar_targets(b_fp, b_desc)
    terms: list[torch.Tensor] = []
    if bool(b_pm.any().item()):
        terms.append(1.2 * F.huber_loss(sar_out["pred_potency"][b_pm], b_pot[b_pm], delta=0.75))
    if bool(b_am.any().item()):
        terms.append(0.8 * F.binary_cross_entropy_with_logits(sar_out["active_logits"][b_am], b_act[b_am]))
    return torch.stack(terms).sum() if terms else None


def _solve_ridge_np(x: np.ndarray, y: np.ndarray, lam: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form centered Ridge regression returning `(weight_out_by_in, bias_out)`."""
    xm = x.mean(axis=0, keepdims=True).astype(np.float64)
    ym = y.mean(axis=0, keepdims=True).astype(np.float64)
    xc = (x - xm).astype(np.float64)
    yc = (y - ym).astype(np.float64)
    d = xc.shape[1]
    gram = xc.T @ xc + np.eye(d, dtype=np.float64) * float(lam)
    w = np.linalg.solve(gram, xc.T @ yc)  # (d, out)
    b = (ym - xm @ w).reshape(-1)
    return w.T.astype(np.float32), b.astype(np.float32)


def _solve_zero_bias_ridge_np(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None, lam: float = 140.0) -> np.ndarray:
    """Closed-form zero-intercept Ridge regression returning `weight_out_by_in` (`0.0` when `x == 0`)."""
    xd = x.astype(np.float64)
    yd = y.astype(np.float64)
    d_in = xd.shape[1]
    d_out = yd.shape[1]
    w_out = np.zeros((d_out, d_in), dtype=np.float32)
    eye = np.eye(d_in, dtype=np.float64) * float(lam)
    if mask is None:
        if len(xd) >= 8:
            w = np.linalg.solve(xd.T @ xd + eye, xd.T @ yd)
            w_out = w.T.astype(np.float32)
        return w_out
    for j in range(d_out):
        mj = mask[:, j] > 0.5
        if int(mj.sum()) >= 8:
            xj = xd[mj]
            yj = yd[mj, j]
            wj = np.linalg.solve(xj.T @ xj + eye, xj.T @ yj)
            w_out[j] = wj.astype(np.float32)
    return w_out


def _record_hits_forbidden_chemistry(
    rec: dict[str, Any],
    forbid_smiles: set[str] | None,
    forbid_ik14: set[str] | None,
    forbid_scaffolds: set[str] | None,
) -> bool:
    """Return True if any single- or multi-drug chemical field in `rec` hits held-out chemistry."""
    if not forbid_smiles and not forbid_ik14 and not forbid_scaffolds:
        return False
    from ferroflux.data.dataset import canonicalize_smiles_and_keys

    f_s = forbid_smiles or set()
    f_ik = forbid_ik14 or set()
    f_sc = forbid_scaffolds or set()
    for k_s in ("canonical_smiles", "fin_modifier_smiles", "smiles_a", "smiles_b"):
        sv = str(rec.get(k_s, "") or "").strip()
        if not sv or sv.lower() == "nan":
            continue
        if sv in f_s:
            return True
        c_s, ik_s, sc_s = canonicalize_smiles_and_keys(sv)
        if (c_s and c_s in f_s) or (ik_s and ik_s in f_ik) or (sc_s and sc_s in f_sc):
            return True
    for k_ik in ("ik14", "ik14_a", "ik14_b"):
        iv = str(rec.get(k_ik, "") or "").strip()[:14]
        if iv and iv.lower() != "nan" and iv in f_ik:
            return True
    for k_sc in ("murcko_scaffold", "murcko_scaffold_a", "murcko_scaffold_b"):
        scv = str(rec.get(k_sc, "") or "").strip()
        if scv and scv.lower() != "nan" and scv in f_sc:
            return True
    return False


def _compute_standardized_extra_ctx_np(
    idxr: Any,
    cids: list[int] | np.ndarray,
    ctx_mean: np.ndarray,
    ctx_scale: np.ndarray,
    ablate_physical_anchors: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build mask-aware standardized `[N, 234]` multi-omic context matching `FerroFluxFM.forward()`."""
    n_c = len(cids)
    if ablate_physical_anchors:
        z = np.zeros((n_c, 234), dtype=np.float32)
        return z, z.copy()
    gen = idxr.cell_genomic[cids].astype(np.float32)
    gen_m = (np.abs(gen).sum(axis=1, keepdims=True) > 1e-6).astype(np.float32) * np.ones_like(gen)
    p64_m = idxr.cell_prot64_mask[cids].astype(np.float32)
    p64 = idxr.cell_prot64[cids].astype(np.float32) * p64_m
    l32_m = idxr.cell_lipid32_mask[cids].astype(np.float32)
    l32 = idxr.cell_lipid32[cids].astype(np.float32) * l32_m
    raw_ctx = np.concatenate([gen, p64, l32], axis=1)
    ctx_m = np.concatenate([gen_m, p64_m, l32_m], axis=1)
    sc = np.maximum(ctx_scale[None, :], 2e-2)
    std_ctx = np.clip(((raw_ctx - ctx_mean[None, :]) / sc) * ctx_m, -6.0, 6.0).astype(np.float32)
    return std_ctx, ctx_m.astype(np.float32)


def calibrate_mgm_decoder_from_train_indices(
    model: FerroFluxFM,
    dataset: Any,
    train_indices: list[int],
    *,
    forbid_test_set: set[int] | None = None,
    device: torch.device,
    seed: int = 101,
) -> None:
    """Calibrate affine-equivariant row-z-score `mgm_svd_proj` + `mgm_ctx_decoder` strictly on `train_indices` (`0` test or Stage-0 exposure)."""
    if forbid_test_set:
        assert forbid_test_set.isdisjoint(set(train_indices)), "FATAL: test index present in MGM calibration!"
    from ferroflux.data.dataset import get_fast_indexer
    from sklearn.decomposition import TruncatedSVD

    idxr = get_fast_indexer(dataset)
    if idxr is None or not hasattr(dataset, "records") or model.n_genes < 128:
        return

    rng = np.random.default_rng(seed + 31)
    n_genes = model.n_genes
    recs = dataset.records
    tr_arr = np.asarray(train_indices, dtype=np.int64)

    pmap = model._pathway_idx_map
    sterol_req = pmap.get("sterol_7dhc_mask", pmap.get("sterol_up", []) + pmap.get("sterol_down", []))
    basal_cids = sorted(
        {
            int(idxr.row_cell_idx[int(ti)])
            for ti in tr_arr
            if recs[int(ti)].get("kind") == "basal"
        }
    )
    cand_cids = basal_cids if len(basal_cids) >= 64 else np.unique(idxr.row_cell_idx[tr_arr]).tolist()
    valid_cids = [
        int(ci)
        for ci in cand_cids
        if float(idxr.cell_mask[int(ci)].sum()) >= 256.0
        and (not sterol_req or float(idxr.cell_mask[int(ci), sterol_req].min()) > 0.5)
    ]
    if len(valid_cids) < 24:
        valid_cids = [int(ci) for ci in np.unique(idxr.row_cell_idx[tr_arr]) if float(idxr.cell_mask[int(ci)].sum()) >= 64.0]
    if len(valid_cids) < 24:
        return

    xc_base = idxr.cell_expr[valid_cids].astype(np.float32)
    mc_base = idxr.cell_mask[valid_cids].astype(np.float32)
    n0 = len(xc_base)
    xm_base = np.concatenate([xc_base, xc_base, xc_base], axis=0)
    mm_base = np.concatenate([mc_base, mc_base, mc_base], axis=0)

    sting_idx = sorted(set(pmap.get("cgas_sting", []) + pmap.get("pdl1_ifn", [])))
    sterol_idx = sorted(set(sterol_req))
    km_mat = np.ones_like(mm_base)
    if sting_idx:
        km_mat[:n0, sting_idx] = 0.0
    if sterol_idx:
        km_mat[n0 : 2 * n0, sterol_idx] = 0.0
    km_mat *= (rng.random(km_mat.shape) >= 0.15).astype(np.float32)

    eff_m = mm_base * km_mat
    eff_cnt = np.maximum(eff_m.sum(axis=1, keepdims=True), 1.0)
    xm_obs_mean = (xm_base * eff_m).sum(axis=1, keepdims=True) / eff_cnt
    xm_obs_var = (((xm_base - xm_obs_mean) * eff_m) ** 2).sum(axis=1, keepdims=True) / eff_cnt
    xm_obs_std = np.sqrt(xm_obs_var + 1e-4)
    xz_obs = (((xm_base - xm_obs_mean) / xm_obs_std) * eff_m).astype(np.float32)
    yz_full = ((xm_base - xm_obs_mean) / xm_obs_std).astype(np.float32)

    k_svd = min(256, xz_obs.shape[0] - 1, n_genes - 1)
    if k_svd < 16:
        return
    svd_m = TruncatedSVD(n_components=k_svd, random_state=seed + 5)
    z_svd = svd_m.fit_transform(xz_obs)
    col_sc = np.maximum(np.std(z_svd, axis=0, keepdims=True), 1e-2).T
    w_svd = (svd_m.components_ / col_sc).astype(np.float32)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model.mgm_svd_proj.weight.zero_()
        model.mgm_svd_proj.weight[:k_svd].copy_(torch.from_numpy(w_svd).to(device=device))

        mgm_lat_chunks: list[np.ndarray] = []
        trunk_rel_chunks: list[np.ndarray] = []
        for s_i in range(0, len(xm_base), 256):
            sl_r = slice(s_i, min(len(xm_base), s_i + 256))
            xb_t = torch.from_numpy(xm_base[sl_r]).to(device=device)
            mb_t = torch.from_numpy(mm_base[sl_r]).to(device=device)
            kb_t = torch.from_numpy(km_mat[sl_r]).to(device=device)
            xz_t = torch.from_numpy(xz_obs[sl_r]).to(device=device)
            t_out = model.perceiver_trunk(xb_t, expr_mask=mb_t, mgm_keep_mask=kb_t)
            tr_al = t_out["mgm_recon"]
            tr_rel = (tr_al - tr_al.mean(dim=-1, keepdim=True)) / (tr_al.std(dim=-1, keepdim=True) + 1.0)
            mgm_lat_chunks.append(model.mgm_svd_proj(xz_t).cpu().numpy())
            trunk_rel_chunks.append(tr_rel.cpu().numpy())

        mgm_lat_all = np.concatenate(mgm_lat_chunks, axis=0)
        trunk_rel_all = np.concatenate(trunk_rel_chunks, axis=0)
        y_res_z = yz_full - 0.01 * trunk_rel_all
        w_mgm, b_mgm = _solve_ridge_np(mgm_lat_all, y_res_z, lam=180.0)
        model.mgm_ctx_decoder.weight.zero_()
        model.mgm_ctx_decoder.weight[:, model.d_model :].copy_(torch.from_numpy(w_mgm).to(device=device))
        model.mgm_ctx_decoder.bias.copy_(torch.from_numpy(b_mgm).to(device=device))
    if was_training:
        model.train()


def warmstart_from_train_indices(
    model: FerroFluxFM,
    dataset: Any,
    train_indices: list[int],
    *,
    forbid_test_set: set[int] | None = None,
    forbid_smiles: set[str] | None = None,
    forbid_ik14: set[str] | None = None,
    forbid_scaffolds: set[str] | None = None,
    device: torch.device,
    seed: int = 101,
) -> None:
    """Strictly `train_indices`-only spectral, physical-anchor, chemical-scaler & EM-SVD CRISPR warm-start (`0` test exposure)."""
    if forbid_test_set:
        assert forbid_test_set.isdisjoint(set(train_indices)), "FATAL: test index present in warm-start train_indices!"
    from ferroflux.data.dataset import get_fast_indexer
    from ferroflux.layers.perceiver_trunk import METABOLITE_TO_IDX
    from sklearn.decomposition import TruncatedSVD

    idxr = get_fast_indexer(dataset)
    if idxr is None or not hasattr(dataset, "records"):
        return

    rng = np.random.default_rng(seed)
    n_genes = model.n_genes
    recs = dataset.records
    tr_arr = np.asarray(train_indices, dtype=np.int64)

    # 1. Row-standardized transcriptome spectral basis (`expr_ctx_proj[0]`) & `anchor_mrna_ref` strictly from `train_indices`
    tr_cell_ids = np.unique(idxr.row_cell_idx[tr_arr])
    valid_cids = [int(ci) for ci in tr_cell_ids if float(idxr.cell_mask[int(ci)].sum()) > 0.0]
    icb_norm_cids = list(
        {
            int(idxr.row_cell_idx[int(ti)])
            for ti in tr_arr[: min(len(tr_arr), 250000)]
            if recs[int(ti)].get("kind") in ("icb", "normal_sfti")
            and float(idxr.cell_mask[int(idxr.row_cell_idx[int(ti)])].sum()) > 0.0
        }
    )
    if len(valid_cids) >= 24:
        xv_raw = idxr.cell_expr[valid_cids]
        mv_raw = idxr.cell_mask[valid_cids]
        cnt_v = np.maximum(mv_raw.sum(axis=1, keepdims=True), 1.0)
        xm_v = (xv_raw * mv_raw).sum(axis=1, keepdims=True) / cnt_v
        xvar_v = (((xv_raw - xm_v) * mv_raw) ** 2).sum(axis=1, keepdims=True) / cnt_v
        xrel_v = (((xv_raw - xm_v) / np.sqrt(xvar_v + 0.25)) * mv_raw).astype(np.float32)

        anc_idx_np = model.perceiver_trunk.anchor_gene_idx.detach().cpu().numpy()
        anc_m_v = mv_raw[:, anc_idx_np]
        anc_cnt = np.maximum(anc_m_v.sum(axis=0), 1.0)
        anc_ref = ((xrel_v[:, anc_idx_np] * anc_m_v).sum(axis=0) / anc_cnt).astype(np.float32)
        anc_basal = np.clip(0.55 * ((xv_raw[:, anc_idx_np] * anc_m_v).sum(axis=0) / anc_cnt), 0.15, 6.5).astype(np.float32)
        with torch.no_grad():
            model.perceiver_trunk.anchor_mrna_ref.copy_(torch.from_numpy(anc_ref).to(device=device))
            model.perceiver_trunk.anchor_mrna_log_basal.copy_(torch.from_numpy(anc_basal).to(device=device))

        if not model.ablate_physical_anchors:
            gen_v = idxr.cell_genomic[valid_cids].astype(np.float32)
            gen_mv = (np.abs(gen_v).sum(axis=1, keepdims=True) > 1e-6).astype(np.float32) * np.ones_like(gen_v)
            p64_mv = idxr.cell_prot64_mask[valid_cids].astype(np.float32)
            l32_mv = idxr.cell_lipid32_mask[valid_cids].astype(np.float32)
            raw_ctx_v = np.concatenate(
                [gen_v, idxr.cell_prot64[valid_cids] * p64_mv, idxr.cell_lipid32[valid_cids] * l32_mv], axis=1
            )
            ctx_mv = np.concatenate([gen_mv, p64_mv, l32_mv], axis=1)
            cnt_ctx = np.maximum(ctx_mv.sum(axis=0), 1.0)
            mu_ctx = ((raw_ctx_v * ctx_mv).sum(axis=0) / cnt_ctx).astype(np.float32)
            var_ctx = ((((raw_ctx_v - mu_ctx[None, :]) * ctx_mv) ** 2).sum(axis=0) / cnt_ctx).astype(np.float32)
            sd_ctx = np.where(np.sqrt(var_ctx) < 2e-2, 1.0, np.sqrt(var_ctx)).astype(np.float32)
            with torch.no_grad():
                model.ctx_obs_mean.copy_(torch.from_numpy(mu_ctx).to(device=device))
                model.ctx_obs_scale.copy_(torch.from_numpy(sd_ctx).to(device=device))

            pl_std_v = np.clip(
                ((raw_ctx_v[:, 138:] - mu_ctx[None, 138:]) / np.maximum(sd_ctx[None, 138:], 2e-2)) * ctx_mv[:, 138:],
                -6.0,
                6.0,
            ).astype(np.float32)
            model.eval()
            with torch.no_grad():
                plog_mrna_chunks: list[np.ndarray] = []
                mlog_mrna_chunks: list[np.ndarray] = []
                for s_i in range(0, len(valid_cids), 256):
                    sl_c = valid_cids[s_i : s_i + 256]
                    xb_t = torch.from_numpy(idxr.cell_expr[sl_c]).to(device=device)
                    mb_t = torch.from_numpy(idxr.cell_mask[sl_c]).to(device=device)
                    epi_t = torch.from_numpy(
                        idxr.cell_cis_reg[sl_c] * idxr.cell_cis_mask[sl_c][..., None]
                    ).to(device=device, dtype=xb_t.dtype)
                    cis_tok = model.cis_reg_slot(epi_t, batch_size=len(sl_c), device=device, dtype=xb_t.dtype)
                    tr_o = model.perceiver_trunk(xb_t, expr_mask=mb_t, cis_reg_tokens=cis_tok)
                    plog_mrna_chunks.append(tr_o["p_hat_log"].cpu().numpy())
                    mlog_mrna_chunks.append(tr_o["m0_hat_log"].cpu().numpy())
                plog_mrna_v = np.concatenate(plog_mrna_chunks, axis=0)
                mlog_mrna_v = np.concatenate(mlog_mrna_chunks, axis=0)

                prot_res_y = idxr.cell_prot_target[valid_cids] - plog_mrna_v
                prot_m_v = idxr.cell_prot_mask[valid_cids]
                w_prot = _solve_zero_bias_ridge_np(pl_std_v, prot_res_y, mask=prot_m_v, lam=140.0)
                model.multiomic_to_prot_anchor.weight.copy_(torch.from_numpy(w_prot).to(device=device))

                met_keys_9 = (
                    "GSH",
                    "GSSG",
                    "NADP+",
                    "putrescine",
                    "spermidine",
                    "PC_PUFA2",
                    "PE_PUFA",
                    "ePL_PUFA",
                    "MUFA_PL",
                )
                met_idx_9 = [METABOLITE_TO_IDX[mk] for mk in met_keys_9]
                met_res_y = idxr.cell_metab_target[valid_cids, :9] - mlog_mrna_v[:, met_idx_9]
                met_m_v = idxr.cell_metab_mask[valid_cids, :9]
                w_met9 = _solve_zero_bias_ridge_np(pl_std_v, met_res_y, mask=met_m_v, lam=140.0)
                model.multiomic_to_metab_anchor.weight.zero_()
                for j_loc, m_col in enumerate(met_idx_9):
                    model.multiomic_to_metab_anchor.weight[m_col].copy_(torch.from_numpy(w_met9[j_loc]).to(device=device))

    if len(valid_cids) >= 32 and n_genes >= 128:
        aug_cids = valid_cids + icb_norm_cids * 6
        x_raw = idxr.cell_expr[aug_cids]
        m_raw = idxr.cell_mask[aug_cids]
        cnt_m = np.maximum(m_raw.sum(axis=1, keepdims=True), 1.0)
        x_mean = (x_raw * m_raw).sum(axis=1, keepdims=True) / cnt_m
        x_var = (((x_raw - x_mean) * m_raw) ** 2).sum(axis=1, keepdims=True) / cnt_m
        x_rel = (((x_raw - x_mean) / np.sqrt(x_var + 0.25)) * m_raw).astype(np.float32)

        k_expr = min(128, x_rel.shape[0] - 1, n_genes - 1)
        if k_expr >= 16:
            svd_x = TruncatedSVD(n_components=k_expr, random_state=seed)
            z_svd = svd_x.fit_transform(x_rel)
            col_scale = np.maximum(np.std(z_svd, axis=0, keepdims=True), 1e-2).T  # (k_expr, 1)
            w_proj = (svd_x.components_ / col_scale).astype(np.float32)
            with torch.no_grad():
                lin0 = model.expr_ctx_proj[0]
                lin0.weight.zero_()
                lin0.bias.zero_()
                lin0.weight[:k_expr].copy_(torch.from_numpy(w_proj).to(device=device))

    # 2. Fit observed-only multi-omic cell standardization stats (`full_cell_dim`) strictly on `train_indices`
    if len(valid_cids) >= 24:
        ctx_m_np = model.ctx_obs_mean.detach().cpu().numpy()
        ctx_s_np = model.ctx_obs_scale.detach().cpu().numpy()
        ex_v_np, ex_m_v_np = _compute_standardized_extra_ctx_np(
            idxr, valid_cids, ctx_m_np, ctx_s_np, model.ablate_physical_anchors
        )
        model.eval()
        with torch.no_grad():
            fc_chunks: list[np.ndarray] = []
            fm_chunks: list[np.ndarray] = []
            for s_i in range(0, len(valid_cids), 256):
                sl_c = valid_cids[s_i : s_i + 256]
                xb_t = torch.from_numpy(idxr.cell_expr[sl_c]).to(device=device, dtype=torch.float32)
                mb_t = torch.from_numpy(idxr.cell_mask[sl_c]).to(device=device, dtype=torch.float32)
                ex_t = torch.from_numpy(ex_v_np[s_i : s_i + 256]).to(device=device, dtype=torch.float32)
                ex_m_t = torch.from_numpy(ex_m_v_np[s_i : s_i + 256]).to(device=device, dtype=torch.float32)
                if model.ablate_physical_anchors:
                    pl_t = torch.zeros((len(sl_c), 28), device=device, dtype=torch.float32)
                    ml_t = torch.zeros((len(sl_c), 12), device=device, dtype=torch.float32)
                    ph_t = torch.ones((len(sl_c), 28), device=device, dtype=torch.float32)
                else:
                    epi_t = torch.from_numpy(
                        idxr.cell_cis_reg[sl_c] * idxr.cell_cis_mask[sl_c][..., None]
                    ).to(device=device, dtype=torch.float32)
                    cis_tok = model.cis_reg_slot(epi_t, batch_size=len(sl_c), device=device, dtype=xb_t.dtype)
                    tr_o = model.perceiver_trunk(xb_t, expr_mask=mb_t, cis_reg_tokens=cis_tok)
                    pl_ctx_t = ex_t[:, 138:]
                    pl_t = tr_o["p_hat_log"] + model.multiomic_to_prot_anchor(pl_ctx_t)
                    ml_t = tr_o["m0_hat_log"] + model.multiomic_to_metab_anchor(pl_ctx_t)
                    ph_t = F.softplus(pl_t + model.perceiver_trunk.anchor_mrna_log_basal.unsqueeze(0)) + 0.05
                if model.ablate_cis_reg:
                    ci_t = torch.zeros((len(sl_c), 64), device=device, dtype=torch.float32)
                    ci_m_t = torch.zeros((len(sl_c), 64), device=device, dtype=torch.float32)
                else:
                    ep_r = idxr.cell_cis_reg[sl_c] * idxr.cell_cis_mask[sl_c][..., None]
                    ci_np = (0.35 * ep_r[:, :64, 0] + 0.35 * ep_r[:, :64, 2] + 0.30 * ep_r[:, :64, 1:].mean(axis=-1)).astype(
                        np.float32
                    )
                    ci_t = torch.from_numpy(ci_np).to(device=device)
                    ci_m_t = torch.from_numpy(idxr.cell_cis_mask[sl_c, :64].astype(np.float32)).to(device=device)
                sec_b_t = model.compute_basal_sec46_features(xb_t, mb_t, ph_t, ml_t)
                fc_t, fm_t = model.diff_ferro_gem.build_raw_cell_greybox_features(
                    xb_t, mb_t, pl_t, ml_t, ex_t, ci_t, ctx_mask=ex_m_t, cis_mask=ci_m_t, sec_basal_3=sec_b_t
                )
                fc_chunks.append(fc_t.cpu().numpy())
                fm_chunks.append(fm_t.cpu().numpy())
            fc_mat = np.concatenate(fc_chunks, axis=0)
            fm_mat = np.concatenate(fm_chunks, axis=0)
            if fc_mat.shape[1] == model.diff_ferro_gem.cell_expr_mean.numel():
                cnt_fc = np.maximum(fm_mat.sum(axis=0), 1.0)
                fc_mean = ((fc_mat * fm_mat).sum(axis=0) / cnt_fc).astype(np.float32)
                fc_var = ((((fc_mat - fc_mean[None, :]) * fm_mat) ** 2).sum(axis=0) / cnt_fc).astype(np.float32)
                fc_std = np.sqrt(fc_var)
                fc_scale = np.where(fc_std < 2e-2, 1.0, fc_std).astype(np.float32)
                model.diff_ferro_gem.cell_expr_mean.copy_(torch.from_numpy(fc_mean).to(device=device))
                model.diff_ferro_gem.cell_expr_scale.copy_(torch.from_numpy(fc_scale).to(device=device))

    # 3. Train-only Scaffold-Invariant Chemical Feature Standardization & Shared-ECFP Bit Mask
    tr_pharm_list = [
        int(ti)
        for ti in tr_arr
        if recs[int(ti)].get("kind") == "pharmaco"
        and int(idxr.row_chem_a_idx[int(ti)]) > 0
        and not _record_hits_forbidden_chemistry(recs[int(ti)], forbid_smiles, forbid_ik14, forbid_scaffolds)
    ]
    if len(tr_pharm_list) >= 32:
        tr_ph_arr = np.asarray(tr_pharm_list, dtype=np.int64)
        uniq_ca = np.unique(idxr.row_chem_a_idx[tr_ph_arr])
        ecfp_sub_d = model.perturbation_op.ecfp_sub_dim
        if len(uniq_ca) >= 4 and idxr.chem_fp_table.shape[1] >= ecfp_sub_d:
            fp_doc_freq = (idxr.chem_fp_table[uniq_ca, :ecfp_sub_d] > 0.5).sum(axis=0)
            sh_mask = (fp_doc_freq >= 2).astype(np.float32)
            with torch.no_grad():
                model.perturbation_op.shared_ecfp_mask.copy_(torch.from_numpy(sh_mask).to(device=device))

        sub_ph = (
            tr_ph_arr[rng.choice(len(tr_ph_arr), size=8192, replace=False)]
            if len(tr_ph_arr) > 8192
            else tr_ph_arr
        )
        model.eval()
        with torch.no_grad():
            cd_chunks: list[np.ndarray] = []
            for s_i in range(0, len(sub_ph), 512):
                sl_p = sub_ph[s_i : s_i + 512]
                ca_b = idxr.row_chem_a_idx[sl_p]
                fp_t = model.perturbation_op._align_vec(
                    torch.from_numpy(idxr.chem_fp_table[ca_b]).to(device=device), model.perturbation_op.chem_fp_dim
                )
                ds_t = model.perturbation_op._align_vec(
                    torch.from_numpy(idxr.chem_desc_table[ca_b]).to(device=device), model.perturbation_op.chem_desc_dim
                )
                d_log10 = torch.from_numpy(idxr.dose_a[sl_p]).to(device=device)
                d_uM = (torch.pow(10.0, d_log10.clamp(-4.0, 3.5)) - 1e-3).clamp(min=0.0)
                cd_t = model.perturbation_op.build_raw_chem_dense_features(fp_t, ds_t, d_uM, d_log10)
                cd_chunks.append(cd_t.cpu().numpy())
            cd_all = np.concatenate(cd_chunks, axis=0)
            c_mean = cd_all.mean(axis=0).astype(np.float32)
            c_scale = np.where(cd_all.std(axis=0) < 2e-2, 1.0, cd_all.std(axis=0)).astype(np.float32)
            model.perturbation_op.chem_dense_mean.copy_(torch.from_numpy(c_mean).to(device=device))
            model.perturbation_op.chem_dense_scale.copy_(torch.from_numpy(c_scale).to(device=device))

    # 4. Masked pathway-block Ridge warm-start for `mgm_svd_proj` + `mgm_ctx_decoder`
    calibrate_mgm_decoder_from_train_indices(
        model,
        dataset,
        train_indices,
        forbid_test_set=forbid_test_set,
        device=device,
        seed=seed,
    )

    # 5. Exact Block Schur-Complement + Soft-Impute EM-SVD CRISPR Solve (`cell_chronos_offset` + `gene_essentiality_prior`)
    warmstart_crispr_joint_and_svd(
        model,
        dataset,
        train_indices,
        forbid_test_set=forbid_test_set,
        device=device,
        seed=seed,
    )


def fit_post_curriculum_greybox_from_train_indices(
    model: FerroFluxFM,
    dataset: Any,
    train_indices: list[int],
    *,
    forbid_test_set: set[int] | None = None,
    forbid_smiles: set[str] | None = None,
    forbid_ik14: set[str] | None = None,
    forbid_scaffolds: set[str] | None = None,
    include_non_ferro: bool = False,
    split_name: str = "",  # noqa: ARG001 — unused; blend is 100% inner-CV + train-Tanimoto driven!
    device: torch.device,
    seed: int = 101,
) -> None:
    """Post-Stage-5 APHYNITY Grey-Box Residual Fit STRICTLY on `train_indices` (`0` test exposure, `0` split-name oracle)."""
    if forbid_test_set:
        assert forbid_test_set.isdisjoint(set(train_indices)), "FATAL: test index present in post-curriculum greybox!"
    if model.ablate_diff_ferro_gem or model.ablate_aphynity_residual:
        return
    from ferroflux.data.dataset import get_fast_indexer
    from scipy.stats import spearmanr

    idxr = get_fast_indexer(dataset)
    if idxr is None or not hasattr(dataset, "records"):
        return

    rng = np.random.default_rng(seed + 91)
    recs = dataset.records
    tr_arr = np.asarray(train_indices, dtype=np.int64)
    ferro_ph: list[int] = []
    other_ph: list[int] = []
    for ti in tr_arr:
        ti_int = int(ti)
        rc = recs[ti_int]
        if (
            rc.get("kind") != "pharmaco"
            or int(idxr.row_chem_a_idx[ti_int]) <= 0
            or float(idxr.viability_mask[ti_int]) <= 0.5
            or _record_hits_forbidden_chemistry(rc, forbid_smiles, forbid_ik14, forbid_scaffolds)
        ):
            continue
        if int(rc.get("is_ferroptosis_compound", 1)) > 0:
            ferro_ph.append(ti_int)
        else:
            other_ph.append(ti_int)

    if len(ferro_ph) < 128 or model.diff_ferro_gem.cell_expr_mean.numel() != model.diff_ferro_gem.full_cell_dim:
        return

    f_sub = np.asarray(ferro_ph, dtype=np.int64)
    if len(f_sub) > 50000:
        f_sub = np.sort(f_sub[rng.choice(len(f_sub), size=50000, replace=False)])
    if include_non_ferro and len(other_ph) >= 32:
        o_arr = np.asarray(other_ph, dtype=np.int64)
        if len(o_arr) > 10000:
            o_arr = np.sort(o_arr[rng.choice(len(o_arr), size=10000, replace=False)])
        tr_f_arr = np.concatenate([f_sub, o_arr], axis=0)
    else:
        tr_f_arr = f_sub

    if forbid_test_set:
        assert forbid_test_set.isdisjoint(set(tr_f_arr.tolist())), "FATAL: test leak in greybox calibration!"

    # Deterministically re-initialize post-curriculum grey-box modules (`gb_mods`) so Stage 1-5 decoupled
    # weight decay / in-split scaffold memorization never distorts `hf_wb_f` or collapses starting weights.
    rng_state_cpu = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed) + 701)
        for mod_init in (
            model.perturbation_op.chem_scaffold_inv_enc,
            model.perturbation_op.chem_shared_ecfp_enc,
            model.perturbation_op.chem_scaffold_fuse,
            model.diff_ferro_gem.aphynity_scaf_cell_net,
            model.diff_ferro_gem.aphynity_scaf_head,
            model.diff_ferro_gem.aphynity_scaf_ferro_gate,
        ):
            for sub_m in mod_init.modules():
                if hasattr(sub_m, "reset_parameters"):
                    sub_m.reset_parameters()
        with torch.no_grad():
            model.diff_ferro_gem.aphynity_scaf_head[-1].weight.zero_()
            model.diff_ferro_gem.aphynity_scaf_head[-1].bias.zero_()
            model.diff_ferro_gem.aphynity_scaf_ferro_gate.weight.zero_()
            model.diff_ferro_gem.aphynity_scaf_ferro_gate.bias.fill_(4.0)
    finally:
        torch.random.set_rng_state(rng_state_cpu)

    ctx_m_np = model.ctx_obs_mean.detach().cpu().numpy()
    ctx_s_np = model.ctx_obs_scale.detach().cpu().numpy()

    model.eval()
    with torch.no_grad():
        model.diff_ferro_gem.scaf_gb_ready.zero_()
        u_cells, cell_inv_f = np.unique(idxr.row_cell_idx[tr_f_arr], return_inverse=True)
        ex_u_np, ex_m_u_np = _compute_standardized_extra_ctx_np(
            idxr, u_cells, ctx_m_np, ctx_s_np, model.ablate_physical_anchors
        )
        raw_fc_chunks: list[np.ndarray] = []
        fmask_chunks: list[np.ndarray] = []
        for s_i in range(0, len(u_cells), 256):
            sl_c = u_cells[s_i : s_i + 256]
            gm_t = torch.from_numpy(idxr.cell_mask[sl_c]).to(device=device, dtype=torch.float32)
            xb_t = torch.from_numpy(idxr.cell_expr[sl_c]).to(device=device, dtype=torch.float32)
            ex_t = torch.from_numpy(ex_u_np[s_i : s_i + 256]).to(device=device, dtype=torch.float32)
            ex_m_t = torch.from_numpy(ex_m_u_np[s_i : s_i + 256]).to(device=device, dtype=torch.float32)
            if model.ablate_physical_anchors:
                pl_t = torch.zeros((len(sl_c), 28), device=device, dtype=torch.float32)
                ml_t = torch.zeros((len(sl_c), 12), device=device, dtype=torch.float32)
                ph_t = torch.ones((len(sl_c), 28), device=device, dtype=torch.float32)
            else:
                epi_t = torch.from_numpy(idxr.cell_cis_reg[sl_c] * idxr.cell_cis_mask[sl_c][..., None]).to(
                    device=device, dtype=torch.float32
                )
                cis_tok = model.cis_reg_slot(epi_t, batch_size=len(sl_c), device=device, dtype=xb_t.dtype)
                tr_o = model.perceiver_trunk(xb_t, expr_mask=gm_t, cis_reg_tokens=cis_tok)
                pl_ctx_t = ex_t[:, 138:]
                pl_t = tr_o["p_hat_log"] + model.multiomic_to_prot_anchor(pl_ctx_t)
                ml_t = tr_o["m0_hat_log"] + model.multiomic_to_metab_anchor(pl_ctx_t)
                ph_t = F.softplus(pl_t + model.perceiver_trunk.anchor_mrna_log_basal.unsqueeze(0)) + 0.05
            if model.ablate_cis_reg:
                ci_t = torch.zeros((len(sl_c), 64), device=device, dtype=torch.float32)
                ci_m_t = torch.zeros((len(sl_c), 64), device=device, dtype=torch.float32)
            else:
                ep_r = idxr.cell_cis_reg[sl_c] * idxr.cell_cis_mask[sl_c][..., None]
                ci_np = (0.35 * ep_r[:, :64, 0] + 0.35 * ep_r[:, :64, 2] + 0.30 * ep_r[:, :64, 1:].mean(axis=-1)).astype(
                    np.float32
                )
                ci_t = torch.from_numpy(ci_np).to(device=device)
                ci_m_t = torch.from_numpy(idxr.cell_cis_mask[sl_c, :64].astype(np.float32)).to(device=device)
            sec_b_t = model.compute_basal_sec46_features(xb_t, gm_t, ph_t, ml_t)
            fc_t, fm_t = model.diff_ferro_gem.build_raw_cell_greybox_features(
                xb_t, gm_t, pl_t, ml_t, ex_t, ci_t, ctx_mask=ex_m_t, cis_mask=ci_m_t, sec_basal_3=sec_b_t
            )
            raw_fc_chunks.append(fc_t.cpu().numpy())
            fmask_chunks.append(fm_t.cpu().numpy())

        raw_fc_u = np.concatenate(raw_fc_chunks, axis=0)
        fmask_u = np.concatenate(fmask_chunks, axis=0)
        cnt_fmask = np.maximum(fmask_u.sum(axis=0), 1.0)
        cm_new = ((raw_fc_u * fmask_u).sum(axis=0) / cnt_fmask).astype(np.float32)
        cs_var = ((((raw_fc_u - cm_new[None, :]) * fmask_u) ** 2).sum(axis=0) / cnt_fmask).astype(np.float32)
        cs_std = np.sqrt(cs_var)
        cs_new = np.where(cs_std < 2e-2, 1.0, cs_std).astype(np.float32)
        model.diff_ferro_gem.cell_expr_mean.copy_(torch.from_numpy(cm_new).to(device=device))
        model.diff_ferro_gem.cell_expr_scale.copy_(torch.from_numpy(cs_new).to(device=device))
        cell_std_u = (
            model.diff_ferro_gem.standardize_cell_greybox_features(
                torch.from_numpy(raw_fc_u).to(device=device),
                torch.from_numpy(fmask_u).to(device=device),
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        is_ferro_f = idxr.is_ferroptosis_compound[tr_f_arr].astype(np.float32)
        ca_f = idxr.row_chem_a_idx[tr_f_arr]
        ferro_ca = np.unique(ca_f[is_ferro_f > 0.5])
        all_tr_ca = np.unique(ca_f)
        if len(all_tr_ca) >= 1:
            n_bk = min(len(all_tr_ca), model.perturbation_op.max_train_chem_bank)
            bk_fp = (idxr.chem_fp_table[all_tr_ca[:n_bk], :2048] > 0.5).astype(np.float32)
            model.perturbation_op.train_chem_ecfp_bank.zero_()
            model.perturbation_op.train_chem_ecfp_bank[:n_bk].copy_(torch.from_numpy(bk_fp).to(device=device))
            model.perturbation_op.n_train_chem_bank.fill_(int(n_bk))
        if len(ferro_ca) >= 4:
            chem_ecfp_u = (idxr.chem_fp_table[ferro_ca, : model.perturbation_op.ecfp_sub_dim] > 0.5).astype(np.float32)
            bit_cnt = chem_ecfp_u.sum(axis=0)
            sh_mask = ((bit_cnt >= 1.0) & (bit_cnt <= max(4.0, 0.90 * float(len(ferro_ca))))).astype(np.float32)
            if float(sh_mask.sum()) >= 16.0:
                model.perturbation_op.shared_ecfp_mask.copy_(torch.from_numpy(sh_mask).to(device=device))

        _, ca_inv, ca_cnts = np.unique(ca_f, return_inverse=True, return_counts=True)
        cmpd_w_raw = 1.0 / np.sqrt(np.maximum(ca_cnts[ca_inv].astype(np.float32), 16.0))
        ferro_w_mean = float(cmpd_w_raw[is_ferro_f > 0.5].mean()) if np.any(is_ferro_f > 0.5) else float(cmpd_w_raw.mean())
        cmpd_w_f = (cmpd_w_raw / max(1e-6, ferro_w_mean)).astype(np.float32)

        dlog_f = idxr.dose_a[tr_f_arr].astype(np.float32)
        duM_f = np.maximum(np.power(10.0, np.clip(dlog_f, -4.0, 3.5)) - 1e-3, 0.0).astype(np.float32)
        raw_cd_chunks: list[np.ndarray] = []
        ecfp_sh_chunks: list[np.ndarray] = []
        for s_i in range(0, len(ca_f), 1024):
            sl_r = slice(s_i, min(len(ca_f), s_i + 1024))
            fp_r = model.perturbation_op._align_vec(
                torch.from_numpy(idxr.chem_fp_table[ca_f[sl_r]]).to(device=device, dtype=torch.float32),
                model.perturbation_op.chem_fp_dim,
            )
            ds_r = model.perturbation_op._align_vec(
                torch.from_numpy(idxr.chem_desc_table[ca_f[sl_r]]).to(device=device, dtype=torch.float32),
                model.perturbation_op.chem_desc_dim,
            )
            du_r = torch.from_numpy(duM_f[sl_r]).to(device=device, dtype=torch.float32)
            dl_r = torch.from_numpy(dlog_f[sl_r]).to(device=device, dtype=torch.float32)
            raw_cd_chunks.append(model.perturbation_op.build_raw_chem_dense_features(fp_r, ds_r, du_r, dl_r).cpu().numpy())
            ecfp_sh_chunks.append(model.perturbation_op.build_shared_ecfp_features(fp_r).cpu().numpy())

        raw_cd_f = np.concatenate(raw_cd_chunks, axis=0).astype(np.float32)
        ferro_cd = raw_cd_f[is_ferro_f > 0.5] if np.any(is_ferro_f > 0.5) else raw_cd_f
        cdm_new = ferro_cd.mean(axis=0).astype(np.float32)
        cds_std = ferro_cd.std(axis=0).astype(np.float32)
        cds_new = np.where(cds_std < 2e-2, 1.0, cds_std).astype(np.float32)
        model.perturbation_op.chem_dense_mean.copy_(torch.from_numpy(cdm_new).to(device=device))
        model.perturbation_op.chem_dense_scale.copy_(torch.from_numpy(cds_new).to(device=device))
        cd_std_f = np.clip((raw_cd_f - cdm_new[None, :]) / cds_new[None, :], -8.0, 8.0).astype(np.float32)

        ecfp_sh_f = np.concatenate(ecfp_sh_chunks, axis=0).astype(np.float32)
        y_viab_f = idxr.viability_target[tr_f_arr].astype(np.float32)

        hf_wb_list: list[np.ndarray] = []
        h_oth_list: list[np.ndarray] = []
        rsh_list: list[np.ndarray] = []
        for s_i in range(0, len(tr_f_arr), 512):
            b_sub = idxr.get_batch(tr_f_arr[s_i : s_i + 512], device=device)
            out_sub = model(b_sub)
            hf_cur = out_sub["h_ferro"].clamp(min=0.0)
            tot_h = (out_sub["hazards"].sum(dim=-1) + 0.10 * out_sub["rcd_22_hazards"][:, 5:].sum(dim=-1)).clamp(min=0.0)
            hf_wb_list.append(hf_cur.cpu().numpy())
            h_oth_list.append((tot_h - hf_cur).clamp(min=0.0).cpu().numpy())
            rsh_list.append(out_sub["rescue_shield"].cpu().numpy().astype(np.float32))
        hf_wb_f = np.concatenate(hf_wb_list, axis=0)
        h_oth_f = np.concatenate(h_oth_list, axis=0)
        rsh_f = np.concatenate(rsh_list, axis=0)

    gb_mods = (
        model.perturbation_op.chem_scaffold_inv_enc,
        model.perturbation_op.chem_shared_ecfp_enc,
        model.perturbation_op.chem_scaffold_fuse,
        model.diff_ferro_gem.aphynity_scaf_cell_net,
        model.diff_ferro_gem.aphynity_scaf_head,
        model.diff_ferro_gem.aphynity_scaf_ferro_gate,
    )

    def _fit_gb_on_mask(sub_idx_local: np.ndarray, n_epochs_run: int, fixed_blend: float) -> None:
        with torch.no_grad():
            model.diff_ferro_gem.raw_scaf_gb_blend.fill_(float(math.log(fixed_blend / (1.0 - fixed_blend))))
        gb_opt_loc = torch.optim.AdamW(
            [
                {
                    "params": list(model.perturbation_op.chem_scaffold_inv_enc.parameters())
                    + list(model.perturbation_op.chem_scaffold_fuse.parameters())
                    + list(model.diff_ferro_gem.aphynity_scaf_head.parameters())
                    + list(model.diff_ferro_gem.aphynity_scaf_ferro_gate.parameters()),
                    "lr": 1.0e-3,
                    "weight_decay": 1.5e-2,
                },
                {
                    "params": list(model.diff_ferro_gem.aphynity_scaf_cell_net.parameters()),
                    "lr": 1.0e-3,
                    "weight_decay": 1.5e-2,
                },
                {
                    "params": list(model.perturbation_op.chem_shared_ecfp_enc.parameters()),
                    "lr": 4.0e-4,
                    "weight_decay": 1.6e-1,
                },
            ]
        )
        bs_loc = min(256, max(32, len(sub_idx_local) // 10))
        gb_sched_loc = torch.optim.lr_scheduler.CosineAnnealingLR(
            gb_opt_loc, T_max=max(1, n_epochs_run * ((len(sub_idx_local) + bs_loc - 1) // bs_loc))
        )
        for mod_g in gb_mods:
            mod_g.train()
        swa_gb: dict[int, dict[str, torch.Tensor]] | None = None
        swa_gb_n = 0
        for ep_gb in range(n_epochs_run):
            perm_gb = sub_idx_local[rng.permutation(len(sub_idx_local))]
            for s_i in range(0, len(perm_gb), bs_loc):
                bi = perm_gb[s_i : s_i + bs_loc]
                if len(bi) < 8:
                    continue
                cd_t = torch.from_numpy(cd_std_f[bi]).to(device=device)
                ef_t = torch.from_numpy(ecfp_sh_f[bi]).to(device=device)
                cx_t = torch.from_numpy(cell_std_u[cell_inv_f[bi]]).to(device=device)
                yt_t = torch.from_numpy(y_viab_f[bi]).to(device=device)
                hf_t = torch.from_numpy(hf_wb_f[bi]).to(device=device)
                ho_t = torch.from_numpy(h_oth_f[bi]).to(device=device)
                rs_t = torch.from_numpy(rsh_f[bi]).to(device=device)
                isf_t = torch.from_numpy(is_ferro_f[bi]).to(device=device)

                zc_t = F.silu(
                    model.perturbation_op.chem_scaffold_fuse(
                        torch.cat(
                            [
                                model.perturbation_op.chem_scaffold_inv_enc(cd_t),
                                model.perturbation_op.chem_shared_ecfp_enc(ef_t),
                            ],
                            dim=-1,
                        )
                    )
                )
                zx_t = model.diff_ferro_gem.aphynity_scaf_cell_net(cx_t)
                raw_add_t = model.diff_ferro_gem.aphynity_scaf_head(
                    torch.cat([zc_t, zx_t, zc_t * zx_t], dim=-1)
                ).squeeze(-1)
                fgate_logit = model.diff_ferro_gem.aphynity_scaf_ferro_gate(zc_t).squeeze(-1)
                fgate_p = torch.sigmoid(fgate_logit)
                add_haz_t = F.softplus(raw_add_t - 1.2) * rs_t
                w_gb = torch.sigmoid(model.diff_ferro_gem.raw_scaf_gb_blend) * fgate_p
                h_ferro_new = (1.0 - w_gb) * hf_t + w_gb * add_haz_t
                h_tot_t = h_ferro_new + ho_t
                vp_t = 0.99 * torch.exp(-h_tot_t.clamp(max=25.0)) + 0.01 / (1.0 + 0.25 * h_tot_t).pow(2)

                f_mask = isf_t > 0.5
                if int(f_mask.sum().item()) >= 6:
                    vp_f = vp_t[f_mask]
                    yt_f = yt_t[f_mask]
                    if model.ablate_survivorship:
                        l_hub = F.huber_loss(vp_f, yt_f, reduction="mean", delta=0.20)
                        l_cor = l_hub.new_zeros(())
                        l_rnk = l_hub.new_zeros(())
                    else:
                        yt_c = yt_f.clamp(0.0, 1.0)
                        cw_t = torch.from_numpy(cmpd_w_f[bi]).to(device=device)[f_mask]
                        sw_t = (1.0 + 0.35 * (4.0 * yt_c * (1.0 - yt_c))) * cw_t
                        l_hub = (F.huber_loss(vp_f, yt_f, reduction="none", delta=0.20) * sw_t).sum() / sw_t.sum().clamp(min=1e-6)
                        w_sq = sw_t.sqrt()
                        pc = (vp_f - (vp_f * sw_t).sum() / sw_t.sum().clamp(min=1e-6)) * w_sq
                        tc = (yt_f - (yt_f * sw_t).sum() / sw_t.sum().clamp(min=1e-6)) * w_sq
                        l_cor = 1.0 - (pc * tc).sum() / ((pc.pow(2).sum() + 1e-6).sqrt() * (tc.pow(2).sum() + 1e-6).sqrt())
                        diff_y = yt_f.unsqueeze(1) - yt_f.unsqueeze(0)
                        pm = (diff_y.abs() > 0.05).float()
                        l_rnk = (
                            F.softplus(-8.0 * (vp_f.unsqueeze(1) - vp_f.unsqueeze(0)) * torch.sign(diff_y)) * pm
                        ).sum() / pm.sum().clamp(min=1.0)
                else:
                    l_hub = vp_t.new_zeros(())
                    l_cor = vp_t.new_zeros(())
                    l_rnk = vp_t.new_zeros(())

                if float((1.0 - isf_t).sum().item()) > 0.5:
                    l_fgate = F.binary_cross_entropy_with_logits(fgate_logit, isf_t) * 0.35
                    non_ferro_pen = ((1.0 - isf_t) * (fgate_p * add_haz_t).pow(2)).mean() * 0.65
                else:
                    l_fgate = vp_t.new_zeros(())
                    non_ferro_pen = vp_t.new_zeros(())

                loss_gb = l_hub + 0.40 * l_cor + 0.35 * l_rnk + l_fgate + non_ferro_pen
                gb_opt_loc.zero_grad(set_to_none=True)
                loss_gb.backward()
                gb_opt_loc.step()
                gb_sched_loc.step()

            if ep_gb >= max(0, n_epochs_run - 5):
                with torch.no_grad():
                    if swa_gb is None:
                        swa_gb = {
                            m_i: {k: v.detach().clone() for k, v in mod_g.state_dict().items()}
                            for m_i, mod_g in enumerate(gb_mods)
                        }
                        swa_gb_n = 1
                    else:
                        swa_gb_n += 1
                        w_new = 1.0 / float(swa_gb_n)
                        for m_i, mod_g in enumerate(gb_mods):
                            for k, v in mod_g.state_dict().items():
                                if v.is_floating_point():
                                    swa_gb[m_i][k].mul_(1.0 - w_new).add_(v.detach(), alpha=w_new)
                                else:
                                    swa_gb[m_i][k].copy_(v)
        if swa_gb is not None and swa_gb_n >= 1:
            for m_i, mod_g in enumerate(gb_mods):
                mod_g.load_state_dict(swa_gb[m_i])

    init_gb_state = {m_i: copy.deepcopy(mod_g.state_dict()) for m_i, mod_g in enumerate(gb_mods)}

    # 100% Data-Driven Dual Inner Validation (Out-of-Cell -> w_in, Out-of-Scaffold -> w_ood) strictly inside `train_indices`
    row_scafs = np.array(
        [str(recs[int(ti)].get("murcko_scaffold") or f"cid_{int(idxr.row_chem_a_idx[int(ti)])}") for ti in tr_f_arr]
    )
    ferro_scafs = np.unique(row_scafs[is_ferro_f > 0.5])
    ferro_cells = np.unique(cell_inv_f[is_ferro_f > 0.5])
    oth_chems = np.unique(ca_f[is_ferro_f < 0.5])
    has_sar = int(model.perturbation_op.has_sar_snapshot.item()) > 0 and not model.ablate_chem_prior

    best_w_in = 0.0 if include_non_ferro else 0.16
    best_w_ood = 0.0 if include_non_ferro else (0.88 if has_sar else 0.38)
    best_epochs_full = 15
    if len(ferro_scafs) >= 5 and len(ferro_cells) >= 12:
        val_scaf_set = set(rng.permutation(ferro_scafs)[: max(2, len(ferro_scafs) // 4)].tolist())
        val_cell_set = set(rng.permutation(ferro_cells)[: max(3, len(ferro_cells) // 5)].tolist())
        val_oth_set = (
            set(rng.permutation(oth_chems)[: max(1, len(oth_chems) // 4)].tolist()) if len(oth_chems) >= 4 else set()
        )

        is_val_scaf = np.array([(s in val_scaf_set) and (isf > 0.5) for s, isf in zip(row_scafs, is_ferro_f)], dtype=bool)
        is_val_cell = np.array(
            [(int(c) in val_cell_set) and (isf > 0.5) and (s not in val_scaf_set) for c, s, isf in zip(cell_inv_f, row_scafs, is_ferro_f)],
            dtype=bool,
        )
        is_val_oth = np.array([(int(cid) in val_oth_set) and (isf < 0.5) for cid, isf in zip(ca_f, is_ferro_f)], dtype=bool)
        in_tr_mask = (~is_val_scaf) & (~is_val_cell) & (~is_val_oth)
        in_tr_idx = np.where(in_tr_mask)[0]
        in_va_all = np.where(~in_tr_mask)[0]

        if len(in_tr_idx) >= 96 and int(is_val_scaf.sum()) >= 24 and int(is_val_cell.sum()) >= 24:
            def _eval_inner_gb() -> tuple[np.ndarray, np.ndarray]:
                model.eval()
                with torch.no_grad():
                    radd_va_list: list[np.ndarray] = []
                    fg_va_list: list[np.ndarray] = []
                    for s_i in range(0, len(in_va_all), 512):
                        vi = in_va_all[s_i : s_i + 512]
                        cd_v = torch.from_numpy(cd_std_f[vi]).to(device=device)
                        ef_v = torch.from_numpy(ecfp_sh_f[vi]).to(device=device)
                        cx_v = torch.from_numpy(cell_std_u[cell_inv_f[vi]]).to(device=device)
                        zc_v = F.silu(
                            model.perturbation_op.chem_scaffold_fuse(
                                torch.cat(
                                    [
                                        model.perturbation_op.chem_scaffold_inv_enc(cd_v),
                                        model.perturbation_op.chem_shared_ecfp_enc(ef_v),
                                    ],
                                    dim=-1,
                                )
                            )
                        )
                        zx_v = model.diff_ferro_gem.aphynity_scaf_cell_net(cx_v)
                        ra_v = model.diff_ferro_gem.aphynity_scaf_head(torch.cat([zc_v, zx_v, zc_v * zx_v], dim=-1)).squeeze(-1)
                        fg_l = model.diff_ferro_gem.aphynity_scaf_ferro_gate(zc_v).squeeze(-1)
                        radd_va_list.append(ra_v.cpu().numpy())
                        fg_va_list.append(torch.sigmoid(fg_l).cpu().numpy())
                    radd_all = np.zeros(len(tr_f_arr), dtype=np.float32)
                    fg_all = np.ones(len(tr_f_arr), dtype=np.float32)
                    radd_all[in_va_all] = np.concatenate(radd_va_list, axis=0)
                    fg_all[in_va_all] = np.concatenate(fg_va_list, axis=0)
                    add_all = np.log1p(np.exp(np.clip(radd_all - 1.2, -20.0, 20.0))) * rsh_f
                    return add_all, fg_all

            def _rho_sub_arr(add_arr: np.ndarray, fg_arr: np.ndarray, mask_b: np.ndarray, w_cand: float) -> float:
                if int(mask_b.sum()) < 8:
                    return 0.0
                eff_w = w_cand * fg_arr[mask_b]
                ht = (1.0 - eff_w) * hf_wb_f[mask_b] + eff_w * add_arr[mask_b] + h_oth_f[mask_b]
                vp = 0.99 * np.exp(-np.clip(ht, 0.0, 25.0)) + 0.01 / ((1.0 + 0.25 * ht) ** 2)
                yv = y_viab_f[mask_b]
                if np.std(yv) < 1e-8 or np.std(vp) < 1e-8:
                    return 0.0
                rv = float(spearmanr(yv, vp).statistic)
                return 0.0 if np.isnan(rv) else rv

            if include_non_ferro and int(is_val_oth.sum()) >= 24:
                _fit_gb_on_mask(in_tr_idx, 10, 0.35)
                add_all, fg_all = _eval_inner_gb()
                ood_mask = is_val_scaf | is_val_oth
                r_ood0 = _rho_sub_arr(add_all, fg_all, ood_mask, 0.0)
                best_ood = r_ood0
                best_w_ood = 0.0
                for w_c in (0.15, 0.30, 0.50):
                    ro_v = _rho_sub_arr(add_all, fg_all, ood_mask, w_c)
                    if ro_v > best_ood + 0.004:
                        best_ood = ro_v
                        best_w_ood = w_c
                best_w_in = min(0.04, best_w_ood)
            else:
                _fit_gb_on_mask(in_tr_idx, 12, 0.88 if has_sar else 0.40)
                add_all, fg_all = _eval_inner_gb()
                best_cell_r = _rho_sub_arr(add_all, fg_all, is_val_cell, 0.08)
                best_w_in = 0.08
                for w_ci in (0.18, 0.30, 0.45, 0.58):
                    rc_v = _rho_sub_arr(add_all, fg_all, is_val_cell, w_ci)
                    if rc_v > best_cell_r + 0.0015:
                        best_cell_r = rc_v
                        best_w_in = w_ci

            for m_i, mod_g in enumerate(gb_mods):
                mod_g.load_state_dict(init_gb_state[m_i])

    if max(best_w_in, best_w_ood) <= 1e-3:
        with torch.no_grad():
            model.diff_ferro_gem.scaf_gb_ready.zero_()
        model.eval()
        return

    fit_blend = float(np.clip(max(0.55, best_w_in, best_w_ood), 0.20, 0.88))
    _fit_gb_on_mask(np.arange(len(tr_f_arr), dtype=np.int64), best_epochs_full, fit_blend)
    with torch.no_grad():
        win_c = float(np.clip(best_w_in, 1e-4, 0.999))
        wood_c = float(np.clip(best_w_ood, 1e-4, 0.999))
        model.diff_ferro_gem.raw_scaf_gb_blend.fill_(float(math.log(win_c / (1.0 - win_c))))
        model.diff_ferro_gem.raw_scaf_gb_blend_ood.fill_(float(math.log(wood_c / (1.0 - wood_c))))
        model.diff_ferro_gem.scaf_gb_ready.fill_(1.0)
    model.eval()


def warmstart_crispr_joint_and_svd(
    model: FerroFluxFM,
    dataset: Any,
    train_indices: list[int],
    *,
    forbid_test_set: set[int] | None = None,
    device: torch.device,
    seed: int = 101,
) -> None:
    """Simultaneously deconvolve cell-batch offsets, per-gene Chronos essentiality, and 5-iter Soft-Impute EM-SVD co-essentiality strictly on `train_indices`."""
    if forbid_test_set:
        assert forbid_test_set.isdisjoint(set(train_indices)), "FATAL: test index present in CRISPR warm-start!"
    from ferroflux.data.dataset import get_fast_indexer
    from sklearn.decomposition import TruncatedSVD

    idxr = get_fast_indexer(dataset)
    if idxr is None or not hasattr(dataset, "records"):
        return

    n_genes = model.n_genes
    recs = dataset.records
    tr_chr_list = [
        ti
        for ti in train_indices
        if recs[ti].get("kind") == "crispr"
        and int(recs[ti].get("is_crispr_fin_epistasis", 0)) == 0
        and 0 <= int(idxr.row_ko_idx[ti]) < n_genes
        and float(idxr.cell_mask[int(idxr.row_cell_idx[ti])].sum()) > 0.0
    ]
    if len(tr_chr_list) < 64:
        return

    tr_chr = np.asarray(tr_chr_list, dtype=np.int64)
    c_rows = idxr.row_cell_idx[tr_chr]
    g_rows = idxr.row_ko_idx[tr_chr]
    y_rows = idxr.chronos_target[tr_chr].astype(np.float64)

    uniq_c, c_inv = np.unique(c_rows, return_inverse=True)
    xc_u = idxr.cell_expr[uniq_c]
    mc_u = idxr.cell_mask[uniq_c]
    cnt_m_u = np.maximum(mc_u.sum(axis=1, keepdims=True), 1.0)
    xm_u = (xc_u * mc_u).sum(axis=1, keepdims=True) / cnt_m_u
    xvar_u = (((xc_u - xm_u) * mc_u) ** 2).sum(axis=1, keepdims=True) / cnt_m_u
    xcent_u = (((xc_u - xm_u) / np.sqrt(xvar_u + 0.25)) * mc_u).astype(np.float32)

    ctx_m_np = model.ctx_obs_mean.detach().cpu().numpy()
    ctx_s_np = model.ctx_obs_scale.detach().cpu().numpy()
    extra_u, _ = _compute_standardized_extra_ctx_np(
        idxr, uniq_c, ctx_m_np, ctx_s_np, model.ablate_physical_anchors
    )

    model.eval()
    with torch.no_grad():
        gx_chunks: list[np.ndarray] = []
        ko_hid_chunks: list[np.ndarray] = []
        ko_hid_zero_ex_chunks: list[np.ndarray] = []
        for s_i in range(0, len(uniq_c), 256):
            sl_u = uniq_c[s_i : s_i + 256]
            xc_t = torch.from_numpy(xcent_u[s_i : s_i + 256]).to(device=device)
            gx_t = model.expr_ctx_proj(xc_t)
            gx_chunks.append(gx_t.cpu().numpy())

            xb_t = torch.from_numpy(idxr.cell_expr[sl_u]).to(device=device)
            mb_t = torch.from_numpy(idxr.cell_mask[sl_u]).to(device=device)
            epi_t = torch.from_numpy(idxr.cell_cis_reg[sl_u] * idxr.cell_cis_mask[sl_u][..., None]).to(device=device)
            ex_t = torch.from_numpy(extra_u[s_i : s_i + 256]).to(device=device)
            ex_z = torch.zeros_like(ex_t)
            cis_tok = model.cis_reg_slot(epi_t, batch_size=len(sl_u), device=device, dtype=xb_t.dtype)
            cis_mean = 0.25 * cis_tok.mean(dim=1)
            tr_o = model.perceiver_trunk(xb_t, expr_mask=mb_t, cis_reg_tokens=cis_tok)
            zc_t = tr_o["z_cell"] + model.multiomic_ctx_fusion(ex_t) + cis_mean
            ko_in_t = torch.cat([zc_t, tr_o["p_hat_log"], tr_o["m0_hat_log"], ex_t, gx_t], dim=-1)
            ko_hid_chunks.append(model.cell_to_gene_ctx[:-1](ko_in_t).cpu().numpy())

            zc_z = tr_o["z_cell"] + model.multiomic_ctx_fusion(ex_z) + cis_mean
            ko_in_z = torch.cat([zc_z, tr_o["p_hat_log"], tr_o["m0_hat_log"], ex_z, gx_t], dim=-1)
            ko_hid_zero_ex_chunks.append(model.cell_to_gene_ctx[:-1](ko_in_z).cpu().numpy())

        gx_u = np.concatenate(gx_chunks, axis=0)
        ko_hid_u = np.concatenate(ko_hid_chunks, axis=0)
        ko_hid_z = np.concatenate(ko_hid_zero_ex_chunks, axis=0)

        # Exact Block Schur-Complement Ridge on `[1, gx_u]` (100% invariant to 2D-vs-3D `extra_u` missingness!)
        u_c = np.concatenate([np.ones((len(uniq_c), 1), dtype=np.float64), gx_u.astype(np.float64)], axis=1)
        d_c = u_c.shape[1]
        cnt_c = np.bincount(c_inv, minlength=len(uniq_c)).astype(np.float64)
        a_mat = (u_c.T * cnt_c) @ u_c
        lam_c = np.full(d_c, 120.0, dtype=np.float64)
        lam_c[0] = 1e-6
        a_mat += np.diag(lam_c)
        rhs_c = u_c.T @ np.bincount(c_inv, weights=y_rows, minlength=len(uniq_c))

        m_cg = np.zeros((len(uniq_c), n_genes), dtype=np.float64)
        np.add.at(m_cg, (c_inv, g_rows), 1.0)
        b_mat = u_c.T @ m_cg
        cnt_g = m_cg.sum(axis=0) + 8.0
        rhs_g = np.bincount(g_rows, weights=y_rows, minlength=n_genes)

        schur = a_mat - (b_mat / cnt_g[None, :]) @ b_mat.T
        w_c = np.linalg.solve(schur, rhs_c - (b_mat / cnt_g[None, :]) @ rhs_g)
        b_g = (rhs_g - b_mat.T @ w_c) / cnt_g
        pred_off_u = (u_c @ w_c).astype(np.float64)

        # Within-gene standardized expression slope (`cov(x_cent, y_res) / (var(x_cent) + 35.0)`)
        y_res1 = y_rows - pred_off_u[c_inv] - b_g[g_rows]
        x_at_ko = xcent_u[c_inv, g_rows].astype(np.float64)
        cov_xy = np.bincount(g_rows, weights=x_at_ko * y_res1, minlength=n_genes)
        var_xx = np.bincount(g_rows, weights=x_at_ko * x_at_ko, minlength=n_genes)
        eb_prior_scaled = (b_g / 0.95).astype(np.float32)
        eb_slope_scaled = np.clip((cov_xy / (var_xx + 35.0)) / 0.95, -0.25, 0.25).astype(np.float32)

        model.gene_essentiality_prior.copy_(torch.from_numpy(eb_prior_scaled).to(device=device))
        model.gene_expr_slope_prior.copy_(torch.from_numpy(eb_slope_scaled).to(device=device))
        model.cell_chronos_offset.bias.fill_(float(w_c[0]))
        model.cell_chronos_offset.weight.copy_(torch.from_numpy(w_c[1:].reshape(1, -1).astype(np.float32)).to(device=device))

        if len(uniq_c) >= 24 and n_genes >= 64:
            y_res2 = y_res1 - (0.95 * eb_slope_scaled[g_rows]) * x_at_ko
            obs_m_gc = (m_cg.T > 0).astype(np.float32)
            res_obs = np.zeros((n_genes, len(uniq_c)), dtype=np.float32)
            res_obs[g_rows, c_inv] = (y_res2 / 0.95).astype(np.float32)
            k_svd = min(model.gene_rank_d, len(uniq_c) - 1, n_genes - 1)
            if k_svd >= 8:
                # 5-iteration Soft-Impute Masked EM-SVD strictly on `train_indices` (`7%` observed density)
                mat_imp = res_obs.copy()
                svd_g = TruncatedSVD(n_components=k_svd, random_state=seed)
                u_sig = np.zeros((n_genes, k_svd), dtype=np.float32)
                s_shrunk = np.ones(k_svd, dtype=np.float32)
                n_em = 5 if len(tr_chr) >= 8000 else 2
                for _ in range(n_em):
                    u_sig = svd_g.fit_transform(mat_imp)
                    s_raw = np.maximum(svd_g.singular_values_, 1e-5)
                    s_shrunk = np.maximum(s_raw - (1.35 if n_em > 2 else 0.35), 1e-4).astype(np.float32)
                    u_shrunk = u_sig * (s_shrunk / s_raw)[None, :]
                    recon = (u_shrunk @ svd_g.components_).astype(np.float32)
                    mat_imp = np.where(obs_m_gc > 0.5, res_obs, recon)

                s_raw_fin = np.maximum(svd_g.singular_values_, 1e-5).astype(np.float32)
                sqrt_s = np.sqrt(np.maximum(s_shrunk, 1e-4)).astype(np.float32)
                g_factors = ((u_sig / s_raw_fin[None, :]) * sqrt_s[None, :]).astype(np.float32)
                c_factors = (svd_g.components_.T * sqrt_s[None, :]).astype(np.float32)

                model.gene_ctx_emb.zero_()
                model.gene_ctx_emb[:, :k_svd].copy_(torch.from_numpy(g_factors).to(device=device))
                # Fit on both multi-omic and RNA-only (`extra_u=0`) views so 3D organoid transfer is seamless!
                ko_aug_x = np.concatenate([ko_hid_u, ko_hid_z], axis=0)
                c_aug_y = np.concatenate([c_factors, c_factors], axis=0)
                w_ko, b_ko = _solve_ridge_np(ko_aug_x, c_aug_y, lam=140.0)
                last_ko_lin = model.cell_to_gene_ctx[-1]
                last_ko_lin.weight.zero_()
                last_ko_lin.bias.zero_()
                last_ko_lin.weight[:k_svd].copy_(torch.from_numpy(w_ko).to(device=device))
                last_ko_lin.bias[:k_svd].copy_(torch.from_numpy(b_ko).to(device=device))


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
    else:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if "OMP_NUM_THREADS" in os.environ:
            torch.set_num_threads(max(1, int(os.environ["OMP_NUM_THREADS"])))

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
    from ferroflux.data.dataset import get_fast_indexer

    fast_idxr = get_fast_indexer(dataset) if (not smoke_test and not use_ddp) else None
    sar_replay_tensors: tuple[torch.Tensor, ...] | None = None
    rng_seed = np.random.default_rng(seed + rank)

    # Warm-start Stage-0.5 SAR chemical representation first, then spectral/physical-anchor/CRISPR priors
    has_recs = hasattr(dataset, "records") and isinstance(dataset.records, list)
    all_tr_idx = list(range(len(dataset.records))) if has_recs else []
    if start_stage == 1 and start_epoch == 0 and not smoke_test:
        sar_info = pretrain_stage05_chem_sar(
            raw_model,
            data_dir=data_dir,
            license_track=license_track,
            device=device,
            train_dataset=dataset if has_recs else None,
            train_indices=all_tr_idx if has_recs else None,
            epochs=8,
            batch_size=256,
            seed=seed,
            smoke_test=smoke_test,
        )
        sar_replay_tensors = sar_info.get("sar_replay_tensors")
        if rank == 0 and not sar_info.get("skipped", True):
            print(
                f"[Seed {seed}] Stage-0.5 SAR pretraining done on {sar_info['n_molecules']:,} molecules "
                f"(loss={sar_info['final_sar_loss']})",
                flush=True,
            )

        if has_recs:
            warmstart_from_train_indices(
                raw_model,
                dataset,
                all_tr_idx,
                device=device,
                seed=seed,
            )

    # Build stage-matched index pools when running on FerroFluxMultimodalDataset
    ckpt_stage_pools: dict[int, np.ndarray] = {}
    if fast_idxr is not None and hasattr(dataset, "records"):
        by_k: dict[str, list[int]] = {}
        cr_3d: list[int] = []
        for idx_r, rc in enumerate(dataset.records):
            kd = str(rc.get("kind", "other"))
            by_k.setdefault(kd, []).append(idx_r)
            if kd == "crispr" and int(rc.get("is_3d_organoid", 0)) > 0:
                cr_3d.append(idx_r)

        def _cap_pool(lst: list[int], k_cap: int) -> list[int]:
            if len(lst) <= k_cap:
                return list(lst)
            return [int(x) for x in rng_seed.choice(lst, size=k_cap, replace=False).tolist()]

        icb_aug = by_k.get("icb", []) * 6 + by_k.get("normal_sfti", []) * 6
        ckpt_stage_pools = {
            1: np.asarray(by_k.get("basal", []) + icb_aug + _cap_pool(by_k.get("perturb", []), 3500) + _cap_pool(by_k.get("pharmaco", []), 2000), dtype=np.int64),
            2: np.asarray(_cap_pool(by_k.get("crispr", []), 48000) + by_k.get("perturb", []), dtype=np.int64),
            3: np.asarray(_cap_pool(by_k.get("pharmaco", []), 65000) + by_k.get("rescue", []) * 2 + _cap_pool(by_k.get("synergy", []), 5000) + _cap_pool(by_k.get("crispr", []), 6000), dtype=np.int64),
            4: np.asarray(_cap_pool(by_k.get("pharmaco", []), 22000) + cr_3d + _cap_pool(by_k.get("oxylipid", []), 4000) + icb_aug, dtype=np.int64),
        }

    for stage_num, stage_name, n_epochs, batch_size, stage_lr in schedule:
        if stage_num < start_stage:
            continue

        eff_bs = min(batch_size, max(2, len(dataset)))
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed) if use_ddp else None
        loader = (
            None
            if (fast_idxr is not None and stage_num in ckpt_stage_pools and ckpt_stage_pools[stage_num].size >= 8)
            else DataLoader(
                dataset,
                batch_size=eff_bs,
                shuffle=(sampler is None),
                sampler=sampler,
                collate_fn=collate_fn,
                drop_last=False,
            )
        )

        sar_prefixes = (
            "perturbation_op.chem_fp_proj.",
            "perturbation_op.chem_desc_proj.",
            "perturbation_op.chem_in_proj.",
            "perturbation_op.chem_blocks.",
            "perturbation_op.chem_out_norm.",
        )
        warm_anchor_names = (
            "gene_essentiality_prior",
            "gene_expr_slope_prior",
            "expr_ctx_proj.0.weight",
            "expr_ctx_proj.0.bias",
            "cell_chronos_offset.weight",
            "cell_chronos_offset.bias",
        )
        dense_p: list[torch.nn.Parameter] = []
        sar_p: list[torch.nn.Parameter] = []
        sar_head_p: list[torch.nn.Parameter] = []
        anchor_p: list[torch.nn.Parameter] = []
        gene_emb_p: list[torch.nn.Parameter] = []
        ecfp_shared_p: list[torch.nn.Parameter] = []
        greybox_p: list[torch.nn.Parameter] = []
        for n, p in ddp_or_raw.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith(warm_anchor_names):
                anchor_p.append(p)
            elif n.endswith("gene_ctx_emb"):
                gene_emb_p.append(p)
            elif "perturbation_op.chem_shared_ecfp_enc." in n:
                ecfp_shared_p.append(p)
            elif (
                "diff_ferro_gem.aphynity_" in n
                or "perturbation_op.chem_scaffold_" in n
                or n.endswith(("raw_scaf_gb_blend", "raw_scaf_gb_blend_ood"))
            ):
                greybox_p.append(p)
            elif "perturbation_op.sar_pretrain_head." in n:
                sar_head_p.append(p)
            elif (not raw_model.ablate_sar_pretrain and not raw_model.ablate_chem_prior) and any(sp in n for sp in sar_prefixes):
                sar_p.append(p)
            else:
                dense_p.append(p)
        param_groups = [{"params": dense_p, "lr": stage_lr, "weight_decay": weight_decay}]
        if greybox_p:
            param_groups.append({"params": greybox_p, "lr": stage_lr * 0.18, "weight_decay": 1.5e-2})
        if ecfp_shared_p:
            param_groups.append({"params": ecfp_shared_p, "lr": stage_lr * 0.12, "weight_decay": 5.0e-2})
        if sar_p:
            param_groups.append({"params": sar_p, "lr": stage_lr * 0.65, "weight_decay": 1e-4})
        if sar_head_p:
            param_groups.append({"params": sar_head_p, "lr": stage_lr * 0.08, "weight_decay": 1e-4})
        if anchor_p:
            param_groups.append({"params": anchor_p, "lr": min(stage_lr * 0.45, 2.0e-4), "weight_decay": 1e-4})
        if gene_emb_p:
            param_groups.append({"params": gene_emb_p, "lr": min(stage_lr * 2.0, 8.0e-4), "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(param_groups)
        if stage_num == start_stage and saved_opt_state is not None:
            try:
                optimizer.load_state_dict(saved_opt_state)
            except ValueError:
                pass
            saved_opt_state = None

        if loader is None:
            pool_sz = int(ckpt_stage_pools[stage_num].size)
            n_batches_avail = max(1, (pool_sz + eff_bs - 1) // eff_bs)
        else:
            n_batches_avail = len(loader)
        steps_per_ep = min(n_batches_avail, max_steps_per_stage) if max_steps_per_stage is not None else n_batches_avail
        total_steps = max(1, n_epochs * max(1, steps_per_ep))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=stage_lr * 0.1)

        ep_begin = start_epoch if stage_num == start_stage else 0
        for ep in range(ep_begin, n_epochs):
            if sampler is not None:
                sampler.set_epoch(ep)

            ddp_or_raw.train()
            ep_losses: list[float] = []
            t_ep = time.perf_counter()

            if loader is None:
                assert fast_idxr is not None
                perm_pool = rng_seed.permutation(ckpt_stage_pools[stage_num])
                step_iter: Any = (
                    (s_i, fast_idxr.get_batch(perm_pool[s_i * eff_bs : (s_i + 1) * eff_bs], device=device))
                    for s_i in range(steps_per_ep)
                )
            else:
                step_iter = enumerate(loader)

            for step_idx, raw_batch in step_iter:
                if max_steps_per_stage is not None and step_idx >= max_steps_per_stage:
                    break

                batch = raw_batch if loader is None else _move_batch_to_device(raw_batch, device)
                # Structured pathway/gene MGM corruption in Stage 1 and every 2nd step of Stage 4
                batch["enable_mgm_masking"] = bool(stage_num == 1 or (stage_num == 4 and step_idx % 2 == 0) or (step_idx % 6 == 0))
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    outputs = ddp_or_raw(batch)
                    loss_dict = loss_fn(outputs, batch, stage=stage_num)
                    loss = loss_dict["loss"]
                    if stage_num in (3, 4) and step_idx % 2 == 0 and sar_replay_tensors is not None:
                        sar_rep = compute_sar_replay_loss(raw_model, sar_replay_tensors, device, rng_seed)
                        if sar_rep is not None:
                            loss = loss + 0.20 * sar_rep

                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ddp_or_raw.parameters(), grad_clip)
                    optimizer.step()
                    scheduler.step()
                    ep_losses.append(float(loss.detach().item()))

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
            if rank == 0 and not smoke_test and (ep + 1 == n_epochs):
                print(
                    f"[Seed {seed}] Stage {stage_num}/4 ({stage_name}) epoch {ep+1}/{n_epochs} loss={mean_loss:.4f} ({rec['epoch_sec']:.1f}s)",
                    flush=True,
                )

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

        if stage_num == 1 and not smoke_test and not use_ddp and has_recs:
            warmstart_crispr_joint_and_svd(
                raw_model,
                dataset,
                all_tr_idx,
                device=device,
                seed=seed,
            )

    if not smoke_test and not use_ddp and has_recs:
        with torch.no_grad():
            raw_model.perturbation_op.cell_bioact_film[-1].weight.mul_(0.18)
            if raw_model.perturbation_op.cell_bioact_film[-1].bias is not None:
                raw_model.perturbation_op.cell_bioact_film[-1].bias.mul_(0.18)
        fit_post_curriculum_greybox_from_train_indices(
            raw_model,
            dataset,
            all_tr_idx,
            device=device,
            seed=seed,
        )
        calibrate_mgm_decoder_from_train_indices(
            raw_model,
            dataset,
            all_tr_idx,
            device=device,
            seed=seed,
        )

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
        import io

        buf = io.BytesIO()
        torch.save(ckpt_payload, buf)
        if final_ckpt_path.exists():
            try:
                final_ckpt_path.chmod(0o664)
            except OSError:
                pass
            final_ckpt_path.write_bytes(buf.getvalue())
        else:
            tmp_final = final_ckpt_path.with_name(f".tmp_{os.getpid()}_{final_ckpt_path.name}")
            tmp_final.write_bytes(buf.getvalue())
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


def smoke_test_safe_write(out_dir: Path) -> bool:
    return not str(out_dir).startswith("/anvil/projects/")


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

    if rank == 0 and (len(args.seeds) > 1 or int(args.seeds[0]) == 101 or smoke_test_safe_write(args.output_dir)):
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
