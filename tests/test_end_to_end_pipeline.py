"""End-to-End Integration Tests: Real Parquet Shards -> FerroFluxMultimodalDataset -> FerroFluxFM.

Verifies:
1. Real processed Parquet shards load cleanly through FerroFluxMultimodalDataset with canonical gene_vocab.json.
2. 'open' vs 'academic' license filtering works end-to-end on real processed shards.
3. Disjoint OOD splitters (scaffold, leave-out-lineage Kidney/Melanoma, 2D->3D organoids, dual CRISPR+FIN).
4. Finite loss and parameter gradients across all 4 curriculum stages + spatial wave rollout + all 5 ablation flags.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def test_real_dataset_loading_license_tracks_and_ood_splitters() -> None:
    from ferroflux.data import dataset as ds_mod

    ds_cls = getattr(ds_mod, "FerroFluxMultimodalDataset")
    sig = inspect.signature(ds_cls.__init__)
    base_kw: dict[str, Any] = {}
    if "data_dir" in sig.parameters:
        base_kw["data_dir"] = str(PROCESSED_DIR)
    elif "processed_dir" in sig.parameters:
        base_kw["processed_dir"] = str(PROCESSED_DIR)

    track_key = "license_track" if "license_track" in sig.parameters else ("track" if "track" in sig.parameters else None)
    assert track_key is not None, "FerroFluxMultimodalDataset must accept license_track='open'|'academic'"

    ds_open = ds_cls(**base_kw, **{track_key: "open"})
    ds_acad = ds_cls(**base_kw, **{track_key: "academic"})

    assert len(ds_open) > 0, "Open-track dataset has 0 samples!"
    assert len(ds_acad) >= len(ds_open), "Academic-track dataset must be a superset of Open-track dataset!"

    sample = ds_acad[0]
    assert isinstance(sample, dict) and "gene_expr" in sample
    assert torch.isfinite(sample["gene_expr"]).all()

    # Check OOD splitters produce non-empty, strictly disjoint train/test splits
    for splitter_name in [
        "scaffold_split",
        "lineage_split",
        "organoid_2d_to_3d_split",
        "crispr_fin_epistasis_split",
    ]:
        if hasattr(ds_acad, splitter_name):
            train_idx, test_idx = getattr(ds_acad, splitter_name)()
            assert len(train_idx) > 0 and len(test_idx) > 0
            assert set(train_idx).isdisjoint(set(test_idx)), f"Data leak between train & test in {splitter_name}!"
        elif hasattr(ds_mod, splitter_name):
            train_idx, test_idx = getattr(ds_mod, splitter_name)(ds_acad)
            assert len(train_idx) > 0 and len(test_idx) > 0
            assert set(train_idx).isdisjoint(set(test_idx)), f"Data leak between train & test in {splitter_name}!"


def test_end_to_end_four_stage_losses_spatial_wave_and_ablations() -> None:
    from ferroflux.data.dataset import FerroFluxMultimodalDataset
    from ferroflux.models.ferroflux_fm import FerroFluxFM
    from ferroflux.models import losses as loss_mod

    sig_ds = inspect.signature(FerroFluxMultimodalDataset.__init__)
    ds_kw: dict[str, Any] = {}
    if "data_dir" in sig_ds.parameters:
        ds_kw["data_dir"] = str(PROCESSED_DIR)
    elif "processed_dir" in sig_ds.parameters:
        ds_kw["processed_dir"] = str(PROCESSED_DIR)
    ds = FerroFluxMultimodalDataset(**ds_kw)

    collate = getattr(ds, "collate_fn", None)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate)
    batch = next(iter(loader))

    n_genes = batch["gene_expr"].shape[-1]
    sig_m = inspect.signature(FerroFluxFM.__init__)
    m_kw: dict[str, Any] = {}
    if "n_genes" in sig_m.parameters:
        m_kw["n_genes"] = n_genes

    model = FerroFluxFM(**m_kw)
    loss_cls = getattr(loss_mod, "FerroFluxCurriculumLoss", getattr(loss_mod, "FerroFluxLoss", None))
    loss_fn = loss_cls() if loss_cls is not None else getattr(loss_mod, "compute_curriculum_loss")

    for stage in (1, 2, 3, 4):
        model.train()
        model.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss_out = loss_fn(outputs, batch, stage=stage)
        loss = loss_out["loss"] if isinstance(loss_out, dict) else (loss_out[0] if isinstance(loss_out, tuple) else loss_out)
        assert torch.isfinite(loss).all(), f"Non-finite loss in curriculum stage {stage}: {loss}"
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0 and all(gn == gn and gn < 1e8 for gn in grad_norms), (
            f"Invalid/NaN gradients in curriculum stage {stage}"
        )

    # Verify 2D spatial reaction-diffusion wave rollout API
    model.eval()
    assert hasattr(model, "predict_spatial_wave"), "FerroFluxFM must implement predict_spatial_wave()"
    coords = torch.rand(16, 2) * 100.0
    expr_grid = batch["gene_expr"][:1].repeat(16, 1)
    wave_sig = inspect.signature(model.predict_spatial_wave)
    if "coords" in wave_sig.parameters:
        wave_out = model.predict_spatial_wave(expr_grid, coords=coords)
    else:
        wave_out = model.predict_spatial_wave(expr_grid, coords)
    assert isinstance(wave_out, dict)
    for req_key in ("c_wave", "morans_i"):
        matched = [v for k, v in wave_out.items() if req_key in k.lower()]
        assert matched, f"predict_spatial_wave output missing {req_key!r}: got keys {list(wave_out.keys())}"
        t_val = matched[0]
        if isinstance(t_val, torch.Tensor):
            assert torch.isfinite(t_val).all()

    # Verify all 5 Reviewer #2 ablation flags instantiate and execute forward cleanly
    for ablate_flag in [
        "ablate_diff_ferro_gem",
        "ablate_sec46_gate",
        "ablate_survivorship",
        "ablate_chem_prior",
        "ablate_physical_anchors",
    ]:
        assert ablate_flag in sig_m.parameters, f"FerroFluxFM.__init__ missing required ablation flag {ablate_flag!r}"
        ab_model = FerroFluxFM(**m_kw, **{ablate_flag: True})
        ab_model.eval()
        with torch.no_grad():
            ab_out = ab_model(batch)
        assert isinstance(ab_out, dict)
