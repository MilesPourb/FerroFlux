"""Reviewer #2 Multi-Omic Tabular, Chemical-MLP & Linear Perturbation Baselines.

Implements six strong non-mechanistic baselines against which FerroFlux-FM is benchmarked
across all six compositional Out-Of-Distribution (OOD) splits while strictly enforcing
Tier-3 Bradbury/Overholtzer/Wiesner quarantine and 'open' vs 'academic' license tracks:

1. `RidgeMultiOmicBaseline`: L2-regularized `RidgeCV` on basal `TPM + ProCan + CCLE-Metab + ChemDesc/Dose/Media/NP/KO`.
2. `ElasticNetMultiOmicBaseline`: L1/L2 `ElasticNetCV` / `SGDRegressor(penalty='elasticnet')` on multi-omic & perturbation features.
3. `LightGBMMultiOmicBaseline`: Histogram-binned `LGBMRegressor` / `LGBMClassifier` on full multi-omic + 2048-bit Morgan FP + NP features.
4. `XGBoostMultiOmicBaseline`: `XGBRegressor(tree_method='hist')` on full multi-omic + 2048-bit Morgan FP + NP features.
5. `ChempropCellStateMLPBaseline`: Permutation-symmetric 2-branch PyTorch MLP on `2048`-bit Morgan FP + `12`-dim covalent/RTA descriptors
   fused via bilinear Hadamard interaction with a basal multi-omic cell-state MLP (AdamW + cosine annealing).
6. `LinearPerturbBaseline`: *Ahlmann-Eltze et al., Nature Methods 2025* additive linear perturbation model predicting
   post-perturbation transcriptional shift `delta_expr` (`(B, G)`) and downstream viability / genetic epistasis.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ElasticNetCV, RidgeCV, SGDRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ferroflux.security.license_guard import filter_by_license_track
from ferroflux.security.quarantine import audit_parquet_directory

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_OUT_CSV = ROOT_DIR / "results" / "tables" / "baselines_ood_summary.csv"

OOD_SPLIT_NAMES: tuple[str, ...] = (
    "1_leave_out_scaffold",
    "2_leave_out_lineage_ccrcc_melanoma",
    "3_2d_to_3d_organoid_pdx",
    "4_dual_crispr_fin_epistasis",
    "5_cuproptosis_elesclomol_specificity",
    "6_sfti_tumor_vs_normal_til",
)

OOD_DATASET_METHOD_MAP: dict[str, str] = {
    "1_leave_out_scaffold": "scaffold_split",
    "2_leave_out_lineage_ccrcc_melanoma": "lineage_split",
    "3_2d_to_3d_organoid_pdx": "organoid_2d_to_3d_split",
    "4_dual_crispr_fin_epistasis": "crispr_fin_epistasis_split",
    "5_cuproptosis_elesclomol_specificity": "cuproptosis_specificity_split",
    "6_sfti_tumor_vs_normal_til": "sfti_normal_til_split",
}


# ---------------------------------------------------------------------------
# Statistical Metrics & Vectorized Non-Parametric Bootstrap 95% CIs
# ---------------------------------------------------------------------------


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.size < 3 or np.std(yt) < 1e-12 or np.std(yp) < 1e-12:
        return 0.0
    val = float(spearmanr(yt, yp).statistic)
    return 0.0 if np.isnan(val) else val


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.size < 3 or np.std(yt) < 1e-12 or np.std(yp) < 1e-12:
        return 0.0
    val = float(pearsonr(yt, yp).statistic)
    return 0.0 if np.isnan(val) else val


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(yt, yp)))


def _safe_auroc(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    yt = np.asarray(y_true_binary, dtype=np.int32).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    if yt.size < 4 or np.unique(yt).size < 2:
        return float("nan")
    return float(roc_auc_score(yt, ys))


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 400,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute `(point_estimate, ci_low, ci_high)` via paired non-parametric bootstrap."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    point = float(metric_fn(yt, yp))
    n = yt.shape[0]
    if n < 4 or n_boot <= 0 or np.isnan(point):
        return point, point, point

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = float(metric_fn(yt[idx], yp[idx]))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, point, point
    lo = float(np.percentile(vals, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(vals, 100.0 * (1.0 - alpha / 2.0)))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Multimodal Feature Extraction & Synthetic Smoke-Test Generator
# ---------------------------------------------------------------------------


def _to_np(x: Any, default_shape: tuple[int, ...], fill: float = 0.0) -> np.ndarray:
    if x is None:
        return np.full(default_shape, fill, dtype=np.float32)
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        arr = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(arr, nan=fill, posinf=10.0, neginf=-10.0)


def _slice_x_tabular(arrays: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    """Build compact multi-omic + perturbation matrix for `idx` on demand (<200 MB RSS)."""
    if "x_tabular" in arrays:
        return arrays["x_tabular"][idx]
    m = len(idx)
    parts = [
        arrays["gene_expr"][idx],
        arrays["prot_anchor"][idx],
        arrays["metab_anchor"][idx],
    ]
    for ctx_key in ("genomic_context", "prot64_context", "lipid32_context"):
        if ctx_key in arrays:
            parts.append(arrays[ctx_key][idx])
    parts.extend(
        [
            arrays["chem_desc_a"][idx],
            arrays["chem_desc_b"][idx],
            arrays["dose_a"][idx],
            arrays["dose_b"][idx],
            arrays["media_ext"][idx],
            arrays["np_params"][idx],
            arrays["ko_core64"][idx],
            arrays["is_3d_organoid"][idx].reshape(m, 1),
        ]
    )
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _slice_x_gbdt(arrays: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    """Build full multi-omic + permutation-invariant 2048-bit Morgan FP matrix for `idx` on demand (<400 MB RSS)."""
    if "x_gbdt" in arrays:
        return arrays["x_gbdt"][idx]
    x_tab = _slice_x_tabular(arrays, idx)
    fpa = arrays["chem_fp_a"][idx].astype(np.float32)
    fpb = arrays["chem_fp_b"][idx].astype(np.float32)
    fp_sum = fpa + fpb
    fp_max = np.maximum(fpa, fpb)
    return np.concatenate([x_tab, fp_sum, fp_max], axis=1)


def extract_multimodal_arrays(
    dataset_or_dict: Any,
    max_samples: int | None = None,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Extract canonical numpy arrays adhering to the FerroFluxMultimodalDataset (#40) schema with bounded (<3 GB) RSS."""
    if isinstance(dataset_or_dict, dict):
        raw = dict(dataset_or_dict)
        n = np.asarray(raw["gene_expr"]).shape[0]
        if max_samples is not None and n > max_samples:
            raw = {k: np.asarray(v)[:max_samples] for k, v in raw.items()}
            n = max_samples

        gene_expr = _to_np(raw.get("gene_expr"), (n, 2048), 0.0)
        g_dim = gene_expr.shape[1]
        prot_mask = _to_np(raw.get("prot_anchor_mask"), (n, 28), 1.0 if "prot_anchor" in raw else 0.0)
        prot_raw = _to_np(raw.get("prot_anchor", raw.get("prot_anchor_target")), (n, 28), 0.0)
        prot_anchor = prot_raw * prot_mask
        metab_mask = _to_np(raw.get("metab_anchor_mask"), (n, 10), 1.0 if "metab_anchor" in raw else 0.0)
        metab_raw = _to_np(raw.get("metab_anchor", raw.get("metab_anchor_target")), (n, 10), 0.0)
        metab_anchor = metab_raw * metab_mask
        genomic_context = _to_np(raw.get("genomic_context"), (n, 138), 0.0)
        prot64_context = _to_np(raw.get("prot64_context"), (n, 64), 0.0)
        lipid32_context = _to_np(raw.get("lipid32_context"), (n, 32), 0.0)

        raw_fpa = raw.get("chem_fp_a")
        fp_dim = int(np.asarray(raw_fpa).shape[-1]) if raw_fpa is not None and np.asarray(raw_fpa).ndim >= 2 else 2048
        raw_da = raw.get("chem_desc_a")
        desc_dim = int(np.asarray(raw_da).shape[-1]) if raw_da is not None and np.asarray(raw_da).ndim >= 2 else 12

        chem_fp_a = (_to_np(raw_fpa, (n, fp_dim), 0.0) > 0.5).astype(np.uint8)
        chem_fp_b = (_to_np(raw.get("chem_fp_b"), (n, fp_dim), 0.0) > 0.5).astype(np.uint8)
        chem_desc_a = _to_np(raw_da, (n, desc_dim), 0.0)
        chem_desc_b = _to_np(raw.get("chem_desc_b"), (n, desc_dim), 0.0)
        dose_a = _to_np(raw.get("dose_a"), (n,), 0.0).reshape(n, 1)
        dose_b = _to_np(raw.get("dose_b"), (n,), 0.0).reshape(n, 1)
        media_ext = _to_np(raw.get("media_ext"), (n, 10), 1.0)
        np_params = _to_np(raw.get("np_params"), (n, 5), 0.0)
        gene_ko_f32 = _to_np(raw.get("gene_ko_mult"), (n, g_dim), 1.0)
        gene_ko_mult = gene_ko_f32.astype(np.float16)
        is_3d = _to_np(raw.get("is_3d_organoid"), (n,), 0.0).reshape(n)
        is_normal_til = _to_np(raw.get("is_normal_organ_or_til"), (n,), 0.0).reshape(n)
        is_ferro_cmpd = _to_np(raw.get("is_ferroptosis_compound"), (n,), 1.0).reshape(n)
        elesclomol_conc = _to_np(raw.get("elesclomol_conc"), (n,), 0.0).reshape(n)
        gpx4_inhib = _to_np(raw.get("gpx4_inhib"), (n,), 0.0).reshape(n, 1)
        xc_inhib = _to_np(raw.get("xc_inhib"), (n,), 0.0).reshape(n, 1)
        rta_conc = _to_np(raw.get("rta_conc"), (n,), 0.0).reshape(n, 1)
        chelator_conc = _to_np(raw.get("chelator_conc"), (n,), 0.0).reshape(n, 1)

        viability_raw = _to_np(raw.get("viability_target"), (n,), 0.5).reshape(n)
        viability_mask = _to_np(raw.get("viability_mask"), (n,), 1.0).reshape(n)
        chronos_target = _to_np(raw.get("chronos_target"), (n,), 0.0).reshape(n)
        chronos_mask = _to_np(raw.get("chronos_mask"), (n,), 0.0).reshape(n)
        synergy_target = _to_np(raw.get("synergy_target"), (n,), 0.0).reshape(n)
        synergy_mask = _to_np(raw.get("synergy_mask"), (n,), 0.0).reshape(n)
        hazard_target = _to_np(raw.get("hazard_target"), (n, 5), 0.0)
        hazard_mask = _to_np(raw.get("hazard_mask"), (n, 5), 1.0)
        delta_expr_f32 = _to_np(raw.get("delta_expr_target"), (n, g_dim), 0.0)
        delta_expr_target = delta_expr_f32.astype(np.float16)
        delta_expr_mask = (_to_np(raw.get("delta_expr_mask"), (n, g_dim), 1.0) > 0.5)
        has_dx_row = (np.abs(delta_expr_f32).max(axis=1) > 1e-6) & (delta_expr_mask.sum(axis=1) >= 8)
        rescue_delta_target = _to_np(raw.get("rescue_delta_target"), (n,), 0.0).reshape(n)
        rescue_mask = _to_np(raw.get("rescue_mask"), (n,), 0.0).reshape(n)

        ko_shift_all = 1.0 - gene_ko_f32
        ko_proj_rng = np.random.default_rng(2026)
        ko_proj_mat = ko_proj_rng.standard_normal((max(1, g_dim - min(64, g_dim)), 128)).astype(np.float32) / np.sqrt(128.0)
        ko_tail = ko_shift_all[:, min(64, g_dim) :]
        ko_sketch128 = (ko_tail @ ko_proj_mat[: ko_tail.shape[1]]) if ko_tail.shape[1] > 0 else np.zeros((n, 128), dtype=np.float32)
        ko_basal_expr = (gene_expr * ko_shift_all).sum(axis=1, keepdims=True)
        ko_min_mult = gene_ko_f32.min(axis=1, keepdims=True)
        ko_n_genes = (gene_ko_f32 < 0.85).sum(axis=1, keepdims=True).astype(np.float32)
        ko_core64 = np.concatenate(
            [
                ko_shift_all[:, : min(64, g_dim)],
                ko_sketch128,
                ko_basal_expr,
                ko_min_mult,
                ko_n_genes,
                gpx4_inhib,
                xc_inhib,
                rta_conc,
                chelator_conc,
                elesclomol_conc.reshape(n, 1),
            ],
            axis=1,
        ).astype(np.float32)

        lineage = np.asarray(raw.get("lineage", ["Other"] * n)).astype(str)
        scaffold_id = np.asarray(raw.get("scaffold_id", [f"scaf_{i % 20}" for i in range(n)])).astype(str)
        group_id = np.asarray(raw.get("group_id", scaffold_id)).astype(str)
        license_track = np.asarray(raw.get("license_track", ["open"] * n)).astype(str)
    else:
        n_total = len(dataset_or_dict)
        n = min(n_total, max_samples) if max_samples is not None else n_total
        collate = getattr(dataset_or_dict, "collate_fn", None)
        eff_bs = 2048 if n > 2048 else batch_size
        loader = DataLoader(
            dataset_or_dict,
            batch_size=eff_bs,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )
        sample0 = dataset_or_dict[0]
        g_dim = int(sample0["gene_expr"].shape[-1])
        fp_dim = int(sample0["chem_fp_a"].shape[-1]) if "chem_fp_a" in sample0 else 2048
        desc_dim = int(sample0["chem_desc_a"].shape[-1]) if "chem_desc_a" in sample0 else 12
        gen_dim = int(sample0["genomic_context"].shape[-1]) if "genomic_context" in sample0 else 138
        p64_dim = int(sample0["prot64_context"].shape[-1]) if "prot64_context" in sample0 else 64
        l32_dim = int(sample0["lipid32_context"].shape[-1]) if "lipid32_context" in sample0 else 32
        k64_dim = min(64, g_dim) + 128 + 8
        ko_proj_rng = np.random.default_rng(2026)
        ko_proj_mat = ko_proj_rng.standard_normal((max(1, g_dim - min(64, g_dim)), 128)).astype(np.float32) / np.sqrt(128.0)

        gene_expr = np.empty((n, g_dim), dtype=np.float32)
        prot_anchor = np.empty((n, 28), dtype=np.float32)
        metab_anchor = np.empty((n, 10), dtype=np.float32)
        genomic_context = np.empty((n, gen_dim), dtype=np.float32)
        prot64_context = np.empty((n, p64_dim), dtype=np.float32)
        lipid32_context = np.empty((n, l32_dim), dtype=np.float32)
        chem_fp_a = np.empty((n, fp_dim), dtype=np.uint8)
        chem_fp_b = np.empty((n, fp_dim), dtype=np.uint8)
        chem_desc_a = np.empty((n, desc_dim), dtype=np.float32)
        chem_desc_b = np.empty((n, desc_dim), dtype=np.float32)
        dose_a = np.empty((n, 1), dtype=np.float32)
        dose_b = np.empty((n, 1), dtype=np.float32)
        media_ext = np.empty((n, 10), dtype=np.float32)
        np_params = np.empty((n, 5), dtype=np.float32)
        gene_ko_mult = np.empty((n, g_dim), dtype=np.float16)
        ko_core64 = np.empty((n, k64_dim), dtype=np.float32)
        is_3d = np.empty(n, dtype=np.float32)
        is_normal_til = np.empty(n, dtype=np.float32)
        is_ferro_cmpd = np.empty(n, dtype=np.float32)
        elesclomol_conc = np.empty(n, dtype=np.float32)
        viability_raw = np.empty(n, dtype=np.float32)
        viability_mask = np.empty(n, dtype=np.float32)
        chronos_target = np.empty(n, dtype=np.float32)
        chronos_mask = np.empty(n, dtype=np.float32)
        synergy_target = np.empty(n, dtype=np.float32)
        synergy_mask = np.empty(n, dtype=np.float32)
        hazard_target = np.empty((n, 5), dtype=np.float32)
        hazard_mask = np.empty((n, 5), dtype=np.float32)
        delta_expr_target = np.empty((n, g_dim), dtype=np.float16)
        delta_expr_mask = np.empty((n, g_dim), dtype=bool)
        has_dx_row = np.empty(n, dtype=bool)
        rescue_delta_target = np.empty(n, dtype=np.float32)
        rescue_mask = np.empty(n, dtype=np.float32)
        lin_list: list[str] = []
        scaf_list: list[str] = []
        lic_list: list[str] = []

        seen = 0
        for batch in loader:
            bsz = batch["gene_expr"].shape[0]
            take = min(bsz, n - seen)
            sl = slice(seen, seen + take)

            gx_b = _to_np(batch.get("gene_expr"), (bsz, g_dim), 0.0)[:take]
            gene_expr[sl] = gx_b

            pm_b = _to_np(batch.get("prot_anchor_mask"), (bsz, 28), 1.0 if "prot_anchor" in batch else 0.0)[:take]
            pr_b = _to_np(batch.get("prot_anchor", batch.get("prot_anchor_target")), (bsz, 28), 0.0)[:take]
            prot_anchor[sl] = pr_b * pm_b

            mm_b = _to_np(batch.get("metab_anchor_mask"), (bsz, 10), 1.0 if "metab_anchor" in batch else 0.0)[:take]
            mr_b = _to_np(batch.get("metab_anchor", batch.get("metab_anchor_target")), (bsz, 10), 0.0)[:take]
            metab_anchor[sl] = mr_b * mm_b
            genomic_context[sl] = _to_np(batch.get("genomic_context"), (bsz, gen_dim), 0.0)[:take]
            prot64_context[sl] = _to_np(batch.get("prot64_context"), (bsz, p64_dim), 0.0)[:take]
            lipid32_context[sl] = _to_np(batch.get("lipid32_context"), (bsz, l32_dim), 0.0)[:take]

            chem_fp_a[sl] = (_to_np(batch.get("chem_fp_a"), (bsz, fp_dim), 0.0)[:take] > 0.5).astype(np.uint8)
            chem_fp_b[sl] = (_to_np(batch.get("chem_fp_b"), (bsz, fp_dim), 0.0)[:take] > 0.5).astype(np.uint8)
            chem_desc_a[sl] = _to_np(batch.get("chem_desc_a"), (bsz, desc_dim), 0.0)[:take]
            chem_desc_b[sl] = _to_np(batch.get("chem_desc_b"), (bsz, desc_dim), 0.0)[:take]
            dose_a[sl] = _to_np(batch.get("dose_a"), (bsz,), 0.0)[:take].reshape(take, 1)
            dose_b[sl] = _to_np(batch.get("dose_b"), (bsz,), 0.0)[:take].reshape(take, 1)
            media_ext[sl] = _to_np(batch.get("media_ext"), (bsz, 10), 1.0)[:take]
            np_params[sl] = _to_np(batch.get("np_params"), (bsz, 5), 0.0)[:take]

            ko_b = _to_np(batch.get("gene_ko_mult"), (bsz, g_dim), 1.0)[:take]
            gene_ko_mult[sl] = ko_b.astype(np.float16)
            ko_sh = 1.0 - ko_b
            ko_tail_b = ko_sh[:, min(64, g_dim) :]
            ko_sk_b = (ko_tail_b @ ko_proj_mat[: ko_tail_b.shape[1]]) if ko_tail_b.shape[1] > 0 else np.zeros((take, 128), dtype=np.float32)
            el_b = _to_np(batch.get("elesclomol_conc"), (bsz,), 0.0)[:take]
            elesclomol_conc[sl] = el_b
            ko_core64[sl] = np.concatenate(
                [
                    ko_sh[:, : min(64, g_dim)],
                    ko_sk_b,
                    (gx_b * ko_sh).sum(axis=1, keepdims=True),
                    ko_b.min(axis=1, keepdims=True),
                    (ko_b < 0.85).sum(axis=1, keepdims=True).astype(np.float32),
                    _to_np(batch.get("gpx4_inhib"), (bsz,), 0.0)[:take].reshape(take, 1),
                    _to_np(batch.get("xc_inhib"), (bsz,), 0.0)[:take].reshape(take, 1),
                    _to_np(batch.get("rta_conc"), (bsz,), 0.0)[:take].reshape(take, 1),
                    _to_np(batch.get("chelator_conc"), (bsz,), 0.0)[:take].reshape(take, 1),
                    el_b.reshape(take, 1),
                ],
                axis=1,
            )

            is_3d[sl] = _to_np(batch.get("is_3d_organoid"), (bsz,), 0.0)[:take].reshape(take)
            is_normal_til[sl] = _to_np(batch.get("is_normal_organ_or_til"), (bsz,), 0.0)[:take].reshape(take)
            is_ferro_cmpd[sl] = _to_np(batch.get("is_ferroptosis_compound"), (bsz,), 1.0)[:take].reshape(take)
            viability_raw[sl] = _to_np(batch.get("viability_target"), (bsz,), 0.5)[:take].reshape(take)
            viability_mask[sl] = _to_np(batch.get("viability_mask"), (bsz,), 1.0)[:take].reshape(take)
            chronos_target[sl] = _to_np(batch.get("chronos_target"), (bsz,), 0.0)[:take].reshape(take)
            chronos_mask[sl] = _to_np(batch.get("chronos_mask"), (bsz,), 0.0)[:take].reshape(take)
            synergy_target[sl] = _to_np(batch.get("synergy_target"), (bsz,), 0.0)[:take].reshape(take)
            synergy_mask[sl] = _to_np(batch.get("synergy_mask"), (bsz,), 0.0)[:take].reshape(take)
            hazard_target[sl] = _to_np(batch.get("hazard_target"), (bsz, 5), 0.0)[:take]
            hazard_mask[sl] = _to_np(batch.get("hazard_mask"), (bsz, 5), 1.0)[:take]

            dxt_b = _to_np(batch.get("delta_expr_target"), (bsz, g_dim), 0.0)[:take]
            dxm_b = _to_np(batch.get("delta_expr_mask"), (bsz, g_dim), 1.0)[:take] > 0.5
            delta_expr_target[sl] = dxt_b.astype(np.float16)
            delta_expr_mask[sl] = dxm_b
            has_dx_row[sl] = (np.abs(dxt_b).max(axis=1) > 1e-6) & (dxm_b.sum(axis=1) >= 8)

            rescue_delta_target[sl] = _to_np(batch.get("rescue_delta_target"), (bsz,), 0.0)[:take].reshape(take)
            rescue_mask[sl] = _to_np(batch.get("rescue_mask"), (bsz,), 0.0)[:take].reshape(take)

            lin_b = batch.get("lineage")
            lin_list.extend([str(x) for x in lin_b[:take]] if lin_b is not None else ["Other"] * take)
            scaf_b = batch.get("scaffold_id")
            scaf_list.extend([str(x) for x in scaf_b[:take]] if scaf_b is not None else [f"scaf_{(seen + j) % 20}" for j in range(take)])
            lic_b = batch.get("license_track")
            lic_list.extend([str(x) for x in lic_b[:take]] if lic_b is not None else ["open"] * take)

            seen += take
            if seen >= n:
                break

        lineage = np.asarray(lin_list, dtype=str)
        scaffold_id = np.asarray(scaf_list, dtype=str)
        license_track = np.asarray(lic_list, dtype=str)
        if hasattr(dataset_or_dict, "records") and len(dataset_or_dict.records) >= n:
            recs_sub = dataset_or_dict.records[:n]
            group_id = np.asarray(
                [
                    str(
                        r.get("compound_name")
                        or r.get("compound_a")
                        or (f"GENE_{r.get('gene_idx')}" if r.get("kind") == "crispr" else "")
                        or r.get("fin_modifier_compound")
                        or scaf_list[idx_i]
                    )
                    for idx_i, r in enumerate(recs_sub)
                ],
                dtype=str,
            )
        else:
            group_id = scaffold_id.copy()

    chronos_surv = (1.0 / (1.0 + np.exp(-2.0 * np.clip(chronos_target + 0.5, -8.0, 8.0)))).astype(np.float32)
    synergy_surv = np.clip(0.50 - 0.02 * synergy_target, 0.02, 0.98).astype(np.float32)
    rescue_surv = np.clip(0.25 + 0.70 * rescue_delta_target, 0.02, 0.98).astype(np.float32)

    viability_target = viability_raw.copy()
    m_chr = (viability_mask <= 0.5) & (chronos_mask > 0.5)
    viability_target[m_chr] = chronos_surv[m_chr]
    m_syn = (viability_mask <= 0.5) & (chronos_mask <= 0.5) & (synergy_mask > 0.5)
    viability_target[m_syn] = synergy_surv[m_syn]
    m_res = (viability_mask <= 0.5) & (chronos_mask <= 0.5) & (synergy_mask <= 0.5) & (rescue_mask > 0.5)
    viability_target[m_res] = rescue_surv[m_res]

    has_fitness_label = (
        (viability_mask > 0.5)
        | (chronos_mask > 0.5)
        | (synergy_mask > 0.5)
        | (rescue_mask > 0.5)
    ).astype(np.float32)

    out: dict[str, np.ndarray] = {
        "gene_expr": gene_expr,
        "prot_anchor": prot_anchor,
        "metab_anchor": metab_anchor,
        "genomic_context": genomic_context,
        "prot64_context": prot64_context,
        "lipid32_context": lipid32_context,
        "chem_fp_a": chem_fp_a,
        "chem_fp_b": chem_fp_b,
        "chem_desc_a": chem_desc_a,
        "chem_desc_b": chem_desc_b,
        "dose_a": dose_a,
        "dose_b": dose_b,
        "media_ext": media_ext,
        "np_params": np_params,
        "gene_ko_mult": gene_ko_mult,
        "ko_core64": ko_core64,
        "is_3d_organoid": is_3d,
        "is_normal_organ_or_til": is_normal_til,
        "is_ferroptosis_compound": is_ferro_cmpd,
        "elesclomol_conc": elesclomol_conc,
        "viability_target": viability_target,
        "viability_raw": viability_raw,
        "viability_mask": viability_mask,
        "chronos_target": chronos_target,
        "chronos_mask": chronos_mask,
        "has_fitness_label": has_fitness_label,
        "has_dx_row": has_dx_row,
        "hazard_target": hazard_target,
        "hazard_mask": hazard_mask,
        "delta_expr_target": delta_expr_target,
        "delta_expr_mask": delta_expr_mask,
        "rescue_delta_target": rescue_delta_target,
        "rescue_mask": rescue_mask,
        "lineage": lineage,
        "scaffold_id": scaffold_id,
        "group_id": group_id,
        "license_track": license_track,
    }
    if n <= 4096:
        all_i = np.arange(n, dtype=np.int64)
        out["x_tabular"] = _slice_x_tabular(out, all_i)
        out["x_gbdt"] = _slice_x_gbdt(out, all_i)
    return out


def build_synthetic_multimodal_arrays(
    n_samples: int = 360,
    n_genes: int = 256,
    seed: int = 2026,
) -> dict[str, Any]:
    """Generate a deterministic synthetic cohort obeying the #40 multimodal schema for smoke tests."""
    rng = np.random.default_rng(seed)
    gene_expr = rng.normal(loc=2.5, scale=1.1, size=(n_samples, n_genes)).astype(np.float32)
    prot_anchor = rng.normal(loc=0.0, scale=0.8, size=(n_samples, 28)).astype(np.float32)
    metab_anchor = rng.normal(loc=0.0, scale=0.8, size=(n_samples, 10)).astype(np.float32)

    chem_fp_a = (rng.random((n_samples, 2048)) < 0.04).astype(np.float32)
    chem_fp_b = (rng.random((n_samples, 2048)) < 0.015).astype(np.float32)
    chem_desc_a = rng.normal(size=(n_samples, 12)).astype(np.float32)
    chem_desc_b = rng.normal(size=(n_samples, 12)).astype(np.float32) * 0.3
    dose_a = rng.uniform(-2.0, 1.5, size=(n_samples,)).astype(np.float32)
    dose_b = rng.uniform(-3.0, 0.8, size=(n_samples,)).astype(np.float32)
    media_ext = np.ones((n_samples, 10), dtype=np.float32)
    media_ext[:, 0] = rng.choice([1.0, 0.1, 0.02], size=n_samples, p=[0.7, 0.2, 0.1])

    has_np = rng.random(n_samples) < 0.18
    np_params = np.zeros((n_samples, 5), dtype=np.float32)
    np_params[has_np, 0] = rng.uniform(5.0, 40.0, size=int(has_np.sum()))
    np_params[has_np, 2] = rng.uniform(0.2, 1.5, size=int(has_np.sum()))
    np_params[has_np, 4] = rng.uniform(1.0, 15.0, size=int(has_np.sum()))

    gene_ko_mult = np.ones((n_samples, n_genes), dtype=np.float32)
    ko_rows = rng.random(n_samples) < 0.30
    for idx in np.where(ko_rows)[0]:
        ko_gene = int(rng.integers(0, min(32, n_genes)))
        gene_ko_mult[idx, ko_gene] = 0.05

    is_3d = (rng.random(n_samples) < 0.20).astype(np.float32)
    is_normal_til = (rng.random(n_samples) < 0.18).astype(np.float32)
    elesclomol_conc = np.where(rng.random(n_samples) < 0.16, rng.uniform(0.5, 5.0, size=n_samples), 0.0).astype(np.float32)
    lineages = rng.choice(["Lung", "Breast", "Kidney", "Melanoma", "Pancreas", "Liver"], size=n_samples)
    scaffolds = np.array([f"murcko_{i % 18:02d}" for i in range(n_samples)])
    licenses = rng.choice(["permissive", "CC-BY-4.0", "CC-BY-NC-SA-4.0"], size=n_samples, p=[0.5, 0.3, 0.2])

    # Mechanistically structured ground-truth signal (GPX4=0, SLC7A11=1, ACSL4=24, DHCR7=36)
    gpx4_act = (gene_expr[:, 0] + prot_anchor[:, 0]) * gene_ko_mult[:, 0]
    acsl4_sig = gene_expr[:, min(24, n_genes - 1)] * gene_ko_mult[:, min(24, n_genes - 1)]
    warhead = chem_desc_a[:, 6] * np.maximum(dose_a + 1.5, 0.0)
    rta_rescue = np.maximum(chem_desc_b[:, 7], 0.0) * np.maximum(dose_b + 2.0, 0.0)
    np_stress = (np_params[:, 0] > 0.0).astype(np.float32) * (10.0 / (np_params[:, 0] + 4.0)) * (1.2 - media_ext[:, 0])

    h_ferro = np.clip(0.35 * acsl4_sig + 0.45 * warhead + 0.60 * np_stress - 0.30 * gpx4_act - 0.50 * rta_rescue, 0.0, 4.0)
    h_other = np.clip(rng.exponential(0.15, size=(n_samples, 4)), 0.0, 2.0).astype(np.float32)
    hazard_target = np.column_stack([h_ferro.astype(np.float32), h_other])
    viability_target = np.exp(-0.55 * h_ferro - 0.15 * h_other.sum(axis=1)).astype(np.float32)

    w_true = rng.normal(scale=0.15, size=(12, n_genes)).astype(np.float32)
    delta_expr_target = (
        (chem_desc_a * dose_a[:, None]) @ w_true
        - 1.2 * (1.0 - gene_ko_mult)
        + rng.normal(scale=0.08, size=(n_samples, n_genes)).astype(np.float32)
    )
    rescue_mask = (np.abs(chem_desc_b[:, 7]) > 0.4).astype(np.float32)
    rescue_delta = np.clip(0.25 * rta_rescue, 0.0, 1.0).astype(np.float32)

    return {
        "gene_expr": gene_expr,
        "prot_anchor": prot_anchor,
        "metab_anchor": metab_anchor,
        "chem_fp_a": chem_fp_a,
        "chem_fp_b": chem_fp_b,
        "chem_desc_a": chem_desc_a,
        "chem_desc_b": chem_desc_b,
        "dose_a": dose_a,
        "dose_b": dose_b,
        "media_ext": media_ext,
        "np_params": np_params,
        "gene_ko_mult": gene_ko_mult,
        "is_3d_organoid": is_3d,
        "is_normal_organ_or_til": is_normal_til,
        "elesclomol_conc": elesclomol_conc,
        "viability_target": viability_target,
        "viability_mask": np.ones(n_samples, dtype=np.float32),
        "hazard_target": hazard_target,
        "hazard_mask": np.ones((n_samples, 5), dtype=np.float32),
        "delta_expr_target": delta_expr_target,
        "delta_expr_mask": np.ones((n_samples, n_genes), dtype=np.float32),
        "rescue_delta_target": rescue_delta,
        "rescue_mask": rescue_mask,
        "lineage": lineages,
        "scaffold_id": scaffolds,
        "license_track": licenses,
    }


# ---------------------------------------------------------------------------
# Six Reviewer #2 Baseline Models
# ---------------------------------------------------------------------------


class _CompressedTabularPreprocessor:
    """Train-only StandardScaler + optional TruncatedSVD when gene/FP dimension exceeds max_components."""

    def __init__(self, max_gene_components: int = 256, seed: int = 42) -> None:
        self.max_gene_components = max_gene_components
        self.seed = seed
        self.scaler = StandardScaler()
        self.svd: TruncatedSVD | None = None
        self.g_split: int = 0

    def fit_transform(self, x: np.ndarray, g_dim: int) -> np.ndarray:
        self.g_split = g_dim
        if g_dim > self.max_gene_components and x.shape[0] > 32:
            n_comp = min(self.max_gene_components, x.shape[0] - 2, g_dim - 1)
            if n_comp >= 8:
                self.svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
                g_red = self.svd.fit_transform(x[:, :g_dim])
                x = np.concatenate([g_red, x[:, g_dim:]], axis=1)
        self.scaler.fit(x)
        self.scaler.scale_ = np.where(self.scaler.scale_ < 1e-2, 1.0, self.scaler.scale_)
        return np.clip(self.scaler.transform(x), -12.0, 12.0).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.svd is not None:
            g_red = self.svd.transform(x[:, : self.g_split])
            x = np.concatenate([g_red, x[:, self.g_split :]], axis=1)
        return np.clip(self.scaler.transform(x), -12.0, 12.0).astype(np.float32)


class RidgeMultiOmicBaseline:
    """Reviewer #2 Baseline 1: Multi-omic RidgeCV (`TPM + ProCan + CCLE-Metab + ChemDesc/Dose/NP`)."""

    name = "Ridge_MultiOmic"

    def __init__(self, alphas: np.ndarray | None = None, seed: int = 42, **_: Any) -> None:
        self.alphas = alphas if alphas is not None else np.logspace(-1, 4, 12)
        self.prep = _CompressedTabularPreprocessor(max_gene_components=256, seed=seed)
        self.model = RidgeCV(alphas=self.alphas)

    def fit(self, arrays: dict[str, np.ndarray], train_idx: np.ndarray, target_key: str = "viability_target") -> "RidgeMultiOmicBaseline":
        g_dim = arrays["gene_expr"].shape[1]
        x_tr = self.prep.fit_transform(_slice_x_tabular(arrays, train_idx), g_dim=g_dim)
        y_tr = arrays[target_key][train_idx]
        self.model.fit(x_tr, y_tr)
        return self

    def predict(self, arrays: dict[str, np.ndarray], eval_idx: np.ndarray) -> np.ndarray:
        chunks = [
            np.asarray(self.model.predict(self.prep.transform(_slice_x_tabular(arrays, eval_idx[s : s + 8192]))), dtype=np.float32)
            for s in range(0, len(eval_idx), 8192)
        ]
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]


class ElasticNetMultiOmicBaseline:
    """Reviewer #2 Baseline 2: L1/L2 ElasticNetCV (with fast SGD elasticnet scaling for >8k samples)."""

    name = "ElasticNet_MultiOmic"

    def __init__(self, n_jobs: int = 8, seed: int = 42, **_: Any) -> None:
        self.n_jobs = n_jobs
        self.seed = seed
        self.prep = _CompressedTabularPreprocessor(max_gene_components=256, seed=seed)
        self.model: Any = None

    def fit(self, arrays: dict[str, np.ndarray], train_idx: np.ndarray, target_key: str = "viability_target") -> "ElasticNetMultiOmicBaseline":
        g_dim = arrays["gene_expr"].shape[1]
        x_tr = self.prep.fit_transform(_slice_x_tabular(arrays, train_idx), g_dim=g_dim)
        y_tr = arrays[target_key][train_idx]
        if y_tr.ndim == 1 and x_tr.shape[0] <= 3500:
            self.model = ElasticNetCV(
                l1_ratio=[0.15, 0.5, 0.85],
                alphas=np.logspace(-3, 2, 8),
                cv=min(3, max(2, x_tr.shape[0] // 20)),
                max_iter=400,
                n_jobs=self.n_jobs,
                random_state=self.seed,
            )
        elif y_tr.ndim == 1:
            self.model = SGDRegressor(
                penalty="elasticnet",
                l1_ratio=0.35,
                alpha=1e-3,
                max_iter=500,
                tol=1e-3,
                random_state=self.seed,
            )
        else:
            base = SGDRegressor(penalty="elasticnet", l1_ratio=0.35, alpha=1e-3, max_iter=500, tol=1e-3, random_state=self.seed)
            self.model = MultiOutputRegressor(base, n_jobs=self.n_jobs)
        self.model.fit(x_tr, y_tr)
        return self

    def predict(self, arrays: dict[str, np.ndarray], eval_idx: np.ndarray) -> np.ndarray:
        chunks = [
            np.asarray(self.model.predict(self.prep.transform(_slice_x_tabular(arrays, eval_idx[s : s + 8192]))), dtype=np.float32)
            for s in range(0, len(eval_idx), 8192)
        ]
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]


class LightGBMMultiOmicBaseline:
    """Reviewer #2 Baseline 3: Histogram LightGBM on Multi-Omics + 2048-bit Morgan FP + NP features."""

    name = "LightGBM_MultiOmic_MorganFP"

    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        n_jobs: int = 8,
        seed: int = 42,
    ) -> None:
        self.task = task
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.n_jobs = n_jobs
        self.seed = seed
        self.models: list[Any] = []

    def _make_estimator(self) -> Any:
        cls = lgb.LGBMClassifier if self.task == "classification" else lgb.LGBMRegressor
        return cls(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_samples=max(5, 10),
            n_jobs=self.n_jobs,
            random_state=self.seed,
            verbosity=-1,
        )

    def fit(self, arrays: dict[str, np.ndarray], train_idx: np.ndarray, target_key: str = "viability_target") -> "LightGBMMultiOmicBaseline":
        x_tr = _slice_x_gbdt(arrays, train_idx)
        y_tr = arrays[target_key][train_idx]
        self.models = []
        if y_tr.ndim == 1:
            est = self._make_estimator()
            est.fit(x_tr, y_tr)
            self.models.append(est)
        else:
            for col in range(y_tr.shape[1]):
                est = self._make_estimator()
                est.fit(x_tr, y_tr[:, col])
                self.models.append(est)
        return self

    def predict(self, arrays: dict[str, np.ndarray], eval_idx: np.ndarray) -> np.ndarray:
        out_chunks: list[np.ndarray] = []
        for s in range(0, len(eval_idx), 8192):
            x_ev = _slice_x_gbdt(arrays, eval_idx[s : s + 8192])
            if len(self.models) == 1:
                if self.task == "classification":
                    out_chunks.append(np.asarray(self.models[0].predict_proba(x_ev)[:, 1], dtype=np.float32))
                else:
                    out_chunks.append(np.asarray(self.models[0].predict(x_ev), dtype=np.float32))
            else:
                out_chunks.append(np.column_stack([np.asarray(m.predict(x_ev), dtype=np.float32) for m in self.models]))
        return np.concatenate(out_chunks, axis=0) if len(out_chunks) > 1 else out_chunks[0]


class XGBoostMultiOmicBaseline:
    """Reviewer #2 Baseline 4: Histogram XGBoost on Multi-Omics + 2048-bit Morgan FP + NP features."""

    name = "XGBoost_Hist_MultiOmic_MorganFP"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        n_jobs: int = 8,
        seed: int = 42,
    ) -> None:
        self.model = xgb.XGBRegressor(
            tree_method="hist",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.5,
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=0,
        )

    def fit(self, arrays: dict[str, np.ndarray], train_idx: np.ndarray, target_key: str = "viability_target") -> "XGBoostMultiOmicBaseline":
        x_tr = _slice_x_gbdt(arrays, train_idx)
        y_tr = arrays[target_key][train_idx]
        self.model.fit(x_tr, y_tr)
        return self

    def predict(self, arrays: dict[str, np.ndarray], eval_idx: np.ndarray) -> np.ndarray:
        chunks = [
            np.asarray(self.model.predict(_slice_x_gbdt(arrays, eval_idx[s : s + 8192])), dtype=np.float32)
            for s in range(0, len(eval_idx), 8192)
        ]
        return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]


class _TwoBranchChemCellNet(nn.Module):
    """Permutation-invariant 2-branch Morgan-FP/Warhead x Basal-Cell-State bilinear MLP."""

    def __init__(self, chem_in_dim: int, cell_in_dim: int, hidden_dim: int = 192, out_dim: int = 1, dropout: float = 0.15) -> None:
        super().__init__()
        self.chem_enc = nn.Sequential(
            nn.Linear(chem_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.cell_enc = nn.Sequential(
            nn.Linear(cell_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, chem_a: torch.Tensor, chem_b: torch.Tensor, cell_x: torch.Tensor) -> torch.Tensor:
        z_chem = self.chem_enc(chem_a) + self.chem_enc(chem_b)
        z_cell = self.cell_enc(cell_x)
        fused = torch.cat([z_chem, z_cell, z_chem * z_cell], dim=-1)
        return self.readout(fused)


class ChempropCellStateMLPBaseline:
    """Reviewer #2 Baseline 5: 2-branch Morgan-FP + Physicochemical/Warhead x Cell-State PyTorch MLP."""

    name = "ChempropStyle_MorganFP_CellState_MLP"

    def __init__(
        self,
        hidden_dim: int = 192,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 2.5e-3,
        n_jobs: int = 8,
        seed: int = 42,
        device: str | None = None,
        **_: Any,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_jobs = n_jobs
        self.seed = seed
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.chem_scaler = StandardScaler()
        self.cell_scaler = StandardScaler()
        self.net: _TwoBranchChemCellNet | None = None
        self.out_is_1d: bool = True

    @staticmethod
    def _pack_inputs(arrays: dict[str, np.ndarray], idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chem_a = np.concatenate([arrays["chem_fp_a"][idx].astype(np.float32), arrays["chem_desc_a"][idx], arrays["dose_a"][idx]], axis=1)
        chem_b = np.concatenate([arrays["chem_fp_b"][idx].astype(np.float32), arrays["chem_desc_b"][idx], arrays["dose_b"][idx]], axis=1)
        cell_parts = [
            arrays["gene_expr"][idx],
            arrays["prot_anchor"][idx],
            arrays["metab_anchor"][idx],
        ]
        for ctx_key in ("genomic_context", "prot64_context", "lipid32_context"):
            if ctx_key in arrays:
                cell_parts.append(arrays[ctx_key][idx])
        cell_parts.extend(
            [
                arrays["media_ext"][idx],
                arrays["np_params"][idx],
                arrays["ko_core64"][idx],
                arrays["is_3d_organoid"][idx].reshape(-1, 1),
            ]
        )
        cell_x = np.concatenate(cell_parts, axis=1)
        return chem_a.astype(np.float32), chem_b.astype(np.float32), cell_x.astype(np.float32)

    def fit(self, arrays: dict[str, np.ndarray], train_idx: np.ndarray, target_key: str = "viability_target") -> "ChempropCellStateMLPBaseline":
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, min(16, int(self.n_jobs))))
        torch.manual_seed(self.seed)
        chem_a_raw, chem_b_raw, cell_raw = self._pack_inputs(arrays, train_idx)
        self.chem_scaler.fit(np.vstack([chem_a_raw, chem_b_raw]))
        self.chem_scaler.scale_ = np.where(self.chem_scaler.scale_ < 1e-2, 1.0, self.chem_scaler.scale_)
        chem_a = np.clip(self.chem_scaler.transform(chem_a_raw), -12.0, 12.0).astype(np.float32)
        chem_b = np.clip(self.chem_scaler.transform(chem_b_raw), -12.0, 12.0).astype(np.float32)
        self.cell_scaler.fit(cell_raw)
        self.cell_scaler.scale_ = np.where(self.cell_scaler.scale_ < 1e-2, 1.0, self.cell_scaler.scale_)
        cell_x = np.clip(self.cell_scaler.transform(cell_raw), -12.0, 12.0).astype(np.float32)
        y_raw = arrays[target_key][train_idx].astype(np.float32)
        self.out_is_1d = y_raw.ndim == 1
        y = y_raw.reshape(-1, 1) if self.out_is_1d else y_raw

        self.net = _TwoBranchChemCellNet(
            chem_in_dim=chem_a.shape[1],
            cell_in_dim=cell_x.shape[1],
            hidden_dim=self.hidden_dim,
            out_dim=y.shape[1],
        ).to(self.device)

        eff_bs = 512 if len(train_idx) > 4096 else min(self.batch_size, max(16, len(train_idx) // 6))
        ds = TensorDataset(torch.from_numpy(chem_a), torch.from_numpy(chem_b), torch.from_numpy(cell_x), torch.from_numpy(y))
        loader = DataLoader(ds, batch_size=eff_bs, shuffle=True)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.epochs))
        loss_fn = nn.SmoothL1Loss()

        self.net.train()
        for _ in range(self.epochs):
            for ba, bb, bc, by in loader:
                opt.zero_grad(set_to_none=True)
                pred = self.net(ba.to(self.device), bb.to(self.device), bc.to(self.device))
                loss = loss_fn(pred, by.to(self.device))
                loss.backward()
                opt.step()
            sched.step()
        return self

    def predict(self, arrays: dict[str, np.ndarray], eval_idx: np.ndarray) -> np.ndarray:
        assert self.net is not None, "ChempropCellStateMLPBaseline must be fit before predict"
        chem_a_raw, chem_b_raw, cell_raw = self._pack_inputs(arrays, eval_idx)
        chem_a = np.clip(self.chem_scaler.transform(chem_a_raw), -12.0, 12.0).astype(np.float32)
        chem_b = np.clip(self.chem_scaler.transform(chem_b_raw), -12.0, 12.0).astype(np.float32)
        cell_x = np.clip(self.cell_scaler.transform(cell_raw), -12.0, 12.0).astype(np.float32)
        ds = TensorDataset(torch.from_numpy(chem_a), torch.from_numpy(chem_b), torch.from_numpy(cell_x))
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        self.net.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for ba, bb, bc in loader:
                out = self.net(ba.to(self.device), bb.to(self.device), bc.to(self.device)).cpu().numpy()
                preds.append(out)
        res = np.concatenate(preds, axis=0).astype(np.float32)
        return res.ravel() if self.out_is_1d else res


class LinearPerturbBaseline:
    """Reviewer #2 Baseline 6: Additive Linear Perturbation Model (Ahlmann-Eltze et al., Nat. Methods 2025).

    Solves regularized closed-form additive superposition over basal state + single-gene KOs + chemical
    perturbations to jointly predict post-perturbation transcriptome `delta_expr` (`(N, G)`) and
    viability / genetic epistasis without nonlinear mechanistic feedback.
    """

    name = "LinearPerturb_NatMethods2025"

    def __init__(self, ridge_lambda: float = 1.0, max_basal_pcs: int = 32, seed: int = 42, **_: Any) -> None:
        self.ridge_lambda = float(ridge_lambda)
        self.max_basal_pcs = max_basal_pcs
        self.seed = seed
        self.basal_svd: TruncatedSVD | None = None
        self.u_mean: np.ndarray | None = None
        self.w_delta: np.ndarray | None = None
        self.w_target: np.ndarray | None = None

    def _design_matrix(self, arrays: dict[str, np.ndarray], idx: np.ndarray, fit: bool = False) -> np.ndarray:
        gx = arrays["gene_expr"][idx]
        if fit:
            n_comp = min(self.max_basal_pcs, max(4, gx.shape[0] - 2), gx.shape[1] - 1)
            self.basal_svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
            gx_red = self.basal_svd.fit_transform(gx)
        else:
            assert self.basal_svd is not None
            gx_red = self.basal_svd.transform(gx)

        # Additive perturbation vector: KO indicator (1 - mult) + dose-scaled chemical descriptors + media + NP + basal multiomics
        ko_shift = 1.0 - arrays["gene_ko_mult"][idx].astype(np.float32)
        chem_add = arrays["chem_desc_a"][idx] * arrays["dose_a"][idx] + arrays["chem_desc_b"][idx] * arrays["dose_b"][idx]
        u_parts = [gx_red]
        for ctx_key in ("genomic_context", "prot64_context", "lipid32_context"):
            if ctx_key in arrays:
                u_parts.append(arrays[ctx_key][idx])
        u_parts.extend([ko_shift, chem_add, arrays["media_ext"][idx], arrays["np_params"][idx]])
        u = np.concatenate(u_parts, axis=1)
        if fit:
            self.u_mean = u.mean(axis=0, keepdims=True)
        assert self.u_mean is not None
        u_centered = u - self.u_mean
        ones = np.ones((u_centered.shape[0], 1), dtype=np.float32)
        return np.concatenate([ones, u_centered.astype(np.float32)], axis=1)

    @staticmethod
    def _solve_ridge(u: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        gram = (u.T @ u).astype(np.float64)
        reg = np.eye(gram.shape[0], dtype=np.float64) * lam
        reg[0, 0] = 1e-6  # Do not penalize intercept
        rhs = (u.T @ y).astype(np.float64)
        return np.linalg.solve(gram + reg, rhs).astype(np.float32)

    def fit(
        self,
        arrays: dict[str, np.ndarray],
        train_idx: np.ndarray,
        target_key: str = "viability_target",
        delta_train_idx: np.ndarray | None = None,
    ) -> "LinearPerturbBaseline":
        u_tr = self._design_matrix(arrays, train_idx, fit=True)
        self.w_target = self._solve_ridge(u_tr, arrays[target_key][train_idx], self.ridge_lambda)
        dx_idx = delta_train_idx if (delta_train_idx is not None and delta_train_idx.size >= 8) else train_idx
        u_dx = self._design_matrix(arrays, dx_idx, fit=False)
        self.w_delta = self._solve_ridge(u_dx, arrays["delta_expr_target"][dx_idx].astype(np.float32), self.ridge_lambda)
        return self

    def predict(self, arrays: dict[str, np.ndarray], eval_idx: np.ndarray) -> np.ndarray:
        assert self.w_target is not None
        u_ev = self._design_matrix(arrays, eval_idx, fit=False)
        return (u_ev @ self.w_target).astype(np.float32)

    def predict_delta_expr(self, arrays: dict[str, np.ndarray], eval_idx: np.ndarray) -> np.ndarray:
        assert self.w_delta is not None
        u_ev = self._design_matrix(arrays, eval_idx, fit=False)
        return (u_ev @ self.w_delta).astype(np.float32)


# ---------------------------------------------------------------------------
# Six Compositional OOD Split Resolvers & Full Benchmark Runner
# ---------------------------------------------------------------------------


def resolve_ood_splits(
    arrays: dict[str, np.ndarray],
    dataset_obj: Any = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Resolve all 6 canonical compositional OOD splits (#44) without train-test overlap."""
    n = arrays["gene_expr"].shape[0]
    all_idx = np.arange(n, dtype=np.int64)
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if dataset_obj is not None:
        for sname, method_name in OOD_DATASET_METHOD_MAP.items():
            for cand in (sname, method_name):
                if hasattr(dataset_obj, cand):
                    tr, te = getattr(dataset_obj, cand)()
                    tr_a, te_a = np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)
                    tr_a, te_a = tr_a[tr_a < n], te_a[te_a < n]
                    if tr_a.size > 0 and te_a.size > 0:
                        splits[sname] = (tr_a, te_a)
                        break

    def _ensure_nonempty(mask_test: np.ndarray, fallback_mod: int = 5) -> tuple[np.ndarray, np.ndarray]:
        te = all_idx[mask_test]
        tr = all_idx[~mask_test]
        if te.size < 4 or tr.size < 8:
            fb_mask = (all_idx % fallback_mod) == 0
            te = all_idx[fb_mask]
            tr = all_idx[~fb_mask]
        return tr, te

    if "1_leave_out_scaffold" not in splits:
        uniq_scaf = sorted(set(arrays["scaffold_id"].tolist()))
        hold_scaf = set(uniq_scaf[::5]) if len(uniq_scaf) >= 5 else set(uniq_scaf[-1:])
        splits["1_leave_out_scaffold"] = _ensure_nonempty(np.isin(arrays["scaffold_id"], list(hold_scaf)), 5)

    if "2_leave_out_lineage_ccrcc_melanoma" not in splits:
        lin_upper = np.char.upper(arrays["lineage"].astype(str))
        is_hold_lin = np.isin(lin_upper, ["KIDNEY", "MELANOMA", "RENAL", "SKIN", "CCRCC"])
        splits["2_leave_out_lineage_ccrcc_melanoma"] = _ensure_nonempty(is_hold_lin, 6)

    if "3_2d_to_3d_organoid_pdx" not in splits:
        splits["3_2d_to_3d_organoid_pdx"] = _ensure_nonempty(arrays["is_3d_organoid"] > 0.5, 5)

    if "4_dual_crispr_fin_epistasis" not in splits:
        has_ko = np.min(arrays["gene_ko_mult"], axis=1) < 0.85
        has_chem = (np.max(np.abs(arrays["chem_fp_a"]), axis=1) > 0) | (arrays["dose_a"].ravel() > -2.5)
        splits["4_dual_crispr_fin_epistasis"] = _ensure_nonempty(has_ko & has_chem, 5)

    if "5_cuproptosis_elesclomol_specificity" not in splits:
        if dataset_obj is not None and hasattr(dataset_obj, "records") and len(dataset_obj.records) == n:
            recs = dataset_obj.records
            te_cand = [
                idx
                for idx in range(n)
                if (
                    recs[idx].get("kind") == "pharmaco"
                    and (
                        recs[idx].get("probe_mechanism") == "cuproptosis"
                        or int(recs[idx].get("hazard_channel_idx", -1)) == 3
                    )
                )
                or (
                    recs[idx].get("kind") == "crispr"
                    and "ELESCLOMOL" in str(recs[idx].get("fin_modifier_compound", "")).upper()
                )
                or (
                    recs[idx].get("kind") == "perturb"
                    and recs[idx].get("perturbation_type") in ("bionano", "nanoparticle")
                )
            ]
            m_cup = np.zeros(n, dtype=bool)
            m_cup[te_cand] = True
            splits["5_cuproptosis_elesclomol_specificity"] = _ensure_nonempty(m_cup, 6)
        else:
            has_cupro = (arrays["elesclomol_conc"] > 0.0) | (arrays["hazard_target"][:, 3] > 0.35)
            ref_ferro = (arrays["viability_mask"] > 0.5) & ((all_idx % 11) == 0)
            splits["5_cuproptosis_elesclomol_specificity"] = _ensure_nonempty(has_cupro | ref_ferro, 6)

    if "6_sfti_tumor_vs_normal_til" not in splits:
        if dataset_obj is not None and hasattr(dataset_obj, "records") and len(dataset_obj.records) == n:
            recs = dataset_obj.records
            te_cand = [
                idx
                for idx in range(n)
                if recs[idx].get("kind") in ("normal_sfti", "rescue")
                or (
                    recs[idx].get("kind") == "pharmaco"
                    and recs[idx].get("probe_mechanism") in ("gpx4_inhib", "xc_inhib", "gsh_depletion")
                    and (idx % 5 == 0)
                )
            ]
            m_sfti = np.zeros(n, dtype=bool)
            m_sfti[te_cand] = True
            splits["6_sfti_tumor_vs_normal_til"] = _ensure_nonempty(m_sfti, 5)
        else:
            is_norm = arrays["is_normal_organ_or_til"] > 0.5
            tum_holdout = (arrays["viability_mask"] > 0.5) & (~is_norm) & ((all_idx % 9) == 0)
            splits["6_sfti_tumor_vs_normal_til"] = _ensure_nonempty(is_norm | tum_holdout, 5)

    for sname, (tr, te) in splits.items():
        assert set(tr.tolist()).isdisjoint(set(te.tolist())), f"Train/test leak in {sname}"
    return splits


def _load_dataset_or_synthetic(
    data_dir: Path,
    license_track: str,
    smoke_test: bool,
    max_samples: int | None,
) -> tuple[dict[str, np.ndarray], Any]:
    if smoke_test:
        syn = build_synthetic_multimodal_arrays(n_samples=360, n_genes=256)
        df_lic = pd.DataFrame({"idx": np.arange(len(syn["license_track"])), "license_track": syn["license_track"]})
        keep_idx = filter_by_license_track(df_lic, track=license_track)["idx"].to_numpy()
        filtered = {k: np.asarray(v)[keep_idx] for k, v in syn.items()}
        return extract_multimodal_arrays(filtered, max_samples=max_samples), None

    audit_parquet_directory(data_dir)
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
    arrays = extract_multimodal_arrays(ds_obj, max_samples=max_samples)
    splits = resolve_ood_splits(arrays, dataset_obj=ds_obj)
    del ds_obj
    gc.collect()
    return arrays, splits


def evaluate_all_baselines_ood(
    arrays: dict[str, np.ndarray],
    dataset_obj: Any = None,
    precomputed_splits: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    license_track: str = "open",
    n_jobs: int = 8,
    n_boot: int = 300,
    mlp_epochs: int = 25,
    max_fit_samples: int = 16000,
    seed: int = 42,
) -> pd.DataFrame:
    """Fit and evaluate all 6 Reviewer #2 baselines across all 6 compositional OOD splits."""
    splits = precomputed_splits if precomputed_splits is not None else resolve_ood_splits(arrays, dataset_obj=dataset_obj)
    baseline_factories: list[Callable[[], Any]] = [
        lambda: RidgeMultiOmicBaseline(seed=seed),
        lambda: ElasticNetMultiOmicBaseline(n_jobs=n_jobs, seed=seed),
        lambda: LightGBMMultiOmicBaseline(n_jobs=n_jobs, seed=seed),
        lambda: XGBoostMultiOmicBaseline(n_jobs=n_jobs, seed=seed),
        lambda: ChempropCellStateMLPBaseline(epochs=mlp_epochs, n_jobs=n_jobs, seed=seed),
        lambda: LinearPerturbBaseline(seed=seed),
    ]

    m_viab_all = arrays["viability_mask"] > 0.5
    m_chronos_all = arrays.get("chronos_mask", np.zeros_like(arrays["viability_mask"])) > 0.5
    is_norm_all = arrays["is_normal_organ_or_til"] > 0.5
    has_fit = arrays.get("has_fitness_label", np.ones(arrays["gene_expr"].shape[0], dtype=np.float32)) > 0.5
    if "has_dx_row" in arrays:
        has_dx = arrays["has_dx_row"].astype(bool)
    else:
        has_dx = (np.abs(arrays["delta_expr_target"]).max(axis=1) > 1e-6) & (arrays["delta_expr_mask"].sum(axis=1) >= 8)
    rng = np.random.default_rng(seed)

    records: list[dict[str, Any]] = []
    for split_name, (raw_tr, raw_te) in splits.items():
        # Evaluate strictly on measured experimental viability or Chronos fitness (never unperturbed dose=0 normal rows)
        if int(m_viab_all[raw_te].sum()) >= 8 and "viability_raw" in arrays:
            mask_mod = m_viab_all
            target_key = "viability_raw"
        elif int(m_chronos_all[raw_te].sum()) >= 8 and "chronos_target" in arrays:
            mask_mod = m_chronos_all
            target_key = "chronos_target"
        else:
            mask_mod = has_fit
            target_key = "viability_target"

        fit_tr = raw_tr[mask_mod[raw_tr]]
        if fit_tr.size < 16:
            fit_tr = raw_tr
        eval_te = raw_te[mask_mod[raw_te]]
        if eval_te.size < 8:
            eval_te = raw_te

        if max_fit_samples > 0 and fit_tr.size > max_fit_samples:
            fit_tr = np.sort(rng.choice(fit_tr, size=max_fit_samples, replace=False))

        dx_tr = raw_tr[has_dx[raw_tr]]
        if dx_tr.size > 8000:
            dx_tr = np.sort(rng.choice(dx_tr, size=8000, replace=False))
        dx_te = raw_te[has_dx[raw_te]]
        if dx_te.size < 4:
            cand_dx = np.where(has_dx)[0]
            if cand_dx.size >= 16:
                dx_te = cand_dx[::5]
                dx_tr = np.setdiff1d(cand_dx, dx_te, assume_unique=False)
                if dx_tr.size > 8000:
                    dx_tr = np.sort(rng.choice(dx_tr, size=8000, replace=False))
        if dx_te.size > 1024:
            dx_te = np.sort(rng.choice(dx_te, size=1024, replace=False))

        y_true = arrays[target_key][eval_te]
        thresh = float(np.median(y_true))
        y_bin = (y_true < thresh).astype(np.int32)
        if np.unique(y_bin).size < 2:
            y_bin = (y_true <= thresh).astype(np.int32)
        if np.unique(y_bin).size < 2 and y_true.size >= 4:
            q25 = float(np.quantile(y_true, 0.25))
            y_bin = (y_true <= q25).astype(np.int32)

        for make_model in baseline_factories:
            t0 = time.perf_counter()
            model = make_model()
            if isinstance(model, LinearPerturbBaseline):
                model.fit(arrays, fit_tr, target_key=target_key, delta_train_idx=dx_tr)
            else:
                model.fit(arrays, fit_tr, target_key=target_key)
            y_pred = model.predict(arrays, eval_te)
            elapsed = time.perf_counter() - t0

            rho, rho_lo, rho_hi = bootstrap_metric_ci(y_true, y_pred, _safe_spearman, n_boot=n_boot, seed=seed)
            prs, prs_lo, prs_hi = bootstrap_metric_ci(y_true, y_pred, _safe_pearson, n_boot=n_boot, seed=seed)
            rmse, rmse_lo, rmse_hi = bootstrap_metric_ci(y_true, y_pred, _safe_rmse, n_boot=n_boot, seed=seed)
            auroc = _safe_auroc(y_bin, -y_pred)

            delta_r = float("nan")
            if hasattr(model, "predict_delta_expr") and dx_te.size >= 4:
                dx_pred = model.predict_delta_expr(arrays, dx_te)
                dx_true = arrays["delta_expr_target"][dx_te].astype(np.float32)
                dx_mask = arrays["delta_expr_mask"][dx_te].astype(bool)
                row_corrs: list[float] = []
                for r_i in range(dx_te.size):
                    mr = dx_mask[r_i]
                    if int(mr.sum()) >= 8:
                        yt_r = dx_true[r_i, mr]
                        yp_r = dx_pred[r_i, mr]
                        if np.std(yt_r) > 1e-8 and np.std(yp_r) > 1e-8:
                            row_corrs.append(_safe_pearson(yt_r, yp_r))
                delta_r = float(np.mean(row_corrs)) if row_corrs else _safe_pearson(dx_true, dx_pred)

            records.append(
                {
                    "license_track": license_track,
                    "ood_split": split_name,
                    "baseline_model": model.name,
                    "n_train": int(fit_tr.size),
                    "n_test": int(eval_te.size),
                    "spearman_rho": round(rho, 4),
                    "spearman_ci95_low": round(rho_lo, 4),
                    "spearman_ci95_high": round(rho_hi, 4),
                    "pearson_r": round(prs, 4),
                    "pearson_ci95_low": round(prs_lo, 4),
                    "pearson_ci95_high": round(prs_hi, 4),
                    "rmse": round(rmse, 4),
                    "rmse_ci95_low": round(rmse_lo, 4),
                    "rmse_ci95_high": round(rmse_hi, 4),
                    "lethal_auroc": round(auroc, 4) if not np.isnan(auroc) else np.nan,
                    "delta_expr_pearson_r": round(delta_r, 4) if not np.isnan(delta_r) else np.nan,
                    "fit_eval_sec": round(elapsed, 3),
                }
            )
            del model
            gc.collect()
        print(
            f"[baselines] track={license_track} split={split_name} target={target_key} "
            f"n_train={fit_tr.size} n_test={eval_te.size} done",
            file=sys.stderr,
            flush=True,
        )
    return pd.DataFrame(records)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FerroFlux-FM Reviewer #2 Baselines Across 6 OOD Splits")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Processed Parquet directory")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT_CSV, help="Output CSV summary table")
    parser.add_argument("--license-track", type=str, default="open", choices=["open", "academic", "both"])
    parser.add_argument("--n-jobs", type=int, default=8, help="Parallel CPU threads for GBDT / ElasticNet")
    parser.add_argument("--n-boot", type=int, default=300, help="Bootstrap resamples for 95% CIs")
    parser.add_argument("--mlp-epochs", type=int, default=20, help="Training epochs for ChempropCellStateMLPBaseline")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on samples loaded")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Run fast synthetic verification without shards")
    args = parser.parse_args(argv)

    tracks = ["open", "academic"] if args.license_track == "both" else [args.license_track]
    dfs: list[pd.DataFrame] = []
    for trk in tracks:
        arrays, pre_splits = _load_dataset_or_synthetic(
            data_dir=args.data_dir,
            license_track=trk,
            smoke_test=args.smoke_test,
            max_samples=args.max_samples,
        )
        df_res = evaluate_all_baselines_ood(
            arrays=arrays,
            precomputed_splits=pre_splits,
            license_track=trk,
            n_jobs=args.n_jobs,
            n_boot=args.n_boot,
            mlp_epochs=args.mlp_epochs,
            seed=args.seed,
        )
        dfs.append(df_res)
        del arrays, pre_splits
        gc.collect()

    full_df = pd.concat(dfs, ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(args.output_csv, index=False)
    print(
        json.dumps(
            {
                "status": "OK",
                "rows_written": int(len(full_df)),
                "output_csv": str(args.output_csv),
                "models": sorted(full_df["baseline_model"].unique().tolist()),
                "splits": sorted(full_df["ood_split"].unique().tolist()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
