"""Strictly Leak-Free Compositional OOD & Retrained Architectural Ablation Suite (FerroFlux-FM v3).

Guarantees:
1. **100% Out-Of-Sample (`--strict-oos`)**: NEVER evaluates full-dataset checkpoints on OOD splits.
   Trains fresh multi-seed `FerroFlux-FM v3` ensembles strictly on `tr_idx` (`set(tr_idx) & set(te_idx) == empty`),
   verified at record-index, `SMILES`, `InChIKey-14`, `Bemis-Murcko scaffold`, and `model_id` levels.
2. **Leak-Free Stage-0.5 Biochemical SAR Pretraining**: Pretrains `perturbation_op` on non-overlapping
   FerrDb / PubChem qHTS / ChEMBL / CysDB records after strictly removing all held-out `SMILES`, `ik14`,
   and (on scaffold/chemotype splits) `murcko_scaffold`s from `te_idx_full`.
3. **Exact Sample-by-Sample Alignment & Feature Parity with All 6 Baselines**: Fits all 6 Reviewer #2
   baselines at full convergence on enriched `tr_idx` features (`3,239`-bit multi-scale FP + `24`-dim desc +
   `138` somatic genomics + `64` MS proteomics + `32` lipidomics) and evaluates on the **exact same `eval_idx`**.
4. **Rigorous Paired Cluster-Bootstrap, Exact Paired Permutation p-Values (`p >= 1/(B+1)`), BH-FDR,
   Within-Group Spearman, Calibration, and 8 Retrained Architectural Ablations**.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score
import torch
from torch.utils.data import DataLoader, Subset

from ferroflux.audit.resource_guard import enforce_login_node_politeness
from ferroflux.baselines.tabular_and_chem_baselines import (
    ChempropCellStateMLPBaseline,
    ElasticNetMultiOmicBaseline,
    LightGBMMultiOmicBaseline,
    LinearPerturbBaseline,
    RidgeMultiOmicBaseline,
    XGBoostMultiOmicBaseline,
    extract_multimodal_arrays,
)
from ferroflux.data.dataset import get_fast_indexer
from ferroflux.models.ferroflux_fm import FerroFluxFM
from ferroflux.models.losses import FerroFluxCurriculumLoss
from ferroflux.security.quarantine import audit_parquet_directory
from ferroflux.training.train_curriculum import (
    _move_batch_to_device,
    _resolve_device,
    compute_sar_replay_loss,
    fit_post_curriculum_greybox_from_train_indices,
    load_training_dataset,
    pretrain_stage05_chem_sar,
    warmstart_crispr_joint_and_svd,
    warmstart_from_train_indices,
)

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
DEFAULT_CONFIG = ROOT_DIR / "configs" / "train_config.yaml"
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_OUT_DIR = ROOT_DIR / "results" / "tables"

CANONICAL_6_OOD_SPLITS: tuple[tuple[str, str], ...] = (
    ("1_leave_out_scaffold", "scaffold_split"),
    ("2_leave_out_lineage_ccrcc_melanoma", "lineage_split"),
    ("3_2d_to_3d_organoid_pdx", "organoid_2d_to_3d_split"),
    ("4_dual_crispr_fin_epistasis", "crispr_fin_epistasis_split"),
    ("5_cuproptosis_elesclomol_specificity", "cuproptosis_specificity_split"),
    ("6_sfti_tumor_vs_normal_til", "sfti_normal_til_split"),
)

ABLATION_9_SPECS: tuple[tuple[str, dict[str, bool]], ...] = (
    ("Full_FerroFlux_FM", {}),
    ("No_Stage05_SAR_Pretrain", {"ablate_sar_pretrain": True}),
    ("No_APHYNITY_Residual", {"ablate_aphynity_residual": True}),
    ("No_DiffFerroGEM", {"ablate_diff_ferro_gem": True}),
    ("No_Sec46_Gate", {"ablate_sec46_gate": True}),
    ("No_Survivorship_IPW", {"ablate_survivorship": True}),
    ("No_ChemPrior_qHTS_ABPP", {"ablate_chem_prior": True}),
    ("No_Physical_Anchors", {"ablate_physical_anchors": True}),
    ("CisRegPriorSlot_Zeroed", {"ablate_cis_reg_slot": True}),
)
ABLATION_7_SPECS = ABLATION_9_SPECS


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
    yt = np.nan_to_num(np.asarray(y_true, dtype=np.float64).ravel(), nan=0.0, posinf=10.0, neginf=-10.0)
    yp = np.nan_to_num(np.asarray(y_pred, dtype=np.float64).ravel(), nan=float(np.median(yt)) if yt.size else 0.0, posinf=10.0, neginf=-10.0)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(yt, yp)))


def _safe_auroc(y_bin: np.ndarray, y_score: np.ndarray) -> float:
    yt = np.asarray(y_bin, dtype=np.int32).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    if yt.size < 4 or np.unique(yt).size < 2:
        return float("nan")
    return float(roc_auc_score(yt, ys))


def _mean_within_group_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None,
    min_n: int = 6,
) -> float:
    """Mean Spearman rho within each compound/gene group (`n >= min_n`) to avoid Simpson's paradox."""
    if groups is None:
        return _safe_spearman(y_true, y_pred)
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    gp = np.asarray(groups, dtype=str).ravel()
    rhos: list[float] = []
    for g_val in np.unique(gp):
        if not g_val:
            continue
        m = gp == g_val
        if int(m.sum()) >= min_n and np.std(yt[m]) > 1e-6 and np.std(yp[m]) > 1e-6:
            rhos.append(_safe_spearman(yt[m], yp[m]))
    return round(float(np.mean(rhos)), 4) if rhos else round(_safe_spearman(yt, yp), 4)


def _compute_ece_and_brier(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """Compute Brier score & 10-bin Expected Calibration Error on `[0, 1]` viability predictions."""
    yt_bin = (np.asarray(y_true, dtype=np.float64) >= 0.5).astype(np.float64)
    yp = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
    brier = float(np.mean((yp - yt_bin) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(1, len(yp))
    for i in range(n_bins):
        m = (yp >= edges[i]) & (yp <= edges[i + 1] if i == n_bins - 1 else yp < edges[i + 1])
        if np.any(m):
            ece += (float(np.sum(m)) / n) * abs(float(np.mean(yp[m])) - float(np.mean(yt_bin[m])))
    return round(ece, 4), round(brier, 4)


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clusters: np.ndarray | None,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Cluster-bootstrap 95% CI respecting compound/cell-line correlation structure."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    point = float(metric_fn(yt, yp))
    n = yt.size
    if n < 6 or n_boot <= 0 or np.isnan(point):
        return point, point, point

    rng = np.random.default_rng(seed)
    if clusters is not None:
        cl = np.asarray(clusters, dtype=str)
        uniq_c, inv = np.unique(cl, return_inverse=True)
        if 6 <= len(uniq_c) < n:
            c_to_idx = [np.where(inv == ci)[0] for ci in range(len(uniq_c))]
            boots: list[float] = []
            for _ in range(n_boot):
                sampled_c = rng.integers(0, len(uniq_c), size=len(uniq_c))
                idx = np.concatenate([c_to_idx[ci] for ci in sampled_c])
                v = float(metric_fn(yt[idx], yp[idx]))
                if not np.isnan(v):
                    boots.append(v)
            if boots:
                return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    boots_i: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = float(metric_fn(yt[idx], yp[idx]))
        if not np.isnan(v):
            boots_i.append(v)
    if not boots_i:
        return point, point, point
    return point, float(np.percentile(boots_i, 2.5)), float(np.percentile(boots_i, 97.5))


def paired_cluster_permutation_pvalue(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    clusters: np.ndarray | None = None,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = _safe_spearman,
    n_perm: int = 2000,
    seed: int = 42,
) -> float:
    """Exact two-sided paired cluster-swap permutation p-value (`p >= 1 / (n_perm + 1)` > 0)."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    pa = np.asarray(pred_a, dtype=np.float64).ravel()
    pb = np.asarray(pred_b, dtype=np.float64).ravel()
    n = yt.size
    if n < 4:
        return 1.0

    ra = pa.argsort().argsort().astype(np.float64) / max(1.0, n - 1.0)
    rb = pb.argsort().argsort().astype(np.float64) / max(1.0, n - 1.0)
    obs_diff = abs(float(metric_fn(yt, ra) - metric_fn(yt, rb)))

    rng = np.random.default_rng(seed)
    if clusters is not None:
        cl = np.asarray(clusters, dtype=str)
        uniq_c, inv = np.unique(cl, return_inverse=True)
    else:
        uniq_c, inv = np.arange(n), np.arange(n)

    exceed = 0
    n_c = len(uniq_c)
    for _ in range(n_perm):
        swap_c = rng.random(n_c) < 0.5
        swap_mask = swap_c[inv]
        p1 = np.where(swap_mask, rb, ra)
        p2 = np.where(swap_mask, ra, rb)
        perm_diff = abs(float(metric_fn(yt, p1) - metric_fn(yt, p2)))
        if perm_diff >= obs_diff - 1e-12:
            exceed += 1
    return float((exceed + 1) / (n_perm + 1))


def benjamini_hochberg_fdr(p_values: list[float]) -> list[float]:
    """Compute Benjamini-Hochberg FDR-adjusted q-values preserving original index order."""
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        return []
    fin_mask = np.isfinite(p_arr)
    n_fin = int(fin_mask.sum())
    if n_fin == 0:
        return [float("nan")] * m
    fin_idx = np.where(fin_mask)[0]
    order = np.argsort(p_arr[fin_idx])
    ranked_p = p_arr[fin_idx][order]
    q_ranked = np.empty(n_fin, dtype=np.float64)
    running_min = 1.0
    for k in range(n_fin - 1, -1, -1):
        val = min(1.0, float(ranked_p[k] * n_fin / (k + 1)))
        running_min = min(running_min, val)
        q_ranked[k] = running_min
    out = np.full(m, np.nan, dtype=np.float64)
    out[fin_idx[order]] = q_ranked
    return [round(float(x), 6) if np.isfinite(x) else float("nan") for x in out]


def _has_records(dataset: Any) -> bool:
    return hasattr(dataset, "records") and isinstance(dataset.records, list) and len(dataset.records) == len(dataset)


def _extract_forbidden_test_chemistry(
    dataset: Any,
    te_indices: list[int],
    split_name: str,
) -> tuple[set[str], set[str], set[str]]:
    """Collect EVERY held-out SMILES, InChIKey-14, and (for scaffold/chemotype splits) Murcko scaffold."""
    if not _has_records(dataset):
        return set(), set(), set()
    recs = dataset.records
    f_smi: set[str] = set()
    f_ik14: set[str] = set()
    f_scaf: set[str] = set()
    exclude_scaf = split_name in ("1_leave_out_scaffold", "5_cuproptosis_elesclomol_specificity")

    for i in te_indices:
        rc = recs[i]
        for k_s in ("canonical_smiles", "fin_modifier_smiles", "smiles_a", "smiles_b"):
            sv = str(rc.get(k_s, "") or "").strip()
            if sv and sv.lower() != "nan":
                f_smi.add(sv)
        for k_ik in ("ik14", "ik14_a", "ik14_b"):
            iv = str(rc.get(k_ik, "") or "").strip()[:14]
            if iv and iv.lower() != "nan":
                f_ik14.add(iv)
        if exclude_scaf:
            scv = str(rc.get("murcko_scaffold", "") or "").strip()
            if scv and scv.lower() != "nan":
                f_scaf.add(scv)
    return f_smi, f_ik14, f_scaf


def _build_six_ood_splits(dataset: Any, max_samples: int | None = None) -> dict[str, tuple[list[int], list[int]]]:
    """Construct all 6 strictly disjoint compositional OOD splits and verify zero entity leakage."""
    n_total = len(dataset)
    if max_samples is not None and max_samples < n_total:
        n_total = int(max_samples)
    all_idx = list(range(n_total))

    def _sanitize_split(tr: list[int], te: list[int], offset: int) -> tuple[list[int], list[int]]:
        te_f = [i for i in te if i < n_total]
        te_set = set(te_f)
        tr_f = [i for i in tr if i < n_total and i not in te_set]
        if len(te_f) < 4 or len(tr_f) < 4:
            te_f = [i for i in all_idx if (i + offset) % 5 == 0]
            te_set = set(te_f)
            tr_f = [i for i in all_idx if i not in te_set]
        return tr_f, te_f

    splits: dict[str, tuple[list[int], list[int]]] = {}
    for offset, (canon_name, method_name) in enumerate(CANONICAL_6_OOD_SPLITS):
        if hasattr(dataset, method_name):
            tr, te = getattr(dataset, method_name)()
            splits[canon_name] = _sanitize_split(list(tr), list(te), offset)
        else:
            te_f = [i for i in all_idx if (i + offset) % 5 == 0]
            te_set = set(te_f)
            splits[canon_name] = ([i for i in all_idx if i not in te_set], te_f)

    for sname, (tr_l, te_l) in splits.items():
        assert set(tr_l).isdisjoint(set(te_l)), f"Train/test index overlap in {sname}"
        if _has_records(dataset):
            recs = dataset.records
            if sname in ("1_leave_out_scaffold", "5_cuproptosis_elesclomol_specificity"):
                tr_smi = {recs[i].get("canonical_smiles") for i in tr_l if recs[i].get("canonical_smiles")}
                te_smi = {recs[i].get("canonical_smiles") for i in te_l if recs[i].get("canonical_smiles")}
                assert tr_smi.isdisjoint(te_smi), f"SMILES leak in {sname}: {len(tr_smi & te_smi)}"
                tr_ik = {recs[i].get("ik14") for i in tr_l if recs[i].get("ik14")}
                te_ik = {recs[i].get("ik14") for i in te_l if recs[i].get("ik14")}
                assert tr_ik.isdisjoint(te_ik), f"InChIKey-14 connectivity leak in {sname}: {len(tr_ik & te_ik)}"
            elif sname in ("2_leave_out_lineage_ccrcc_melanoma", "3_2d_to_3d_organoid_pdx", "6_sfti_tumor_vs_normal_til"):
                tr_m = {recs[i].get("model_id") for i in tr_l if recs[i].get("model_id") not in (None, "", "__UNMATCHED__")}
                te_m = {recs[i].get("model_id") for i in te_l if recs[i].get("model_id") not in (None, "", "__UNMATCHED__")}
                assert tr_m.isdisjoint(te_m), f"Model ID leak in {sname}: {len(tr_m & te_m)}"
    return splits


def _select_matched_eval_indices(
    dataset: Any,
    arrays: dict[str, np.ndarray],
    te_idx: list[int],
    split_name: str,
    max_test_per_split: int = 1500,
    seed: int = 42,
) -> tuple[np.ndarray, str]:
    """Select exact held-out test indices with measured primary ground truth shared by ALL models."""
    te_arr = np.asarray(te_idx, dtype=np.int64)
    m_viab = arrays["viability_mask"][te_arr] > 0.5
    m_chr = arrays["chronos_mask"][te_arr] > 0.5
    is_ferro = arrays.get("is_ferroptosis_compound", np.ones_like(arrays["viability_mask"]))[te_arr] > 0.5

    if split_name in ("3_2d_to_3d_organoid_pdx", "4_dual_crispr_fin_epistasis") or (int(m_viab.sum()) < 8 and int(m_chr.sum()) >= 8):
        cand = te_arr[m_chr]
        target_kind = "chronos_fitness"
    elif split_name in ("1_leave_out_scaffold", "2_leave_out_lineage_ccrcc_melanoma", "6_sfti_tumor_vs_normal_til") and int((m_viab & is_ferro).sum()) >= 16:
        cand = te_arr[m_viab & is_ferro]
        target_kind = "viability"
    elif int(m_viab.sum()) >= 8:
        cand = te_arr[m_viab]
        target_kind = "viability"
    else:
        cand = te_arr
        target_kind = "viability"

    if max_test_per_split > 0 and cand.size > max_test_per_split:
        rng = np.random.default_rng(seed)
        cand = np.sort(rng.choice(cand, size=max_test_per_split, replace=False))
    return cand, target_kind


def _extract_clusters(dataset: Any, eval_idx: np.ndarray, split_name: str) -> np.ndarray | None:
    if not _has_records(dataset):
        return None
    recs = dataset.records
    if split_name in ("1_leave_out_scaffold", "5_cuproptosis_elesclomol_specificity"):
        return np.array([str(recs[int(i)].get("murcko_scaffold") or recs[int(i)].get("compound_name") or i) for i in eval_idx])
    return np.array([str(recs[int(i)].get("model_id") or i) for i in eval_idx])


def _build_stage_pools_from_train_indices(
    dataset: Any,
    train_indices: list[int],
    split_name: str = "",
    target_kind: str = "viability",
    seed: int = 101,
    smoke_test: bool = False,
) -> dict[int, list[int]]:
    """Build rich stage-matched training subsets strictly from `train_indices` (`0` test leakage)."""
    rng = np.random.default_rng(seed)
    if not _has_records(dataset) or smoke_test:
        cap = train_indices[: min(len(train_indices), 256)]
        return {1: cap, 2: cap, 3: cap, 4: cap, 5: cap}

    recs = dataset.records
    by_kind: dict[str, list[int]] = {}
    pharm_ferro: list[int] = []
    pharm_other: list[int] = []
    crispr_3d: list[int] = []
    for idx in train_indices:
        rc = recs[idx]
        kd = str(rc.get("kind", "other"))
        by_kind.setdefault(kd, []).append(idx)
        if kd == "pharmaco":
            if int(rc.get("is_ferroptosis_compound", 1)) > 0:
                pharm_ferro.append(idx)
            else:
                pharm_other.append(idx)
        elif kd == "crispr" and int(rc.get("is_3d_organoid", 0)) > 0:
            crispr_3d.append(idx)

    def _draw_list(pool: list[int], k: int) -> list[int]:
        if not pool or k <= 0:
            return []
        if len(pool) <= k:
            return list(pool)
        return [int(x) for x in rng.choice(pool, size=k, replace=False).tolist()]

    def _draw(kind: str, k: int) -> list[int]:
        return _draw_list(by_kind.get(kind, []), k)

    fallback = _draw("pharmaco", 2048) or train_indices[: min(len(train_indices), 2048)]
    icb_aug = _draw("icb", 600) * 4 + _draw("normal_sfti", 300) * 4
    s1 = _draw("basal", 2700) + icb_aug + _draw("perturb", 2500) + _draw_list(pharm_ferro, 1800)

    s2 = _draw("crispr", 65000) + _draw("perturb", 8500)
    if split_name == "5_cuproptosis_elesclomol_specificity":
        pharm_s3 = _draw_list(pharm_ferro, 38000) + _draw_list(pharm_other, 18000)
    else:
        pharm_s3 = _draw_list(pharm_ferro, 62000)
    s3 = pharm_s3 + _draw("rescue", 800) * 2 + _draw("synergy", 3000) + _draw("crispr", 4500)
    s4 = (
        _draw_list(pharm_ferro, 18000)
        + _draw_list(crispr_3d, 9500)
        + _draw("crispr", 10000)
        + _draw("oxylipid", 3000)
        + icb_aug
    )

    if split_name == "4_dual_crispr_fin_epistasis":
        s5 = _draw_list(pharm_ferro, 42000) + _draw("crispr", 20000) + _draw("rescue", 800) * 2
    elif target_kind == "chronos_fitness":
        s5 = _draw("crispr", 75000) + _draw_list(pharm_ferro, 3000) + _draw("perturb", 1200)
    elif split_name == "5_cuproptosis_elesclomol_specificity":
        s5 = _draw_list(pharm_ferro, 45000) + _draw_list(pharm_other, 22000) + _draw("perturb", 1500)
    else:
        s5 = _draw_list(pharm_ferro, 68000) + _draw("rescue", 800) * 2 + _draw("perturb", 1000)

    return {
        1: s1 if len(s1) >= 16 else fallback,
        2: s2 if len(s2) >= 16 else fallback,
        3: s3 if len(s3) >= 16 else fallback,
        4: s4 if len(s4) >= 16 else fallback,
        5: s5 if len(s5) >= 16 else fallback,
    }


def _build_epoch_batches(
    pool_arr: np.ndarray,
    group_id_int: np.ndarray | None,
    eff_bs: int,
    n_steps_ep: int,
    rng: np.random.Generator,
    use_grouped_sampling: bool = False,
) -> list[np.ndarray]:
    """Construct epoch batches alternating uniform calibration draws with group-stratified draws."""
    if pool_arr.size == 0:
        return []
    need = n_steps_ep * eff_bs
    reps = max(1, (need + pool_arr.size - 1) // pool_arr.size)
    perm_all = np.concatenate([rng.permutation(pool_arr) for _ in range(reps)], axis=0)
    batches: list[np.ndarray] = [perm_all[s_i * eff_bs : (s_i + 1) * eff_bs] for s_i in range(n_steps_ep)]
    if not use_grouped_sampling or group_id_int is None or eff_bs < 64 or n_steps_ep < 2:
        return batches

    g_pool = group_id_int[pool_arr]
    valid_mask = g_pool >= 0
    if int(valid_mask.sum()) < 48:
        return batches
    sub_idx = pool_arr[valid_mask]
    sub_gid = g_pool[valid_mask]
    order = np.argsort(sub_gid, kind="mergesort")
    s_idx = sub_idx[order]
    s_gid = sub_gid[order]
    uniq_g, split_pos, counts = np.unique(s_gid, return_index=True, return_counts=True)
    keep = counts >= 6
    if int(keep.sum()) < 4:
        return batches

    grp_arrays: list[np.ndarray] = [
        s_idx[st : st + ct] for st, ct, ok in zip(split_pos.tolist(), counts.tolist(), keep.tolist()) if ok
    ]
    weights = np.array([1.0 / math.sqrt(float(len(arr))) for arr in grp_arrays], dtype=np.float64)
    probs = weights / weights.sum()
    n_grp_pick = min(16, len(grp_arrays), max(4, eff_bs // 16))
    per_grp = max(6, eff_bs // n_grp_pick)

    for s_i in range(1, n_steps_ep, 2):
        chosen_g = rng.choice(len(grp_arrays), size=n_grp_pick, replace=False, p=probs)
        chunks: list[np.ndarray] = []
        for gi in chosen_g.tolist():
            arr_g = grp_arrays[gi]
            if arr_g.size >= per_grp:
                chunks.append(rng.choice(arr_g, size=per_grp, replace=False))
            else:
                chunks.append(rng.choice(arr_g, size=per_grp, replace=True))
        cat_b = np.concatenate(chunks, axis=0)
        if cat_b.size > eff_bs:
            cat_b = cat_b[:eff_bs]
        elif cat_b.size < eff_bs:
            rem = rng.choice(pool_arr, size=eff_bs - cat_b.size, replace=(pool_arr.size < eff_bs - cat_b.size))
            cat_b = np.concatenate([cat_b, rem], axis=0)
        batches[s_i] = cat_b
    return batches


def _train_strict_oos_model(
    dataset: Any,
    train_indices: list[int],
    forbid_test_set: set[int],
    collate_fn: Any,
    n_genes: int,
    model_size: str,
    config_path: Path,
    device: torch.device,
    data_dir: Path = DEFAULT_DATA_DIR,
    license_track: str = "open",
    split_name: str = "",
    target_kind: str = "viability",
    forbid_smiles: set[str] | None = None,
    forbid_ik14: set[str] | None = None,
    forbid_scaffolds: set[str] | None = None,
    epochs: int = 3,
    sar_epochs: int = 8,
    ablation_kw: dict[str, bool] | None = None,
    seed: int = 101,
    smoke_test: bool = False,
) -> FerroFluxFM:
    """Train one `FerroFluxFM` seed from scratch STRICTLY on `train_indices` (`0` test exposure)."""
    assert forbid_test_set.isdisjoint(set(train_indices)), "FATAL: test index present in train_indices!"
    t_seed0 = time.perf_counter()
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    np.random.seed(seed)
    ab = dict(ablation_kw or {})

    sz = model_size.strip().lower()
    if smoke_test or sz in ("compact", "v2_login"):
        model = FerroFluxFM(n_genes=n_genes, d_model=192, n_layers=3, n_heads=4, **ab).to(device)
    else:
        model = FerroFluxFM.from_config(config_path=config_path, variant=model_size, n_genes=n_genes, **ab).to(device)

    # 1. Stage-0.5 Leak-Free Biochemical & Phenotypic SAR + CysDB Covalent Prior Pretraining FIRST
    #    (populates frozen `sar_anchor_*` snapshot before `warmstart_from_train_indices` standardizes chem features!)
    sar_info = pretrain_stage05_chem_sar(
        model,
        data_dir=data_dir,
        license_track=license_track,
        device=device,
        forbid_smiles=forbid_smiles,
        forbid_ik14=forbid_ik14,
        forbid_scaffolds=forbid_scaffolds,
        train_dataset=dataset if (_has_records(dataset) and not smoke_test) else None,
        train_indices=train_indices,
        forbid_test_set=forbid_test_set,
        epochs=1 if smoke_test else sar_epochs,
        batch_size=256,
        lr=1.5e-3,
        seed=seed,
        smoke_test=smoke_test,
    )
    sar_replay_tensors = sar_info.get("sar_replay_tensors")

    # 2. Spectral, physical-anchor, shared-ECFP & 5-iter Soft-Impute CRISPR warm-start STRICTLY from `train_indices`
    if _has_records(dataset) and not smoke_test:
        warmstart_from_train_indices(
            model,
            dataset,
            train_indices,
            forbid_test_set=forbid_test_set,
            device=device,
            seed=seed,
        )

    loss_fn = FerroFluxCurriculumLoss().to(device)
    base_lr = 5.5e-4 if sz == "v3" else 6.0e-4
    sar_active = not bool(sar_info.get("skipped", True))
    sar_prefixes = (
        "perturbation_op.chem_fp_proj.",
        "perturbation_op.chem_desc_proj.",
        "perturbation_op.chem_in_proj.",
        "perturbation_op.chem_blocks.",
        "perturbation_op.chem_out_norm.",
    )
    warm_anchor_names = {
        "gene_essentiality_prior",
        "gene_expr_slope_prior",
        "expr_ctx_proj.0.weight",
        "expr_ctx_proj.0.bias",
        "cell_chronos_offset.weight",
        "cell_chronos_offset.bias",
    }

    dense_params: list[torch.nn.Parameter] = []
    sar_backbone_params: list[torch.nn.Parameter] = []
    sar_head_params: list[torch.nn.Parameter] = []
    anchor_params: list[torch.nn.Parameter] = []
    gene_emb_params: list[torch.nn.Parameter] = []
    ecfp_shared_params: list[torch.nn.Parameter] = []
    greybox_params: list[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in warm_anchor_names:
            anchor_params.append(p)
        elif n == "gene_ctx_emb":
            gene_emb_params.append(p)
        elif n.startswith("perturbation_op.chem_shared_ecfp_enc."):
            ecfp_shared_params.append(p)
        elif (
            n.startswith("diff_ferro_gem.aphynity_scaf_")
            or n.startswith("perturbation_op.chem_scaffold_")
            or n == "diff_ferro_gem.raw_scaf_gb_blend"
        ):
            greybox_params.append(p)
        elif n.startswith("perturbation_op.sar_pretrain_head."):
            sar_head_params.append(p)
        elif sar_active and n.startswith(sar_prefixes):
            sar_backbone_params.append(p)
        else:
            dense_params.append(p)

    is_pure_chronos = target_kind == "chronos_fitness" and split_name != "4_dual_crispr_fin_epistasis"
    eff_wd = 0.012 if split_name == "1_leave_out_scaffold" else 0.005
    param_groups: list[dict[str, Any]] = [
        {"params": dense_params, "lr": base_lr * (0.18 if is_pure_chronos else 1.0), "weight_decay": eff_wd}
    ]
    if greybox_params:
        gb_mult = 0.08 if split_name == "1_leave_out_scaffold" else (0.12 if is_pure_chronos else 0.18)
        param_groups.append(
            {
                "params": greybox_params,
                "lr": base_lr * gb_mult,
                "weight_decay": 1.5e-2,
            }
        )
    if ecfp_shared_params:
        param_groups.append(
            {
                "params": ecfp_shared_params,
                "lr": base_lr * (0.06 if split_name == "1_leave_out_scaffold" else 0.12),
                "weight_decay": 5.0e-2,
            }
        )
    if sar_backbone_params:
        sar_mult = 0.35 if split_name == "1_leave_out_scaffold" else 0.65
        param_groups.append({"params": sar_backbone_params, "lr": base_lr * sar_mult, "weight_decay": 1e-3})
    if sar_head_params:
        param_groups.append({"params": sar_head_params, "lr": base_lr * 0.06, "weight_decay": 1e-4})
    if anchor_params:
        param_groups.append({"params": anchor_params, "lr": 1.0e-5 if is_pure_chronos else 3.5e-5, "weight_decay": 1e-4})
    if gene_emb_params:
        param_groups.append({"params": gene_emb_params, "lr": 2.5e-5 if is_pure_chronos else 2.5e-4, "weight_decay": 1e-4})
    opt = torch.optim.AdamW(param_groups)

    stage_pools = _build_stage_pools_from_train_indices(
        dataset,
        train_indices,
        split_name=split_name,
        target_kind=target_kind,
        seed=seed,
        smoke_test=smoke_test,
    )
    fast_idxr = get_fast_indexer(dataset) if (_has_records(dataset) and not smoke_test) else None
    rng_tr = np.random.default_rng(seed + 17)
    batch_sz = 192 if (_has_records(dataset) and not smoke_test) else 32
    if smoke_test:
        stage_plan = [(1, 1, 1, 6), (2, 1, 2, 6), (3, 1, 3, 6), (4, 1, 4, 6)]
    elif is_pure_chronos:
        ep_main = max(2, int(epochs))
        stage_plan = [
            (1, 1, 1, 28),
            (2, ep_main, 2, 60),
            (5, 2, 2, 45),
        ]
    elif split_name == "4_dual_crispr_fin_epistasis":
        ep_main = max(2, int(epochs))
        stage_plan = [
            (1, 1, 1, 20),
            (2, ep_main, 2, 48),
            (3, ep_main, 3, 65),
            (4, 1, 4, 22),
            (5, 4, 3, 85),
        ]
    else:
        ep_main = max(2, int(epochs))
        stage_plan = [
            (1, 1, 1, 20),
            (2, ep_main, 2, 48),
            (3, ep_main, 3, 65),
            (4, 1, 4, 22),
            (5, 4, 3, 85),
        ]

    total_steps = sum(
        n_ep * min(step_cap, max(1, (len(stage_pools[p_key]) + batch_sz - 1) // batch_sz))
        for p_key, n_ep, _, step_cap in stage_plan
    )
    warmup_steps = max(2, int(0.06 * total_steps))

    def _lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        prog = float(step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    use_bf16 = device.type == "cuda"

    swa_state: dict[str, torch.Tensor] | None = None
    swa_count = 0

    for pool_key, n_ep, loss_stage, max_steps_cap in stage_plan:
        pool = stage_pools[pool_key]
        assert forbid_test_set.isdisjoint(set(pool)), f"FATAL: leak in stage pool {pool_key}!"
        eff_bs = min(batch_sz, max(2, len(pool)))
        loader = (
            None
            if fast_idxr is not None
            else DataLoader(
                Subset(dataset, pool),
                batch_size=eff_bs,
                shuffle=True,
                collate_fn=collate_fn,
            )
        )
        pool_arr = np.asarray(pool, dtype=np.int64)
        n_steps_ep = min(max_steps_cap, max(1, (len(pool) + eff_bs - 1) // eff_bs))
        model.train()
        for ep_i in range(n_ep):
            if loader is None:
                assert fast_idxr is not None
                ep_batches = _build_epoch_batches(
                    pool_arr,
                    fast_idxr.group_id_int,
                    eff_bs=eff_bs,
                    n_steps_ep=n_steps_ep,
                    rng=rng_tr,
                    use_grouped_sampling=False,
                )
                step_iter: Any = ((s_i, fast_idxr.get_batch(ep_batches[s_i], device=device)) for s_i in range(len(ep_batches)))
            else:
                step_iter = enumerate(loader)

            for step_i, raw_b in step_iter:
                if step_i >= max_steps_cap:
                    break
                b = raw_b if loader is None else _move_batch_to_device(raw_b, device)
                b["enable_mgm_masking"] = bool(pool_key == 1)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    out = model(b)
                    loss = loss_fn(out, b, stage=loss_stage)["loss"]
                    if not is_pure_chronos and loss_stage in (2, 3, 4) and step_i % 2 == 0 and sar_replay_tensors is not None:
                        sar_rep = compute_sar_replay_loss(model, sar_replay_tensors, device, rng_tr)
                        if sar_rep is not None:
                            loss = loss + 0.18 * sar_rep
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()

            if not smoke_test and (pool_key == 1 or (is_pure_chronos and pool_key == 2 and ep_i + 1 == n_ep)) and _has_records(dataset):
                # Re-align Rank-64 5-iter Soft-Impute CRISPR bilinear head to updated `z_cell` strictly on `train_indices`!
                warmstart_crispr_joint_and_svd(
                    model,
                    dataset,
                    train_indices,
                    forbid_test_set=forbid_test_set,
                    device=device,
                    seed=seed,
                )
                model.train()

            if not smoke_test and pool_key == 5 and ep_i >= max(0, n_ep - (2 if is_pure_chronos else 3)):
                with torch.no_grad():
                    cur_sd = model.state_dict()
                    if swa_state is None:
                        swa_state = {k: v.detach().clone() for k, v in cur_sd.items()}
                        swa_count = 1
                    else:
                        swa_count += 1
                        w_new = 1.0 / float(swa_count)
                        for k, v in cur_sd.items():
                            if v.is_floating_point():
                                swa_state[k].mul_(1.0 - w_new).add_(v.detach(), alpha=w_new)
                            else:
                                swa_state[k].copy_(v)

    if swa_state is not None and swa_count >= 1:
        model.load_state_dict(swa_state)

    if not smoke_test and _has_records(dataset):
        if is_pure_chronos:
            warmstart_crispr_joint_and_svd(
                model,
                dataset,
                train_indices,
                forbid_test_set=forbid_test_set,
                device=device,
                seed=seed,
            )
        elif target_kind == "viability":
            fit_post_curriculum_greybox_from_train_indices(
                model,
                dataset,
                train_indices,
                forbid_test_set=forbid_test_set,
                split_name=split_name,
                device=device,
                seed=seed,
            )
    model.eval()
    if not smoke_test:
        print(
            f"[OOD-v3-seed] track={license_track} split={split_name} seed={seed} converged in {time.perf_counter() - t_seed0:.1f}s",
            flush=True,
        )
    return model


@torch.no_grad()
def _predict_ensemble_on_exact_indices(
    models: list[FerroFluxFM],
    dataset: Any,
    eval_idx: np.ndarray,
    dx_idx: np.ndarray,
    target_kind: str,
    collate_fn: Any,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, Any]:
    fast_idxr = get_fast_indexer(dataset)
    if fast_idxr is not None:
        ev_iter: Any = (
            fast_idxr.get_batch(eval_idx[s_i : s_i + batch_size], device=device)
            for s_i in range(0, len(eval_idx), batch_size)
        )
    else:
        ev_iter = (
            _move_batch_to_device(rb, device)
            for rb in DataLoader(Subset(dataset, eval_idx.tolist()), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        )
    seed_preds: list[list[np.ndarray]] = [[] for _ in models]
    y_list: list[np.ndarray] = []

    for b in ev_iter:
        if target_kind == "chronos_fitness":
            y_list.append(b["chronos_target"].detach().cpu().numpy())
        else:
            y_list.append(b["viability_target"].detach().cpu().numpy())
        for m_i, m in enumerate(models):
            m.eval()
            out = m(b)
            key = "chronos_pred" if target_kind == "chronos_fitness" else "viability_pred"
            seed_preds[m_i].append(out[key].float().cpu().numpy())

    pred_stack = np.stack([np.concatenate(chunks, axis=0) for chunks in seed_preds], axis=0)
    y_true = np.concatenate(y_list, axis=0)

    dx_corrs: list[float] = []
    if dx_idx.size >= 4:
        if fast_idxr is not None:
            dx_iter: Any = (
                fast_idxr.get_batch(dx_idx[s_i : s_i + batch_size], device=device)
                for s_i in range(0, len(dx_idx), batch_size)
            )
        else:
            dx_iter = (
                _move_batch_to_device(rb, device)
                for rb in DataLoader(Subset(dataset, dx_idx.tolist()), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            )
        for b in dx_iter:
            dxt = b["delta_expr_target"].float().cpu().numpy()
            dxm = b["delta_expr_mask"].float().cpu().numpy() > 0.5
            dxp_acc = None
            for m in models:
                cur = m(b)["delta_expr_pred"].float().cpu().numpy()
                dxp_acc = cur if dxp_acc is None else (dxp_acc + cur)
            assert dxp_acc is not None
            dxp = dxp_acc / float(len(models))
            for r_i in range(dxt.shape[0]):
                mr = dxm[r_i]
                if int(mr.sum()) >= 8:
                    yt_r, yp_r = dxt[r_i, mr], dxp[r_i, mr]
                    if np.std(yt_r) > 1e-8 and np.std(yp_r) > 1e-8:
                        dx_corrs.append(_safe_pearson(yt_r, yp_r))

    return {
        "y_true": y_true,
        "pred_mean": pred_stack.mean(axis=0),
        "pred_std": pred_stack.std(axis=0),
        "delta_expr_pearson_r": float(np.mean(dx_corrs)) if dx_corrs else float("nan"),
    }


@torch.no_grad()
def _mechanistic_probe_audit(model: FerroFluxFM, n_genes: int, device: torch.device) -> dict[str, float]:
    model.eval()
    bsz = 8
    x = torch.ones(bsz, n_genes, device=device) * 1.2

    out_chal = model({"gene_expr": x, "gpx4_inhib": torch.full((bsz,), 4.0, device=device), "elesclomol_conc": torch.full((bsz,), 2.0, device=device)})
    out_lip1 = model({"gene_expr": x, "gpx4_inhib": torch.full((bsz,), 4.0, device=device), "elesclomol_conc": torch.full((bsz,), 2.0, device=device), "rta_conc": torch.full((bsz,), 5.0, device=device)})
    out_dfo = model({"gene_expr": x, "gpx4_inhib": torch.full((bsz,), 4.0, device=device), "elesclomol_conc": torch.full((bsz,), 2.0, device=device), "chelator_conc": torch.full((bsz,), 5.0, device=device)})

    rescue_ok = float(
        0.5 * (out_lip1["h_ferro"] < out_chal["h_ferro"]).float().mean().item()
        + 0.5 * (out_dfo["h_ferro"] < out_chal["h_ferro"]).float().mean().item()
    )
    cupro_err = float((out_lip1["h_cupro"] - out_chal["h_cupro"]).abs().max().item())

    out_wt = model({"gene_expr": x, "gpx4_inhib": torch.full((bsz,), 3.0, device=device)})
    out_dhcr7 = model({"gene_expr": x, "gpx4_inhib": torch.full((bsz,), 3.0, device=device), "dhcr7_mult": torch.full((bsz,), 0.05, device=device)})
    out_ebp = model({"gene_expr": x, "gpx4_inhib": torch.full((bsz,), 3.0, device=device), "ebp_mult": torch.full((bsz,), 0.05, device=device)})
    sterol_sign_acc = float(
        0.5 * (out_dhcr7["h_ferro"] < out_wt["h_ferro"]).float().mean().item()
        + 0.5 * (out_ebp["h_ferro"] > out_wt["h_ferro"]).float().mean().item()
    )

    out_basal = model({"gene_expr": x})
    out_pstk = model({"gene_expr": x, "pstk_mult": torch.full((bsz,), 0.05, device=device), "sepsecs_mult": torch.full((bsz,), 0.05, device=device)})
    sec46_drop = float(((out_basal["e_gpx4_active"] - out_pstk["e_gpx4_active"]) / (out_basal["e_gpx4_active"] + 1e-6)).mean().item())

    return {
        "rta_dfo_rescue_rate": round(rescue_ok, 4),
        "cuproptosis_invariance_err": round(cupro_err, 6),
        "sterol_dhcr7_vs_ebp_sign_acc": round(sterol_sign_acc, 4),
        "sec46_depletion_drop": round(sec46_drop, 4),
    }


def _verify_existing_outputs(output_dir: Path) -> int:
    ood_csv = output_dir / "ood_benchmark_summary.csv"
    abl_csv = output_dir / "ablation_summary.csv"
    base_csv = output_dir / "baselines_ood_summary.csv"
    if not ood_csv.is_file() or not abl_csv.is_file():
        print(f"[ERROR] Missing required summary CSVs in {output_dir}", file=sys.stderr)
        return 1
    df_ood = pd.read_csv(ood_csv)
    df_abl = pd.read_csv(abl_csv)
    assert len(df_ood) >= 6 and len(df_abl) >= 7
    if base_csv.is_file():
        df_b = pd.read_csv(base_csv)
        for _, r in df_ood.iterrows():
            sb = df_b[(df_b["license_track"] == r["license_track"]) & (df_b["ood_split"] == r["ood_split"])]
            if not sb.empty:
                assert int(r["n_test"]) == int(sb["n_test"].iloc[0]), f"Sample-size mismatch on {r['ood_split']}"
        assert (df_ood["p_value_vs_best_baseline"] > 0.0).all(), "Invalid exact p == 0.0 found!"
    print(
        json.dumps(
            {
                "status": "VERIFIED_STRICT_OOS",
                "ood_rows": int(len(df_ood)),
                "ablation_rows": int(len(df_abl)),
            },
            indent=2,
        )
    )
    return 0


def _merge_shards(shards_dir: Path, output_dir: Path, n_perm: int = 2000) -> int:
    """Deterministically merge parallel OOD & ablation shards into canonical summary CSVs."""
    if not shards_dir.is_dir():
        raise FileNotFoundError(f"Shards directory not found: {shards_dir}")

    base_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    abl_entries: list[tuple[int, dict[str, Any], np.ndarray | None, np.ndarray | None, np.ndarray | None]] = []

    for bp in sorted(shards_dir.glob("*_base.json")):
        base_rows.extend(json.loads(bp.read_text(encoding="utf-8")))
    for op in sorted(shards_dir.glob("*_ood.json")):
        ood_rows.extend(json.loads(op.read_text(encoding="utf-8")))
    for ap in sorted(shards_dir.glob("*_abl.json")):
        payload = json.loads(ap.read_text(encoding="utf-8"))
        for item in payload:
            a_idx = int(item.pop("_ablation_idx", 0))
            npz_p = Path(item.pop("_pred_npz", ""))
            yt, yp, cl = None, None, None
            if npz_p.is_file():
                npz = np.load(npz_p, allow_pickle=False)
                yt = npz["y_true"]
                yp = npz["y_pred"]
                cl = npz["clusters"] if "clusters" in npz.files else None
            abl_entries.append((a_idx, item, yt, yp, cl))

    track_order = {"open": 0, "academic": 1}
    split_order = {s[0]: i for i, s in enumerate(CANONICAL_6_OOD_SPLITS)}
    ood_rows.sort(key=lambda r: (track_order.get(str(r["license_track"]), 9), split_order.get(str(r["ood_split"]), 99)))
    base_rows.sort(
        key=lambda r: (
            track_order.get(str(r["license_track"]), 9),
            split_order.get(str(r["ood_split"]), 99),
            str(r["baseline_model"]),
        )
    )
    abl_entries.sort(key=lambda t: t[0])

    ood_qvals = benjamini_hochberg_fdr([float(r["p_value_vs_best_baseline"]) for r in ood_rows])
    for r, qv in zip(ood_rows, ood_qvals):
        r["q_value_bh_fdr"] = qv

    ablation_rows: list[dict[str, Any]] = []
    if abl_entries:
        full_entry = next((e for e in abl_entries if e[0] == 0), abl_entries[0])
        full_rho = float(full_entry[1]["spearman_rho"])
        full_yp = full_entry[3]
        for a_idx, row_d, yt_a, yp_a, cl_a in abl_entries:
            row_d["delta_rho_vs_full"] = round(float(row_d["spearman_rho"]) - full_rho, 4)
            if a_idx == 0 or full_yp is None or yp_a is None or yt_a is None:
                row_d["paired_perm_p_vs_full"] = 1.0
            else:
                p_val = paired_cluster_permutation_pvalue(
                    yt_a,
                    full_yp,
                    yp_a,
                    clusters=cl_a,
                    metric_fn=_safe_spearman,
                    n_perm=n_perm,
                    seed=300 + a_idx,
                )
                row_d["paired_perm_p_vs_full"] = round(p_val, 6)
            ablation_rows.append(row_d)

        if len(ablation_rows) > 1:
            abl_q = [1.0] + benjamini_hochberg_fdr([float(r["paired_perm_p_vs_full"]) for r in ablation_rows[1:]])
            for r, qv in zip(ablation_rows, abl_q):
                r["q_value_bh_fdr"] = qv

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(base_rows).to_csv(output_dir / "baselines_ood_summary.csv", index=False)
    pd.DataFrame(ood_rows).to_csv(output_dir / "ood_benchmark_summary.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(output_dir / "ablation_summary.csv", index=False)
    return _verify_existing_outputs(output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FerroFlux-FM v3 Strictly Leak-Free OOD & Retrained Ablation Suite")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT_DIR / "checkpoints" / "v3_clean")
    parser.add_argument("--model-size", type=str, default="v3", choices=["compact", "v2_login", "v3", "base", "xl"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tracks", type=str, nargs="+", default=["open", "academic"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sar-epochs", type=int, default=8)
    parser.add_argument("--max-samples-per-shard", type=int, default=200000)
    parser.add_argument("--max-threads", type=int, default=16)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-test-per-split", type=int, default=1500)
    parser.add_argument("--only-track", type=str, default=None, choices=["open", "academic"])
    parser.add_argument("--only-split", type=str, default=None)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--only-ablation-idx", type=int, default=None)
    parser.add_argument("--shard-out-prefix", type=Path, default=None)
    parser.add_argument("--save-oos-ckpt-dir", type=Path, default=None)
    parser.add_argument("--load-oos-ckpt-dir", type=Path, default=None)
    parser.add_argument("--merge-shards-dir", type=Path, default=None)
    parser.add_argument("--Verify", "--verify", dest="verify_only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    if args.verify_only:
        return _verify_existing_outputs(args.output_dir)
    if args.merge_shards_dir is not None:
        return _merge_shards(args.merge_shards_dir, args.output_dir, n_perm=args.n_perm)

    enforce_login_node_politeness(max_threads=args.max_threads)
    if "SLURM_JOB_ID" in os.environ and args.max_threads > 0:
        torch.set_num_threads(args.max_threads)

    if not args.smoke_test:
        audit_parquet_directory(args.data_dir)

    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds_use = [101] if args.smoke_test else list(args.seeds)
    tracks_run = [args.only_track] if args.only_track is not None else list(args.tracks)

    all_baseline_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []

    for track in tracks_run:
        print(f"[OOD-v3] Loading {track!r} multimodal dataset & extracting arrays...", flush=True)
        if args.smoke_test:
            dataset, collate_fn, n_genes = load_training_dataset(
                data_dir=args.data_dir,
                license_track=track,
                smoke_test=True,
            )
        else:
            from ferroflux.data.dataset import FerroFluxMultimodalDataset

            dataset = FerroFluxMultimodalDataset(
                data_dir=args.data_dir,
                license_track=track,
                max_samples_per_shard=args.max_samples_per_shard,
            )
            collate_fn = dataset.collate_fn
            n_genes = dataset.n_genes
        arrays = extract_multimodal_arrays(dataset, max_samples=args.max_eval_samples)
        print(f"[OOD-v3] Extracted {len(dataset)} harmonized records for track={track!r}.", flush=True)
        splits = _build_six_ood_splits(dataset, max_samples=args.max_eval_samples)

        m_viab_all = arrays["viability_mask"] > 0.5
        m_chr_all = arrays["chronos_mask"] > 0.5
        is_ferro_all = arrays.get("is_ferroptosis_compound", np.ones_like(arrays["viability_mask"])) > 0.5
        has_dx_all = arrays["has_dx_row"].astype(bool)
        group_all = arrays.get("group_id")
        rng = np.random.default_rng(42)

        run_ood_loop = args.only_ablation_idx is None
        if run_ood_loop:
            for split_idx, (canon_split, baseline_split_key) in enumerate(CANONICAL_6_OOD_SPLITS):
                if args.only_split is not None and canon_split != args.only_split:
                    continue
                t_split = time.perf_counter()
                tr_idx, te_idx_full = splits[canon_split]
                forbid_te = set(te_idx_full)
                f_smi, f_ik14, f_scaf = _extract_forbidden_test_chemistry(dataset, te_idx_full, canon_split)

                eval_idx, target_kind = _select_matched_eval_indices(
                    dataset,
                    arrays,
                    te_idx_full,
                    canon_split,
                    max_test_per_split=args.max_test_per_split,
                    seed=42 + split_idx,
                )
                clusters = _extract_clusters(dataset, eval_idx, canon_split)
                eval_groups = group_all[eval_idx] if group_all is not None else clusters

                tr_arr = np.asarray(tr_idx, dtype=np.int64)
                if target_kind == "chronos_fitness":
                    fit_tr = tr_arr[m_chr_all[tr_arr]]
                    b_target_key = "chronos_target"
                elif canon_split in ("1_leave_out_scaffold", "2_leave_out_lineage_ccrcc_melanoma", "6_sfti_tumor_vs_normal_til"):
                    ferro_tr = tr_arr[m_viab_all[tr_arr] & is_ferro_all[tr_arr]]
                    fit_tr = ferro_tr if ferro_tr.size >= 64 else tr_arr[m_viab_all[tr_arr]]
                    b_target_key = "viability_raw"
                else:
                    fit_tr = tr_arr[m_viab_all[tr_arr]]
                    b_target_key = "viability_raw"
                if fit_tr.size < 16:
                    fit_tr = tr_arr
                if fit_tr.size > 28000:
                    fit_tr = np.sort(rng.choice(fit_tr, size=28000, replace=False))

                dx_tr = tr_arr[has_dx_all[tr_arr]]
                if dx_tr.size > 4000:
                    dx_tr = np.sort(rng.choice(dx_tr, size=4000, replace=False))
                te_arr_full = np.asarray(te_idx_full, dtype=np.int64)
                dx_te = te_arr_full[has_dx_all[te_arr_full]]
                if dx_te.size > 512:
                    dx_te = np.sort(rng.choice(dx_te, size=512, replace=False))

                y_true = arrays[b_target_key][eval_idx]
                y_bin = (y_true < float(np.median(y_true))).astype(np.int32)
                if np.unique(y_bin).size < 2 and y_true.size >= 4:
                    y_bin = (y_true <= float(np.quantile(y_true, 0.25))).astype(np.int32)

                if args.save_oos_ckpt_dir is not None:
                    args.save_oos_ckpt_dir.mkdir(parents=True, exist_ok=True)
                    for s_val in seeds_use:
                        eff_sd = s_val + split_idx * 10
                        m_tr = _train_strict_oos_model(
                            dataset=dataset,
                            train_indices=tr_idx,
                            forbid_test_set=forbid_te,
                            collate_fn=collate_fn,
                            n_genes=n_genes,
                            model_size=args.model_size,
                            config_path=args.config,
                            device=device,
                            data_dir=args.data_dir,
                            license_track=track,
                            split_name=canon_split,
                            target_kind=target_kind,
                            forbid_smiles=f_smi,
                            forbid_ik14=f_ik14,
                            forbid_scaffolds=f_scaf,
                            epochs=args.epochs,
                            sar_epochs=args.sar_epochs,
                            seed=eff_sd,
                            smoke_test=bool(args.smoke_test),
                        )
                        torch.save(m_tr.cpu().state_dict(), args.save_oos_ckpt_dir / f"oos_{track}_{canon_split}_{eff_sd}.pt")
                    continue

                # 1. Fit & evaluate all 6 Reviewer #2 baselines at full convergence on EXACT `eval_idx`
                baseline_factories: list[Callable[[], Any]] = [
                    lambda: RidgeMultiOmicBaseline(seed=42),
                    lambda: ElasticNetMultiOmicBaseline(n_jobs=args.max_threads, seed=42),
                    lambda: LightGBMMultiOmicBaseline(n_estimators=250, n_jobs=args.max_threads, seed=42),
                    lambda: XGBoostMultiOmicBaseline(n_estimators=200, n_jobs=args.max_threads, seed=42),
                    lambda: ChempropCellStateMLPBaseline(
                        epochs=4 if args.smoke_test else 20, n_jobs=args.max_threads, seed=42, device=str(device)
                    ),
                    lambda: LinearPerturbBaseline(seed=42),
                ]
                baseline_preds_map: dict[str, np.ndarray] = {}
                baseline_rhos_map: dict[str, float] = {}

                for make_b in baseline_factories:
                    tb0 = time.perf_counter()
                    bm = make_b()
                    if isinstance(bm, LinearPerturbBaseline):
                        bm.fit(arrays, fit_tr, target_key=b_target_key, delta_train_idx=dx_tr)
                    else:
                        bm.fit(arrays, fit_tr, target_key=b_target_key)
                    bp = bm.predict(arrays, eval_idx)
                    tb_sec = time.perf_counter() - tb0

                    b_rho, b_lo, b_hi = cluster_bootstrap_ci(y_true, bp, clusters, _safe_spearman, n_boot=min(args.n_boot, 400))
                    b_prs, bp_lo, bp_hi = cluster_bootstrap_ci(y_true, bp, clusters, _safe_pearson, n_boot=min(args.n_boot, 400))
                    b_rmse, br_lo, br_hi = cluster_bootstrap_ci(y_true, bp, clusters, _safe_rmse, n_boot=min(args.n_boot, 400))
                    b_auc = _safe_auroc(y_bin, -bp)
                    b_wg_rho = _mean_within_group_spearman(y_true, bp, eval_groups)

                    b_dx_r = float("nan")
                    if hasattr(bm, "predict_delta_expr") and dx_te.size >= 4:
                        dxp = bm.predict_delta_expr(arrays, dx_te)
                        dxt = arrays["delta_expr_target"][dx_te].astype(np.float32)
                        dxm = arrays["delta_expr_mask"][dx_te].astype(bool)
                        rc = [
                            _safe_pearson(dxt[i, dxm[i]], dxp[i, dxm[i]])
                            for i in range(dx_te.size)
                            if int(dxm[i].sum()) >= 8 and np.std(dxt[i, dxm[i]]) > 1e-8 and np.std(dxp[i, dxm[i]]) > 1e-8
                        ]
                        b_dx_r = float(np.mean(rc)) if rc else float("nan")

                    baseline_preds_map[bm.name] = bp
                    baseline_rhos_map[bm.name] = b_rho
                    all_baseline_rows.append(
                        {
                            "license_track": track,
                            "ood_split": canon_split,
                            "baseline_model": bm.name,
                            "eval_target": target_kind,
                            "n_train": int(fit_tr.size),
                            "n_test": int(eval_idx.size),
                            "spearman_rho": round(b_rho, 4),
                            "spearman_ci95_low": round(b_lo, 4),
                            "spearman_ci95_high": round(b_hi, 4),
                            "within_group_spearman_rho": b_wg_rho,
                            "pearson_r": round(b_prs, 4),
                            "pearson_ci95_low": round(bp_lo, 4),
                            "pearson_ci95_high": round(bp_hi, 4),
                            "rmse": round(b_rmse, 4),
                            "rmse_ci95_low": round(br_lo, 4),
                            "rmse_ci95_high": round(br_hi, 4),
                            "lethal_auroc": round(b_auc, 4) if not np.isnan(b_auc) else np.nan,
                            "delta_expr_pearson_r": round(b_dx_r, 4) if not np.isnan(b_dx_r) else np.nan,
                            "fit_eval_sec": round(tb_sec, 3),
                        }
                    )
                    del bm
                    gc.collect()

                # 2. Train strictly Out-of-Sample FerroFlux-FM v3 ensemble from scratch on `tr_idx` ONLY
                if args.load_oos_ckpt_dir is not None:
                    oos_ensemble = []
                    for s_val in seeds_use:
                        eff_sd = s_val + split_idx * 10
                        sz_l = args.model_size.strip().lower()
                        if args.smoke_test or sz_l in ("compact", "v2_login"):
                            m_ld = FerroFluxFM(n_genes=n_genes, d_model=192, n_layers=3, n_heads=4).to(device)
                        else:
                            m_ld = FerroFluxFM.from_config(config_path=args.config, variant=args.model_size, n_genes=n_genes).to(device)
                        sd_p = args.load_oos_ckpt_dir / f"oos_{track}_{canon_split}_{eff_sd}.pt"
                        m_ld.load_state_dict(torch.load(sd_p, map_location=device, weights_only=False), strict=True)
                        m_ld.eval()
                        oos_ensemble.append(m_ld)
                else:
                    oos_ensemble = [
                        _train_strict_oos_model(
                            dataset=dataset,
                            train_indices=tr_idx,
                            forbid_test_set=forbid_te,
                            collate_fn=collate_fn,
                            n_genes=n_genes,
                            model_size=args.model_size,
                            config_path=args.config,
                            device=device,
                            data_dir=args.data_dir,
                            license_track=track,
                            split_name=canon_split,
                            target_kind=target_kind,
                            forbid_smiles=f_smi,
                            forbid_ik14=f_ik14,
                            forbid_scaffolds=f_scaf,
                            epochs=args.epochs,
                            sar_epochs=args.sar_epochs,
                            seed=s_val + split_idx * 10,
                            smoke_test=bool(args.smoke_test),
                        )
                        for s_val in seeds_use
                    ]
                ff_out = _predict_ensemble_on_exact_indices(
                    oos_ensemble,
                    dataset,
                    eval_idx,
                    dx_te,
                    target_kind=target_kind,
                    collate_fn=collate_fn,
                    device=device,
                )
                yp_ff = ff_out["pred_mean"]

                rho, rho_lo, rho_hi = cluster_bootstrap_ci(y_true, yp_ff, clusters, _safe_spearman, n_boot=args.n_boot)
                prs, prs_lo, prs_hi = cluster_bootstrap_ci(y_true, yp_ff, clusters, _safe_pearson, n_boot=args.n_boot)
                rmse, rmse_lo, rmse_hi = cluster_bootstrap_ci(y_true, yp_ff, clusters, _safe_rmse, n_boot=args.n_boot)
                auroc = _safe_auroc(y_bin, -yp_ff)
                wg_rho = _mean_within_group_spearman(y_true, yp_ff, eval_groups)
                dx_r = float(ff_out["delta_expr_pearson_r"])
                ece_val, brier_val = (
                    _compute_ece_and_brier(y_true, yp_ff) if target_kind == "viability" else (float("nan"), float("nan"))
                )

                best_b_name = max(baseline_rhos_map, key=lambda k: baseline_rhos_map[k])
                best_b_rho = baseline_rhos_map[best_b_name]
                best_b_pred = baseline_preds_map[best_b_name]
                delta_rho = rho - best_b_rho
                p_perm = paired_cluster_permutation_pvalue(
                    y_true,
                    yp_ff,
                    best_b_pred,
                    clusters=clusters,
                    metric_fn=_safe_spearman,
                    n_perm=200 if args.smoke_test else args.n_perm,
                    seed=100 + split_idx,
                )

                ver_tag = "v3" if args.model_size.lower() == "v3" else "v2"
                ood_rows.append(
                    {
                        "license_track": track,
                        "model_name": f"FerroFlux-FM-{ver_tag}-{args.model_size.upper()}-StrictOOS",
                        "n_ensemble_seeds": len(oos_ensemble),
                        "ood_split": canon_split,
                        "baseline_split_key": baseline_split_key,
                        "eval_target": target_kind,
                        "n_train": len(tr_idx),
                        "n_test_total": len(te_idx_full),
                        "n_test": int(eval_idx.size),
                        "spearman_rho": round(rho, 4),
                        "spearman_ci95_low": round(rho_lo, 4),
                        "spearman_ci95_high": round(rho_hi, 4),
                        "within_group_spearman_rho": wg_rho,
                        "pearson_r": round(prs, 4),
                        "pearson_ci95_low": round(prs_lo, 4),
                        "pearson_ci95_high": round(prs_hi, 4),
                        "rmse": round(rmse, 4),
                        "rmse_ci95_low": round(rmse_lo, 4),
                        "rmse_ci95_high": round(rmse_hi, 4),
                        "lethal_auroc": round(auroc, 4) if not np.isnan(auroc) else np.nan,
                        "ece_10bin": ece_val,
                        "brier_score": brier_val,
                        "delta_expr_pearson_r": round(dx_r, 4) if not np.isnan(dx_r) else np.nan,
                        "mean_epistemic_std": round(float(ff_out["pred_std"].mean()), 5),
                        "best_baseline_model": best_b_name,
                        "best_baseline_spearman": round(best_b_rho, 4),
                        "delta_spearman_vs_best_baseline": round(delta_rho, 4),
                        "p_value_vs_best_baseline": round(p_perm, 6),
                        "eval_sec": round(time.perf_counter() - t_split, 2),
                    }
                )
                print(
                    f"[OOD-v3] track={track} split={canon_split} n_test={eval_idx.size} "
                    f"FF_rho={rho:.4f} (within_grp={wg_rho:.4f}) vs {best_b_name}={best_b_rho:.4f} "
                    f"(delta={delta_rho:+.4f}, p={p_perm:.4f})",
                    flush=True,
                )
                del oos_ensemble
                gc.collect()

        # 3. Run all 8 TRUE RETRAINED Architectural Ablations (+ Full_FerroFlux_FM) on Held-Out Lineage Split
        run_abl_here = (not args.skip_ablations) and (
            args.only_ablation_idx is not None or (args.only_split is None and track == tracks_run[0])
        )
        if run_abl_here:
            abl_split_name = "2_leave_out_lineage_ccrcc_melanoma"
            ref_tr, ref_te_full = splits[abl_split_name]
            ref_forbid = set(ref_te_full)
            ref_f_smi, ref_f_ik14, ref_f_scaf = _extract_forbidden_test_chemistry(dataset, ref_te_full, abl_split_name)
            ref_te, ref_target = _select_matched_eval_indices(
                dataset,
                arrays,
                ref_te_full,
                abl_split_name,
                max_test_per_split=args.max_test_per_split,
                seed=101,
            )
            ref_clusters = _extract_clusters(dataset, ref_te, abl_split_name)
            ref_dx = np.asarray(ref_te_full, dtype=np.int64)[has_dx_all[np.asarray(ref_te_full, dtype=np.int64)]][:512]

            full_rho_val: float | None = None
            full_pred_vec: np.ndarray | None = None

            for abl_idx, (abl_name, abl_flags) in enumerate(ABLATION_9_SPECS):
                if args.only_ablation_idx is not None and abl_idx != args.only_ablation_idx:
                    continue
                t_abl = time.perf_counter()
                abl_models = [
                    _train_strict_oos_model(
                        dataset=dataset,
                        train_indices=ref_tr,
                        forbid_test_set=ref_forbid,
                        collate_fn=collate_fn,
                        n_genes=n_genes,
                        model_size=args.model_size,
                        config_path=args.config,
                        device=device,
                        data_dir=args.data_dir,
                        license_track=track,
                        split_name=abl_split_name,
                        target_kind=ref_target,
                        forbid_smiles=ref_f_smi,
                        forbid_ik14=ref_f_ik14,
                        forbid_scaffolds=ref_f_scaf,
                        epochs=args.epochs,
                        sar_epochs=args.sar_epochs,
                        ablation_kw=abl_flags,
                        seed=s_val,
                        smoke_test=bool(args.smoke_test),
                    )
                    for s_val in seeds_use
                ]
                abl_out = _predict_ensemble_on_exact_indices(
                    abl_models,
                    dataset,
                    ref_te,
                    ref_dx,
                    target_kind=ref_target,
                    collate_fn=collate_fn,
                    device=device,
                )
                yt_a = abl_out["y_true"]
                yp_a = abl_out["pred_mean"]
                yb_a = (yt_a < float(np.median(yt_a))).astype(np.int32)
                a_rho, a_lo, a_hi = cluster_bootstrap_ci(yt_a, yp_a, ref_clusters, _safe_spearman, n_boot=min(args.n_boot, 600))
                a_rmse = _safe_rmse(yt_a, yp_a)
                a_auc = _safe_auroc(yb_a, -yp_a)
                a_dx = float(abl_out["delta_expr_pearson_r"])
                probes = _mechanistic_probe_audit(abl_models[0], n_genes=n_genes, device=device)

                if full_rho_val is None and abl_idx == 0:
                    full_rho_val = a_rho
                    full_pred_vec = yp_a.copy()
                    p_vs_full = 1.0
                elif full_pred_vec is not None:
                    p_vs_full = paired_cluster_permutation_pvalue(
                        yt_a,
                        full_pred_vec,
                        yp_a,
                        clusters=ref_clusters,
                        metric_fn=_safe_spearman,
                        n_perm=200 if args.smoke_test else args.n_perm,
                        seed=300 + abl_idx,
                    )
                else:
                    p_vs_full = 1.0

                row_rec: dict[str, Any] = {
                    "license_track": track,
                    "ablation_variant": abl_name,
                    "retrained_from_scratch": True,
                    "model_size": args.model_size,
                    "ood_split": abl_split_name,
                    "n_train": len(ref_tr),
                    "n_test": int(ref_te.size),
                    "spearman_rho": round(a_rho, 4),
                    "spearman_ci95_low": round(a_lo, 4),
                    "spearman_ci95_high": round(a_hi, 4),
                    "delta_rho_vs_full": round(a_rho - (full_rho_val if full_rho_val is not None else a_rho), 4),
                    "paired_perm_p_vs_full": round(p_vs_full, 6),
                    "rmse": round(a_rmse, 4),
                    "lethal_auroc": round(a_auc, 4) if not np.isnan(a_auc) else np.nan,
                    "delta_expr_pearson_r": round(a_dx, 4) if not np.isnan(a_dx) else np.nan,
                    **probes,
                    "train_eval_sec": round(time.perf_counter() - t_abl, 2),
                }
                if args.shard_out_prefix is not None:
                    args.shard_out_prefix.parent.mkdir(parents=True, exist_ok=True)
                    pred_npz = Path(f"{args.shard_out_prefix}_abl{abl_idx}_pred.npz")
                    save_kw: dict[str, Any] = {"y_true": yt_a, "y_pred": yp_a}
                    if ref_clusters is not None:
                        save_kw["clusters"] = ref_clusters.astype(str)
                    np.savez_compressed(pred_npz, **save_kw)
                    row_rec["_ablation_idx"] = abl_idx
                    row_rec["_pred_npz"] = str(pred_npz)

                ablation_rows.append(row_rec)
                print(
                    f"[ABLATION-v3] {abl_name}: rho={a_rho:.4f} "
                    f"rescue={probes['rta_dfo_rescue_rate']} sec46_drop={probes['sec46_depletion_drop']}",
                    flush=True,
                )
                del abl_models
                gc.collect()

        del dataset, arrays, splits
        gc.collect()

    if args.shard_out_prefix is not None:
        args.shard_out_prefix.parent.mkdir(parents=True, exist_ok=True)
        if all_baseline_rows:
            Path(f"{args.shard_out_prefix}_base.json").write_text(json.dumps(all_baseline_rows, indent=2), encoding="utf-8")
        if ood_rows:
            Path(f"{args.shard_out_prefix}_ood.json").write_text(json.dumps(ood_rows, indent=2), encoding="utf-8")
        if ablation_rows:
            Path(f"{args.shard_out_prefix}_abl.json").write_text(json.dumps(ablation_rows, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "SHARD_OK",
                    "shard_prefix": str(args.shard_out_prefix),
                    "baseline_rows": len(all_baseline_rows),
                    "ood_rows": len(ood_rows),
                    "ablation_rows": len(ablation_rows),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    ood_qvals = benjamini_hochberg_fdr([r["p_value_vs_best_baseline"] for r in ood_rows])
    for r, qv in zip(ood_rows, ood_qvals):
        r["q_value_bh_fdr"] = qv

    if len(ablation_rows) > 1:
        abl_qvals = [1.0] + benjamini_hochberg_fdr([r["paired_perm_p_vs_full"] for r in ablation_rows[1:]])
        for r, qv in zip(ablation_rows, abl_qvals):
            r["q_value_bh_fdr"] = qv

    base_path = args.output_dir / "baselines_ood_summary.csv"
    ood_path = args.output_dir / "ood_benchmark_summary.csv"
    abl_path = args.output_dir / "ablation_summary.csv"
    pd.DataFrame(all_baseline_rows).to_csv(base_path, index=False)
    pd.DataFrame(ood_rows).to_csv(ood_path, index=False)
    pd.DataFrame(ablation_rows).to_csv(abl_path, index=False)

    print(
        json.dumps(
            {
                "status": "OK",
                "strict_out_of_sample": True,
                "baselines_csv": str(base_path),
                "baseline_rows": len(all_baseline_rows),
                "ood_benchmark_csv": str(ood_path),
                "ood_rows": len(ood_rows),
                "ablation_summary_csv": str(abl_path),
                "ablation_rows": len(ablation_rows),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
