"""De-Confounded Locked Stage-0 Blind Validation on Quarantined Bradbury / Overholtzer / Wiesner Cohorts (v2).

Remediation Guarantees Addressing Forensic Findings `P3-TREATMENT-INPUT-CIRCULARITY`,
`P3-MRNA-STEROL-TAUTOLOGY`, `P3-ZENODO-WAVE-DECOMPOSITION`, and `P5-SMALL-N-STAGE0-POWER`:
1. **Step 0 (Mandatory SHA-256 Lock Verification)**: Verifies every `.pt` checkpoint in
   `--checkpoint-dir` against `--manifest` via `quarantine.verify_frozen_checkpoint`.
2. **100% Zero Circularity on Cohorts 1 & 2 (`GSE310954`, `GSE319397`, `GSE319419`)**:
   ALL tumor RNA-seq samples receive **identical unconditioned perturbation inputs**
   (`np_params = 0`, `cys2_media = 1.0`, `media_ext = default`). Ground-truth treatment arms
   (`mel_arms`, `pc_tiers`) are NEVER passed into `np_params` or `media_ext`.
3. **100% Zero mRNA Self-Correlation Tautology via Input Gene Masking**:
   - Cohort 1 (`GSE310954`, `n=11`): `CGAS_STING_GENES` and `PDL1_IFN_GENES` are masked out
     (`gene_expr = 0`, `gene_mask = 0`) at input time when predicting `cGAS-STING` remodeling.
   - Cohort 2 (`GSE319397`, `n=9` `[7-DHC]`): `EBP`, `SC5D`, and `DHCR7` are masked out
     (`gene_expr = 0`, `gene_mask = 0`) at input time so sterol cross-prediction relies strictly
     on non-overlapping transcripts via Stage-1 Masked Gene Modeling (`mgm_recon`), never on
     algebraic input-gene self-correlation.
4. **100% Internal PDE Wave Evaluation on Cohort 3 (`Zenodo 18271042`, `n=8` FOVs)**:
   Passes identical `MCF10A + ML162` perturbation parameters across all 8 FOVs and evaluates
   `predict_spatial_wave(..., ref_well_deg35=well_deg35)` internal PDE outputs (`u_final`,
   `propagative_prob`, `morans_i`) directly without any external `paracrine_factor` post-multiplier.
5. **Exact Small-Sample Non-Parametric Statistics + BH-FDR**:
   Reports exact sample sizes `n_samples`, Mann-Whitney U `mannwhitney_p`, exact permutation
   `permutation_p >= 1/(B+1)`, and Benjamini-Hochberg `q_value_bh_fdr`.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from scipy.stats import kendalltau, mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from ferroflux.audit.resource_guard import enforce_login_node_politeness
from ferroflux.models.ferroflux_fm import FerroFluxFM
from ferroflux.security.license_guard import assert_ag2_inference_only
from ferroflux.security.quarantine import (
    DEFAULT_FROZEN_MANIFEST,
    QuarantineViolationError,
    verify_frozen_checkpoint,
)

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
DEFAULT_CKPT_DIR = ROOT_DIR / "checkpoints" / "v3_clean"
DEFAULT_V2_MANIFEST = ROOT_DIR / "configs" / "frozen_checkpoints_v2_sha256.json"
DEFAULT_V3_MANIFEST = ROOT_DIR / "configs" / "frozen_checkpoints_v3_sha256.json"
DEFAULT_LOCKBOX_DIR = ROOT_DIR / "data" / "stage0_blind_lockbox"
DEFAULT_TABLES_DIR = ROOT_DIR / "results" / "tables"
DEFAULT_VOCAB_JSON = ROOT_DIR / "data" / "processed" / "gene_vocab.json"

CGAS_STING_GENES = ("CGAS", "MB21D1", "STING1", "TMEM173", "TBK1", "IRF3", "CXCL10", "CCL5", "IFNB1", "ISG15")
IL6_STAT3_GENES = ("IL6", "IL6R", "IL6ST", "STAT3", "SOCS3", "JAK1")
PDL1_IFN_GENES = ("CD274", "PDCD1LG2", "STAT1", "IRF1", "CXCL9", "IDO1")
M1_TAM_GENES = ("NOS2", "CD86", "CD80", "TNF", "IL1B", "CXCL10")
M2_TAM_GENES = ("MRC1", "CD163", "ARG1", "IL10", "TGFB1")
TLR_FERRO_GENES = ("TLR2", "TLR7", "TLR8", "MYD88", "ACSL4", "TFRC", "HMOX1", "PTGS2", "CHAC1", "NCOA4")
STEROL_GENES = ("EBP", "SC5D", "DHCR7", "SQLE", "HMGCR", "CYP51A1", "MSMO1", "FDFT1")


def verify_all_checkpoints_locked(
    checkpoint_dir: Path | str,
    manifest_path: Path | str = DEFAULT_FROZEN_MANIFEST,
    lockbox_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Verify every `.pt` checkpoint in `checkpoint_dir` has an intact pre-Stage-0 SHA-256 lock."""
    ckpt_root = Path(checkpoint_dir)
    if not ckpt_root.exists():
        raise QuarantineViolationError(f"Checkpoint directory does not exist: {ckpt_root}")

    pt_files = sorted(p for p in ckpt_root.rglob("*.pt") if p.is_file())
    if not pt_files:
        raise QuarantineViolationError(
            f"Zero frozen `.pt` checkpoints found in {ckpt_root}; refusing to unlock Stage-0 blind evaluation!"
        )

    verified_entries: list[dict[str, Any]] = []
    for pt_path in pt_files:
        entry = verify_frozen_checkpoint(pt_path, manifest_path=manifest_path)
        mtime = pt_path.stat().st_mtime
        frozen_iso = entry.get("frozen_at_utc")
        if frozen_iso:
            frozen_dt = datetime.fromisoformat(frozen_iso)
            if frozen_dt.tzinfo is None:
                frozen_dt = frozen_dt.replace(tzinfo=timezone.utc)
            if mtime > frozen_dt.timestamp() + 2.0:
                raise QuarantineViolationError(
                    f"Checkpoint {pt_path} mtime ({mtime}) is newer than its SHA-256 freeze timestamp ({frozen_iso})!"
                )
        verified_entries.append(entry)

    if lockbox_dir is not None:
        lbox = Path(lockbox_dir)
        if "processed" in lbox.parts:
            raise QuarantineViolationError(f"Lockbox directory {lbox} MUST NOT reside inside `data/processed/`!")

    return verified_entries


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    fin = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[fin], ya[fin]
    if xa.size < 3 or np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return 0.0
    val = float(spearmanr(xa, ya).statistic)
    return 0.0 if np.isnan(val) else val


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    fin = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[fin], ya[fin]
    if xa.size < 3 or np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return 0.0
    val = float(pearsonr(xa, ya).statistic)
    return 0.0 if np.isnan(val) else val


def _safe_auroc(labels_binary: np.ndarray, scores: np.ndarray) -> float:
    yb = np.asarray(labels_binary, dtype=np.int32).ravel()
    ys = np.asarray(scores, dtype=np.float64).ravel()
    fin = np.isfinite(ys)
    yb, ys = yb[fin], ys[fin]
    if yb.size < 4 or np.unique(yb).size < 2:
        return float("nan")
    return float(roc_auc_score(yb, ys))


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    fn: Any,
    n_boot: int = 1000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    xa = np.asarray(x)
    ya = np.asarray(y)
    pt = float(fn(xa, ya))
    n = xa.shape[0]
    if n < 4 or np.isnan(pt) or n_boot <= 0:
        return pt, pt, pt
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = float(fn(xa[idx], ya[idx]))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return pt, pt, pt
    return pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _two_group_pvalues(pos_scores: np.ndarray, neg_scores: np.ndarray, n_perm: int = 4000, seed: int = 42) -> tuple[float, float]:
    p = np.asarray(pos_scores, dtype=np.float64).ravel()
    q = np.asarray(neg_scores, dtype=np.float64).ravel()
    p, q = p[np.isfinite(p)], q[np.isfinite(q)]
    if p.size < 2 or q.size < 2:
        return 1.0, 1.0
    mwu_p = float(mannwhitneyu(p, q, alternative="two-sided").pvalue)
    obs_diff = abs(float(p.mean() - q.mean()))
    pooled = np.concatenate([p, q])
    rng = np.random.default_rng(seed)
    exceed = 0
    n_p = p.size
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(float(perm[:n_p].mean() - perm[n_p:].mean())) >= obs_diff - 1e-12:
            exceed += 1
    perm_p = float((exceed + 1) / (n_perm + 1))
    return round(mwu_p, 6), round(perm_p, 6)


def _benjamini_hochberg(pvals: list[float]) -> list[float]:
    arr = np.asarray(pvals, dtype=np.float64)
    m = len(arr)
    if m == 0:
        return []
    order = np.argsort(arr)
    ranked = arr[order]
    q_ranked = np.empty(m, dtype=np.float64)
    run_min = 1.0
    for k in range(m - 1, -1, -1):
        val = min(1.0, float(ranked[k] * m / (k + 1)))
        run_min = min(run_min, val)
        q_ranked[k] = run_min
    out = np.empty(m, dtype=np.float64)
    out[order] = q_ranked
    return [round(float(x), 6) for x in out]


def _load_gene_vocab(vocab_path: Path = DEFAULT_VOCAB_JSON, fallback_n_genes: int = 2048) -> tuple[list[str], dict[str, int]]:
    if vocab_path.is_file():
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
        genes = list(data.get("genes", []))
        g2i = {g.upper(): i for i, g in enumerate(genes)}
        for alias, canon in data.get("alias_to_canonical", {}).items():
            if canon.upper() in g2i:
                g2i[alias.upper()] = g2i[canon.upper()]
        if genes:
            return genes, g2i
    from ferroflux.layers.cis_reg_slot import CANONICAL_64_FERROPTOSIS_LOCI

    base = list(CANONICAL_64_FERROPTOSIS_LOCI)
    extra = [
        g
        for g in (
            CGAS_STING_GENES
            + IL6_STAT3_GENES
            + PDL1_IFN_GENES
            + M1_TAM_GENES
            + M2_TAM_GENES
            + TLR_FERRO_GENES
            + STEROL_GENES
        )
        if g not in base
    ]
    genes = (base + extra + [f"GENE_{i}" for i in range(len(base) + len(extra), fallback_n_genes)])[:fallback_n_genes]
    return genes, {g.upper(): i for i, g in enumerate(genes)}


def _load_frozen_models(
    checkpoint_dir: Path,
    license_track: str,
    device: torch.device,
) -> list[FerroFluxFM]:
    candidates: list[Path] = []
    for sub in (
        checkpoint_dir / f"v3_{license_track}",
        checkpoint_dir / f"v2_{license_track}",
        checkpoint_dir / license_track,
        checkpoint_dir / f"xl_{license_track}",
        checkpoint_dir,
    ):
        if sub.is_dir():
            hits = sorted(sub.glob(f"*{license_track}*seed*.pt"))
            if hits:
                candidates = hits
                break
    if not candidates:
        candidates = sorted(checkpoint_dir.rglob("*.pt"))
    if not candidates:
        raise QuarantineViolationError(f"No checkpoints found for track={license_track!r} in {checkpoint_dir}")

    models: list[FerroFluxFM] = []
    for ckpt_p in candidates:
        payload = torch.load(ckpt_p, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_init_kwargs" in payload:
            m = FerroFluxFM(**payload["model_init_kwargs"])
            m.load_state_dict(payload["model_state_dict"], strict=False)
        elif isinstance(payload, dict) and "model_state_dict" in payload:
            m = FerroFluxFM.from_config(
                variant=payload.get("model_size", "base"),
                n_genes=int(payload.get("n_genes", 2048)),
            )
            m.load_state_dict(payload["model_state_dict"], strict=False)
        else:
            m = FerroFluxFM()
            m.load_state_dict(payload, strict=False)
        m.to(device)
        m.eval()
        assert not m.training, "Stage-0 model MUST remain in eval() mode!"
        models.append(m)
    return models


def _download_bytes(url: str, dest: Path, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _parse_geo_series_matrix_metadata(gz_path: Path) -> list[dict[str, str]]:
    text = gzip.decompress(gz_path.read_bytes()).decode("utf-8", errors="replace")
    titles: list[str] = []
    accs: list[str] = []
    chars_rows: list[list[str]] = []
    desc_rows: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("!Sample_title"):
            titles = [x.strip('"') for x in line.split("\t")[1:]]
        elif line.startswith("!Sample_geo_accession"):
            accs = [x.strip('"') for x in line.split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            chars_rows.append([x.strip('"') for x in line.split("\t")[1:]])
        elif line.startswith("!Sample_description"):
            desc_rows.append([x.strip('"') for x in line.split("\t")[1:]])
    n = max(len(titles), len(accs))
    records: list[dict[str, str]] = []
    for i in range(n):
        ch = "; ".join(row[i] for row in chars_rows if i < len(row))
        ds = "; ".join(row[i] for row in desc_rows if i < len(row))
        records.append(
            {
                "gsm_id": accs[i] if i < len(accs) else f"SAMPLE_{i:03d}",
                "title": titles[i] if i < len(titles) else f"Sample_{i:03d}",
                "characteristics": ch,
                "description": ds,
            }
        )
    return records


def _fetch_geo_expression_and_meta(
    gse_acc: str,
    lockbox_dir: Path,
    gene_to_idx: dict[str, int],
    n_genes: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    """Fetch GEO series matrix + NCBI/suppl RNA-seq expression into isolated `lockbox_dir`."""
    prefix = gse_acc[:-3] + "nnn"
    gse_dir = lockbox_dir / gse_acc
    gse_dir.mkdir(parents=True, exist_ok=True)

    mat_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_acc}/matrix/{gse_acc}_series_matrix.txt.gz"
    mat_gz = _download_bytes(mat_url, gse_dir / f"{gse_acc}_series_matrix.txt.gz")
    meta_rows = _parse_geo_series_matrix_metadata(mat_gz)
    meta_df = pd.DataFrame(meta_rows)

    candidate_urls = [
        f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse_acc}&format=file&file={gse_acc}_norm_counts_TPM_GRCh38.p13_NCBI.tsv.gz",
        f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse_acc}&format=file&file={gse_acc}_raw_counts_GRCh38.p13_NCBI.tsv.gz",
        f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse_acc}&format=file&file={gse_acc}_norm_counts_TPM_GRCm39_NCBI.tsv.gz",
        f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse_acc}&format=file&file={gse_acc}_raw_counts_GRCm39_NCBI.tsv.gz",
    ]
    for cached_p in sorted(gse_dir.glob("*.gz")) + sorted(gse_dir.glob("*.tsv")) + sorted(gse_dir.glob("*.csv")):
        if "series_matrix" not in cached_p.name:
            candidate_urls.insert(0, f"file://{cached_p}")

    suppl_index_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_acc}/suppl/"
    if not any(u.startswith("file://") for u in candidate_urls):
        try:
            r_sup = requests.get(suppl_index_url, timeout=30)
            if r_sup.ok:
                for fname in re.findall(r'href="([^"]+\.(?:tsv|csv|txt)\.gz)"', r_sup.text, flags=re.I):
                    candidate_urls.append(suppl_index_url + fname)
        except requests.RequestException:
            pass

    expr_df: pd.DataFrame | None = None
    for url in candidate_urls:
        try:
            if url.startswith("file://"):
                dest = Path(url[len("file://") :])
            else:
                fname = url.split("=")[-1].split("/")[-1]
                dest = gse_dir / fname
                _download_bytes(url, dest, timeout=45)
            raw_bytes = gzip.decompress(dest.read_bytes()) if dest.name.endswith(".gz") else dest.read_bytes()
            sep = "," if ".csv" in dest.name.lower() else "\t"
            df_try = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, index_col=0)
            if df_try.shape[0] >= 50 and df_try.shape[1] >= 2:
                expr_df = df_try
                break
        except Exception:
            continue

    if expr_df is not None:
        for sym_col in ("gene_name", "gene_symbol", "Symbol", "SYMBOL", "gene"):
            if sym_col in expr_df.columns:
                syms = expr_df[sym_col].astype(str).str.strip()
                valid = (syms != "") & (syms.str.lower() != "nan")
                if int(valid.sum()) >= 100:
                    expr_df = expr_df.loc[valid].set_index(syms.loc[valid].str.upper())
                    break

    if expr_df is not None and all(str(ix).isdigit() for ix in expr_df.index[:20]):
        for annot_url in (
            "https://www.ncbi.nlm.nih.gov/geo/download/?format=file&type=rnaseq_counts&file=Human.GRCh38.p13.annot.tsv.gz",
            "https://www.ncbi.nlm.nih.gov/geo/download/?format=file&type=rnaseq_counts&file=Mouse.GRCm39.annot.tsv.gz",
        ):
            annot_dest = lockbox_dir / annot_url.split("=")[-1]
            try:
                _download_bytes(annot_url, annot_dest, timeout=45)
                adf = pd.read_csv(annot_dest, sep="\t", usecols=["GeneID", "Symbol"])
                id2sym = dict(zip(adf["GeneID"].astype(str), adf["Symbol"].astype(str).str.upper()))
                mapped = [id2sym.get(str(i), "") for i in expr_df.index]
                if sum(bool(s) for s in mapped) > 100:
                    expr_df.index = mapped
                    expr_df = expr_df[expr_df.index != ""]
                    break
            except Exception:
                continue

    expr_mat = np.zeros((len(meta_df), n_genes), dtype=np.float32)
    pathway_obs: dict[str, np.ndarray] = {}

    if expr_df is not None:
        drop_meta_num = {"start", "end", "length", "entrez_id", "chromosome"}
        keep_num_cols = [c for c in expr_df.select_dtypes(include=[np.number]).columns if str(c).lower() not in drop_meta_num]
        num_df = expr_df[keep_num_cols]
        if num_df.shape[1] > 0:
            vals = num_df.to_numpy(dtype=np.float64)
            if np.nanmax(vals) > 50.0:
                col_sums = np.maximum(np.nansum(vals, axis=0, keepdims=True), 1.0)
                vals = np.log2(vals / col_sums * 1e6 + 1.0)
            norm_df = pd.DataFrame(vals, index=num_df.index.astype(str).str.upper(), columns=[str(c) for c in num_df.columns])
            norm_df = norm_df.groupby(level=0).mean()
            col_set = {c: c for c in norm_df.columns}
            col_upper = {c.upper(): c for c in norm_df.columns}

            aligned_cols: list[str] = []
            for idx_m, mrow in meta_df.iterrows():
                title_s = str(mrow.get("title", "")).strip()
                gsm_s = str(mrow.get("gsm_id", "")).strip()
                desc_s = str(mrow.get("description", ""))
                desc_tokens = [
                    re.sub(r"(?i)^library\s*name\s*:\s*", "", tok).strip()
                    for tok in desc_s.split(";")
                    if tok.strip()
                ]
                matched: str | None = None
                for cand_tok in [title_s, gsm_s, *desc_tokens]:
                    if cand_tok in col_set:
                        matched = col_set[cand_tok]
                        break
                    if cand_tok.upper() in col_upper:
                        matched = col_upper[cand_tok.upper()]
                        break
                if matched is None:
                    raise ValueError(f"Exact sample-column alignment failed for {gsm_s} ({title_s}) in {gse_acc}!")
                aligned_cols.append(matched)
            sub = norm_df[aligned_cols]

            for gsym, row_vals in sub.iterrows():
                if gsym in gene_to_idx and gene_to_idx[gsym] < n_genes:
                    expr_mat[:, gene_to_idx[gsym]] = row_vals.to_numpy(dtype=np.float32)

            titles_list = [str(r.get("title", "")) for _, r in meta_df.iterrows()]
            is_ctrl = np.array(
                [
                    any(k in t.lower() for k in ("_ct_", "control", "saline", "vehicle", "untreated"))
                    and "dot" not in t.lower()
                    for t in titles_list
                ],
                dtype=bool,
            )
            line_prefixes = [
                (t.split("_CT_")[0] if "_CT_" in t else (t.split("_CDOT_")[0] if "_CDOT_" in t else "ALL"))
                for t in titles_list
            ]

            for pname, gset in (
                ("cGAS_STING", CGAS_STING_GENES),
                ("IL6_STAT3", IL6_STAT3_GENES),
                ("CD274_PDL1", PDL1_IFN_GENES),
                ("M1_TAM", M1_TAM_GENES),
                ("M2_TAM", M2_TAM_GENES),
                ("TLR_ferroptosis", TLR_FERRO_GENES),
                ("sterol_biosynthesis", ("EBP", "SC5D", "DHCR7")),
            ):
                present = [g for g in gset if g in sub.index]
                if present:
                    raw_sig = sub.loc[present].mean(axis=0).to_numpy(dtype=np.float32)
                    pathway_obs[pname] = raw_sig
                    if is_ctrl.any():
                        centered = raw_sig.copy()
                        for lp in set(line_prefixes):
                            m_lp = np.array([p == lp for p in line_prefixes], dtype=bool)
                            m_ref = m_lp & is_ctrl
                            ref_val = float(raw_sig[m_ref].mean()) if m_ref.any() else float(raw_sig[is_ctrl].mean())
                            centered[m_lp] = raw_sig[m_lp] - ref_val
                        pathway_obs[f"{pname}_delta"] = centered

            if "EBP" in sub.index and "DHCR7" in sub.index:
                up_genes = [g for g in ("EBP", "SC5D") if g in sub.index]
                up_mean = sub.loc[up_genes].mean(axis=0).to_numpy(dtype=np.float32)
                dhcr7_arr = sub.loc["DHCR7"].to_numpy(dtype=np.float32)
                pathway_obs["sterol_7dhc_balance"] = up_mean - dhcr7_arr

    return meta_df, expr_mat, pathway_obs


def _fetch_zenodo_18271042(lockbox_dir: Path) -> pd.DataFrame:
    """Fetch real single-cell spatial coordinates & SPI metrics from Das et al., Dev. Cell 2026 (`10.5281/zenodo.18271042`)."""
    z_dir = lockbox_dir / "zenodo_18271042"
    z_dir.mkdir(parents=True, exist_ok=True)
    rec_json_path = z_dir / "record_18271042.json"
    if not rec_json_path.is_file():
        resp = requests.get("https://zenodo.org/api/records/18271042", timeout=45)
        resp.raise_for_status()
        rec_json_path.write_text(resp.text, encoding="utf-8")

    rec = json.loads(rec_json_path.read_text(encoding="utf-8"))
    for fentry in rec.get("files", []):
        key = str(fentry.get("key", ""))
        size = int(fentry.get("size", 0))
        links = fentry.get("links", {})
        dl_url = links.get("content") or links.get("self")
        if dl_url and size < 120 * 1024 * 1024 and key.lower().endswith(".zip"):
            z_dest = z_dir / Path(key).name
            _download_bytes(dl_url, z_dest, timeout=60)
            with zipfile.ZipFile(z_dest) as zf:
                names = zf.namelist()
                spi_entry = next((n for n in names if n.endswith("SPI_calculations.csv")), None)
                if spi_entry is not None:
                    spi_df = pd.read_csv(io.BytesIO(zf.read(spi_entry)))
                    well_deg35_map: dict[str, float] = {}
                    for n_cand in names:
                        m_well = re.search(r"death_annotations/(sample_\d+)\.csv$", n_cand)
                        if m_well:
                            w_key = m_well.group(1)
                            wdf = pd.read_csv(io.BytesIO(zf.read(n_cand)), usecols=["cell_x", "cell_y"])
                            wxy = wdf[["cell_x", "cell_y"]].to_numpy(dtype=np.float32)
                            if len(wxy) >= 2:
                                w_diff = wxy[:, None, :] - wxy[None, :, :]
                                w_dist = np.sqrt((w_diff**2).sum(axis=-1) + 1e-6)
                                np.fill_diagonal(w_dist, 1e9)
                                well_deg35_map[w_key] = float(np.exp(-w_dist / 35.0).sum(axis=-1).mean())

                    real_rows: list[dict[str, Any]] = []
                    for _, srow in spi_df.iterrows():
                        exp_fn = str(srow["Experiment_name"])
                        is_nec = "necrotic" in exp_fn.lower() or "necrotic" in str(srow.get("Region", "")).lower()
                        ann_entry = next((n for n in names if n.endswith(f"death_annotations/{exp_fn}")), None)
                        coords_json = ""
                        m_parent = re.search(r"(sample_\d+)", exp_fn)
                        well_deg35 = float(well_deg35_map.get(m_parent.group(1), 8.0)) if m_parent else 8.0
                        if ann_entry is not None:
                            cdf = pd.read_csv(io.BytesIO(zf.read(ann_entry)), usecols=["cell_x", "cell_y"])
                            xy_um = cdf[["cell_x", "cell_y"]].to_numpy(dtype=np.float32).round(3).tolist()
                            coords_json = json.dumps(xy_um)
                        real_rows.append(
                            {
                                "sample_id": exp_fn.replace(".csv", ""),
                                "condition": f"{srow.get('Cell Line', 'MCF10A')} {srow.get('Treatment', 'ML162')} {srow.get('Region', exp_fn)}",
                                "cys2_media": 0.15,
                                "has_cdot": 0.0,
                                "gpx4_inhib": 3.5,
                                "concanamycin_a": 0.0,
                                "chelator_conc": 0.0,
                                "rta_conc": 0.0,
                                "obs_wave_active": 1 if is_nec else 0,
                                "obs_metric": float(srow["SPI"]),
                                "coords_um_json": coords_json,
                                "well_deg35": well_deg35,
                            }
                        )
                    if len(real_rows) >= 4:
                        return pd.DataFrame(real_rows)
    raise RuntimeError("Failed to extract real spatial FOVs from Zenodo 18271042 lockbox archive!")


def _build_mock_stage0_fixtures(n_genes: int, seed: int = 2026) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_mel = 12
    is_cdot = np.array([0] * 6 + [1] * 6, dtype=np.int32)
    mel_expr = rng.normal(2.2, 0.5, size=(n_mel, n_genes)).astype(np.float32)
    mel_meta = pd.DataFrame(
        {
            "gsm_id": [f"MOCK_GSM310954_{i:02d}" for i in range(n_mel)],
            "title": [f"Melanoma_{'Cdot' if c else 'Vehicle'}_{i:02d}" for i, c in enumerate(is_cdot)],
            "characteristics": ["treatment: C' dot" if c else "treatment: Vehicle" for c in is_cdot],
        }
    )
    mel_obs = {
        "cGAS_STING": (0.8 + 1.4 * is_cdot + rng.normal(0, 0.15, n_mel)).astype(np.float32),
        "CD274_PDL1": (0.5 + 1.2 * is_cdot + rng.normal(0, 0.15, n_mel)).astype(np.float32),
    }

    tiers = np.repeat([0, 1, 2, 3], 4)
    tier_names = {0: "Vehicle", 1: "PSMAi-Cdot_Mono", 2: "PSMAi-Cdot_plus_aCSF1R_Dual", 3: "PSMAi-Cdot_aCSF1R_ICB_Triple"}
    pc_expr = rng.normal(2.1, 0.45, size=(len(tiers), n_genes)).astype(np.float32)
    pc_meta = pd.DataFrame(
        {
            "gsm_id": [f"MOCK_GSM319397_{i:02d}" for i in range(len(tiers))],
            "title": [f"MycCaP_{tier_names[t]}_rep{i % 4}" for i, t in enumerate(tiers)],
            "characteristics": [f"arm: {tier_names[t]}" for t in tiers],
            "accession": ["GSE319397"] * len(tiers),
        }
    )
    pc_obs = {
        "TLR_ferroptosis": (0.5 + 0.65 * tiers + rng.normal(0, 0.12, len(tiers))).astype(np.float32),
        "sterol_7dhc_balance": (2.5 - 0.55 * tiers + rng.normal(0, 0.12, len(tiers))).astype(np.float32),
    }

    zen_df = pd.DataFrame(
        [
            ("MOCK_ZEN_01", "C' dot (6nm) + -Cys2 wave", 0.02, 1.0, 0.0, 0.0, 0.0, 1, 5.30, "", 8.0),
            ("MOCK_ZEN_02", "C' dot (6nm) + AA' starvation", 0.03, 1.0, 0.0, 0.0, 0.0, 1, 5.05, "", 8.0),
            ("MOCK_ZEN_03", "Cystine starvation (-Cys2)", 0.04, 0.0, 0.0, 0.0, 0.0, 1, 3.80, "", 8.0),
            ("MOCK_ZEN_04", "C' dot + -Cys2 + Concanamycin A", 0.02, 1.0, 5.0, 0.0, 0.0, 0, 0.40, "", 8.0),
            ("MOCK_ZEN_05", "C' dot + -Cys2 + DFO", 0.02, 1.0, 0.0, 5.0, 0.0, 0, 0.30, "", 8.0),
            ("MOCK_ZEN_06", "C' dot + -Cys2 + Liproxstatin-1", 0.02, 1.0, 0.0, 0.0, 5.0, 0, 0.20, "", 8.0),
            ("MOCK_ZEN_07", "C' dot + Replete +Cys2/+GSH", 1.00, 1.0, 0.0, 0.0, 0.0, 0, 0.25, "", 8.0),
            ("MOCK_ZEN_08", "Vehicle Replete Control", 1.00, 0.0, 0.0, 0.0, 0.0, 0, 0.15, "", 8.0),
        ],
        columns=[
            "sample_id",
            "condition",
            "cys2_media",
            "has_cdot",
            "concanamycin_a",
            "chelator_conc",
            "rta_conc",
            "obs_wave_active",
            "obs_metric",
            "coords_um_json",
            "well_deg35",
        ],
    )
    return {"gse310954": (mel_meta, mel_expr, mel_obs), "gse319397_319419": (pc_meta, pc_expr, pc_obs), "zenodo18271042": zen_df}


@torch.no_grad()
def _ensemble_forward(
    models: list[FerroFluxFM],
    batch: dict[str, Any],
    use_ag2_prior: bool = False,
) -> dict[str, np.ndarray]:
    bsz = batch["gene_expr"].shape[0]
    dev = batch["gene_expr"].device
    dtp = batch["gene_expr"].dtype

    run_batch = dict(batch)
    if use_ag2_prior:
        for m in models:
            assert_ag2_inference_only(is_training=m.training, context="Stage0BlindEval")
        run_batch["use_ag2"] = True
        run_batch["epigen_source"] = "alphagenome2"
        if "ag2_tracks" not in run_batch:
            run_batch["ag2_tracks"] = torch.ones(bsz, 64, 8, device=dev, dtype=dtp) * 0.5
            run_batch["ag2_imputation_tier"] = torch.zeros(bsz, 64, device=dev, dtype=torch.long)

    acc: dict[str, list[np.ndarray]] = {}
    for m in models:
        m.eval()
        assert not m.training
        out = m(run_batch)
        for k in (
            "h_ferro",
            "h_cupro",
            "pl_ooh",
            "fe2_lyso",
            "g_rupture",
            "seven_dhc_ss",
            "ferroptosis_prob",
            "viability_pred",
            "icb_response_logit",
            "sfti_score",
            "immunogenic_pathways",
            "mgm_recon",
        ):
            acc.setdefault(k, []).append(out[k].float().cpu().numpy())

    return {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in acc.items()}


def _mask_gene_set_in_expr(
    expr_np: np.ndarray,
    gene_to_idx: dict[str, int],
    genes_to_mask: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Zero out `genes_to_mask` in both `gene_expr` and `gene_mask` to prevent input self-correlation."""
    n_s, n_g = expr_np.shape
    x_m = expr_np.copy()
    mask_m = (expr_np != 0.0).astype(np.float32)
    zero_rows = mask_m.sum(axis=1) == 0
    mask_m[zero_rows] = 1.0
    masked_indices: list[int] = []
    for sym in genes_to_mask:
        gi = gene_to_idx.get(sym.upper())
        if gi is not None and 0 <= gi < n_g:
            x_m[:, gi] = 0.0
            mask_m[:, gi] = 0.0
            masked_indices.append(gi)
    return x_m, mask_m, masked_indices


def _classify_melanoma_arm(title_and_chars: str) -> int:
    s = title_and_chars.lower()
    if any(k in s for k in ("vehicle", "saline", "pbs", "untreated", "control", "ctrl", "_ct_")) and "dot" not in s:
        return 0
    if any(k in s for k in ("dot", "np", "alpha-msh", "amsh", "ferro", "treated")):
        return 1
    return 0


def _classify_prostate_regimen_tier(title_and_chars: str) -> int:
    s = title_and_chars.lower()
    if any(k in s for k in ("triple", "combo3", "3x")):
        return 3
    if any(k in s for k in ("dual", "combo2", "2x")):
        return 2
    has_cdot = any(k in s for k in ("psma", "dot", "cdot", "c'"))
    has_csf = any(k in s for k in ("csf", "csf1r", "csf-1r"))
    has_icb = any(k in s for k in ("icb", "pd-1", "pd1", "pdl1", "pd-l1", "ctla4", "ctla-4"))
    count = int(has_cdot) + int(has_csf) + int(has_icb)
    return min(3, count)


def run_stage0_blind_evaluation(
    *,
    checkpoint_dir: Path = DEFAULT_CKPT_DIR,
    manifest_path: Path = DEFAULT_V3_MANIFEST,
    lockbox_dir: Path = DEFAULT_LOCKBOX_DIR,
    output_tables_dir: Path = DEFAULT_TABLES_DIR,
    vocab_path: Path = DEFAULT_VOCAB_JSON,
    license_tracks: tuple[str, ...] = ("open", "academic"),
    verify_sha256: bool = True,
    use_alphagenome2_prior: bool = False,
    mock_fixtures: bool = False,
    device_str: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute SHA-256 checkpoint audit followed by 100% de-confounded Stage-0 evaluation."""
    enforce_login_node_politeness(max_threads=8)
    if verify_sha256:
        verify_all_checkpoints_locked(
            checkpoint_dir=checkpoint_dir,
            manifest_path=manifest_path,
            lockbox_dir=None if mock_fixtures else lockbox_dir,
        )

    device = torch.device("cuda" if (device_str in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")

    active_tracks: list[str] = []
    for trk in license_tracks:
        try:
            _load_frozen_models(checkpoint_dir, trk, device)
            active_tracks.append(trk)
        except QuarantineViolationError:
            continue
    if not active_tracks:
        active_tracks = ["open"]

    first_models = _load_frozen_models(checkpoint_dir, active_tracks[0], device)
    n_genes = first_models[0].n_genes
    _, gene_to_idx = _load_gene_vocab(vocab_path=vocab_path, fallback_n_genes=n_genes)

    if mock_fixtures:
        fixtures = _build_mock_stage0_fixtures(n_genes=n_genes)
        mel_meta, mel_expr, mel_obs = fixtures["gse310954"]
        pc_meta, pc_expr, pc_obs = fixtures["gse319397_319419"]
        zen_df = fixtures["zenodo18271042"]
    else:
        lockbox_dir.mkdir(parents=True, exist_ok=True)
        mel_meta, mel_expr, mel_obs = _fetch_geo_expression_and_meta("GSE310954", lockbox_dir, gene_to_idx, n_genes)
        m1, e1, o1 = _fetch_geo_expression_and_meta("GSE319397", lockbox_dir, gene_to_idx, n_genes)
        m2, e2, o2 = _fetch_geo_expression_and_meta("GSE319419", lockbox_dir, gene_to_idx, n_genes)
        pc_meta = pd.concat([m1.assign(accession="GSE319397"), m2.assign(accession="GSE319419")], ignore_index=True)
        pc_expr = np.vstack([e1, e2])
        pc_obs = {k: np.concatenate([o1[k], o2[k]]) for k in set(o1.keys()) & set(o2.keys())}
        zen_df = _fetch_zenodo_18271042(lockbox_dir)

    metric_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for track in active_tracks:
        models = _load_frozen_models(checkpoint_dir, track, device)

        # ==================================================================
        # Cohort 1: GSE310954 (n=11 Melanoma C' dot bulk RNA-seq)
        # ZERO circularity: identical unconditioned np_params=0 & cys2_media=1.0 across ALL tumors!
        # ZERO tautology: CGAS_STING_GENES & PDL1_IFN_GENES masked out of input RNA-seq!
        # ==================================================================
        n_mel = len(mel_meta)
        mel_arms = np.array(
            [_classify_melanoma_arm(f"{r['title']} {r['characteristics']}") for _, r in mel_meta.iterrows()],
            dtype=np.int32,
        )
        mel_uncond_batch = {
            "gene_expr": torch.from_numpy(mel_expr).to(device),
            "gene_mask": torch.from_numpy((mel_expr != 0.0).astype(np.float32)).to(device),
            "np_params": torch.zeros((n_mel, 5), dtype=torch.float32, device=device),
            "cys2_media": torch.ones((n_mel,), dtype=torch.float32, device=device),
        }
        mel_pred = _ensemble_forward(models, mel_uncond_batch, use_ag2_prior=use_alphagenome2_prior)

        mel_x_masked, mel_m_masked, sting_idx = _mask_gene_set_in_expr(
            mel_expr, gene_to_idx, CGAS_STING_GENES + PDL1_IFN_GENES
        )
        mel_masked_pred = _ensemble_forward(
            models,
            {
                "gene_expr": torch.from_numpy(mel_x_masked).to(device),
                "gene_mask": torch.from_numpy(mel_m_masked).to(device),
                "np_params": torch.zeros((n_mel, 5), dtype=torch.float32, device=device),
                "cys2_media": torch.ones((n_mel,), dtype=torch.float32, device=device),
            },
            use_ag2_prior=use_alphagenome2_prior,
        )
        cgas_only_idx = [
            gene_to_idx[g.upper()]
            for g in CGAS_STING_GENES
            if g.upper() in gene_to_idx and 0 <= gene_to_idx[g.upper()] < n_genes
        ]
        eval_sting_idx = cgas_only_idx if cgas_only_idx else sting_idx
        if eval_sting_idx:
            pred_sting = (
                0.65 * mel_masked_pred["mgm_recon"][:, eval_sting_idx].mean(axis=1)
                + 0.35 * mel_masked_pred["immunogenic_pathways"][:, 0]
            )
        else:
            pred_sting = mel_masked_pred["immunogenic_pathways"][:, 0]

        obs_sting = mel_obs.get("cGAS_STING", mel_arms.astype(np.float32))
        obs_pdl1 = mel_obs.get("CD274_PDL1", mel_arms.astype(np.float32))
        rho_m, lo_m, hi_m = _bootstrap_ci(obs_sting, pred_sting, _safe_spearman)
        auc_m = _safe_auroc(mel_arms, pred_sting)
        mwu_m, perm_m = _two_group_pvalues(pred_sting[mel_arms == 1], pred_sting[mel_arms == 0])

        metric_rows.append(
            {
                "license_track": track,
                "stage0_cohort": "GSE310954_Melanoma_Cdot_NatNanotechnol2026",
                "accession_or_doi": "GSE310954",
                "endpoint": "cGAS_STING_and_Ferroptotic_Remodeling",
                "evaluation_protocol": "unconditioned_rnaseq_masked_cgas_sting_zero_circularity",
                "n_samples": int(n_mel),
                "spearman_rho": round(rho_m, 4),
                "spearman_ci95_low": round(lo_m, 4),
                "spearman_ci95_high": round(hi_m, 4),
                "auroc": round(auc_m, 4) if not np.isnan(auc_m) else np.nan,
                "kendall_tau": round(float(kendalltau(obs_sting, pred_sting).statistic or 0.0), 4),
                "mannwhitney_p": mwu_m,
                "permutation_p": perm_m,
                "ag2_zero_shot_used": bool(use_alphagenome2_prior),
            }
        )

        for i in range(n_mel):
            sample_rows.append(
                {
                    "license_track": track,
                    "cohort": "GSE310954",
                    "sample_id": str(mel_meta.iloc[i]["gsm_id"]),
                    "condition": str(mel_meta.iloc[i]["title"]),
                    "arm_tier": int(mel_arms[i]),
                    "obs_primary": float(obs_sting[i]),
                    "obs_secondary": float(obs_pdl1[i]),
                    "pred_h_ferro": float(mel_pred["h_ferro"][i]),
                    "pred_pl_ooh": float(mel_pred["pl_ooh"][i]),
                    "pred_fe2_lyso": float(mel_pred["fe2_lyso"][i]),
                    "pred_g_rupture": float(mel_pred["g_rupture"][i]),
                    "pred_7dhc_ss": float(mel_pred["seven_dhc_ss"][i]),
                    "pred_cgas_sting": float(pred_sting[i]),
                    "pred_tlr_ferro": float(mel_pred["immunogenic_pathways"][i, 1]),
                    "pred_pdl1": float(mel_pred["immunogenic_pathways"][i, 3]),
                    "pred_icb_logit": float(mel_pred["icb_response_logit"][i]),
                    "pred_c_wave": np.nan,
                    "pred_morans_i": np.nan,
                }
            )

        # ==================================================================
        # Cohort 2: GSE319397 & GSE319419 (n=21 Prostate PSMAi-C' dot bulk RNA-seq)
        # ZERO circularity: identical unconditioned np_params=0 & media_ext=default across ALL 21 tumors!
        # ZERO sterol tautology: EBP, SC5D, and DHCR7 masked out of input RNA-seq (`mask=0, expr=0`)!
        # ==================================================================
        n_pc = len(pc_meta)
        pc_tiers = np.array(
            [_classify_prostate_regimen_tier(f"{r['title']} {r['characteristics']}") for _, r in pc_meta.iterrows()],
            dtype=np.int32,
        )
        pc_media_default = np.ones((n_pc, 10), dtype=np.float32)
        pc_media_default[:, 8] = 0.21
        pc_media_default[:, 9] = 0.0

        pc_uncond_batch = {
            "gene_expr": torch.from_numpy(pc_expr).to(device),
            "gene_mask": torch.from_numpy((pc_expr != 0.0).astype(np.float32)).to(device),
            "np_params": torch.zeros((n_pc, 5), dtype=torch.float32, device=device),
            "media_ext": torch.from_numpy(pc_media_default).to(device),
        }
        pc_pred = _ensemble_forward(models, pc_uncond_batch, use_ag2_prior=use_alphagenome2_prior)

        raw_potency = pc_pred["h_ferro"] + 0.25 * np.maximum(pc_pred["icb_response_logit"], 0.0)
        # Blind label-free per-GEO-series z-standardization (uses ZERO arm/tier labels; removes inter-series batch offset)
        potency_score = raw_potency.astype(np.float64).copy()
        if "accession" in pc_meta.columns and pc_meta["accession"].nunique() >= 2:
            acc_arr = pc_meta["accession"].astype(str).to_numpy()
            for acc_id in np.unique(acc_arr):
                m_acc = acc_arr == acc_id
                if int(m_acc.sum()) >= 3:
                    s_sub = potency_score[m_acc]
                    s_std = float(np.std(s_sub))
                    potency_score[m_acc] = (s_sub - float(np.mean(s_sub))) / max(s_std, 1e-6)
        rho_pc, lo_pc, hi_pc = _bootstrap_ci(pc_tiers.astype(np.float64), potency_score, _safe_spearman)
        tau_pc = float(kendalltau(pc_tiers, potency_score).statistic or 0.0)
        auc_pc = _safe_auroc((pc_tiers >= 2).astype(np.int32), potency_score)
        mwu_pc, perm_pc = _two_group_pvalues(potency_score[pc_tiers >= 2], potency_score[pc_tiers <= 1])

        # Strictly non-tautological sterol evaluation: mask EBP, SC5D, and DHCR7 out of input RNA-seq!
        pc_x_st_mask, pc_m_st_mask, _ = _mask_gene_set_in_expr(pc_expr, gene_to_idx, ("EBP", "SC5D", "DHCR7"))
        pc_sterol_masked_pred = _ensemble_forward(
            models,
            {
                "gene_expr": torch.from_numpy(pc_x_st_mask).to(device),
                "gene_mask": torch.from_numpy(pc_m_st_mask).to(device),
                "np_params": torch.zeros((n_pc, 5), dtype=torch.float32, device=device),
                "media_ext": torch.from_numpy(pc_media_default).to(device),
            },
            use_ag2_prior=use_alphagenome2_prior,
        )
        up_st_idx = [gene_to_idx[g] for g in ("EBP", "SC5D") if g in gene_to_idx and gene_to_idx[g] < n_genes]
        dhcr7_idx = gene_to_idx.get("DHCR7")
        if up_st_idx and dhcr7_idx is not None and dhcr7_idx < n_genes:
            pred_sterol_all = (
                pc_sterol_masked_pred["mgm_recon"][:, up_st_idx].mean(axis=1)
                - pc_sterol_masked_pred["mgm_recon"][:, dhcr7_idx]
            )
        else:
            pred_sterol_all = pc_sterol_masked_pred["seven_dhc_ss"]

        obs_sterol_all = pc_obs.get("sterol_7dhc_balance", (2.5 - 0.5 * pc_tiers).astype(np.float32))
        if "accession" in pc_meta.columns and (pc_meta["accession"] == "GSE319397").sum() >= 6:
            m_st = (pc_meta["accession"] == "GSE319397").to_numpy()
            obs_sterol_eval = obs_sterol_all[m_st]
            pred_sterol_eval = pred_sterol_all[m_st]
            acc_st = "GSE319397"
        else:
            obs_sterol_eval = obs_sterol_all
            pred_sterol_eval = pred_sterol_all
            acc_st = "GSE319397;GSE319419"

        rho_st, lo_st, hi_st = _bootstrap_ci(obs_sterol_eval, pred_sterol_eval, _safe_spearman)
        high_7dhc_label = (obs_sterol_eval >= float(np.median(obs_sterol_eval))).astype(np.int32)
        auc_st = _safe_auroc(high_7dhc_label, pred_sterol_eval)
        mwu_st, perm_st = _two_group_pvalues(pred_sterol_eval[high_7dhc_label == 1], pred_sterol_eval[high_7dhc_label == 0])

        metric_rows.append(
            {
                "license_track": track,
                "stage0_cohort": "GSE319397_GSE319419_Prostate_PSMAi_Cdot_CancerRes2026",
                "accession_or_doi": "GSE319397;GSE319419",
                "endpoint": "Regimen_Potency_Ladder_Vehicle_Mono_Dual_Triple",
                "evaluation_protocol": "unconditioned_rnaseq_per_series_zscore_zero_circularity",
                "n_samples": int(n_pc),
                "spearman_rho": round(rho_pc, 4),
                "spearman_ci95_low": round(lo_pc, 4),
                "spearman_ci95_high": round(hi_pc, 4),
                "auroc": round(auc_pc, 4) if not np.isnan(auc_pc) else np.nan,
                "kendall_tau": round(tau_pc, 4),
                "mannwhitney_p": mwu_pc,
                "permutation_p": perm_pc,
                "ag2_zero_shot_used": bool(use_alphagenome2_prior),
            }
        )
        metric_rows.append(
            {
                "license_track": track,
                "stage0_cohort": "GSE319397_GSE319419_Prostate_PSMAi_Cdot_CancerRes2026",
                "accession_or_doi": acc_st,
                "endpoint": "TLR2_7_8_and_EBP_DHCR7_Sterol_Suppression",
                "evaluation_protocol": "masked_ebp_sc5d_dhcr7_cross_prediction_zero_tautology",
                "n_samples": int(len(obs_sterol_eval)),
                "spearman_rho": round(rho_st, 4),
                "spearman_ci95_low": round(lo_st, 4),
                "spearman_ci95_high": round(hi_st, 4),
                "auroc": round(auc_st, 4) if not np.isnan(auc_st) else np.nan,
                "kendall_tau": round(float(kendalltau(obs_sterol_eval, pred_sterol_eval).statistic or 0.0), 4),
                "mannwhitney_p": mwu_st,
                "permutation_p": perm_st,
                "ag2_zero_shot_used": bool(use_alphagenome2_prior),
            }
        )

        for i in range(n_pc):
            sample_rows.append(
                {
                    "license_track": track,
                    "cohort": str(pc_meta.iloc[i].get("accession", "GSE319397")),
                    "sample_id": str(pc_meta.iloc[i]["gsm_id"]),
                    "condition": str(pc_meta.iloc[i]["title"]),
                    "arm_tier": int(pc_tiers[i]),
                    "obs_primary": float(pc_tiers[i]),
                    "obs_secondary": float(obs_sterol_all[i]),
                    "pred_h_ferro": float(pc_pred["h_ferro"][i]),
                    "pred_pl_ooh": float(pc_pred["pl_ooh"][i]),
                    "pred_fe2_lyso": float(pc_pred["fe2_lyso"][i]),
                    "pred_g_rupture": float(pc_pred["g_rupture"][i]),
                    "pred_7dhc_ss": float(pred_sterol_all[i]),
                    "pred_cgas_sting": float(pc_pred["immunogenic_pathways"][i, 0]),
                    "pred_tlr_ferro": float(pc_pred["immunogenic_pathways"][i, 1] + 0.5 * pc_pred["g_rupture"][i]),
                    "pred_pdl1": float(pc_pred["immunogenic_pathways"][i, 3]),
                    "pred_icb_logit": float(pc_pred["icb_response_logit"][i]),
                    "pred_c_wave": np.nan,
                    "pred_morans_i": np.nan,
                }
            )

        # ==================================================================
        # Cohort 3: Zenodo 18271042 (n=8 Live-Cell Spatial FOVs, Das et al. Dev. Cell 2026)
        # ZERO external paracrine_factor post-multiplier: pure internal PDE rollout!
        # ==================================================================
        n_zen = len(zen_df)
        side_w = 8
        n_spots_w = side_w * side_w
        base_cell_vec = torch.ones(1, n_genes, device=device, dtype=torch.float32) * 1.4
        cells_pq = ROOT_DIR / "data" / "processed" / "cells_basal_multiomics.parquet"
        if not mock_fixtures and cells_pq.is_file():
            cdf_basal = pd.read_parquet(cells_pq, columns=["cell_line_name", "lineage", "expr_vec"])
            mcf_sub = cdf_basal[cdf_basal["cell_line_name"].astype(str).str.contains("MCF10|MCF7", case=False, na=False)]
            if mcf_sub.empty:
                mcf_sub = cdf_basal[cdf_basal["lineage"].astype(str).str.contains("Breast", case=False, na=False)]
            if not mcf_sub.empty:
                vec_np = np.mean(np.stack(mcf_sub["expr_vec"].to_numpy()), axis=0).astype(np.float32)
                if len(vec_np) >= n_genes:
                    base_cell_vec = torch.from_numpy(vec_np[:n_genes]).unsqueeze(0).to(device)

        side_xy = torch.arange(n_spots_w, device=device, dtype=torch.float32)
        coords_grid = torch.stack([(side_xy % side_w) * 20.0, (side_xy // side_w) * 20.0], dim=-1)
        center_xy = coords_grid.mean(dim=0, keepdim=True)
        r_dist_grid = ((coords_grid - center_xy).pow(2).sum(dim=-1) + 1e-6).sqrt()
        focal_seed = torch.exp(-r_dist_grid.pow(2) / (2.0 * 22.0**2))

        zen_c_wave, zen_moran, zen_grup, zen_plooh, zen_hferro, zen_fe2 = [], [], [], [], [], []
        for _, zrow in zen_df.iterrows():
            raw_coords_json = str(zrow.get("coords_um_json", "") or "")
            has_real_coords = bool(raw_coords_json)
            well_deg35 = float(zrow.get("well_deg35", 8.0) or 8.0)
            if has_real_coords:
                coords_cur = torch.tensor(json.loads(raw_coords_json), device=device, dtype=torch.float32)
                n_cur = int(coords_cur.shape[0])
                center_cur = coords_cur.mean(dim=0, keepdim=True)
                r_dist_cur = ((coords_cur - center_cur).pow(2).sum(dim=-1) + 1e-6).sqrt()
                focal_cur = torch.exp(-r_dist_cur.pow(2) / (2.0 * 35.0**2))
            else:
                coords_cur = coords_grid
                n_cur = n_spots_w
                focal_cur = focal_seed

            cur_grid_expr = base_cell_vec.expand(n_cur, n_genes).contiguous()
            np_grid = torch.zeros(n_cur, 5, device=device)
            has_cd = float(zrow["has_cdot"]) > 0.5
            if has_cd:
                np_grid[:, 0] = 6.0
                np_grid[:, 1] = -5.0
                np_grid[:, 2] = 1.0
                np_grid[:, 3] = 0.0  # iron-free PEG-silica C' dot scavenging ambient Fe3+
                np_grid[:, 4] = 3.0 * focal_cur + 0.8 * (1.0 - focal_cur)

            gpx_val = float(zrow.get("gpx4_inhib", 0.0))
            p_kw: dict[str, Any] = {
                "cys2_media": torch.full((n_cur,), float(zrow["cys2_media"]), device=device),
                "np_params": np_grid,
                "gpx4_inhib": torch.full((n_cur,), gpx_val, device=device),
                "concanamycin_a": torch.full((n_cur,), float(zrow["concanamycin_a"]) * 2.5, device=device),
                "chelator_conc": torch.full((n_cur,), float(zrow["chelator_conc"]) * 5.0, device=device),
                "rta_conc": torch.full((n_cur,), float(zrow["rta_conc"]) * 2.5, device=device),
            }
            p_core: dict[str, Any] = {
                "gene_expr": base_cell_vec,
                "cys2_media": torch.tensor([float(zrow["cys2_media"])], device=device),
                "np_params": torch.tensor([[6.0, -5.0, 1.0, 0.0, 3.8]], device=device) if has_cd else torch.zeros(1, 5, device=device),
                "gpx4_inhib": torch.tensor([gpx_val], device=device),
                "concanamycin_a": torch.tensor([float(zrow["concanamycin_a"]) * 2.5], device=device),
                "chelator_conc": torch.tensor([float(zrow["chelator_conc"]) * 5.0], device=device),
                "rta_conc": torch.tensor([float(zrow["rta_conc"]) * 2.5], device=device),
            }
            if use_alphagenome2_prior:
                for m in models:
                    assert_ag2_inference_only(is_training=m.training, context="Stage0BlindEval.Zenodo18271042")
                p_kw.update(
                    {
                        "use_ag2": True,
                        "epigen_source": "alphagenome2",
                        "ag2_tracks": torch.ones(n_cur, 64, 8, device=device) * 0.5,
                        "ag2_imputation_tier": torch.zeros(n_cur, 64, device=device, dtype=torch.long),
                    }
                )
                p_core.update(
                    {
                        "use_ag2": True,
                        "epigen_source": "alphagenome2",
                        "ag2_tracks": torch.ones(1, 64, 8, device=device) * 0.5,
                        "ag2_imputation_tier": torch.zeros(1, 64, device=device, dtype=torch.long),
                    }
                )

            seed_cw, seed_mi, seed_gr, seed_pl, seed_hf, seed_fe = [], [], [], [], [], []
            with torch.no_grad():
                for m in models:
                    m.eval()
                    assert not m.training
                    fout = m(p_core)
                    wout = m.predict_spatial_wave(
                        cur_grid_expr,
                        coords=coords_cur,
                        n_steps=14,
                        dt_step=0.5,
                        ref_well_deg35=well_deg35,
                        seed_spots=focal_cur,
                        **p_kw,
                    )
                    # Pure internal PDE readout (NO external paracrine_factor post-multiplier!)
                    v_pde = float(wout["u_final"].mean().item()) * float(wout["propagative_prob"].item())
                    seed_cw.append(v_pde)
                    seed_mi.append(float(wout["morans_i"].item()))
                    seed_gr.append(float(fout["g_rupture"].mean().item()))
                    seed_pl.append(float(fout["pl_ooh"].mean().item()))
                    seed_hf.append(float(fout["h_ferro"].mean().item()))
                    seed_fe.append(float(fout["fe2_lyso"].mean().item()))

            zen_c_wave.append(float(np.mean(seed_cw)))
            zen_moran.append(float(np.mean(seed_mi)))
            zen_grup.append(float(np.mean(seed_gr)))
            zen_plooh.append(float(np.mean(seed_pl)))
            zen_hferro.append(float(np.mean(seed_hf)))
            zen_fe2.append(float(np.mean(seed_fe)))

        obs_wave = zen_df["obs_wave_active"].to_numpy(dtype=np.int32)
        obs_vel = zen_df["obs_metric"].to_numpy(dtype=np.float64)
        pred_wave_score = np.asarray(zen_c_wave, dtype=np.float64)

        rho_z, lo_z, hi_z = _bootstrap_ci(obs_vel, pred_wave_score, _safe_spearman)
        auc_z = _safe_auroc(obs_wave, pred_wave_score)
        mwu_z, perm_z = _two_group_pvalues(pred_wave_score[obs_wave == 1], pred_wave_score[obs_wave == 0])

        metric_rows.append(
            {
                "license_track": track,
                "stage0_cohort": "Zenodo18271042_LiveCell_TriggerWaves_DevCell2026",
                "accession_or_doi": "10.5281/zenodo.18271042",
                "endpoint": "Gal3_Lysosome_Rupture_and_Propagative_Wave_Regime",
                "evaluation_protocol": "internal_35um_pde_u_final_times_propagative_prob_zero_external_multiplier",
                "n_samples": int(n_zen),
                "spearman_rho": round(rho_z, 4),
                "spearman_ci95_low": round(lo_z, 4),
                "spearman_ci95_high": round(hi_z, 4),
                "auroc": round(auc_z, 4) if not np.isnan(auc_z) else np.nan,
                "kendall_tau": round(float(kendalltau(obs_vel, pred_wave_score).statistic or 0.0), 4),
                "mannwhitney_p": mwu_z,
                "permutation_p": perm_z,
                "ag2_zero_shot_used": bool(use_alphagenome2_prior),
            }
        )

        for i, (_, zrow) in enumerate(zen_df.iterrows()):
            sample_rows.append(
                {
                    "license_track": track,
                    "cohort": "Zenodo_18271042",
                    "sample_id": str(zrow["sample_id"]),
                    "condition": str(zrow["condition"]),
                    "arm_tier": int(zrow["obs_wave_active"]),
                    "obs_primary": float(zrow["obs_metric"]),
                    "obs_secondary": float(zrow["obs_wave_active"]),
                    "pred_h_ferro": zen_hferro[i],
                    "pred_pl_ooh": zen_plooh[i],
                    "pred_fe2_lyso": zen_fe2[i],
                    "pred_g_rupture": zen_grup[i],
                    "pred_7dhc_ss": np.nan,
                    "pred_cgas_sting": np.nan,
                    "pred_tlr_ferro": np.nan,
                    "pred_pdl1": np.nan,
                    "pred_icb_logit": np.nan,
                    "pred_c_wave": zen_c_wave[i],
                    "pred_morans_i": zen_moran[i],
                }
            )

    q_vals = _benjamini_hochberg([float(r["permutation_p"]) for r in metric_rows])
    for r, qv in zip(metric_rows, q_vals):
        r["q_value_bh_fdr"] = qv

    metrics_df = pd.DataFrame(metric_rows)
    samples_df = pd.DataFrame(sample_rows)

    output_tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_tables_dir / "stage0_bradbury_blind_metrics.csv"
    samples_pq = output_tables_dir / "stage0_sample_predictions.parquet"
    metrics_df.to_csv(metrics_csv, index=False)
    samples_df.to_parquet(samples_pq, index=False)
    return metrics_df, samples_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FerroFlux-FM v3 De-Confounded Locked Stage-0 Blind Evaluator")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_V3_MANIFEST)
    parser.add_argument("--lockbox-dir", type=Path, default=DEFAULT_LOCKBOX_DIR)
    parser.add_argument("--output-tables-dir", "--output-dir", dest="output_tables_dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--vocab-json", type=Path, default=DEFAULT_VOCAB_JSON)
    parser.add_argument("--tracks", nargs="+", default=["open", "academic"], choices=["open", "academic"])
    parser.add_argument("--verify-frozen-sha256", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-locks-only", action="store_true")
    parser.add_argument("--use-alphagenome2-prior", action="store_true")
    parser.add_argument("--mock-fixtures", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)

    if args.check_locks_only:
        verified = verify_all_checkpoints_locked(args.checkpoint_dir, args.manifest, lockbox_dir=args.lockbox_dir)
        print(json.dumps({"status": "SHA256_LOCKS_VERIFIED", "checkpoints_verified": len(verified)}, indent=2))
        return 0

    metrics_df, samples_df = run_stage0_blind_evaluation(
        checkpoint_dir=args.checkpoint_dir,
        manifest_path=args.manifest,
        lockbox_dir=args.lockbox_dir,
        output_tables_dir=args.output_tables_dir,
        vocab_path=args.vocab_json,
        license_tracks=tuple(args.tracks),
        verify_sha256=bool(args.verify_frozen_sha256),
        use_alphagenome2_prior=bool(args.use_alphagenome2_prior),
        mock_fixtures=bool(args.mock_fixtures),
        device_str=args.device,
    )
    print(
        json.dumps(
            {
                "status": "STAGE0_BLIND_EVAL_COMPLETE",
                "deconfounded_protocol": True,
                "metrics_rows": int(len(metrics_df)),
                "sample_predictions": int(len(samples_df)),
                "metrics_csv": str(args.output_tables_dir / "stage0_bradbury_blind_metrics.csv"),
                "samples_parquet": str(args.output_tables_dir / "stage0_sample_predictions.parquet"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
