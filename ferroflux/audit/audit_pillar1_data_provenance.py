"""Pillar 1 Forensic Audit: Data Provenance, Mis-Joins, Truncation Bias, Vocab Alignment & Quarantine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ferroflux.audit.resource_guard import enforce_login_node_politeness, get_peak_rss_mb
from ferroflux.layers.perceiver_trunk import ANCHOR_28_PROTEINS
from ferroflux.layers.perturbation_ops import DEFAULT_ANCHOR_28_VOCAB_IDX, FDX1_VOCAB_IDX
from ferroflux.security.quarantine import load_quarantine_sets

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def run_pillar1_audit(output_dir: Path, max_threads: int = 6) -> dict[str, Any]:
    enforce_login_node_politeness(max_threads=max_threads)
    output_dir.mkdir(parents=True, exist_ok=True)

    findings: list[dict[str, Any]] = []
    shard_stats: list[dict[str, Any]] = []

    # 1. Load vocabulary & check hardcoded indices vs gene_vocab.json
    vocab_path = PROCESSED_DIR / "gene_vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    genes: list[str] = vocab["genes"]
    gene_to_idx: dict[str, int] = vocab["gene_to_idx"]
    vocab_anchor_28_idx: list[int] = vocab.get("anchor_proteins_28_indices", [])

    hardcoded_mismatches: list[dict[str, Any]] = []
    for pos, prot_sym in enumerate(ANCHOR_28_PROTEINS):
        true_idx = gene_to_idx.get(prot_sym, -1)
        hc_idx = int(DEFAULT_ANCHOR_28_VOCAB_IDX[pos])
        v_idx = int(vocab_anchor_28_idx[pos]) if pos < len(vocab_anchor_28_idx) else -1
        if hc_idx != true_idx or v_idx != true_idx:
            hardcoded_mismatches.append(
                {
                    "protein": prot_sym,
                    "true_vocab_idx": true_idx,
                    "hardcoded_idx": hc_idx,
                    "hardcoded_points_to_gene": genes[hc_idx] if 0 <= hc_idx < len(genes) else "OOB",
                    "vocab_json_anchor_idx": v_idx,
                }
            )
    true_fdx1_idx = gene_to_idx.get("FDX1", -1)
    if FDX1_VOCAB_IDX != true_fdx1_idx:
        hardcoded_mismatches.append(
            {
                "protein": "FDX1",
                "true_vocab_idx": true_fdx1_idx,
                "hardcoded_idx": FDX1_VOCAB_IDX,
                "hardcoded_points_to_gene": genes[FDX1_VOCAB_IDX] if 0 <= FDX1_VOCAB_IDX < len(genes) else "OOB",
            }
        )

    if hardcoded_mismatches:
        findings.append(
            {
                "id": "P1-VOCAB-IDX",
                "severity": "CRITICAL",
                "title": "Hardcoded anchor/FDX1 gene indices in perceiver_trunk.py & perturbation_ops.py mismatch gene_vocab.json",
                "details": hardcoded_mismatches,
            }
        )
    else:
        findings.append(
            {
                "id": "P1-VOCAB-IDX",
                "severity": "MINOR",
                "title": "Hardcoded DEFAULT_ANCHOR_28_VOCAB_IDX & FDX1_VOCAB_IDX match current gene_vocab.json, but fragile if vocab changes",
                "details": {"n_checked": 29, "mismatches": 0},
            }
        )

    # 2. Load basal cell IDs to test silent mid_fallback mis-joins across all shards
    df_basal = pq.read_table(PROCESSED_DIR / "cells_basal_multiomics.parquet").to_pandas()
    basal_mids = set(df_basal["model_id"].astype(str).str.strip())
    basal_first_mid = str(df_basal["model_id"].iloc[0]).strip()
    basal_names = {
        str(r.get("cell_line_name", "")).strip().upper(): str(r["model_id"]).strip()
        for _, r in df_basal.iterrows()
        if str(r.get("cell_line_name", "")).strip()
    }

    df_norm = pq.read_table(PROCESSED_DIR / "normal_organ_til_baselines.parquet").to_pandas()
    norm_ids = set(df_norm["record_id"].astype(str).str.strip())
    df_icb = pq.read_table(PROCESSED_DIR / "clinical_icb_cohorts.parquet").to_pandas()
    icb_ids = set(df_icb["record_id"].astype(str).str.strip())
    all_known_cell_ids = basal_mids | norm_ids | icb_ids

    # Check Stage-1 proteomics coverage vs mRNA proxy fallback in dataset.py lines 592-596
    df_prot = pq.read_table(PROCESSED_DIR / "physical_anchors_proteomics.parquet").to_pandas()
    prot_mids = set(df_prot["model_id"].astype(str).str.strip())
    basal_with_real_ms_prot = len(basal_mids & prot_mids)
    basal_using_mrna_proxy = len(basal_mids - prot_mids)
    findings.append(
        {
            "id": "P1-PROT-PROXY",
            "severity": "MAJOR" if basal_using_mrna_proxy > 0 else "VERIFIED-CLEAN",
            "title": "Stage-1 basal records without MS proteomics substitute raw mRNA expression as prot_anchor_target",
            "details": {
                "basal_models_total": len(basal_mids),
                "basal_with_real_ms_proteomics": basal_with_real_ms_prot,
                "basal_falling_back_to_mrna_proxy": basal_using_mrna_proxy,
                "frac_mrna_proxy": round(basal_using_mrna_proxy / max(1, len(basal_mids)), 4),
            },
        }
    )

    # 3. Scan every Parquet shard for provenance, duplicates, NaNs, license tracks, silent mid_fallback, and .iloc[:cap] bias
    excl_acc, excl_pmc = load_quarantine_sets()
    cap_map = {
        "crispr_chronos_2d_3d.parquet": 45000,
        "pharmacogenomic_viability.parquet": 45000,
        "perturb_transcriptomes.parquet": 35000,
        "drugcomb_synergy_blocks.parquet": 15000,
        "perturbed_oxylipidomics_sterolomics.parquet": 5000,
        "pmc_oa_rescue_and_bionano.parquet": 5000,
    }

    misjoin_summary: dict[str, Any] = {}
    truncation_summary: dict[str, Any] = {}

    for pq_path in sorted(PROCESSED_DIR.glob("*.parquet")):
        df = pq.read_table(pq_path).to_pandas()
        n_rows = len(df)
        cols = list(df.columns)

        # Scalar duplicate check
        scalar_cols = [c for c in cols if not isinstance(df[c].dropna().iloc[0] if len(df[c].dropna()) else 0, (list, np.ndarray))]
        n_exact_dup = int(df.duplicated(subset=scalar_cols).sum()) if scalar_cols else 0

        src_col = next((c for c in ("dataset_source", "source", "accession_id") if c in cols), None)
        src_counts = df[src_col].astype(str).value_counts().to_dict() if src_col else {}

        lic_col = next((c for c in ("license_track", "license", "data_license") if c in cols), None)
        lic_counts = df[lic_col].astype(str).value_counts().to_dict() if lic_col else {"unlabeled": n_rows}

        # Check quarantine string leakage
        quarantine_hits = 0
        for c in scalar_cols:
            if df[c].dtype == object:
                vals_u = df[c].dropna().astype(str).unique()
                for v in vals_u:
                    vu = v.upper()
                    if any(ex in vu for ex in excl_acc) or any(pm in vu for pm in excl_pmc):
                        quarantine_hits += 1

        # Check model_id match rate against all_known_cell_ids
        mid_match_rate = None
        unmatched_rows = 0
        recoverable_by_cell_name = 0
        if "model_id" in cols and pq_path.name not in (
            "cells_basal_multiomics.parquet",
            "physical_anchors_proteomics.parquet",
            "physical_anchors_metabolomics_lipidomics.parquet",
            "cis_reg_64loci_epigenomics.parquet",
        ):
            mids_s = df["model_id"].fillna("").astype(str).str.strip()
            matched_mask = mids_s.isin(all_known_cell_ids)
            unmatched_rows = int((~matched_mask).sum())
            mid_match_rate = round(float(matched_mask.mean()), 4)
            cname_col = next((c for c in ("cell_line_name", "cell_line", "model_name") if c in cols), None)
            if cname_col and unmatched_rows > 0:
                cnames_u = df.loc[~matched_mask, cname_col].fillna("").astype(str).str.strip().str.upper()
                recoverable_by_cell_name = int(cnames_u.isin(basal_names).sum())
            misjoin_summary[pq_path.name] = {
                "total_rows": n_rows,
                "matched_to_basal_expr": int(matched_mask.sum()),
                "unmatched_falling_back_to_first_cell": unmatched_rows,
                "fallback_cell_id": basal_first_mid,
                "match_rate": mid_match_rate,
                "recoverable_via_cell_line_name": recoverable_by_cell_name,
            }
        elif pq_path.name in ("drugcomb_synergy_blocks.parquet", "perturbed_oxylipidomics_sterolomics.parquet", "pmc_oa_rescue_and_bionano.parquet") and "model_id" not in cols:
            cname_col = next((c for c in ("cell_line_name", "cell_line", "sample_name") if c in cols), None)
            recov = 0
            if cname_col:
                recov = int(df[cname_col].fillna("").astype(str).str.strip().str.upper().isin(basal_names).sum())
            misjoin_summary[pq_path.name] = {
                "total_rows": n_rows,
                "matched_to_basal_expr": 0,
                "unmatched_falling_back_to_first_cell": n_rows,
                "fallback_cell_id": basal_first_mid,
                "match_rate": 0.0,
                "has_model_id_col": False,
                "recoverable_via_cell_line_name": recov,
            }

        # Check .iloc[:cap] truncation distortion
        if pq_path.name in cap_map and n_rows > cap_map[pq_path.name]:
            cap = cap_map[pq_path.name]
            head_df = df.iloc[:cap]
            full_srcs = df[src_col].astype(str).value_counts().to_dict() if src_col else {}
            head_srcs = head_df[src_col].astype(str).value_counts().to_dict() if src_col else {}
            dropped_sources = sorted(set(full_srcs.keys()) - set(head_srcs.keys()))
            truncation_summary[pq_path.name] = {
                "total_rows": n_rows,
                "cap": cap,
                "fraction_kept": round(cap / n_rows, 4),
                "full_source_counts": full_srcs,
                "kept_prefix_source_counts": head_srcs,
                "completely_dropped_sources": dropped_sources,
            }

        shard_stats.append(
            {
                "shard": pq_path.name,
                "rows": n_rows,
                "cols": len(cols),
                "size_mb": round(pq_path.stat().st_size / (1024 * 1024), 3),
                "exact_scalar_duplicates": n_exact_dup,
                "quarantine_hits": quarantine_hits,
                "license_counts": lic_counts,
                "top_sources": dict(list(src_counts.items())[:6]),
                "basal_model_id_match_rate": mid_match_rate,
            }
        )

    # Summarize mis-joins & truncations into severity-graded findings
    severe_misjoins = {k: v for k, v in misjoin_summary.items() if v["unmatched_falling_back_to_first_cell"] > 0}
    if severe_misjoins:
        findings.append(
            {
                "id": "P1-SILENT-MID-FALLBACK",
                "severity": "CRITICAL",
                "title": f"Silent fallback assigns unmatched rows in {len(severe_misjoins)} shards to {basal_first_mid} basal transcriptome",
                "details": severe_misjoins,
            }
        )

    if truncation_summary:
        dropped_any = any(len(v["completely_dropped_sources"]) > 0 for v in truncation_summary.values())
        findings.append(
            {
                "id": "P1-UNSHUFFLED-TRUNCATION",
                "severity": "CRITICAL" if dropped_any else "MAJOR",
                "title": "Unshuffled .iloc[:cap] prefix truncation drops entire downstream data sources sorted at the end of Parquet shards",
                "details": truncation_summary,
            }
        )

    # Check license purity of PRISM / GDSC / ChEMBL / DrugComb in 'open' track
    pharm_df = pq.read_table(PROCESSED_DIR / "pharmacogenomic_viability.parquet", columns=["dataset_source", "license_track"]).to_pandas()
    pharm_lic_by_src = pharm_df.groupby(["dataset_source", "license_track"]).size().unstack(fill_value=0).to_dict()
    findings.append(
        {
            "id": "P1-LICENSE-AUDIT",
            "severity": "MINOR",
            "title": "Pharmacogenomic source x license_track distribution audited (DepMap PRISM CC-BY-4.0, GDSC non-commercial terms noted)",
            "details": pharm_lic_by_src,
        }
    )

    pd.DataFrame(shard_stats).to_csv(output_dir / "pillar1_shard_forensics.csv", index=False)
    report = {
        "pillar": 1,
        "name": "Data Provenance, Mis-Joins, Truncation & Vocabulary Alignment",
        "total_shards": len(shard_stats),
        "total_rows_on_disk": int(sum(s["rows"] for s in shard_stats)),
        "quarantine_leaks_found": int(sum(s["quarantine_hits"] for s in shard_stats)),
        "findings": findings,
        "peak_rss_mb": get_peak_rss_mb(),
    }
    (output_dir / "pillar1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_pillar1_audit(ROOT_DIR / "results" / "audit")
    print(json.dumps({"pillar": 1, "n_findings": len(res["findings"]), "peak_rss_mb": res["peak_rss_mb"]}, indent=2))
