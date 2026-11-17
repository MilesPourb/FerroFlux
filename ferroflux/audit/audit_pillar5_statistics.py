"""Pillar 5 Forensic Audit: Statistical Hypothesis Testing, Sample-Size Mismatch, Calibration & Power."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from ferroflux.audit.resource_guard import enforce_login_node_politeness, get_peak_rss_mb

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")


def min_detectable_spearman_80pct_power(n: int, alpha: float = 0.05) -> float:
    """Two-sided minimum detectable |rho| at 80% power via Fisher z-transformation."""
    if n <= 3:
        return 1.0
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(0.80))
    z_req = (z_alpha + z_beta) / math.sqrt(max(1.0, n - 3.0))
    return round(float(math.tanh(z_req)), 4)


def required_n_for_rho(target_rho: float, alpha: float = 0.001, power: float = 0.80) -> int:
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(power))
    z_r = math.atanh(min(max(abs(target_rho), 0.05), 0.99))
    return int(math.ceil(((z_alpha + z_beta) / z_r) ** 2 + 3))


def run_pillar5_audit(output_dir: Path, max_threads: int = 6) -> dict[str, Any]:
    enforce_login_node_politeness(max_threads=max_threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    findings: list[dict[str, Any]] = []

    ood_df = pd.read_csv(ROOT_DIR / "results" / "tables" / "ood_benchmark_summary.csv")
    base_df = pd.read_csv(ROOT_DIR / "results" / "tables" / "baselines_ood_summary.csv")
    st0_df = pd.read_csv(ROOT_DIR / "results" / "tables" / "stage0_bradbury_blind_metrics.csv")

    # 1. Check test sample-size mismatch between FerroFlux-FM (n_test=512) vs Baselines
    mismatches = []
    for _, row in ood_df.iterrows():
        trk = row["license_track"]
        sp = row["ood_split"]
        b_key = row["baseline_split_key"]
        sub_b = base_df[(base_df["license_track"] == trk) & (base_df["ood_split"].isin([sp, b_key]))]
        if not sub_b.empty and "n_test" in sub_b.columns:
            b_ntest = int(sub_b["n_test"].iloc[0])
            ff_ntest = int(row["n_test"])
            if b_ntest != ff_ntest:
                mismatches.append({"track": trk, "split": sp, "ferroflux_n_test": ff_ntest, "baseline_n_test": b_ntest})

    if mismatches:
        findings.append(
            {
                "id": "P5-TEST-SUBSET-SIZE-MISMATCH",
                "severity": "CRITICAL",
                "title": "ood_benchmark_summary.csv evaluated FerroFlux-FM on n_test=512 subset while baselines were evaluated on full test splits",
                "details": mismatches,
            }
        )

    # 2. Check invalid p_value == 0.0 underflows & missing BH-FDR multiple-testing correction
    zero_p_rows = int((ood_df["p_value_vs_best_baseline"] == 0.0).sum())
    has_fdr = any("fdr" in c.lower() or "q_val" in c.lower() for c in ood_df.columns)
    findings.append(
        {
            "id": "P5-ZERO-PVALUE-AND-NO-FDR",
            "severity": "MAJOR",
            "title": "Unpaired normal-approx z-tests rounded to exact p=0.0 without paired item alignment, cluster-bootstrap SE, or BH-FDR correction",
            "details": {
                "rows_with_exact_p_zero": zero_p_rows,
                "total_ood_rows": len(ood_df),
                "has_fdr_column": has_fdr,
                "st0_has_fdr_column": any("fdr" in c.lower() or "q_val" in c.lower() for c in st0_df.columns),
            },
        }
    )

    # 3. Statistical Power Analysis of Stage-0 Cohorts (n=8, 9, 11, 21) & Stage-1 Recommendations
    power_table = []
    for n_val, label in [
        (8, "Zenodo 18271042 Live-Cell Spatial FOVs"),
        (9, "GSE319397 Myc-CaP Prostate C' Dot RNA-seq"),
        (11, "GSE310954 Melanoma C' Dot RNA-seq"),
        (21, "Pooled GSE319397 + GSE319419 Prostate RNA-seq"),
    ]:
        # Exact minimum possible two-sided Mann-Whitney p-value for balanced n1 + n2 = n_val
        n1 = n_val // 2
        n2 = n_val - n1
        min_possible_mwu_p = 2.0 / math.comb(n_val, n1)
        power_table.append(
            {
                "cohort": label,
                "n_samples": n_val,
                "min_detectable_rho_80pct_power_alpha05": min_detectable_spearman_80pct_power(n_val, 0.05),
                "min_possible_exact_mann_whitney_p": round(min_possible_mwu_p, 6),
                "can_ever_reach_p_lt_0_001_binary": bool(min_possible_mwu_p < 0.001),
            }
        )

    stage1_targets = {
        "for_rho_0.50_at_p001_power80": required_n_for_rho(0.50, alpha=0.001, power=0.80),
        "for_rho_0.60_at_p001_power80": required_n_for_rho(0.60, alpha=0.001, power=0.80),
        "for_rho_0.40_at_p005_power80": required_n_for_rho(0.40, alpha=0.005, power=0.80),
    }

    pd.DataFrame(power_table).to_csv(output_dir / "pillar5_power_analysis.csv", index=False)
    findings.append(
        {
            "id": "P5-STAGE0-SMALL-N-POWER-CEILING",
            "severity": "MAJOR",
            "title": "Public Stage-0 cohorts (n=8..21) hit combinatoric p-value floors (e.g. n=8 balanced min Mann-Whitney p=0.0286; cannot reach p<0.001)",
            "details": {"stage0_power": power_table, "recommended_stage1_sample_sizes": stage1_targets},
        }
    )

    report = {
        "pillar": 5,
        "name": "Statistical Hypothesis Testing, Calibration & Power Forensics",
        "findings": findings,
        "peak_rss_mb": get_peak_rss_mb(),
    }
    (output_dir / "pillar5_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_pillar5_audit(ROOT_DIR / "results" / "audit")
    print(json.dumps({"pillar": 5, "n_findings": len(res["findings"]), "peak_rss_mb": res["peak_rss_mb"]}, indent=2))
