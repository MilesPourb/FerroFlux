"""Pillar 7 Forensic Audit: Figure-to-Data Truthfulness, Hardcoded Constants & Missing Key Fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ferroflux.audit.resource_guard import enforce_login_node_politeness, get_peak_rss_mb

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")


def run_pillar7_audit(output_dir: Path, max_threads: int = 6) -> dict[str, Any]:
    enforce_login_node_politeness(max_threads=max_threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    findings: list[dict[str, Any]] = []

    fig_src = (ROOT_DIR / "ferroflux" / "validation" / "generate_nature_figures.py").read_text(encoding="utf-8")

    # 1. Fig1b hardcoded 9.5e7 counts vs actual Parquet row counts on disk
    shard_rows = {
        p.name: pq.read_metadata(p).num_rows
        for p in sorted((ROOT_DIR / "data" / "processed").glob("*.parquet"))
    }
    total_disk_rows = sum(shard_rows.values())
    has_hardcoded_95m = "9.5e7" in fig_src and "8.2e6" in fig_src
    if has_hardcoded_95m:
        findings.append(
            {
                "id": "P7-FIG1B-HARDCODED-COUNTS",
                "severity": "CRITICAL",
                "title": "Fig1b plots hardcoded upstream single-cell counts (9.5e7, 8.2e6) labeled 'Harmonized Perturbation / Cell Profiles' instead of the 734,267 rows on disk",
                "details": {
                    "hardcoded_open_sum": int(9.5e7 + 2.6e6 + 6.8e6 + 3.6e5 + 1.8e4 + 1.1e6),
                    "actual_parquet_rows_on_disk": total_disk_rows,
                    "per_shard_rows": shard_rows,
                },
            }
        )

    # 2. Fig3d missing mass_balance_residual key check
    fm_src = (ROOT_DIR / "ferroflux" / "models" / "ferroflux_fm.py").read_text(encoding="utf-8")
    missing_resid_key = '"mass_balance_residual" in ol' in fig_src and '"mass_balance_residual":' not in fm_src
    if missing_resid_key:
        findings.append(
            {
                "id": "P7-FIG3D-VACUOUS-RESIDUAL",
                "severity": "MAJOR",
                "title": "Fig3d checks 'if mass_balance_residual in ol' which FerroFluxFM.forward() never emits, printing vacuous 0.00e+00 fallback",
                "details": {"emitted_by_model": False, "printed_fallback": 0.0},
            }
        )

    # 3. Fig1-Fig6 stale timestamp vs XL Saturday run check
    ood_mtime = (ROOT_DIR / "results" / "tables" / "ood_benchmark_summary.csv").stat().st_mtime
    fig2_mtime = (ROOT_DIR / "results" / "figures" / "Fig2_Compositional_OOD_Benchmarks.png").stat().st_mtime
    if ood_mtime > fig2_mtime:
        findings.append(
            {
                "id": "P7-STALE-FIGURES-VS-TABLES",
                "severity": "MAJOR",
                "title": "Fig1-Fig6 on disk predate Saturday's FerroFlux-FM-XL OOD/Ablation CSV updates (figures show Friday BASE numbers while CSVs show Saturday XL numbers)",
                "details": {"ood_csv_mtime": ood_mtime, "fig2_png_mtime": fig2_mtime},
            }
        )

    # 4. Synthetic fallback tables inside generate_nature_figures.py if CSVs missing
    has_synth_fallbacks = "0.76, 0.74, 0.71, 0.75, 0.79, 0.73" in fig_src
    if has_synth_fallbacks:
        findings.append(
            {
                "id": "P7-SYNTHETIC-PLOT-FALLBACKS",
                "severity": "MINOR",
                "title": "generate_nature_figures.py contains synthetic fallback DataFrames when CSVs are missing (should raise FileNotFoundError in production)",
                "details": {"has_synthetic_fallback_code": True},
            }
        )

    report = {
        "pillar": 7,
        "name": "Figure-to-Table Truthfulness, Hardcoded Constants & Claim Forensics",
        "findings": findings,
        "peak_rss_mb": get_peak_rss_mb(),
    }
    (output_dir / "pillar7_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_pillar7_audit(ROOT_DIR / "results" / "audit")
    print(json.dumps({"pillar": 7, "n_findings": len(res["findings"]), "peak_rss_mb": res["peak_rss_mb"]}, indent=2))
