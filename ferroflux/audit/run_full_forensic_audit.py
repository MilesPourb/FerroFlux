"""Master Orchestrator for the 8-Pillar Zero-Trust Forensic Audit (`login06` Safe)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ferroflux.audit.audit_pillar1_data_provenance import run_pillar1_audit
from ferroflux.audit.audit_pillar2_splits_and_leakage import run_pillar2_audit
from ferroflux.audit.audit_pillar3_stage0_forensics import run_pillar3_audit
from ferroflux.audit.audit_pillar4_code_and_math import run_pillar4_audit
from ferroflux.audit.audit_pillar5_statistics import run_pillar5_audit
from ferroflux.audit.audit_pillar6_scientific_accuracy import run_pillar6_audit
from ferroflux.audit.audit_pillar7_figures_and_claims import run_pillar7_audit
from ferroflux.audit.resource_guard import enforce_login_node_politeness, get_peak_rss_mb

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")


def compile_markdown_report(pillar_reports: list[dict[str, Any]], output_dir: Path, elapsed_s: float) -> Path:
    all_findings: list[dict[str, Any]] = []
    for pr in pillar_reports:
        for f in pr.get("findings", []):
            all_findings.append({"pillar": pr["pillar"], "pillar_name": pr["name"], **f})

    sev_order = {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2, "VERIFIED-CLEAN": 3}
    all_findings.sort(key=lambda x: (sev_order.get(x["severity"], 9), x["pillar"]))

    counts = {k: sum(1 for f in all_findings if f["severity"] == k) for k in sev_order}

    lines = [
        "# `FerroFlux-FM` V1 Zero-Trust Deep Forensic Audit Report",
        "",
        f"- **Execution Host:** `{enforce_login_node_politeness()['hostname']}` (Login-Node Safe Mode, Zero SLURM)",
        f"- **Elapsed Wall Time:** `{elapsed_s:.1f}s` | **Peak RSS:** `{get_peak_rss_mb():.1f} MiB`",
        f"- **Severity Tallies:** `CRITICAL = {counts['CRITICAL']}` | `MAJOR = {counts['MAJOR']}` | `MINOR = {counts['MINOR']}` | `VERIFIED-CLEAN = {counts['VERIFIED-CLEAN']}`",
        "",
        "## 1. Master Severity-Ranked Forensic Ledger",
        "",
        "| Severity | ID | Pillar | Finding |",
        "| :--- | :--- | :---: | :--- |",
    ]
    for f in all_findings:
        lines.append(f"| **{f['severity']}** | `{f['id']}` | P{f['pillar']} | {f['title']} |")

    lines.extend(["", "## 2. Detailed Evidence by Pillar", ""])
    for pr in pillar_reports:
        lines.append(f"### Pillar {pr['pillar']}: {pr['name']}")
        for f in pr.get("findings", []):
            lines.append(f"- **[{f['severity']}] `{f['id']}` — {f['title']}**")
            lines.append("  ```json")
            detail_txt = json.dumps(f.get("details", {}), indent=2)
            for dl in detail_txt.splitlines()[:45]:
                lines.append(f"  {dl}")
            if len(detail_txt.splitlines()) > 45:
                lines.append("  ... (full evidence in pillar JSON)")
            lines.append("  ```")
        lines.append("")

    md_path = output_dir / "DEEP_FORENSIC_AUDIT_REPORT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all 8 pillars of the FerroFlux-FM Zero-Trust Forensic Audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results" / "audit")
    parser.add_argument("--max-threads", type=int, default=6)
    args = parser.parse_args()

    env_info = enforce_login_node_politeness(max_threads=args.max_threads)
    print(f"[AUDIT] Starting 8-Pillar Forensic Audit on {env_info['hostname']} (threads={env_info['max_threads']})...", flush=True)
    t0 = time.perf_counter()

    runners = [
        ("Pillar 1 (Data Provenance)", run_pillar1_audit),
        ("Pillar 2 (Splits & Leakage)", run_pillar2_audit),
        ("Pillar 3 (Stage-0 Forensics)", run_pillar3_audit),
        ("Pillar 4 (Code, Gradients & ODE)", run_pillar4_audit),
        ("Pillar 5 (Statistics & Power)", run_pillar5_audit),
        ("Pillar 6 (Science & Literature)", run_pillar6_audit),
        ("Pillar 7 (Figures & Claims)", run_pillar7_audit),
    ]

    reports: list[dict[str, Any]] = []
    for label, fn in runners:
        tp = time.perf_counter()
        print(f"[AUDIT] Running {label}...", flush=True)
        rep = fn(args.output_dir, max_threads=args.max_threads)
        reports.append(rep)
        print(f"[AUDIT] Finished {label} in {time.perf_counter() - tp:.1f}s (RSS={get_peak_rss_mb():.1f} MiB)", flush=True)

    elapsed = time.perf_counter() - t0
    md_path = compile_markdown_report(reports, args.output_dir, elapsed)
    summary = {
        "status": "AUDIT_COMPLETE",
        "elapsed_sec": round(elapsed, 2),
        "peak_rss_mb": get_peak_rss_mb(),
        "report_markdown": str(md_path),
    }
    (args.output_dir / "master_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
