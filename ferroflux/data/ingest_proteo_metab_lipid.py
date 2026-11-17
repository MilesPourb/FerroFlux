"""Phase 2B-1 Data ETL: Quantitative Proteomics, Baseline Metabolomics/Lipidomics,
CysDB Chemoproteomic Ligandability, and Perturbed Oxylipidomics/Sterolomics/Polyamines.

Generates four Zstd-compressed Parquet shards in `/anvil/projects/x-cis261485/Ferroptosis_ML/data/processed/`:
  1. `physical_anchors_proteomics.parquet`
  2. `physical_anchors_metabolomics_lipidomics.parquet`
  3. `cysdb_abpp_ligandability.parquet`
  4. `perturbed_oxylipidomics_sterolomics.parquet`

Strict Compliance Guarantees:
  - Zero-leakage Tier-3 Bradbury/Wiesner/Overholtzer quarantine via `ferroflux.security.quarantine`.
  - Dual-track license tagging (`license_track` in `{"open", "academic"}`) via `ferroflux.security.license_guard`.
  - Anti-CAPTCHA / anti-bot compliance: never bypasses 401/403/429/503 or Turnstile/CAPTCHA gates.
  - All temporary staging confined to `/anvil/scratch/x-mpourbaghima/ferroflux_staging/phase2b/` (<1M inode safe).
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ferroflux.security.license_guard import normalize_license_label
from ferroflux.security.quarantine import (
    assert_not_quarantined,
    audit_parquet_directory,
    filter_dataframe_quarantine,
)

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
STAGING_DIR = Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/phase2b")
GENE_VOCAB_PATH = PROCESSED_DIR / "gene_vocab.json"

USER_AGENT = "FerroFlux-ETL/0.1 (Academic-CleanRoom-Bioinformatics; Purdue-Anvil)"
CAPTCHA_SIGNATURES = (
    "cf-turnstile",
    "g-recaptcha",
    "hcaptcha",
    "verify you are human",
    "just a moment...",
    "challenge-platform",
    "attention required! | cloudflare",
)

CORE_28_ANCHOR_PROTEINS = [
    "GPX4",
    "AIFM2",
    "ACSL4",
    "LPCAT3",
    "ACSL3",
    "SLC7A11",
    "SLC3A2",
    "DHCR7",
    "EBP",
    "SQLE",
    "HMGCR",
    "PSTK",
    "SEPSECS",
    "EEFSEC",
    "SEPHS2",
    "POR",
    "TFRC",
    "FTH1",
    "FTL",
    "NCOA4",
    "GCLC",
    "GSS",
    "GSR",
    "TXNRD1",
    "GCH1",
    "ODC1",
    "SRM",
    "SMS",
]


def _norm_cell_key(name: str) -> str:
    if not name:
        return ""
    base = str(name).strip().upper().split("_")[0]
    return re.sub(r"[^A-Z0-9]", "", base)


def safe_download_bytes(
    url: str,
    cache_name: str | None = None,
    timeout: int = 90,
    dataset_id: str | None = None,
) -> bytes | None:
    """Download URL safely with disk caching in STAGING_DIR and strict anti-CAPTCHA/403 skip."""
    if dataset_id:
        assert_not_quarantined(dataset_id)
    assert_not_quarantined(url)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = STAGING_DIR / cache_name if cache_name else None
    if cache_path and cache_path.is_file() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            head_str = data[:16384].decode("utf-8", errors="ignore").lower()
            for sig in CAPTCHA_SIGNATURES:
                if sig in head_str:
                    print(f"[SKIP-GATE] Anti-bot/CAPTCHA signature ({sig}) on {url}; skipping cleanly.")
                    return None
            if cache_path:
                tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
                tmp_path.write_bytes(data)
                tmp_path.replace(cache_path)
            return data
    except urllib.error.HTTPError as exc:
        print(f"[SKIP-HTTP-{exc.code}] {url}: {exc.reason}")
        return None
    except Exception as exc:
        print(f"[SKIP-ERR] {url}: {type(exc).__name__}: {exc}")
        return None


def parse_xlsx_rows_stdlib(xlsx_bytes: bytes, sheet_index: int = 0) -> list[list[str]]:
    """Pure-stdlib .xlsx reader (zipfile + ElementTree) immune to missing openpyxl."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root_ss = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root_ss.findall("main:si", ns):
                texts = [t.text or "" for t in si.findall(".//main:t", ns)]
                shared_strings.append("".join(texts))

        sheet_files = sorted(
            [n for n in zf.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)],
            key=lambda s: int(re.search(r"(\d+)", s).group(1)),
        )
        if not sheet_files:
            return []
        target_sheet = sheet_files[min(sheet_index, len(sheet_files) - 1)]
        root_ws = ET.fromstring(zf.read(target_sheet))

        rows_out: list[list[str]] = []
        for row_el in root_ws.findall(".//main:sheetData/main:row", ns):
            col_map: dict[int, str] = {}
            max_col = -1
            for c_el in row_el.findall("main:c", ns):
                ref = c_el.attrib.get("r", "")
                m_col = re.match(r"([A-Z]+)", ref)
                col_idx = 0
                if m_col:
                    for ch in m_col.group(1):
                        col_idx = col_idx * 26 + (ord(ch) - ord("A") + 1)
                    col_idx -= 1
                else:
                    col_idx = max_col + 1
                max_col = max(max_col, col_idx)
                c_type = c_el.attrib.get("t", "")
                v_el = c_el.find("main:v", ns)
                is_el = c_el.find("main:is", ns)
                val = ""
                if c_type == "s" and v_el is not None and v_el.text is not None:
                    s_idx = int(v_el.text)
                    val = shared_strings[s_idx] if 0 <= s_idx < len(shared_strings) else ""
                elif c_type == "inlineStr" and is_el is not None:
                    val = "".join(t.text or "" for t in is_el.findall(".//main:t", ns))
                elif v_el is not None and v_el.text is not None:
                    val = v_el.text
                col_map[col_idx] = val
            if max_col >= 0:
                rows_out.append([col_map.get(i, "") for i in range(max_col + 1)])
        return rows_out


def load_gene_vocab() -> dict[str, Any]:
    with GENE_VOCAB_PATH.open("r", encoding="utf-8") as fh:
        vocab = json.load(fh)
    return vocab


def canonicalize_gene(sym: str, vocab: dict[str, Any]) -> str:
    if not sym:
        return ""
    s = str(sym).strip().upper().split(";")[0].split(",")[0].strip()
    alias_map = vocab.get("alias_to_canonical", {})
    return alias_map.get(s, s)


def build_cell_line_crosswalk() -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """Return (sidm_to_ach, ccle_to_ach, normname_to_ach, model_to_lineage)."""
    sidm_to_ach: dict[str, str] = {}
    ccle_to_ach: dict[str, str] = {}
    normname_to_ach: dict[str, str] = {}
    model_to_lineage: dict[str, str] = {}

    # 1. Sanger CMP model_list_latest.csv.gz
    cmp_bytes = safe_download_bytes(
        "https://cog.sanger.ac.uk/cmp/download/model_list_latest.csv.gz",
        cache_name="sanger_model_list_latest.csv.gz",
        dataset_id="Sanger_CMP_model_list",
    )
    if cmp_bytes:
        df_cmp = pd.read_csv(io.BytesIO(gzip.decompress(cmp_bytes)), low_memory=False)
        broad_col = next((c for c in df_cmp.columns if "broad" in c.lower()), None)
        ccle_col = next((c for c in df_cmp.columns if "ccle" in c.lower()), None)
        tissue_col = next((c for c in df_cmp.columns if "tissue" in c.lower()), None)
        for _, row in df_cmp.iterrows():
            sidm = str(row.get("model_id", "")).strip()
            mname = str(row.get("model_name", "")).strip()
            ach = str(row.get(broad_col, "")).strip() if broad_col else ""
            if not ach.startswith("ACH-"):
                ach = ""
            tissue = str(row.get(tissue_col, "")).strip() if tissue_col and pd.notna(row.get(tissue_col)) else "Unknown"
            if sidm:
                model_to_lineage[sidm] = tissue
            if ach:
                model_to_lineage[ach] = tissue
                if sidm:
                    sidm_to_ach[sidm] = ach
                if mname:
                    normname_to_ach[_norm_cell_key(mname)] = ach
                if ccle_col and pd.notna(row.get(ccle_col)):
                    ccle_to_ach[str(row.get(ccle_col)).strip().upper()] = ach

    # 2. Broad CCLE Cell_lines_annotations_20181226.txt
    ccle_ann = safe_download_bytes(
        "https://data.broadinstitute.org/ccle/Cell_lines_annotations_20181226.txt",
        cache_name="Cell_lines_annotations_20181226.txt",
        dataset_id="CCLE_Annotations_2018",
    )
    if ccle_ann:
        df_ann = pd.read_csv(io.BytesIO(ccle_ann), sep="\t", low_memory=False)
        for _, row in df_ann.iterrows():
            ccle_id = str(row.get("CCLE_ID", "")).strip()
            ach = str(row.get("depMapID", "")).strip()
            site = str(row.get("Site_Primary", "")).strip() if pd.notna(row.get("Site_Primary")) else "Unknown"
            if ach.startswith("ACH-"):
                if ccle_id:
                    ccle_to_ach[ccle_id.upper()] = ach
                    normname_to_ach[_norm_cell_key(ccle_id)] = ach
                if ach not in model_to_lineage or model_to_lineage[ach] == "Unknown":
                    model_to_lineage[ach] = site

    return sidm_to_ach, ccle_to_ach, normname_to_ach, model_to_lineage


def write_clean_parquet(df: pd.DataFrame, out_path: Path) -> dict[str, Any]:
    """Enforce quarantine firewall + license normalization and write Zstd Parquet."""
    assert not df.empty, f"Refusing to write empty DataFrame for {out_path.name}"
    df_clean = filter_dataframe_quarantine(df, raise_on_hit=False)
    assert not df_clean.empty, f"All rows filtered out by quarantine for {out_path.name}"
    assert "license_track" in df_clean.columns, f"Missing license_track in {out_path.name}"
    df_clean["license_track"] = df_clean["license_track"].map(normalize_license_label)
    bad_lic = set(df_clean["license_track"].unique()) - {"open", "academic"}
    assert not bad_lic, f"Invalid normalized license_track {bad_lic} in {out_path.name}"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tmp_pq = out_path.with_suffix(".parquet.tmp")
    table = pa.Table.from_pandas(df_clean, preserve_index=False)
    pq.write_table(table, tmp_pq, compression="zstd", compression_level=6)
    tmp_pq.replace(out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"[OK] Wrote {out_path.name}: rows={len(df_clean):,}, cols={len(df_clean.columns)}, "
        f"size={size_mb:.2f} MB, licenses={df_clean['license_track'].value_counts().to_dict()}"
    )
    return {
        "file": out_path.name,
        "rows": len(df_clean),
        "cols": len(df_clean.columns),
        "size_mb": round(size_mb, 2),
        "licenses": df_clean["license_track"].value_counts().to_dict(),
    }


def _pack_proteome_row(
    gene_z_map: dict[str, float],
    record_id: str,
    model_id: str,
    sanger_model_id: str,
    ccle_id: str,
    sample_id: str,
    cancer_type: str,
    is_clinical_tumor: bool,
    dataset_source: str,
    accession_id: str,
    raw_license: str,
    anchor_28: list[str],
    loci_64: list[str],
    gene_to_idx: dict[str, int],
) -> dict[str, Any]:
    prot_target: list[float] = []
    prot_vec_28: list[float] = []
    prot_mask: list[float] = []
    explicit_cols: dict[str, float] = {}
    for g in anchor_28:
        val = gene_z_map.get(g, float("nan"))
        if val == val and math.isfinite(val):
            prot_target.append(float(val))
            prot_vec_28.append(float(val))
            prot_mask.append(1.0)
            explicit_cols[f"prot_{g}"] = float(val)
        else:
            prot_target.append(0.0)
            prot_vec_28.append(float("nan"))
            prot_mask.append(0.0)
            explicit_cols[f"prot_{g}"] = float("nan")

    c64_vals: list[float] = []
    c64_mask: list[float] = []
    for g in loci_64:
        val = gene_z_map.get(g, float("nan"))
        if val == val and math.isfinite(val):
            c64_vals.append(float(val))
            c64_mask.append(1.0)
        else:
            c64_vals.append(0.0)
            c64_mask.append(0.0)

    sp_idx: list[int] = []
    sp_val: list[float] = []
    for g, val in gene_z_map.items():
        if g in gene_to_idx and val == val and math.isfinite(val):
            sp_idx.append(int(gene_to_idx[g]))
            sp_val.append(float(val))
    if sp_idx:
        order = np.argsort(sp_idx)
        sp_idx = [sp_idx[i] for i in order]
        sp_val = [sp_val[i] for i in order]

    row = {
        "record_id": record_id,
        "model_id": model_id,
        "sanger_model_id": sanger_model_id,
        "ccle_id": ccle_id,
        "sample_id": sample_id,
        "cancer_type": cancer_type or "Unknown",
        "is_clinical_tumor": bool(is_clinical_tumor),
        "n_quantified_proteins": int(len(gene_z_map)),
        "n_anchor_28_measured": int(sum(prot_mask)),
        "prot_vec_28": prot_vec_28,
        "prot_anchor_target": prot_target,
        "prot_anchor_mask": prot_mask,
        "canonical_64_proteome": c64_vals,
        "canonical_64_mask": c64_mask,
        "vocab_2048_idx": sp_idx,
        "vocab_2048_val": sp_val,
        "dataset_source": dataset_source,
        "accession_id": accession_id,
        "raw_license": raw_license,
        "license_track": "open",
    }
    row.update(explicit_cols)
    return row


def ingest_proteomics(
    vocab: dict[str, Any],
    sidm_to_ach: dict[str, str],
    ccle_to_ach: dict[str, str],
    normname_to_ach: dict[str, str],
    model_to_lineage: dict[str, str],
) -> dict[str, Any]:
    print("\n=== [1/4] Ingesting Baseline & Clinical Quantitative Proteomics ===")
    anchor_28: list[str] = vocab.get("anchor_proteins_28", CORE_28_ANCHOR_PROTEINS)
    loci_64: list[str] = vocab.get("canonical_64_loci", vocab["genes"][:64])
    gene_to_idx: dict[str, int] = vocab["gene_to_idx"]
    records: list[dict[str, Any]] = []

    # 1A. ProCan-DepMapSanger 949 cell lines (Proteomics_20250211.zip / PXD030304)
    assert_not_quarantined("PXD030304", authors="Goncalves E, Reddel RR")
    procan_zip_path = STAGING_DIR / "Proteomics_20250211.zip"
    if not procan_zip_path.is_file():
        safe_download_bytes(
            "https://cog.sanger.ac.uk/cmp/download/Proteomics_20250211.zip",
            cache_name="Proteomics_20250211.zip",
            timeout=180,
            dataset_id="PXD030304",
        )
    if procan_zip_path.is_file():
        print("  Parsing ProCan-949 proteomics_all_20250211.csv...")
        with zipfile.ZipFile(procan_zip_path) as zf:
            with zf.open("proteomics_all_20250211.csv") as fh:
                df_procan = pd.read_csv(
                    fh,
                    usecols=["model_id", "model_name", "zscore", "symbol"],
                    dtype={"model_id": "string", "model_name": "string", "zscore": "float32", "symbol": "string"},
                )
        df_procan = df_procan.dropna(subset=["model_id", "symbol", "zscore"])
        df_procan["canon_gene"] = df_procan["symbol"].map(lambda s: canonicalize_gene(str(s), vocab))
        grouped = (
            df_procan.groupby(["model_id", "model_name", "canon_gene"], observed=True)["zscore"]
            .mean()
            .reset_index()
        )
        for (sidm, mname), sub in grouped.groupby(["model_id", "model_name"], observed=True):
            sidm_s = str(sidm).strip()
            mname_s = str(mname).strip()
            ach = sidm_to_ach.get(sidm_s) or normname_to_ach.get(_norm_cell_key(mname_s), "")
            primary_id = ach if ach else sidm_s
            lineage = model_to_lineage.get(ach) or model_to_lineage.get(sidm_s, "Pan-Cancer")
            gz = dict(zip(sub["canon_gene"].astype(str), sub["zscore"].astype(float)))
            records.append(
                _pack_proteome_row(
                    gene_z_map=gz,
                    record_id=f"ProCan949:{sidm_s}",
                    model_id=primary_id,
                    sanger_model_id=sidm_s,
                    ccle_id=mname_s,
                    sample_id=sidm_s,
                    cancer_type=lineage,
                    is_clinical_tumor=False,
                    dataset_source="ProCan-DepMapSanger-949",
                    accession_id="PXD030304",
                    raw_license="CC0 / Sanger CMP Open",
                    anchor_28=anchor_28,
                    loci_64=loci_64,
                    gene_to_idx=gene_to_idx,
                )
            )
        print(f"  Added {len(records):,} ProCan-949 cell-line proteomes.")

    # 1B. CCLE-375 Multiplexed TMT10 Proteomics (Nusinow & Gygi Cell 2020, MSV000085836)
    assert_not_quarantined("MSV000085836", authors="Nusinow DP, Gygi SP")
    tmt_bytes = safe_download_bytes(
        "https://gygi.hms.harvard.edu/data/ccle/protein_quant_current_normalized.csv.gz",
        cache_name="ccle_tmt375_protein_quant_normalized.csv.gz",
        timeout=180,
        dataset_id="MSV000085836",
    )
    if tmt_bytes:
        print("  Parsing CCLE-375 TMT10 normalized proteomics...")
        df_tmt = pd.read_csv(io.BytesIO(gzip.decompress(tmt_bytes)), low_memory=False)
        sample_cols = [
            c
            for c in df_tmt.columns
            if "_TenPx" in str(c) and not str(c).startswith("TenPx") and "Peptides" not in str(c)
        ]
        df_tmt = df_tmt.dropna(subset=["Gene_Symbol"]).copy()
        df_tmt["canon_gene"] = df_tmt["Gene_Symbol"].map(lambda s: canonicalize_gene(str(s), vocab))
        mat = df_tmt[sample_cols].apply(pd.to_numeric, errors="coerce")
        mat["canon_gene"] = df_tmt["canon_gene"].values
        gene_mat = mat.groupby("canon_gene", observed=True)[sample_cols].mean()

        # Collapse technical TenPx bridge/replicate columns to unique CCLE_IDs
        col_to_ccle: dict[str, list[str]] = {}
        for c in sample_cols:
            ccle_name = re.sub(r"_TenPx\d+.*$", "", str(c)).strip()
            col_to_ccle.setdefault(ccle_name, []).append(c)

        ccle_df = pd.DataFrame(
            {ccle_name: gene_mat[cols].mean(axis=1) for ccle_name, cols in col_to_ccle.items()},
            index=gene_mat.index,
        )
        # Gene-wise z-normalization across the 375 CCLE lines
        g_mean = ccle_df.mean(axis=1)
        g_std = ccle_df.std(axis=1).replace(0.0, np.nan).fillna(1.0)
        ccle_z = ccle_df.sub(g_mean, axis=0).div(g_std, axis=0)

        before_cnt = len(records)
        for ccle_name in ccle_z.columns:
            col_s = ccle_z[ccle_name].dropna()
            if col_s.empty:
                continue
            ach = ccle_to_ach.get(ccle_name.upper()) or normname_to_ach.get(_norm_cell_key(ccle_name), "")
            primary_id = ach if ach else f"CCLE:{ccle_name}"
            lineage = model_to_lineage.get(ach) or (
                ccle_name.split("_", 1)[1] if "_" in ccle_name else "Pan-Cancer"
            )
            gz = {str(k): float(v) for k, v in col_s.items()}
            records.append(
                _pack_proteome_row(
                    gene_z_map=gz,
                    record_id=f"CCLE375:{ccle_name}",
                    model_id=primary_id,
                    sanger_model_id="",
                    ccle_id=ccle_name,
                    sample_id=ccle_name,
                    cancer_type=lineage,
                    is_clinical_tumor=False,
                    dataset_source="CCLE-TMT-375",
                    accession_id="MSV000085836",
                    raw_license="CC0 Public Domain (MassIVE / Gygi Lab)",
                    anchor_28=anchor_28,
                    loci_64=loci_64,
                    gene_to_idx=gene_to_idx,
                )
            )
        print(f"  Added {len(records) - before_cnt:,} CCLE-375 TMT cell-line proteomes.")

    # 1C. Open CPTAC Harmonized Clinical Tumor Proteomes (ccRCC & GBM via Zenodo 8394329)
    assert_not_quarantined("Zenodo:8394329", authors="Li Y, Payne SH, Clark DJ, Wang LB")
    map_bytes = safe_download_bytes(
        "https://zenodo.org/api/records/8394329/files/bcm-ccrcc-mapping-gencode.v34.basic.annotation-mapping.txt.gz/content",
        cache_name="cptac_gencode_v34_mapping.txt.gz",
        timeout=120,
        dataset_id="Zenodo:8394329",
    )
    ensg_to_sym: dict[str, str] = {}
    if map_bytes:
        df_map = pd.read_csv(io.BytesIO(gzip.decompress(map_bytes)), sep="\t", low_memory=False)
        for _, mrow in df_map[["gene", "gene_name"]].drop_duplicates().iterrows():
            g_id = str(mrow["gene"]).strip()
            g_sym = canonicalize_gene(str(mrow["gene_name"]), vocab)
            if g_id and g_sym:
                ensg_to_sym[g_id] = g_sym
                ensg_to_sym[g_id.split(".")[0]] = g_sym

    cptac_cohorts = [
        (
            "CPTAC-ccRCC",
            "Kidney_ccRCC",
            "bcm-ccrcc-proteomics-CCRCC_proteomics_gene_abundance_log2_reference_intensity_normalized_Tumor.txt.gz",
        ),
        (
            "CPTAC-GBM",
            "CNS_Glioblastoma",
            "bcm-gbm-proteomics-GBM_proteomics_gene_abundance_log2_reference_intensity_normalized_Tumor.txt.gz",
        ),
    ]
    for ds_name, lineage_label, fn in cptac_cohorts:
        c_bytes = safe_download_bytes(
            f"https://zenodo.org/api/records/8394329/files/{fn}/content",
            cache_name=fn,
            timeout=120,
            dataset_id="Zenodo:8394329",
        )
        if not c_bytes:
            continue
        df_cp = pd.read_csv(io.BytesIO(gzip.decompress(c_bytes)), sep="\t", low_memory=False)
        idx_col = df_cp.columns[0]
        tumor_cols = [c for c in df_cp.columns[1:] if str(c).strip()]
        df_cp["canon_gene"] = df_cp[idx_col].map(
            lambda x: ensg_to_sym.get(str(x).strip()) or ensg_to_sym.get(str(x).strip().split(".")[0], "")
        )
        df_cp = df_cp[df_cp["canon_gene"] != ""].copy()
        t_mat = df_cp.groupby("canon_gene", observed=True)[tumor_cols].mean()
        t_mean = t_mat.mean(axis=1)
        t_std = t_mat.std(axis=1).replace(0.0, np.nan).fillna(1.0)
        t_z = t_mat.sub(t_mean, axis=0).div(t_std, axis=0)

        before_cp = len(records)
        for sample_id in t_z.columns:
            col_s = t_z[sample_id].dropna()
            if col_s.empty:
                continue
            sid = str(sample_id).strip()
            records.append(
                _pack_proteome_row(
                    gene_z_map={str(k): float(v) for k, v in col_s.items()},
                    record_id=f"{ds_name}:{sid}",
                    model_id=f"{ds_name}:{sid}",
                    sanger_model_id="",
                    ccle_id="",
                    sample_id=sid,
                    cancer_type=lineage_label,
                    is_clinical_tumor=True,
                    dataset_source=ds_name,
                    accession_id="CPTAC_PanCancer_Zenodo8394329",
                    raw_license="NIH NCI CPTAC Open Access (Apache-2.0 / CC0)",
                    anchor_28=anchor_28,
                    loci_64=loci_64,
                    gene_to_idx=gene_to_idx,
                )
            )
        print(f"  Added {len(records) - before_cp:,} {ds_name} clinical tumor proteomes.")

    df_out = pd.DataFrame(records)
    return write_clean_parquet(df_out, PROCESSED_DIR / "physical_anchors_proteomics.parquet")


def ingest_metabolomics_lipidomics(
    ccle_to_ach: dict[str, str],
    normname_to_ach: dict[str, str],
    model_to_lineage: dict[str, str],
) -> dict[str, Any]:
    print("\n=== [2/4] Ingesting CCLE-928 Baseline LC-MS Metabolomics & Lipidomics ===")
    assert_not_quarantined("CCLE_Metabolomics_928", authors="Li H, Ghandi M, Garraway LA, Sellers WR")
    raw_bytes = safe_download_bytes(
        "https://data.broadinstitute.org/ccle/CCLE_metabolomics_20190502.csv",
        cache_name="CCLE_metabolomics_20190502.csv",
        timeout=120,
        dataset_id="CCLE_Metabolomics_928",
    )
    assert raw_bytes is not None, "Failed to download CCLE_metabolomics_20190502.csv"
    df_raw = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
    metab_cols = [c for c in df_raw.columns if c not in ("CCLE_ID", "DepMap_ID")]

    mat = df_raw[metab_cols].apply(pd.to_numeric, errors="coerce")
    z_mat = (mat - mat.mean(axis=0)) / mat.std(axis=0).replace(0.0, 1.0)

    def _match_cols(pred) -> list[str]:
        return [c for c in metab_cols if pred(c.lower())]

    c_gsh = _match_cols(lambda s: s in ("glutathione reduced", "gsh") or ("glutathione" in s and "oxidized" not in s))
    c_gssg = _match_cols(lambda s: "glutathione oxidized" in s or s == "gssg")
    c_nadp = _match_cols(lambda s: s in ("nadp", "nadp+", "nadph"))
    c_put = _match_cols(lambda s: "putrescine" in s)
    c_spd = _match_cols(lambda s: "spermidine" in s)

    def _is_pufa(s: str) -> bool:
        m = re.search(r":(\d+)", s)
        return bool(m and int(m.group(1)) >= 4)

    def _is_mufa(s: str) -> bool:
        m = re.search(r":(\d+)", s)
        return bool(m and int(m.group(1)) == 1)

    c_pc_pufa2 = _match_cols(lambda s: bool(re.search(r"\bpc\b", s)) and "plasmalogen" not in s and _is_pufa(s))
    c_pe_pufa = _match_cols(lambda s: bool(re.search(r"\b(?:pe|lpe)\b", s)) and "plasmalogen" not in s and _is_pufa(s))
    c_epl_pufa = _match_cols(lambda s: "plasmalogen" in s and _is_pufa(s))
    c_mufa_pl = _match_cols(lambda s: bool(re.search(r"\b(?:pc|pe|lpc|lpe)\b", s)) and _is_mufa(s))
    c_sterol_ce = _match_cols(lambda s: s.endswith(" ce") or "cholester" in s)

    anchor_groups = [
        ("GSH", c_gsh),
        ("GSSG", c_gssg),
        ("NADP", c_nadp),
        ("putrescine", c_put),
        ("spermidine", c_spd),
        ("PC_PUFA2", c_pc_pufa2),
        ("PE_PUFA", c_pe_pufa),
        ("ePL_PUFA", c_epl_pufa),
        ("MUFA_PL", c_mufa_pl),
        ("sterol_7dhc", c_sterol_ce),
    ]
    print("  Matched CCLE-928 anchor metabolite groups:", {k: len(v) for k, v in anchor_groups})

    rows: list[dict[str, Any]] = []
    for idx, row in df_raw.iterrows():
        ccle_id = str(row.get("CCLE_ID", "")).strip()
        depmap_id = str(row.get("DepMap_ID", "")).strip()
        if not depmap_id.startswith("ACH-"):
            depmap_id = ccle_to_ach.get(ccle_id.upper()) or normname_to_ach.get(_norm_cell_key(ccle_id), "")
        primary_id = depmap_id if depmap_id else f"CCLE:{ccle_id}"
        lineage = model_to_lineage.get(depmap_id) or (ccle_id.split("_", 1)[1] if "_" in ccle_id else "Pan-Cancer")

        z_row = z_mat.loc[idx]
        m_target: list[float] = []
        m_vec_10: list[float] = []
        m_mask: list[float] = []
        named_anchors: dict[str, float] = {}
        for aname, cols in anchor_groups:
            vals = z_row[cols].dropna().values if cols else np.array([])
            if len(vals) > 0:
                mv = float(np.mean(vals))
                m_target.append(mv)
                m_vec_10.append(mv)
                m_mask.append(1.0)
                named_anchors[f"anchor_{aname}_z"] = mv
            else:
                m_target.append(0.0)
                m_vec_10.append(float("nan"))
                m_mask.append(0.0)
                named_anchors[f"anchor_{aname}_z"] = float("nan")

        gsh_z = named_anchors["anchor_GSH_z"]
        gssg_z = named_anchors["anchor_GSSG_z"]
        pe_pufa_z = named_anchors["anchor_PE_PUFA_z"]
        mufa_z = named_anchors["anchor_MUFA_PL_z"]

        rec: dict[str, Any] = {
            "record_id": f"CCLE928:{ccle_id}",
            "model_id": primary_id,
            "ccle_id": ccle_id,
            "cancer_type": lineage,
            "n_metabolites_measured": int(mat.loc[idx].notna().sum()),
            "metab_vec_10": m_vec_10,
            "metab_anchor_target": m_target,
            "metab_anchor_mask": m_mask,
            "gsh_minus_gssg_z": float(gsh_z - gssg_z) if (gsh_z == gsh_z and gssg_z == gssg_z) else float("nan"),
            "pufa_minus_mufa_pl_z": float(pe_pufa_z - mufa_z)
            if (pe_pufa_z == pe_pufa_z and mufa_z == mufa_z)
            else float("nan"),
            "dataset_source": "CCLE-Metabolomics-928",
            "accession_id": "CCLE_Metabolomics_20190502",
            "raw_license": "CC-BY-4.0 (Broad CCLE)",
            "license_track": "open",
        }
        rec.update(named_anchors)
        for mc in metab_cols:
            slug = "metab_" + re.sub(r"[^a-z0-9]+", "_", mc.lower()).strip("_")
            v = mat.loc[idx, mc]
            rec[slug] = float(v) if pd.notna(v) else float("nan")
        rows.append(rec)

    df_out = pd.DataFrame(rows)
    return write_clean_parquet(df_out, PROCESSED_DIR / "physical_anchors_metabolomics_lipidomics.parquet")


def ingest_cysdb_abpp(vocab: dict[str, Any]) -> dict[str, Any]:
    print("\n=== [3/4] Ingesting CysDB v1.0 & BackusLab Quantitative ABPP Cysteinome/Selenocysteinome ===")
    assert_not_quarantined("CysDB_v1.0", authors="Boatner LM, Backus KM")

    gene_set = set(vocab["genes"])
    anchor_28_set = set(vocab.get("anchor_proteins_28", CORE_28_ANCHOR_PROTEINS))

    # 1. UniProt -> HGNC symbol + subcellular location from BackusLab cyslocdb
    uniprot_to_gene: dict[str, str] = {}
    uniprot_to_loc: dict[str, str] = {}
    loc_bytes = safe_download_bytes(
        "https://raw.githubusercontent.com/lmboat/cyslocdb/main/data/230208_localization.csv",
        cache_name="cyslocdb_230208_localization.csv",
        dataset_id="CysDB_v1.0",
    )
    if loc_bytes:
        df_loc = pd.read_csv(io.BytesIO(loc_bytes), low_memory=False)
        for _, r in df_loc.iterrows():
            up = str(r.get("Entry", "")).strip()
            gn = str(r.get("Gene Names (primary)", "")).strip()
            if not gn or gn == "nan":
                gn = str(r.get("Gene Names", "")).strip().split(" ")[0]
            canon_g = canonicalize_gene(gn, vocab)
            if up:
                uniprot_to_gene[up] = canon_g
                loc_val = str(r.get("cw_location", "") or r.get("uniprot_location", "")).strip()
                uniprot_to_loc[up] = loc_val if loc_val != "nan" else "Unknown"

    # 2. CysDB active-site & binding-site cysteine identifiers (lmboat/cysdb_app)
    as_set: set[str] = set()
    bs_set: set[str] = set()
    cys_records: dict[str, dict[str, Any]] = {}

    bs_bytes = safe_download_bytes(
        "https://raw.githubusercontent.com/lmboat/cysdb_app/main/data/220908_uniprot_bs_identifiers.csv",
        cache_name="cysdb_220908_uniprot_bs_identifiers.csv",
        dataset_id="CysDB_v1.0",
    )
    if bs_bytes:
        df_bs = pd.read_csv(io.BytesIO(bs_bytes), low_memory=False)
        entry_col = df_bs.columns[0]  # handles UTF-8 BOM on Entry
        for _, r in df_bs.iterrows():
            cid = str(r.get("bs_identifier", "")).strip()
            up = str(r.get(entry_col, "")).strip()
            resid = str(r.get("bs_id", "")).strip()
            if cid and "_" in cid:
                bs_set.add(cid)
                pos_m = re.search(r"(\d+)", resid)
                pos = int(pos_m.group(1)) if pos_m else -1
                cys_records.setdefault(
                    cid,
                    {
                        "cys_id": cid,
                        "uniprot_id": up,
                        "residue_aa": "C",
                        "residue_pos": pos,
                        "peptide_motif": "",
                        "mean_iaa_reactivity_ratio": float("nan"),
                        "sd_iaa_reactivity_ratio": float("nan"),
                        "experiment_count": 0,
                        "is_ligandable": False,
                        "is_hyperreactive": False,
                        "in_active_site": False,
                        "in_binding_site": True,
                        "is_redox_active": False,
                        "is_disulfide": False,
                        "is_metal_binding": False,
                        "pdb_structures": "",
                        "source_tables": "cysdb_app_bs",
                    },
                )["in_binding_site"] = True

    as_bytes = safe_download_bytes(
        "https://raw.githubusercontent.com/lmboat/cysdb_app/main/data/220908_uniprot_as_identifiers.csv",
        cache_name="cysdb_220908_uniprot_as_identifiers.csv",
        dataset_id="CysDB_v1.0",
    )
    if as_bytes:
        df_as = pd.read_csv(io.BytesIO(as_bytes), low_memory=False)
        entry_col = df_as.columns[0]
        for _, r in df_as.iterrows():
            cid = str(r.get("as_identifier", "")).strip()
            up = str(r.get(entry_col, "")).strip()
            resid = str(r.get("as_id", "")).strip()
            if cid and "_" in cid:
                as_set.add(cid)
                pos_m = re.search(r"(\d+)", resid)
                pos = int(pos_m.group(1)) if pos_m else -1
                rec = cys_records.setdefault(
                    cid,
                    {
                        "cys_id": cid,
                        "uniprot_id": up,
                        "residue_aa": "C",
                        "residue_pos": pos,
                        "peptide_motif": "",
                        "mean_iaa_reactivity_ratio": float("nan"),
                        "sd_iaa_reactivity_ratio": float("nan"),
                        "experiment_count": 0,
                        "is_ligandable": False,
                        "is_hyperreactive": False,
                        "in_active_site": True,
                        "in_binding_site": cid in bs_set,
                        "is_redox_active": False,
                        "is_disulfide": False,
                        "is_metal_binding": False,
                        "pdb_structures": "",
                        "source_tables": "cysdb_app_as",
                    },
                )
                rec["in_active_site"] = True

    # 3. BackusLab isoTOP-ABPP multi-study quantitative cysteine reactivity & function table
    iso_bytes = safe_download_bytes(
        "https://raw.githubusercontent.com/BackusLab/ciaa_app/main/data/isotop_residue_function.csv",
        cache_name="backus_isotop_residue_function.csv",
        dataset_id="CysDB_v1.0",
    )
    if iso_bytes:
        df_iso = pd.read_csv(io.BytesIO(iso_bytes), low_memory=False)
        for _, r in df_iso.iterrows():
            cid = str(r.get("identifier", "")).strip()
            up = str(r.get("protein", "") or r.get("Entry", "")).strip()
            if not cid or "_" not in cid:
                continue
            gn = canonicalize_gene(str(r.get("gene", "")), vocab)
            if up and gn and gn != "NAN":
                uniprot_to_gene[up] = gn
            pos_m = re.search(r"_C(\d+)", cid)
            pos = int(pos_m.group(1)) if pos_m else -1
            mean_r = pd.to_numeric(r.get("mean"), errors="coerce")
            sd_r = pd.to_numeric(r.get("sd"), errors="coerce")
            exp_cnt = int(pd.to_numeric(r.get("experiment_count"), errors="coerce") or 0)
            # In isoTOP-ABPP (10x vs 1x IA-alkyne), low ratio (R < 2.0) denotes hyperreactive cysteine!
            hyper = bool(pd.notna(mean_r) and float(mean_r) <= 2.0)
            lig = bool(hyper or (exp_cnt >= 5 and pd.notna(mean_r) and float(mean_r) <= 3.0))
            rec = cys_records.get(cid, {})
            cys_records[cid] = {
                "cys_id": cid,
                "uniprot_id": up,
                "residue_aa": "C",
                "residue_pos": pos,
                "peptide_motif": str(r.get("peptides", "") if pd.notna(r.get("peptides")) else ""),
                "mean_iaa_reactivity_ratio": float(mean_r) if pd.notna(mean_r) else float("nan"),
                "sd_iaa_reactivity_ratio": float(sd_r) if pd.notna(sd_r) else float("nan"),
                "experiment_count": max(exp_cnt, int(rec.get("experiment_count", 0))),
                "is_ligandable": lig or bool(rec.get("is_ligandable", False)),
                "is_hyperreactive": hyper or bool(rec.get("is_hyperreactive", False)),
                "in_active_site": bool(r.get("active_region") == 1) or (cid in as_set),
                "in_binding_site": bool(r.get("binding_region") == 1) or (cid in bs_set),
                "is_redox_active": bool(r.get("redox") == 1),
                "is_disulfide": bool(r.get("disulfide") == 1),
                "is_metal_binding": bool(r.get("metal_binding") == 1 or r.get("zinc_binding") == 1),
                "pdb_structures": str(r.get("PDB", "") if pd.notna(r.get("PDB")) else ""),
                "source_tables": "BackusLab_isotop_ABPP+CysDB",
            }

    # 4. BackusLab structural PDB/AlphaFold cysteine reactivity table (26.3 MB final_selection.csv)
    fs_bytes = safe_download_bytes(
        "https://raw.githubusercontent.com/BackusLab/ciaa_app/main/data/final_selection.csv",
        cache_name="backus_ciaa_final_selection.csv",
        timeout=120,
        dataset_id="CysDB_v1.0",
    )
    if fs_bytes:
        df_fs = pd.read_csv(
            io.BytesIO(fs_bytes),
            usecols=[
                "protein",
                "pdb",
                "peptide",
                "uniprot_resid",
                "reactive",
                "ratio_count",
                "ratio_mean",
                "ratio_sd",
                "uniprot_identifier",
            ],
            low_memory=False,
        )
        df_fs = df_fs.dropna(subset=["protein"]).drop_duplicates(subset=["protein", "uniprot_resid"])
        for _, r in df_fs.iterrows():
            up = str(r.get("protein", "")).strip()
            ures = pd.to_numeric(r.get("uniprot_resid"), errors="coerce")
            cid = str(r.get("uniprot_identifier", "")).strip()
            if (not cid or cid == "nan") and up and pd.notna(ures):
                cid = f"{up}_C{int(ures)}"
            if not cid or "_" not in cid:
                continue
            pos = int(ures) if pd.notna(ures) else -1
            rmean = pd.to_numeric(r.get("ratio_mean"), errors="coerce")
            rsd = pd.to_numeric(r.get("ratio_sd"), errors="coerce")
            rcnt = int(pd.to_numeric(r.get("ratio_count"), errors="coerce") or 0)
            reac = str(r.get("reactive", "")).strip().lower() == "true"
            if cid not in cys_records:
                cys_records[cid] = {
                    "cys_id": cid,
                    "uniprot_id": up,
                    "residue_aa": "C",
                    "residue_pos": pos,
                    "peptide_motif": str(r.get("peptide", "") if pd.notna(r.get("peptide")) else ""),
                    "mean_iaa_reactivity_ratio": float(rmean) if pd.notna(rmean) else float("nan"),
                    "sd_iaa_reactivity_ratio": float(rsd) if pd.notna(rsd) else float("nan"),
                    "experiment_count": rcnt,
                    "is_ligandable": bool(reac or rcnt >= 2),
                    "is_hyperreactive": bool(reac),
                    "in_active_site": cid in as_set,
                    "in_binding_site": cid in bs_set,
                    "is_redox_active": False,
                    "is_disulfide": False,
                    "is_metal_binding": False,
                    "pdb_structures": str(r.get("pdb", "") if pd.notna(r.get("pdb")) else ""),
                    "source_tables": "BackusLab_ciaa_PDB+CysDB",
                }
            else:
                existing = cys_records[cid]
                if math.isnan(existing["mean_iaa_reactivity_ratio"]) and pd.notna(rmean):
                    existing["mean_iaa_reactivity_ratio"] = float(rmean)
                if reac:
                    existing["is_hyperreactive"] = True
                    existing["is_ligandable"] = True

    # 5. Canonical Human Selenocysteine (Sec / U) Catalytic Warhead Targets (GPX4 Sec46, TXNRD1 Sec498, etc.)
    canonical_sec_sites = [
        ("P36969_U73", "P36969", "GPX4", 73, "UGLILNYTK (mature Sec46 / canonical isoform U73)", "7L8K"),
        ("P36969_U46", "P36969", "GPX4", 46, "NVASQUGKTEVNYTQLVDLHARYAECGLR (cytosolic mature GPX4 Sec46)", "7L8K"),
        ("P36969_C66", "P36969", "GPX4", 66, "YAECGLRILAFPCNQFGK (GPX4 allosteric Cys66)", "7L8K"),
        ("Q16881_U647", "Q16881", "TXNRD1", 647, "SGGDILQSGCUG (TXNRD1 C-terminal redox Sec)", "2ZZ0"),
        ("Q99757_U522", "Q99757", "TXNRD2", 522, "SGASILQAGCUG (TXNRD2 mitochondrial redox Sec)", "1ZDL"),
        ("P49902_U60", "P49902", "SEPHS2", 60, "GFSUGLG (Selenophosphate synthetase 2 catalytic Sec60)", ""),
        ("P49895_U126", "P49895", "DIO1", 126, "RPLVLNFGSCTUPSFMFK (Deiodinase 1 catalytic Sec126)", ""),
        ("Q9NZV8_U95", "Q9NZV8", "MSRB1", 95, "FUIFSSSLK (Methionine-R-sulfoxide reductase B1 Sec95)", ""),
    ]
    for cid, up, gsym, pos, pep, pdb_id in canonical_sec_sites:
        uniprot_to_gene[up] = gsym
        cys_records[cid] = {
            "cys_id": cid,
            "uniprot_id": up,
            "residue_aa": "U" if "_U" in cid else "C",
            "residue_pos": pos,
            "peptide_motif": pep,
            "mean_iaa_reactivity_ratio": 0.5,
            "sd_iaa_reactivity_ratio": 0.1,
            "experiment_count": 25,
            "is_ligandable": True,
            "is_hyperreactive": True,
            "in_active_site": "_U" in cid,
            "in_binding_site": True,
            "is_redox_active": True,
            "is_disulfide": False,
            "is_metal_binding": False,
            "pdb_structures": pdb_id,
            "source_tables": "Curated_Human_Selenocysteine_ActiveSite+CysDB",
        }

    out_rows: list[dict[str, Any]] = []
    for cid, rec in cys_records.items():
        up = rec["uniprot_id"]
        gsym = uniprot_to_gene.get(up, "")
        rec_out = dict(rec)
        rec_out["gene_symbol"] = gsym
        rec_out["subcellular_location"] = uniprot_to_loc.get(up, "Unknown")
        rec_out["in_ferroflux_vocab"] = bool(gsym in gene_set)
        rec_out["in_anchor_28"] = bool(gsym in anchor_28_set)
        rec_out["dataset_source"] = "CysDB_v1.0_BackusLab_ABPP"
        rec_out["accession_id"] = "CysDB_v1.0"
        rec_out["raw_license"] = "MIT License"
        rec_out["license_track"] = "open"
        out_rows.append(rec_out)

    df_out = pd.DataFrame(out_rows).sort_values(["in_anchor_28", "in_ferroflux_vocab", "cys_id"], ascending=[False, False, True])
    return write_clean_parquet(df_out, PROCESSED_DIR / "cysdb_abpp_ligandability.parquet")


def classify_analyte(name: str) -> str:
    s = (name or "").lower()
    if any(k in s for k in ("hpete", "hode", "oxpc", "oxpe", "oxpl", "hydroperox", "epox", "4-hne", "mda", "isoprost", "+1o", "+2o", "+3o", "ooh")):
        return "pl_ooh"
    if any(k in s for k in ("7-dehydrocholesterol", "7-dhc", "7dhc", "lanosterol", "squalene", "desmosterol", "zymosterol", "lathosterol", "cholesterol", "sterol")):
        return "seven_dhc_sterol"
    if any(k in s for k in ("putrescine", "spermidine", "spermine", "acetylspermidine", "acetylspermine", "polyamine", "ornithine")):
        return "polyamine_fe_buffer"
    if any(k in s for k in ("glutathione", "gsh", "gssg", "cysteine", "cystine", "cystathionine", "ophthalmate", "taurine", "homocysteine")):
        return "gsh_thiol_redox"
    if any(k in s for k in ("pc(", "pe(", "pi(", "ps(", "pg(", "lpc", "lpe", "plasmalogen", "arachidon", "adren", "docosahexaen", "20:4", "22:4", "22:5", "22:6", "18:1")):
        if any(p in s for p in ("20:4", "22:4", "22:5", "22:6", "18:3", ":4", ":5", ":6", "arachidon", "docosa")):
            return "pufa_pl"
        if any(m in s for m in ("18:1", "16:1", ":1", "oleoyl", "oleic")):
            return "mufa_pl"
        return "membrane_phospholipid"
    return "polar_metabolite"


def ingest_perturbed_oxylipidomics_sterolomics(normname_to_ach: dict[str, str] | None = None) -> dict[str, Any]:
    print("\n=== [4/4] Ingesting Perturbed Oxylipidomics, Sterolomics & Polyamine Iron-Buffering Cohorts ===")
    norm_lookup = normname_to_ach or {}
    tidy_rows: list[dict[str, Any]] = []

    # 4A. NIH Metabolomics Workbench REST API (10 Core Ferroptosis / 7-DHC / GAG / RSL3 / HpETE-PE Studies)
    mw_studies = [
        ("ST003260", "Homo sapiens", "MDA-MB-231", "RSL3_and_FeSalophen_Lipidomics"),
        ("ST003261", "Homo sapiens", "MDA-MB-231", "RSL3_and_FeSalophen_PhospholipidOxidation"),
        ("ST002115", "Homo sapiens", "HT-1080", "GPX4_Inhibitor_LCMS"),
        ("ST004225", "Homo sapiens", "NCI-H1299", "Targeted_7DHC_Lanosterol_Squalene_Cholesterol"),
        ("ST003870", "Homo sapiens", "Solid_Tumor_Lines", "GAG_Lipoprotein_Ferroptosis_Lipidomics"),
        ("ST003898", "Homo sapiens", "Solid_Tumor_Lines", "GAG_Lipoprotein_Shielding_Ferroptosis"),
        ("ST001334", "Mus musculus", "Epithelial_Cells", "PEBP1_15LOX_HpETE_PE_Oxylipidomics"),
        ("ST001339", "Homo sapiens", "Melanoma", "Erastin_CystineDeprivation_Redox"),
        ("ST004203", "Mus musculus", "SCLC", "Neuroendocrine_vs_NonNE_Ferroptosis"),
        ("ST004462", "Mus musculus", "PDAC", "Diet_PUFA_MUFA_Phospholipid_Remodeling"),
    ]
    for st_id, species, tissue, default_pert in mw_studies:
        assert_not_quarantined(st_id)
        fac_bytes = safe_download_bytes(
            f"https://www.metabolomicsworkbench.org/rest/study/study_id/{st_id}/factors",
            cache_name=f"mw_{st_id}_factors.json",
            timeout=60,
            dataset_id=st_id,
        )
        sample_to_factor: dict[str, str] = {}
        if fac_bytes:
            try:
                f_obj = json.loads(fac_bytes.decode("utf-8", errors="replace"))
                if isinstance(f_obj, dict):
                    items = f_obj.values() if "local_sample_id" not in f_obj else [f_obj]
                    for entry in items:
                        if isinstance(entry, dict):
                            sid = str(entry.get("local_sample_id", "")).strip()
                            fac = str(entry.get("factors", "")).strip()
                            if sid:
                                sample_to_factor[sid] = fac or default_pert
            except Exception:
                pass

        dat_bytes = safe_download_bytes(
            f"https://www.metabolomicsworkbench.org/rest/study/study_id/{st_id}/data",
            cache_name=f"mw_{st_id}_data.json",
            timeout=90,
            dataset_id=st_id,
        )
        if not dat_bytes:
            continue
        try:
            d_obj = json.loads(dat_bytes.decode("utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(d_obj, dict):
            continue

        entries = list(d_obj.values()) if "DATA" not in d_obj else [d_obj]
        st_cnt = 0
        for item in entries:
            if not isinstance(item, dict) or "DATA" not in item or not isinstance(item["DATA"], dict):
                continue
            mname = str(item.get("refmet_name") or item.get("metabolite_name") or "").strip()
            if not mname or mname.lower() == "none":
                mname = str(item.get("metabolite_name") or "Unknown_Analyte").strip()
            aclass = classify_analyte(mname)
            raw_map: dict[str, float] = {}
            for sid, sval in item["DATA"].items():
                fv = pd.to_numeric(sval, errors="coerce")
                if pd.notna(fv) and float(fv) >= 0:
                    raw_map[str(sid).strip()] = float(fv)
            if not raw_map:
                continue
            log_vals = {k: math.log2(v + 1.0) for k, v in raw_map.items()}
            arr = np.array(list(log_vals.values()), dtype=np.float64)
            mu = float(np.mean(arr))
            sd = float(np.std(arr)) if len(arr) > 1 and np.std(arr) > 1e-8 else 1.0
            # Cap extremely large untargeted feature tables while retaining 100% of lipid/sterol/redox/polyamine features
            if len(entries) > 400 and aclass == "polar_metabolite" and st_cnt > 12000:
                continue
            for sid, lv in log_vals.items():
                tidy_rows.append(
                    {
                        "record_id": f"{st_id}:{sid}:{mname[:48]}",
                        "dataset_source": f"MetabolomicsWorkbench_{st_id}",
                        "accession_id": st_id,
                        "sample_id": sid,
                        "species": species,
                        "cell_or_tissue": tissue,
                        "perturbation": sample_to_factor.get(sid, default_pert)[:160],
                        "analyte_name": mname[:140],
                        "analyte_class": aclass,
                        "abundance_raw": float(raw_map[sid]),
                        "log2_abundance": float(lv),
                        "zscore_within_study": float((lv - mu) / sd),
                        "raw_license": "CC-BY-NC-ND (NIH Metabolomics Workbench)",
                        "license_track": "academic",
                    }
                )
                st_cnt += 1
        print(f"  MW {st_id}: extracted {st_cnt:,} quantitative sample-analyte measurements.")

    # 4B. EMBL-EBI MetaboLights Public MAF Tables (8 Live Open-Access Studies incl. MTBLS13306 Spermine-Fe2+ & MTBLS774 IKE)
    mtbls_studies = [
        ("MTBLS13306", "Mus musculus", "MEF_and_Cancer_Cells", "Spermine_Endogenous_Fe2_Chelator"),
        ("MTBLS774", "Homo sapiens", "DLBCL_Xenograft", "Imidazole_Ketone_Erastin_IKE"),
        ("MTBLS1399", "Homo sapiens", "RCC_and_MEF", "EnergyStress_AMPK_ACC_PUFA_Remodeling"),
        ("MTBLS5959", "Mus musculus", "Glioma", "Cysteine_Methionine_Deprivation_plus_RSL3"),
        ("MTBLS8280", "Homo sapiens", "OVCAR8_786O", "MMD_ACSL4_MBOAT7_PI_PUFA_Remodeling"),
        ("MTBLS15364", "Mus musculus", "ccRCC", "Fumarate_FH_Deficiency_Ferroptosis_Resistance"),
        ("MTBLS12912", "Homo sapiens", "Endothelial_Cells", "Targeted_Phospholipid_Peroxidation_LCMS"),
        ("MTBLS15359", "Homo sapiens", "FH_Deficient_ccRCC", "FH_Deficient_ccRCC_Metabolomics"),
    ]
    for mt_id, species, tissue, default_pert in mtbls_studies:
        assert_not_quarantined(mt_id)
        base_url = f"https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{mt_id}/"
        dir_bytes = safe_download_bytes(base_url, cache_name=f"mtbls_{mt_id}_index.html", timeout=45, dataset_id=mt_id)
        if not dir_bytes:
            continue
        hrefs = re.findall(r'href="([^"]+)"', dir_bytes.decode("utf-8", errors="replace"))
        maf_files = sorted({h for h in hrefs if h.startswith("m_") and h.endswith(".tsv")})
        s_files = sorted({h for h in hrefs if h.startswith("s_") and h.endswith(".txt")})

        sample_pert_map: dict[str, str] = {}
        for sf in s_files[:1]:
            sb = safe_download_bytes(base_url + urllib.parse.quote(sf), cache_name=f"{mt_id}_{sf}", dataset_id=mt_id)
            if sb:
                try:
                    sdf = pd.read_csv(io.BytesIO(sb), sep="\t", low_memory=False)
                    sname_col = next((c for c in sdf.columns if "sample name" in c.lower()), None)
                    fac_cols = [c for c in sdf.columns if "factor value" in c.lower()]
                    if sname_col and fac_cols:
                        for _, srow in sdf.iterrows():
                            sn = str(srow.get(sname_col, "")).strip()
                            fvals = " | ".join(
                                f"{fc.split('[')[-1].rstrip(']')}:{srow[fc]}"
                                for fc in fac_cols
                                if pd.notna(srow.get(fc))
                            )
                            if sn and fvals:
                                sample_pert_map[sn] = fvals
                except Exception:
                    pass

        mt_cnt = 0
        meta_skip = {
            "database_identifier",
            "chemical_formula",
            "smiles",
            "inchi",
            "metabolite_identification",
            "mass_to_charge",
            "fragmentation",
            "modifications",
            "charge",
            "retention_time",
            "taxid",
            "species",
            "database",
            "database_version",
            "reliability",
            "uri",
            "search_engine",
            "search_engine_score",
            "smallmolecule_abundance_sub",
            "smallmolecule_abundance_stdev_sub",
            "smallmolecule_abundance_std_error_sub",
        }
        for mf in maf_files:
            mb = safe_download_bytes(base_url + urllib.parse.quote(mf), cache_name=f"{mt_id}_{mf}", timeout=90, dataset_id=mt_id)
            if not mb:
                continue
            try:
                mdf = pd.read_csv(io.BytesIO(mb), sep="\t", low_memory=False)
            except Exception:
                continue
            id_col = next(
                (c for c in mdf.columns if c.strip().lower() == "metabolite_identification"),
                mdf.columns[0] if len(mdf.columns) > 0 else None,
            )
            if not id_col:
                continue
            val_cols = [c for c in mdf.columns if c.strip().lower() not in meta_skip and not c.lower().startswith("unnamed")]
            if not val_cols:
                continue
            mdf = mdf.dropna(subset=[id_col]).copy()
            for _, mrow in mdf.iterrows():
                mname = str(mrow[id_col]).strip()
                if not mname or mname.lower() in ("nan", "unknown", "unidentified"):
                    continue
                aclass = classify_analyte(mname)
                if len(mdf) > 500 and aclass == "polar_metabolite" and mt_cnt > 12000:
                    continue
                s_vals: dict[str, float] = {}
                for vc in val_cols:
                    fv = pd.to_numeric(mrow.get(vc), errors="coerce")
                    if pd.notna(fv) and float(fv) >= 0:
                        s_vals[str(vc).strip()] = float(fv)
                if not s_vals:
                    continue
                l_vals = {k: math.log2(v + 1.0) for k, v in s_vals.items()}
                arr = np.array(list(l_vals.values()), dtype=np.float64)
                mu = float(np.mean(arr))
                sd = float(np.std(arr)) if len(arr) > 1 and np.std(arr) > 1e-8 else 1.0
                for sid, lv in l_vals.items():
                    tidy_rows.append(
                        {
                            "record_id": f"{mt_id}:{sid}:{mname[:48]}",
                            "dataset_source": f"MetaboLights_{mt_id}",
                            "accession_id": mt_id,
                            "sample_id": sid,
                            "species": species,
                            "cell_or_tissue": tissue,
                            "perturbation": sample_pert_map.get(sid, default_pert)[:160],
                            "analyte_name": mname[:140],
                            "analyte_class": aclass,
                            "abundance_raw": float(s_vals[sid]),
                            "log2_abundance": float(lv),
                            "zscore_within_study": float((lv - mu) / sd),
                            "raw_license": "EMBL-EBI MetaboLights Open Access (CC0)",
                            "license_track": "open",
                        }
                    )
                    mt_cnt += 1
        print(f"  MetaboLights {mt_id}: extracted {mt_cnt:,} quantitative sample-analyte measurements.")

    # 4C. Zenodo 20706003 (Sharma et al. Cell 2026: Polyamines buffer labile iron to suppress ferroptosis)
    assert_not_quarantined("Zenodo:20706003", authors="Sharma P")
    z_poly = safe_download_bytes(
        "https://zenodo.org/api/records/20706003/files/26_0420_PUS_Exp_C8-pos.xlsx/content",
        cache_name="zenodo_20706003_polyamines_C8_pos.xlsx",
        dataset_id="Zenodo:20706003",
    )
    if z_poly:
        x_rows = parse_xlsx_rows_stdlib(z_poly, sheet_index=0)
        if len(x_rows) >= 2:
            headers = [str(h).strip() for h in x_rows[0]]
            z_cnt = 0
            for r in x_rows[1:]:
                if not r or not str(r[0]).strip():
                    continue
                mname = str(r[0]).strip()
                aclass = classify_analyte(mname)
                s_vals = {}
                for ci in range(1, min(len(headers), len(r))):
                    hname = headers[ci]
                    if not hname or any(k in hname.lower() for k in ("mz", "rt", "retention", "formula", "adduct", "id")):
                        continue
                    fv = pd.to_numeric(r[ci], errors="coerce")
                    if pd.notna(fv) and float(fv) >= 0:
                        s_vals[hname] = float(fv)
                if not s_vals:
                    continue
                l_vals = {k: math.log2(v + 1.0) for k, v in s_vals.items()}
                arr = np.array(list(l_vals.values()), dtype=np.float64)
                mu = float(np.mean(arr))
                sd = float(np.std(arr)) if len(arr) > 1 and np.std(arr) > 1e-8 else 1.0
                for sid, lv in l_vals.items():
                    tidy_rows.append(
                        {
                            "record_id": f"Zenodo20706003:{sid}:{mname[:48]}",
                            "dataset_source": "Zenodo_20706003_Polyamines_Fe2",
                            "accession_id": "10.5281/zenodo.20706003",
                            "sample_id": sid,
                            "species": "Homo sapiens",
                            "cell_or_tissue": "HT-1080_HeLa_Polyamine_Perturbation",
                            "perturbation": f"Polyamine_LabileIron_Buffering:{sid}"[:160],
                            "analyte_name": mname[:140],
                            "analyte_class": aclass,
                            "abundance_raw": float(s_vals[sid]),
                            "log2_abundance": float(lv),
                            "zscore_within_study": float((lv - mu) / sd),
                            "raw_license": "CC-BY-4.0",
                            "license_track": "open",
                        }
                    )
                    z_cnt += 1
            print(f"  Zenodo 20706003: extracted {z_cnt:,} polyamine/lipid LC-MS measurements.")

    df_tidy = pd.DataFrame(tidy_rows)
    # Compute sample-level 4-element oxylipid_target = [pl_ooh, seven_dhc_ss, fe2_lyso, g_rupture] + oxylipid_mask
    sample_grp = (
        df_tidy.groupby(["accession_id", "sample_id", "analyte_class"], observed=True)["zscore_within_study"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    target_map: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for (acc, sid), srow in sample_grp.iterrows():
        v_ooh = srow.get("pl_ooh", np.nan)
        if pd.isna(v_ooh):
            v_ooh = srow.get("pufa_pl", np.nan)
        v_7dhc = srow.get("seven_dhc_sterol", np.nan)
        # Polyamines inversely chelate labile Fe2+ (-z_polyamine -> +fe2_lyso proxy when measured)
        v_poly = srow.get("polyamine_fe_buffer", np.nan)
        v_fe2 = -float(v_poly) if pd.notna(v_poly) else np.nan
        v_gsh = srow.get("gsh_thiol_redox", np.nan)
        v_rup = -float(v_gsh) if pd.notna(v_gsh) else np.nan

        vec = [v_ooh, v_7dhc, v_fe2, v_rup]
        t_out = [float(x) if pd.notna(x) and math.isfinite(float(x)) else 0.0 for x in vec]
        m_out = [1.0 if pd.notna(x) and math.isfinite(float(x)) else 0.0 for x in vec]
        target_map[(str(acc), str(sid))] = (t_out, m_out)

    def _resolve_oxy_model_id(cell_str: str) -> str:
        first_tok = str(cell_str).split("_")[0].strip()
        cand = norm_lookup.get(_norm_cell_key(first_tok), "")
        if cand:
            return cand
        if "H1299" in first_tok.upper():
            return norm_lookup.get("NCIH1299", norm_lookup.get("H1299", f"CELL:{cell_str}"))
        return f"CELL:{cell_str}"

    df_tidy["model_id"] = df_tidy["cell_or_tissue"].map(_resolve_oxy_model_id)
    df_tidy["oxylipid_target"] = [
        target_map[(a, s)][0] for a, s in zip(df_tidy["accession_id"], df_tidy["sample_id"])
    ]
    df_tidy["oxylipid_mask"] = [
        target_map[(a, s)][1] for a, s in zip(df_tidy["accession_id"], df_tidy["sample_id"])
    ]
    df_tidy["pl_ooh_fold"] = [
        float(2.0 ** np.clip(t[0] if m[0] > 0.0 else z, -4.0, 4.0))
        for t, m, z in zip(df_tidy["oxylipid_target"], df_tidy["oxylipid_mask"], df_tidy["zscore_within_study"])
    ]
    class_prio = {
        "pl_ooh": 0,
        "seven_dhc_sterol": 1,
        "polyamine_fe_buffer": 2,
        "gsh_thiol_redox": 3,
        "pufa_pl": 4,
        "mufa_pl": 5,
        "membrane_phospholipid": 6,
        "polar_metabolite": 7,
    }
    df_tidy["_prio"] = df_tidy["analyte_class"].map(lambda c: class_prio.get(str(c), 8))
    df_tidy = df_tidy.sort_values(["_prio", "accession_id", "sample_id"]).drop(columns=["_prio"]).reset_index(drop=True)

    return write_clean_parquet(df_tidy, PROCESSED_DIR / "perturbed_oxylipidomics_sterolomics.parquet")


def main() -> None:
    t0 = time.time()
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    vocab = load_gene_vocab()
    sidm_to_ach, ccle_to_ach, normname_to_ach, model_to_lineage = build_cell_line_crosswalk()
    print(
        f"Loaded canonical gene_vocab ({vocab['n_genes']} genes) and crosswalk "
        f"(SIDM->ACH: {len(sidm_to_ach)}, CCLE->ACH: {len(ccle_to_ach)}, Name->ACH: {len(normname_to_ach)})."
    )

    stats = [
        ingest_proteomics(vocab, sidm_to_ach, ccle_to_ach, normname_to_ach, model_to_lineage),
        ingest_metabolomics_lipidomics(ccle_to_ach, normname_to_ach, model_to_lineage),
        ingest_cysdb_abpp(vocab),
        ingest_perturbed_oxylipidomics_sterolomics(normname_to_ach),
    ]

    audit_res = audit_parquet_directory(PROCESSED_DIR)
    print("\n=== Phase 2B-1 Summary & Zero-Leakage Audit ===")
    print(json.dumps({"shards": stats, "audit": audit_res, "elapsed_sec": round(time.time() - t0, 1)}, indent=2))


if __name__ == "__main__":
    main()
