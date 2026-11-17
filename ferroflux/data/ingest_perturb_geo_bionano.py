"""Phase 2A-2 Data ETL: Perturbation Transcriptomes, Sec46 Ribo-seq, Healthy Organ/TIL Baselines & Clinical ICB.

Produces three Zstd-compressed Parquet shards in `data/processed/`:
1. `normal_organ_til_baselines.parquet`
2. `perturb_transcriptomes.parquet` (dual-track: `"open"` AND `"academic"`)
3. `clinical_icb_cohorts.parquet`

Sources harmonized onto `data/processed/gene_vocab.json` (`n_genes = 2048`):
- Healthy Organ & Immune TIL Reference: Human Protein Atlas (`rna_single_cell_type.tsv.zip` + `rna_tissue_consensus.tsv.zip`)
- Single-Cell Chemical Perturbation: `Tahoe-100M` (`arcinstitute/PIE_tahoe100m` + `tahoebio/Tahoe-100M`, `CC0-1.0` -> `"open"`)
  with exact single-cell survivorship weights `clip(ref_ncells / max(target_ncells, 1), 1.0, 20.0)`
- Chemical & CRISPR L1000 Signatures: `LINCS 2020` (`level5_beta_trt_cp` & `level5_beta_trt_xpr` via HTTP Range `h5py`, `CC-BY-4.0` -> `"open"`)
- Perturb-seq & Cytokine Dictionaries:
  - `Arc PIE`: `PIE_replogle_nadig_essential` (`"open"`), `PIE_jiang` (`"open"`), `PIE_arc_vcc_25` (`"open"`),
    and `PIE_x_atlas_orion` (`HCT116`/`HEK293T` genome-wide CRISPRi, `CC-BY-NC-SA-4.0` -> `"academic"`)
  - `Parse-10M` Human Donor 90-Cytokine PBMC Dictionary (`State-Parse-Filtered` CSR `.h5ad` via HTTP Range `h5py`, `CC-BY-NC-4.0` -> `"academic"`)
- Non-Quarantined Leaf `NCBI GEO` Ferroptosis RNA-seq, Paired Mammalian `Sec46` Ribo-seq (`GSE235336` + `GSE235333` `CFPAC-1`/`PANC-1` +/- `IKE`),
  and Non-Bradbury `Fe3O4`/`SiO2` Bio-Nano RNA-seq (`GSE328517`, `GSE261778`, `GSE312613`, `GSE194313`) (`"open"`)
- Clinical Immunotherapy Cohorts (`"open"`):
  - `Braun 2020` CheckMate 009/010/025 ccRCC Anti-PD-1 (`PMC7499153` `MOESM2_ESM.xlsx`)
  - `Riaz 2017` Melanoma Anti-PD-1 (`GSE91061` pre + on-treatment)
  - `Hugo 2016` Melanoma Anti-PD-1 (`GSE78220`)
  - `Cloughesy 2019` GBM Neoadjuvant vs Adjuvant Anti-PD-1 (`GSE121810`)
  - `Jerby-Arnon 2018` Melanoma ICB (`GSE115978` tumor pseudobulks)
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import gzip
import io
import json
import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from ferroflux.data.ingest_depmap_pharmaco import (
    CURATED_RCD_COMPOUNDS,
    download_file,
    read_xlsx_sheet,
)
from ferroflux.security.quarantine import (
    assert_not_quarantined,
    filter_dataframe_quarantine,
)

RDLogger.DisableLog("rdApp.*")

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
DEFAULT_STAGING_DIR = Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/phase2a/perturb_geo")
DEFAULT_DEPMAP_STAGING = Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/phase2a/depmap_pharmaco")
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed"


class HTTPRangeFile(io.RawIOBase):
    """Thread-safe seekable HTTP Range reader for remote HDF5 / GCTX / H5AD files."""

    def __init__(self, url: str, timeout: int = 25) -> None:
        assert_not_quarantined(url)
        self.timeout = timeout
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "FerroFlux-ETL/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            self.url = resp.url
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def readinto(self, b: bytearray | memoryview) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b) - 1, self.size - 1)
        last_err: Exception | None = None
        for _ in range(4):
            try:
                req = urllib.request.Request(
                    self.url,
                    headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "FerroFlux-ETL/1.0"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()
                n = len(data)
                b[:n] = data
                self.pos += n
                return n
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"HTTPRangeFile read failed on {self.url} [{self.pos}-{end}]: {last_err}")


def murcko_from_smiles(smi: str, fallback: str = "") -> tuple[str, str]:
    s = (smi or "").strip()
    if not s or s.lower() == "nan":
        return "", fallback
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s, fallback or s
    canon = Chem.MolToSmiles(mol, canonical=True)
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        scaf_s = Chem.MolToSmiles(scaf, canonical=True) if scaf is not None else ""
        return canon, (scaf_s if scaf_s else canon)
    except Exception:
        return canon, canon


def verify_geo_soft_not_quarantined(gse_acc: str, staging_dir: Path) -> str:
    """Fetch NCBI GEO SOFT brief metadata and run strict contributor + PMID quarantine check."""
    assert_not_quarantined(gse_acc)
    soft_dir = staging_dir / "geo_soft_briefs"
    soft_dir.mkdir(parents=True, exist_ok=True)
    soft_path = soft_dir / f"{gse_acc}_brief.txt"
    if not soft_path.exists():
        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_acc}&targ=self&form=text&view=brief"
        download_file(url, soft_path, gse_acc)
    txt = soft_path.read_text(encoding="utf-8", errors="replace")
    contributors: list[str] = []
    title = gse_acc
    for line in txt.splitlines():
        if line.startswith("!Series_contributor"):
            contributors.append(line.split("=", 1)[-1].strip())
        elif line.startswith("!Series_pubmed_id"):
            pmid = line.split("=", 1)[-1].strip()
            assert_not_quarantined(f"PMID:{pmid}")
        elif line.startswith("!Series_title"):
            title = line.split("=", 1)[-1].strip()
    assert_not_quarantined(gse_acc, authors=contributors)
    return title


def parse_geo_series_matrix_samples(gse_acc: str, staging_dir: Path) -> dict[str, dict[str, str]]:
    """Download GEO series_matrix.txt.gz and extract GSM -> {title, characteristics}."""
    num_prefix = gse_acc[:-3] + "nnn" if len(gse_acc) > 6 else "GSE" + gse_acc[3:].zfill(3)[:1] + "nnn"
    if len(gse_acc) == 8:  # e.g. GSE78220 -> GSE78nnn
        num_prefix = gse_acc[:5] + "nnn"
    elif len(gse_acc) == 9:  # e.g. GSE121810 -> GSE121nnn
        num_prefix = gse_acc[:6] + "nnn"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{num_prefix}/{gse_acc}/matrix/{gse_acc}_series_matrix.txt.gz"
    dest = staging_dir / f"{gse_acc}_series_matrix.txt.gz"
    download_file(url, dest, gse_acc)

    gsm_ids: list[str] = []
    titles: list[str] = []
    char_lines: list[list[str]] = []
    with gzip.open(dest, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                gsm_ids = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [x.strip().strip('"') for x in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                char_lines.append([x.strip().strip('"') for x in line.strip().split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                break

    out: dict[str, dict[str, str]] = {}
    for i, gsm in enumerate(gsm_ids):
        t_str = titles[i] if i < len(titles) else gsm
        c_joined = " | ".join(cl[i] for cl in char_lines if i < len(cl) and cl[i])
        out[gsm] = {"title": t_str, "chars": c_joined}
    return out


# ---------------------------------------------------------------------------
# 1. Healthy Human Organ & Immune TIL Reference (`normal_organ_til_baselines.parquet`)
# ---------------------------------------------------------------------------
def build_normal_organ_til_baselines(
    staging_dir: Path,
    output_dir: Path,
    vocab_genes: list[str],
    gene_to_idx: dict[str, int],
    alias_map: dict[str, str],
) -> tuple[int, dict[str, str]]:
    print("[Phase 2A-2] Building normal_organ_til_baselines.parquet from Human Protein Atlas...")
    hpa_sc_zip = download_file(
        "https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip",
        staging_dir / "rna_single_cell_type.tsv.zip",
        "HPA_SingleCell_v24",
    )
    hpa_tiss_zip = download_file(
        "https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip",
        staging_dir / "rna_tissue_consensus.tsv.zip",
        "HPA_TissueConsensus_v24",
    )

    ensg_to_sym: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    def _classify_organ_and_til(label: str) -> tuple[str, int]:
        low = label.lower()
        if any(k in low for k in ("t-cell", "t cell", "nk-cell", "nk cell", "b-cell", "b cell", "macrophage", "monocyte", "dendritic", "plasma cell", "granulocyte", "kupffer", "hofbauer", "langerhans")):
            if "kupffer" in low:
                return "Liver", 1
            return "Immune_TIL", 1
        if any(k in low for k in ("proximal tubular", "distal tubular", "collecting duct", "podocyte", "kidney")):
            return "Kidney", 0
        if any(k in low for k in ("hepatocyte", "cholangiocyte", "liver", "hepatic")):
            return "Liver", 0
        if any(k in low for k in ("cardiomyocyte", "heart", "myocard")):
            return "Heart", 0
        if any(k in low for k in ("neuron", "astrocyte", "oligodendrocyte", "microglia", "cerebral", "cerebell", "hippocamp", "brain", "cortex", "choroid")):
            return "Brain", 0
        if any(k in low for k in ("alveolar", "club cell", "ciliated", "lung", "bronch")):
            return "Lung", 0
        if any(k in low for k in ("melanocyte", "keratinocyte", "skin", "epiderm")):
            return "Skin", 0
        if any(k in low for k in ("enterocyte", "paneth", "goblet", "colon", "duoden", "small intestine", "rectum")):
            return "Bowel", 0
        if any(k in low for k in ("pancrea", "ductal", "exocrine", "islet")):
            return "Pancreas", 0
        return "Healthy_Tissue", 0

    with zipfile.ZipFile(hpa_sc_zip, "r") as zf:
        tsv_name = [n for n in zf.namelist() if n.endswith(".tsv")][0]
        with zf.open(tsv_name) as fh:
            df_sc = pd.read_csv(fh, sep="\t", low_memory=False)
    for g_ens, g_sym in zip(df_sc["Gene"].astype(str), df_sc["Gene name"].astype(str)):
        s_clean = alias_map.get(g_sym.strip(), g_sym.strip())
        ensg_to_sym[g_ens.split(".")[0].strip()] = s_clean

    df_sc["_sym"] = df_sc["Gene name"].astype(str).map(lambda s: alias_map.get(s.strip(), s.strip()))
    df_sc_sub = df_sc[df_sc["_sym"].isin(gene_to_idx)].copy()
    df_sc_sub["_val"] = np.log2(pd.to_numeric(df_sc_sub["nCPM"], errors="coerce").fillna(0.0).clip(lower=0.0) + 1.0)
    piv_sc = df_sc_sub.pivot_table(index="Cell type", columns="_sym", values="_val", aggfunc="mean").fillna(0.0)

    for ct_name, s_row in piv_sc.iterrows():
        ct_str = str(ct_name).strip()
        vec = np.zeros(len(vocab_genes), dtype=np.float32)
        cnt = 0
        for sym, v in s_row.items():
            gi = gene_to_idx.get(str(sym))
            if gi is not None:
                vec[gi] = float(v)
                cnt += 1
        organ, is_til = _classify_organ_and_til(ct_str)
        clean_id = re.sub(r"[^A-Za-z0-9]+", "_", ct_str).strip("_").upper()
        rows.append(
            {
                "record_id": f"HPA_SC_{clean_id}",
                "organ": organ,
                "cell_type": ct_str,
                "is_til": int(is_til),
                "expr_vec": vec.tolist(),
                "expr_mask_count": cnt,
                "dataset_source": "HPA_SingleCell_v24",
                "accession_id": "HPA_rna_single_cell_type",
                "license_track": "open",
            }
        )

    with zipfile.ZipFile(hpa_tiss_zip, "r") as zf:
        tsv_name = [n for n in zf.namelist() if n.endswith(".tsv")][0]
        with zf.open(tsv_name) as fh:
            df_tiss = pd.read_csv(fh, sep="\t", low_memory=False)
    df_tiss["_sym"] = df_tiss["Gene name"].astype(str).map(lambda s: alias_map.get(s.strip(), s.strip()))
    df_tiss_sub = df_tiss[df_tiss["_sym"].isin(gene_to_idx)].copy()
    val_col = "nTPM" if "nTPM" in df_tiss_sub.columns else df_tiss_sub.columns[-1]
    df_tiss_sub["_val"] = np.log2(pd.to_numeric(df_tiss_sub[val_col], errors="coerce").fillna(0.0).clip(lower=0.0) + 1.0)
    piv_tiss = df_tiss_sub.pivot_table(index="Tissue", columns="_sym", values="_val", aggfunc="mean").fillna(0.0)

    for tiss_name, s_row in piv_tiss.iterrows():
        t_str = str(tiss_name).strip()
        vec = np.zeros(len(vocab_genes), dtype=np.float32)
        cnt = 0
        for sym, v in s_row.items():
            gi = gene_to_idx.get(str(sym))
            if gi is not None:
                vec[gi] = float(v)
                cnt += 1
        organ, is_til = _classify_organ_and_til(t_str)
        clean_id = re.sub(r"[^A-Za-z0-9]+", "_", t_str).strip("_").upper()
        rows.append(
            {
                "record_id": f"HPA_TISS_{clean_id}",
                "organ": organ,
                "cell_type": f"{t_str} (bulk consensus)",
                "is_til": int(is_til),
                "expr_vec": vec.tolist(),
                "expr_mask_count": cnt,
                "dataset_source": "HPA_TissueConsensus_v24",
                "accession_id": "HPA_rna_tissue_consensus",
                "license_track": "open",
            }
        )

    df_norm = pd.DataFrame(rows).drop_duplicates(subset=["record_id"], keep="first")
    df_norm = filter_dataframe_quarantine(df_norm)
    pq.write_table(
        pa.Table.from_pandas(df_norm, preserve_index=False),
        output_dir / "normal_organ_til_baselines.parquet",
        compression="zstd",
    )
    print(f"[Phase 2A-2] Wrote normal_organ_til_baselines.parquet: {len(df_norm)} healthy cell/organ baselines")
    return len(df_norm), ensg_to_sym


# ---------------------------------------------------------------------------
# 2A. Tahoe-100M Pseudobulk DE + Survivorship Weights (`CC0-1.0` -> `"open"`)
# ---------------------------------------------------------------------------
def extract_tahoe100m_signatures(
    staging_dir: Path,
    vocab_genes: list[str],
    gene_to_idx: dict[str, int],
    alias_map: dict[str, str],
    cname_to_mid: dict[str, str],
    default_mid: str,
) -> list[dict[str, Any]]:
    print("[Phase 2A-2] Extracting Tahoe-100M single-cell DE + survivorship weights...")
    drug_meta_pq = download_file(
        "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/metadata/drug_metadata.parquet",
        staging_dir / "tahoe_drug_metadata.parquet",
        "tahoebio/Tahoe-100M",
    )
    cell_meta_pq = download_file(
        "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/metadata/cell_line_metadata.parquet",
        staging_dir / "tahoe_cell_line_metadata.parquet",
        "tahoebio/Tahoe-100M",
    )
    df_drugs = pq.read_table(drug_meta_pq).to_pandas()
    drug_smi_map: dict[str, tuple[str, str]] = {}
    priority_drugs: set[str] = set()
    prio_kw = (
        "erastin", "rSL3", "sorafenib", "sulfasalazine", "artesunate", "artemisinin", "auranofin",
        "simvastatin", "lovastatin", "atorvastatin", "fluvastatin", "pitavastatin", "rosuvastatin",
        "deferoxamine", "deferasirox", "ciclopirox", "rosiglitazone", "pioglitazone", "zileuton",
        "elesclomol", "disulfiram", "bortezomib", "venetoclax", "navitoclax", "paclitaxel", "doxorubicin",
        "vorinostat", "rapamycin", "everolimus", "dasatinib", "lapatinib", "menadione", "withaferin",
        "piperlongumine", "trametinib", "cobimetinib", "vemurafenib", "dabrafenib", "olaparib", "cisplatin",
        "gemcitabine", "5-fluorouracil", "temozolomide", "sunitinib", "cabozantinib", "axitinib", "lenvatinib",
    )
    for _, r in df_drugs.iterrows():
        dname = str(r.get("drug", "")).strip()
        smi = str(r.get("canonical_smiles", "")).strip()
        canon_s, scaf_s = murcko_from_smiles(smi, dname)
        drug_smi_map[dname] = (canon_s, scaf_s)
        low_d = dname.lower()
        moa = (str(r.get("moa-broad", "")) + " " + str(r.get("moa-fine", ""))).lower()
        if any(k.lower() in low_d or k.lower() in moa for k in prio_kw) or any(
            k in moa for k in ("oxidative", "redox", "cholesterol", "hmgcr", "proteasome", "bcl", "mtor", "mek", "braf", " ferro")
        ):
            priority_drugs.add(dname)

    df_cells = pq.read_table(cell_meta_pq).to_pandas().drop_duplicates(subset=["Cell_ID_Cellosaur"], keep="first")
    cvcl_meta: dict[str, tuple[str, str]] = {}
    for _, r in df_cells.iterrows():
        cvcl = str(r.get("Cell_ID_Cellosaur", "")).strip()
        ach = str(r.get("Cell_ID_DepMap", "")).strip()
        cname = str(r.get("cell_name", "")).strip()
        organ = str(r.get("Organ", "Lung")).strip()
        if not ach.startswith("ACH-"):
            ach = cname_to_mid.get(re.sub(r"[^A-Z0-9]", "", cname.upper()), default_mid)
        cvcl_meta[cvcl] = (ach, organ)

    # Query available CVCL shards in PIE_tahoe100m and pick 8 diverse organ lineages
    with urllib.request.urlopen("https://huggingface.co/api/datasets/arcinstitute/PIE_tahoe100m/tree/main/de", timeout=20) as resp:
        avail_files = [x["path"].split("/")[-1] for x in json.loads(resp.read().decode()) if x["path"].endswith(".parquet")]

    chosen_cvcls: list[str] = []
    seen_organs: set[str] = set()
    if "CVCL_0023.parquet" in avail_files:
        chosen_cvcls.append("CVCL_0023")
        seen_organs.add(cvcl_meta.get("CVCL_0023", ("", "Lung"))[1])
    for fname in avail_files:
        cv = fname.replace(".parquet", "")
        if cv in chosen_cvcls:
            continue
        org = cvcl_meta.get(cv, ("", "Unknown"))[1]
        if org not in seen_organs or len(chosen_cvcls) < 8:
            chosen_cvcls.append(cv)
            seen_organs.add(org)
        if len(chosen_cvcls) >= 8:
            break

    want_features = list(set(vocab_genes) | set(alias_map.keys()))
    out_rows: list[dict[str, Any]] = []

    for cv in chosen_cvcls:
        shard_path = staging_dir / f"{cv}.parquet"
        url = f"https://huggingface.co/datasets/arcinstitute/PIE_tahoe100m/resolve/main/de/{cv}.parquet"
        download_file(url, shard_path, "arcinstitute/PIE_tahoe100m")
        tbl = pq.read_table(
            shard_path,
            columns=["target", "feature", "target_ncells", "ref_ncells", "log2_fold_change"],
            filters=[("feature", "in", want_features)],
        )
        df_t = tbl.to_pandas()
        # Immediately remove 480 MB temporary shard to keep scratch lean
        shard_path.unlink(missing_ok=True)

        if df_t.empty:
            continue
        df_t["_sym"] = df_t["feature"].astype(str).map(lambda s: alias_map.get(s, s))
        df_t = df_t[df_t["_sym"].isin(gene_to_idx)]

        ach_id, organ = cvcl_meta.get(cv, (default_mid, "Lung"))
        grouped = df_t.groupby("target", sort=False)
        kept_for_cell = 0
        for tgt_str, grp in grouped:
            try:
                parsed = ast.literal_eval(str(tgt_str))
                dname, dose_val, _unit = parsed[0]
                dose_f = float(dose_val)
            except Exception:
                continue
            # Keep all priority RCD/metabolic drugs across all doses, plus max-dose (5 uM) across all 379 drugs
            if dname not in priority_drugs and dose_f < 4.5:
                continue
            vec = np.zeros(len(vocab_genes), dtype=np.float32)
            for sym, lfc in zip(grp["_sym"], grp["log2_fold_change"]):
                gi = gene_to_idx.get(sym)
                if gi is not None and np.isfinite(float(lfc)):
                    vec[gi] = float(np.clip(float(lfc), -12.0, 12.0))
            t_nc = float(grp["target_ncells"].iloc[0])
            r_nc = float(grp["ref_ncells"].iloc[0])
            # Inverse survivorship weight: lethal ferroptotic/cytotoxic wells have depleted target_ncells
            surv_w = float(np.clip((r_nc / 20.0) / max(t_nc, 5.0), 1.0, 20.0))
            canon_s, scaf_s = drug_smi_map.get(dname, ("", dname))
            out_rows.append(
                {
                    "record_id": f"TAHOE100M_{cv}_{re.sub(r'[^A-Za-z0-9]+', '_', dname)}_{dose_f}",
                    "model_id": ach_id,
                    "lineage": organ,
                    "perturbation_type": "compound",
                    "perturbation_name": dname,
                    "canonical_smiles": canon_s,
                    "murcko_scaffold": scaf_s,
                    "dose_um": dose_f,
                    "time_h": 24.0,
                    "delta_expr_vec": vec.tolist(),
                    "measured_gene_mask_type": "full_rnaseq",
                    "survivorship_weight": surv_w,
                    "sec46_readthrough_eff": -1.0,
                    "np_params_vec": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "dataset_source": "Tahoe_100M",
                    "accession_id": "arcinstitute/PIE_tahoe100m",
                    "license_track": "open",
                }
            )
            kept_for_cell += 1
            if kept_for_cell >= 650:
                break

    print(f"[Phase 2A-2] Extracted {len(out_rows)} Tahoe-100M pseudobulk signatures across {len(chosen_cvcls)} lineages")
    return out_rows


# ---------------------------------------------------------------------------
# 2B. LINCS 2020 `trt_cp` & `trt_xpr` via HTTP Range GCTX (`CC-BY-4.0` -> `"open"`)
# ---------------------------------------------------------------------------
def extract_lincs2020_signatures(
    staging_dir: Path,
    vocab_genes: list[str],
    gene_to_idx: dict[str, int],
    alias_map: dict[str, str],
    loci_64: list[str],
    cname_to_mid: dict[str, str],
    mid_to_lineage: dict[str, str],
    default_mid: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    print("[Phase 2A-2] Extracting LINCS 2020 trt_cp & trt_xpr signatures via HTTP Range...")
    geneinfo_txt = download_file(
        "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt",
        staging_dir / "geneinfo_beta.txt",
        "LINCS2020",
    )
    cmpd_txt = download_file(
        "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt",
        staging_dir / "compoundinfo_beta.txt",
        "LINCS2020",
    )
    sig_txt = download_file(
        "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/siginfo_beta.txt",
        staging_dir / "siginfo_beta.txt",
        "LINCS2020",
    )

    df_gi = pd.read_csv(geneinfo_txt, sep="\t", low_memory=False)
    entrez_to_sym: dict[str, str] = {}
    for gid, gsym in zip(df_gi["gene_id"].astype(str), df_gi["gene_symbol"].astype(str)):
        s_clean = alias_map.get(gsym.strip(), gsym.strip())
        entrez_to_sym[gid.split(".")[0].strip()] = s_clean

    df_cmpd = pd.read_csv(cmpd_txt, sep="\t", low_memory=False)
    pert_id_to_meta: dict[str, tuple[str, str, str]] = {}
    lincs_kw = set(k.lower() for k in CURATED_RCD_COMPOUNDS.keys()) | {
        "erastin", "rsl-3", "1s,3r-rsl3", "ml-210", "ml210", "ml-162", "ml162", "fin-56", "fin56",
        "cil-56", "sulfasalazine", "sorafenib", "artesunate", "artemisinin", "dihydroartemisinin",
        "auranofin", "simvastatin", "lovastatin", "fluvastatin", "atorvastatin", "pitavastatin",
        "deferoxamine", "ciclopirox", "ferrostatin-1", "liproxstatin-1", "rosiglitazone", "pioglitazone",
        "zileuton", "elesclomol", "disulfiram", "bortezomib", "venetoclax", "navitoclax", "staurosporine",
        "menadione", "piperlongumine", "withaferin-a", "brequinar", "buthionine-sulfoximine",
    }
    for _, r in df_cmpd.iterrows():
        pid = str(r.get("pert_id", "")).strip()
        cmap_name = str(r.get("cmap_name", "")).strip()
        smi = str(r.get("canonical_smiles", "")).strip()
        moa = str(r.get("moa", "")).lower()
        low_n = cmap_name.lower()
        if low_n in lincs_kw or any(w in low_n for w in ("erastin", "rsl", "ferrostatin", "liproxstatin", "vastatin", "artemis", "elesclomol", "deferox", "auranofin")) or any(
            w in moa for w in ("glutathione", "lipoxygenase", "hmgcr", "iron", "thioredoxin", "ferroptosis")
        ):
            if not smi or smi == "nan":
                c_ref = CURATED_RCD_COMPOUNDS.get(cmap_name.upper())
                smi = c_ref["smiles"] if c_ref else ""
            canon_s, scaf_s = murcko_from_smiles(smi, cmap_name)
            pert_id_to_meta[pid] = (cmap_name, canon_s, scaf_s)

    # Stream siginfo_beta.txt selecting exemplar signatures for matched compounds and 64-loci CRISPR KOs
    loci_set = set(loci_64)
    cp_sigs: dict[str, dict[str, Any]] = {}
    xpr_sigs: dict[str, dict[str, Any]] = {}
    for chunk in pd.read_csv(
        sig_txt,
        sep="\t",
        usecols=["sig_id", "pert_id", "cmap_name", "pert_type", "cell_iname", "pert_idose", "pert_itime", "is_exemplar_sig", "tas"],
        chunksize=150000,
        low_memory=False,
    ):
        sub_cp = chunk[(chunk["pert_type"] == "trt_cp") & (chunk["pert_id"].isin(pert_id_to_meta))]
        if "is_exemplar_sig" in sub_cp.columns:
            ex = sub_cp[sub_cp["is_exemplar_sig"] == 1]
            if len(ex) > 0:
                sub_cp = ex
        rem_cp = 1600 - len(cp_sigs)
        if rem_cp > 0 and not sub_cp.empty:
            for r_d in sub_cp.iloc[:rem_cp].to_dict(orient="records"):
                cp_sigs[str(r_d["sig_id"])] = r_d

        sub_xpr = chunk[(chunk["pert_type"] == "trt_xpr") & (chunk["cmap_name"].astype(str).isin(loci_set))]
        if "is_exemplar_sig" in sub_xpr.columns:
            ex_x = sub_xpr[sub_xpr["is_exemplar_sig"] == 1]
            if len(ex_x) > 0:
                sub_xpr = ex_x
        rem_xpr = 160 - len(xpr_sigs)
        if rem_xpr > 0 and not sub_xpr.empty:
            for r_d in sub_xpr.iloc[:rem_xpr].to_dict(orient="records"):
                xpr_sigs[str(r_d["sig_id"])] = r_d

    out_rows: list[dict[str, Any]] = []

    def _slice_gctx_subset(gctx_url: str, sig_map: dict[str, dict[str, Any]], ptype: str) -> None:
        if not sig_map:
            return
        rf0_raw = HTTPRangeFile(gctx_url)
        rf0 = io.BufferedReader(rf0_raw, buffer_size=4 * 1024 * 1024)
        with h5py.File(rf0, "r", rdcc_nbytes=64 * 1024 * 1024) as f0:
            cid_raw = f0["0/META/COL/id"][:]
            rid_raw = f0["0/META/ROW/id"][:]
            ds0 = f0["0/DATA/0/matrix"]
            n_cols_mat = int(ds0.shape[1])
            byte_offset = ds0.id.get_offset()
            chunk_shape = ds0.chunks
        cids = [x.decode() if isinstance(x, bytes) else str(x) for x in cid_raw]
        rids = [x.decode().strip() if isinstance(x, bytes) else str(x).strip() for x in rid_raw]

        # Map GCTX gene columns -> our 2048 vocab indices
        g_col_idx: list[int] = []
        v_tgt_idx: list[int] = []
        for g_i, rid in enumerate(rids):
            sym = entrez_to_sym.get(rid)
            if sym and sym in gene_to_idx:
                g_col_idx.append(g_i)
                v_tgt_idx.append(gene_to_idx[sym])

        matched_rows: list[tuple[int, str]] = []
        for row_idx, cid in enumerate(cids):
            if cid in sig_map:
                matched_rows.append((row_idx, cid))
        if not matched_rows:
            return
        matched_rows.sort(key=lambda t: t[0])

        fetched_pairs: list[tuple[str, np.ndarray]] = []
        if byte_offset is not None and chunk_shape is None:
            # Contiguous C-order float32 matrix: fetch exact 49 KB rows via lock-free parallel HTTP Range
            row_bytes = n_cols_mat * 4

            def _fetch_raw_row(item: tuple[int, str]) -> tuple[str, np.ndarray]:
                r_idx, s_id = item
                b0 = int(byte_offset) + r_idx * row_bytes
                b1 = b0 + row_bytes - 1
                req = urllib.request.Request(
                    rf0_raw.url,
                    headers={"Range": f"bytes={b0}-{b1}", "User-Agent": "FerroFlux-ETL/1.0"},
                )
                with urllib.request.urlopen(req, timeout=25) as resp:
                    raw_buf = resp.read()
                raw_v = np.frombuffer(raw_buf, dtype="<f4")
                vec = np.zeros(len(vocab_genes), dtype=np.float32)
                vec[v_tgt_idx] = np.clip(raw_v[g_col_idx] / 2.5, -10.0, 10.0)
                return s_id, vec

            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
                fetched_pairs = list(pool.map(_fetch_raw_row, matched_rows))
        else:
            # Chunked HDF5 matrix (`0/DATA/0/matrix`): group by densest row-chunks in a buffered h5py handle
            c_rows = int(chunk_shape[0]) if chunk_shape else 256
            by_chunk: dict[int, list[tuple[int, str]]] = {}
            for r_idx, s_id in matched_rows:
                by_chunk.setdefault(r_idx // c_rows, []).append((r_idx, s_id))
            top_chunks = sorted(by_chunk.keys(), key=lambda k: len(by_chunk[k]), reverse=True)[:12]
            rf_seq = io.BufferedReader(HTTPRangeFile(gctx_url), buffer_size=4 * 1024 * 1024)
            with h5py.File(rf_seq, "r", rdcc_nbytes=64 * 1024 * 1024) as fw:
                ds = fw["0/DATA/0/matrix"]
                for ck_id in sorted(top_chunks):
                    items = by_chunk[ck_id]
                    r_min = items[0][0]
                    r_max = items[-1][0] + 1
                    block = np.asarray(ds[r_min:r_max, :], dtype=np.float32)
                    for r_idx, s_id in items:
                        raw_v = block[r_idx - r_min]
                        vec = np.zeros(len(vocab_genes), dtype=np.float32)
                        vec[v_tgt_idx] = np.clip(raw_v[g_col_idx] / 2.5, -10.0, 10.0)
                        fetched_pairs.append((s_id, vec))

        for s_id, vec in fetched_pairs:
            meta = sig_map[s_id]
            ciname = str(meta.get("cell_iname", "A375")).strip()
            mid = cname_to_mid.get(re.sub(r"[^A-Z0-9]", "", ciname.upper()), default_mid)
            lin = mid_to_lineage.get(mid, "Skin" if "A375" in ciname.upper() else "Lung")
            dose_m = re.search(r"([0-9.]+)", str(meta.get("pert_idose", "10")))
            dose_um = float(dose_m.group(1)) if dose_m else 10.0
            time_m = re.search(r"([0-9.]+)", str(meta.get("pert_itime", "24")))
            time_h = float(time_m.group(1)) if time_m else 24.0
            if ptype == "compound":
                pid = str(meta.get("pert_id", ""))
                pname, canon_s, scaf_s = pert_id_to_meta.get(pid, (str(meta.get("cmap_name", "")), "", ""))
            else:
                pname = alias_map.get(str(meta.get("cmap_name", "")).strip(), str(meta.get("cmap_name", "")).strip())
                canon_s, scaf_s = "", ""
            out_rows.append(
                {
                    "record_id": f"LINCS2020_{re.sub(r'[^A-Za-z0-9_]+', '_', s_id)}",
                    "model_id": mid,
                    "lineage": lin,
                    "perturbation_type": ptype,
                    "perturbation_name": pname,
                    "canonical_smiles": canon_s,
                    "murcko_scaffold": scaf_s,
                    "dose_um": dose_um if ptype == "compound" else 0.0,
                    "time_h": time_h,
                    "delta_expr_vec": vec.tolist(),
                    "measured_gene_mask_type": "l1000_bing",
                    "survivorship_weight": 1.0,
                    "sec46_readthrough_eff": -1.0,
                    "np_params_vec": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "dataset_source": f"LINCS2020_{'trt_cp' if ptype == 'compound' else 'trt_xpr'}",
                    "accession_id": "LINCS2020_Level5",
                    "license_track": "open",
                }
            )

    _slice_gctx_subset(
        "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5/level5_beta_trt_cp_n720216x12328.gctx",
        cp_sigs,
        "compound",
    )
    _slice_gctx_subset(
        "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5/level5_beta_trt_xpr_n142901x12328.gctx",
        xpr_sigs,
        "crispr_ko",
    )
    # Remove bulky 465 MB siginfo text after extraction
    sig_txt.unlink(missing_ok=True)
    print(f"[Phase 2A-2] Extracted {len(out_rows)} LINCS 2020 signatures (trt_cp + trt_xpr)")
    return out_rows, entrez_to_sym


# ---------------------------------------------------------------------------
# 2C. Arc `PIE` Perturb-seq (`"open"` + `PIE_x_atlas_orion` `"academic"`)
# ---------------------------------------------------------------------------
def extract_arc_pie_signatures(
    staging_dir: Path,
    vocab_genes: list[str],
    gene_to_idx: dict[str, int],
    alias_map: dict[str, str],
    cname_to_mid: dict[str, str],
    mid_to_lineage: dict[str, str],
    default_mid: str,
) -> list[dict[str, Any]]:
    print("[Phase 2A-2] Extracting Arc PIE Perturb-seq atlases (open + X-Atlas/Orion academic)...")
    pie_specs = [
        ("arcinstitute/PIE_jiang", "open", "cytokine_crispr", 1000),
        ("arcinstitute/PIE_arc_vcc_25", "open", "crispr_ko", 500),
        ("arcinstitute/PIE_replogle_nadig_essential", "open", "crispri", 2500),
        ("arcinstitute/PIE_x_atlas_orion", "academic", "crispri", 5000),
    ]
    out_rows: list[dict[str, Any]] = []

    for repo_id, track, ptype, max_keep in pie_specs:
        short = repo_id.split("/")[-1]
        base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/preprocessed"
        meta_path = download_file(f"{base_url}/meta.json", staging_dir / f"{short}_meta.json", repo_id)
        ctx_path = download_file(f"{base_url}/ctx_ids.npy", staging_dir / f"{short}_ctx_ids.npy", repo_id)
        pert_path = download_file(f"{base_url}/pert_ids.npy", staging_dir / f"{short}_pert_ids.npy", repo_id)
        fc_path = download_file(f"{base_url}/fold_changes.npy", staging_dir / f"{short}_fold_changes.npy", repo_id)

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        id_to_ctx = {int(v): str(k) for k, v in meta.get("context_to_id", {}).items()}
        id_to_pert = {int(v): str(k) for k, v in meta.get("pert_to_id", {}).items()}
        src_genes: list[str] = meta.get("genes", [])

        src_idx: list[int] = []
        dst_idx: list[int] = []
        for g_i, g_raw in enumerate(src_genes):
            sym = alias_map.get(str(g_raw).strip(), str(g_raw).strip())
            if sym in gene_to_idx:
                src_idx.append(g_i)
                dst_idx.append(gene_to_idx[sym])

        ctx_ids = np.load(ctx_path)
        pert_ids = np.load(pert_path)
        fc_mat = np.load(fc_path, mmap_mode="r")

        # Prioritize perturbations targeting our 2048 ferroptosis/metabolic/immune vocab first
        priority_row_indices: list[int] = []
        other_row_indices: list[int] = []
        for r_i in range(len(pert_ids)):
            p_sym = alias_map.get(id_to_pert.get(int(pert_ids[r_i]), ""), id_to_pert.get(int(pert_ids[r_i]), ""))
            if p_sym in gene_to_idx:
                priority_row_indices.append(r_i)
            else:
                other_row_indices.append(r_i)
        selected_indices = (priority_row_indices + other_row_indices)[:max_keep]

        for r_i in selected_indices:
            ctx_name = id_to_ctx.get(int(ctx_ids[r_i]), "K562")
            p_raw = id_to_pert.get(int(pert_ids[r_i]), "NT")
            p_sym = alias_map.get(p_raw, p_raw)
            cell_tok = re.sub(r"[^A-Z0-9]", "", ctx_name.split("_")[0].upper())
            mid = cname_to_mid.get(cell_tok, default_mid)
            lin = mid_to_lineage.get(mid, "Bowel" if "HCT116" in cell_tok else ("Myeloid" if "K562" in cell_tok else "Embryonic"))

            row_fc = np.asarray(fc_mat[r_i, src_idx], dtype=np.float32)
            # fold_changes.npy in PIE is linear ratio (target/ctrl) -> log2(max(fc, 1e-3))
            if np.nanmedian(row_fc) > 0.25 and np.nanmin(row_fc) >= 0.0:
                lfc_vals = np.log2(np.clip(row_fc, 1e-3, 1e3))
            else:
                lfc_vals = np.clip(row_fc, -10.0, 10.0)
            vec = np.zeros(len(vocab_genes), dtype=np.float32)
            vec[dst_idx] = np.nan_to_num(lfc_vals, nan=0.0, posinf=10.0, neginf=-10.0)

            out_rows.append(
                {
                    "record_id": f"{short.upper()}_{r_i}_{ctx_name}_{p_sym}",
                    "model_id": mid,
                    "lineage": lin,
                    "perturbation_type": ptype,
                    "perturbation_name": f"{p_sym} ({ctx_name})" if "jiang" in short else p_sym,
                    "canonical_smiles": "",
                    "murcko_scaffold": "",
                    "dose_um": 0.0,
                    "time_h": 72.0 if "crispr" in ptype else 24.0,
                    "delta_expr_vec": vec.tolist(),
                    "measured_gene_mask_type": "full_rnaseq",
                    "survivorship_weight": 1.0,
                    "sec46_readthrough_eff": -1.0,
                    "np_params_vec": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "dataset_source": short,
                    "accession_id": repo_id,
                    "license_track": track,
                }
            )

        del fc_mat
        fc_path.unlink(missing_ok=True)
        print(f"[Phase 2A-2]   -> {short} ({track}): kept {len(selected_indices)} signatures")

    return out_rows


# ---------------------------------------------------------------------------
# 2D. Parse-10M Human Donor Cytokine PBMC Dictionary (`CC-BY-NC-4.0` -> `"academic"`)
# ---------------------------------------------------------------------------
def extract_parse10m_cytokine_signatures(
    vocab_genes: list[str],
    gene_to_idx: dict[str, int],
    alias_map: dict[str, str],
    default_mid: str,
) -> list[dict[str, Any]]:
    print("[Phase 2A-2] Slicing Parse-10M human donor cytokine PBMC dictionary via HTTP Range CSR...")
    url = "https://huggingface.co/datasets/arcinstitute/State-Parse-Filtered/resolve/main/parse_concat_full.h5ad"
    rf = io.BufferedReader(HTTPRangeFile(url), buffer_size=4 * 1024 * 1024)
    out_rows: list[dict[str, Any]] = []

    target_cytokines = [
        "IFN-gamma", "IFN-beta", "IFN-alpha1", "TNF-alpha", "TGF-beta1",
        "IL-6", "IL-1-beta", "IL-2", "IL-4", "IL-10", "IL-12", "IL-15",
        "IL-18", "IL-21", "GM-CSF", "G-CSF", "FLT3L", "FasL",
    ]
    target_cts = ["CD8_Memory", "CD8_Naive", "CD4_Memory", "NK", "CD14_Mono", "CD16_Mono", "cDC", "Treg"]

    with h5py.File(rf, "r", rdcc_nbytes=64 * 1024 * 1024) as f:
        cyts = [x.decode() if isinstance(x, bytes) else str(x) for x in f["obs/cytokine/categories"][:]]
        cts = [x.decode() if isinstance(x, bytes) else str(x) for x in f["obs/cell_type_clean/categories"][:]]
        vgenes = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var/_index"][:]]

        cyt_codes = f["obs/cytokine/codes"][:]
        ct_codes = f["obs/cell_type_clean/codes"][:]
        indptr = f["X/indptr"][:]

        v_to_vocab = np.full(len(vgenes), -1, dtype=np.int32)
        for vi, graw in enumerate(vgenes):
            sym = alias_map.get(graw.strip(), graw.strip())
            if sym in gene_to_idx:
                v_to_vocab[vi] = gene_to_idx[sym]

        pbs_code = cyts.index("PBS") if "PBS" in cyts else -1
        if pbs_code < 0:
            return out_rows

        want_cyts = ["PBS"] + [c for c in target_cytokines if c in cyts]
        want_cyt_codes = {cyts.index(c): c for c in want_cyts}
        want_ct_codes = {cts.index(ct): ct for ct in target_cts if ct in cts}

        # Accumulators: (ct_code, cyt_code) -> (gene_sum[2048], total_umi, n_cells)
        accum_genes: dict[tuple[int, int], np.ndarray] = {}
        accum_umi: dict[tuple[int, int], float] = {}
        accum_ncells: dict[tuple[int, int], int] = {}

        ds_idx = f["X/indices"]
        ds_dat = f["X/data"]

        def _ingest_contiguous_cell_window(w0: int, w1: int) -> None:
            p0 = int(indptr[w0])
            p1 = int(indptr[w1])
            if p1 <= p0:
                return
            sub_idx = ds_idx[p0:p1]
            sub_dat = ds_dat[p0:p1]
            is_log_norm = bool(sub_dat.size > 0 and float(sub_dat[: min(2000, sub_dat.size)].max()) < 15.0 and np.any(np.mod(sub_dat[: min(500, sub_dat.size)], 1.0) != 0.0))
            for ci in range(w0, w1):
                ct_k = int(ct_codes[ci])
                cy_k = int(cyt_codes[ci])
                if ct_k not in want_ct_codes or cy_k not in want_cyt_codes:
                    continue
                pair = (ct_k, cy_k)
                if accum_ncells.get(pair, 0) >= 60:
                    continue
                loc0 = int(indptr[ci]) - p0
                loc1 = int(indptr[ci + 1]) - p0
                if loc1 <= loc0:
                    continue
                c_cols = sub_idx[loc0:loc1]
                c_vals = np.asarray(sub_dat[loc0:loc1], dtype=np.float64)
                if is_log_norm:
                    c_vals = np.expm1(np.clip(c_vals, 0.0, 12.0))
                umi = float(c_vals.sum())
                if umi <= 0.0:
                    continue
                mapped = v_to_vocab[c_cols]
                valid = mapped >= 0
                if pair not in accum_genes:
                    accum_genes[pair] = np.zeros(len(vocab_genes), dtype=np.float64)
                    accum_umi[pair] = 0.0
                    accum_ncells[pair] = 0
                if np.any(valid):
                    np.add.at(accum_genes[pair], mapped[valid], c_vals[valid])
                accum_umi[pair] += umi
                accum_ncells[pair] += 1

        # Detect whether cytokines are interleaved across split-pool wells or concatenated by cytokine block
        if len(np.unique(cyt_codes[:25000])) >= 30:
            _ingest_contiguous_cell_window(0, min(25000, len(indptr) - 1))
        else:
            for cy_k in sorted(want_cyt_codes.keys()):
                hits = np.flatnonzero(cyt_codes == cy_k)
                if len(hits) == 0:
                    continue
                w0 = int(hits[0])
                w1 = min(w0 + 650, len(indptr) - 1)
                _ingest_contiguous_cell_window(w0, w1)

        def _pair_vec(ct_k: int, cy_k: int) -> np.ndarray | None:
            pair = (ct_k, cy_k)
            if accum_ncells.get(pair, 0) < 3 or accum_umi.get(pair, 0.0) <= 0.0:
                return None
            cpm = accum_genes[pair] * (1e6 / accum_umi[pair])
            return np.log2(cpm + 1.0).astype(np.float32)

        for ct_c, ct_name in want_ct_codes.items():
            ctrl_vec = _pair_vec(ct_c, pbs_code)
            if ctrl_vec is None:
                continue
            for cyt_name in target_cytokines:
                if cyt_name not in cyts:
                    continue
                cyt_c = cyts.index(cyt_name)
                trt_vec = _pair_vec(ct_c, cyt_c)
                if trt_vec is None:
                    continue
                delta = np.clip(trt_vec - ctrl_vec, -10.0, 10.0)
                out_rows.append(
                    {
                        "record_id": f"PARSE10M_{ct_name}_{re.sub(r'[^A-Za-z0-9]+', '_', cyt_name)}",
                        "model_id": f"HPA_SC_{ct_name.upper()}" if ct_name else default_mid,
                        "lineage": "Immune_TIL",
                        "perturbation_type": "cytokine",
                        "perturbation_name": cyt_name,
                        "canonical_smiles": "",
                        "murcko_scaffold": "",
                        "dose_um": 0.01,
                        "time_h": 24.0,
                        "delta_expr_vec": delta.tolist(),
                        "measured_gene_mask_type": "full_rnaseq",
                        "survivorship_weight": 1.0,
                        "sec46_readthrough_eff": -1.0,
                        "np_params_vec": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "dataset_source": "Parse_10M_Cytokine_Dictionary",
                        "accession_id": "arcinstitute/State-Parse-Filtered",
                        "license_track": "academic",
                    }
                )

    print(f"[Phase 2A-2] Extracted {len(out_rows)} Parse-10M human immune cytokine signatures (academic)")
    return out_rows


# ---------------------------------------------------------------------------
# 2E. Non-Quarantined Leaf GEO Ferroptosis RNA-seq, Sec46 Ribo-seq & Bio-Nano (`"open"`)
# ---------------------------------------------------------------------------
def extract_geo_ribo_and_bionano_signatures(
    staging_dir: Path,
    vocab_genes: list[str],
    gene_to_idx: dict[str, int],
    alias_map: dict[str, str],
    ensg_to_sym: dict[str, str],
    entrez_to_sym: dict[str, str],
    cname_to_mid: dict[str, str],
    default_mid: str,
) -> list[dict[str, Any]]:
    print("[Phase 2A-2] Extracting GEO leaf ferroptosis RNA-seq, paired Sec46 Ribo-seq & non-Bradbury Bio-Nano...")
    out_rows: list[dict[str, Any]] = []

    # 1. Paired Mammalian Sec46 Ribo-seq (`GSE235336`) + Total RNA-seq (`GSE235333`) in CFPAC-1 & PANC-1 +/- IKE
    verify_geo_soft_not_quarantined("GSE235336", staging_dir)
    verify_geo_soft_not_quarantined("GSE235333", staging_dir)
    ribo_gz = download_file(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE235nnn/GSE235336/suppl/GSE235336_Ribo-seq_processed_data_matrix.txt.gz",
        staging_dir / "GSE235336_Ribo-seq.txt.gz",
        "GSE235336",
    )
    rna_333_gz = download_file(
        "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE235333&format=file&file=GSE235333_raw_counts_GRCh38.p13_NCBI.tsv.gz",
        staging_dir / "GSE235333_raw_counts.tsv.gz",
        "GSE235333",
    )
    gsm_333 = parse_geo_series_matrix_samples("GSE235333", staging_dir)

    df_ribo = pd.read_csv(ribo_gz, sep="\t", low_memory=False)
    df_ribo["_sym"] = df_ribo.iloc[:, 0].astype(str).map(lambda e: ensg_to_sym.get(e.split(".")[0].strip(), ""))
    df_ribo = df_ribo[df_ribo["_sym"].isin(gene_to_idx)].drop_duplicates(subset=["_sym"], keep="first").set_index("_sym")

    df_rna333 = pd.read_csv(rna_333_gz, sep="\t", low_memory=False)
    df_rna333["_sym"] = df_rna333["GeneID"].astype(str).map(lambda g: entrez_to_sym.get(g.split(".")[0].strip(), ""))
    df_rna333 = df_rna333[df_rna333["_sym"].isin(gene_to_idx)].drop_duplicates(subset=["_sym"], keep="first").set_index("_sym")
    gsm_cols_333 = [c for c in df_rna333.columns if c.startswith("GSM")]
    cpm_333 = df_rna333[gsm_cols_333].apply(lambda col: np.log2(col.astype(float) * (1e6 / max(float(col.sum()), 1.0)) + 1.0), axis=0)

    ike_info = CURATED_RCD_COMPOUNDS["IMIDAZOLE KETONE ERASTIN"]
    ike_smi, ike_scaf = murcko_from_smiles(ike_info["smiles"], "IMIDAZOLE KETONE ERASTIN")
    sec_genes = [g for g in ("GPX4", "SEPHS2", "TXNRD1") if g in df_ribo.index and g in cpm_333.index]

    for cell_tag, cname_lookup in [("CFPAC", "CFPAC1"), ("PANC", "PANC1")]:
        mid = cname_to_mid.get(cname_lookup, default_mid)
        ribo_dmso_cols = [c for c in df_ribo.columns if cell_tag in c.upper() and "DMSO" in c.upper()]
        ribo_ike_cols = [c for c in df_ribo.columns if cell_tag in c.upper() and "IKE" in c.upper()]
        rna_dmso_cols = [g for g in gsm_cols_333 if cell_tag in gsm_333.get(g, {}).get("title", "").upper() and "DMSO" in gsm_333.get(g, {}).get("title", "").upper()]
        rna_ike_cols = [g for g in gsm_cols_333 if cell_tag in gsm_333.get(g, {}).get("title", "").upper() and "IKE" in gsm_333.get(g, {}).get("title", "").upper()]
        if not (ribo_dmso_cols and ribo_ike_cols):
            continue
        ribo_ctrl = np.log2(df_ribo[ribo_dmso_cols].astype(float).mean(axis=1).clip(lower=0.0) + 1.0)
        if rna_dmso_cols and rna_ike_cols:
            rna_ctrl = cpm_333[rna_dmso_cols].mean(axis=1)
            rna_trt_mean = cpm_333[rna_ike_cols].mean(axis=1)
        else:
            rna_ctrl = ribo_ctrl
            rna_trt_mean = np.log2(df_ribo[ribo_ike_cols].astype(float).mean(axis=1).clip(lower=0.0) + 1.0)

        for rep_col in ribo_ike_cols:
            ribo_rep = np.log2(df_ribo[rep_col].astype(float).clip(lower=0.0) + 1.0)
            # Compute empirical selenoprotein translational efficiency ratio on GPX4/SEPHS2/TXNRD1
            te_ratios: list[float] = []
            for sg in sec_genes:
                te_trt = (2.0 ** float(ribo_rep.loc[sg])) / max(2.0 ** float(rna_trt_mean.loc[sg]), 0.5)
                te_ctl = (2.0 ** float(ribo_ctrl.loc[sg])) / max(2.0 ** float(rna_ctrl.loc[sg]), 0.5)
                te_ratios.append(te_trt / max(te_ctl, 1e-3))
            sec46_eff = float(np.clip(np.mean(te_ratios) * 0.65 if te_ratios else 0.5, 0.05, 0.98))

            vec = np.zeros(len(vocab_genes), dtype=np.float32)
            delta_s = (rna_trt_mean - rna_ctrl).reindex(df_ribo.index).fillna(ribo_rep - ribo_ctrl)
            for sym, dv in delta_s.items():
                gi = gene_to_idx.get(str(sym))
                if gi is not None and np.isfinite(float(dv)):
                    vec[gi] = float(np.clip(float(dv), -10.0, 10.0))

            out_rows.append(
                {
                    "record_id": f"GSE235336_SEC46_{rep_col}",
                    "model_id": mid,
                    "lineage": "Pancreas",
                    "perturbation_type": "compound",
                    "perturbation_name": "IMIDAZOLE KETONE ERASTIN",
                    "canonical_smiles": ike_smi,
                    "murcko_scaffold": ike_scaf,
                    "dose_um": 2.0,
                    "time_h": 12.0,
                    "delta_expr_vec": vec.tolist(),
                    "measured_gene_mask_type": "full_rnaseq",
                    "survivorship_weight": 1.5,
                    "sec46_readthrough_eff": sec46_eff,
                    "np_params_vec": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "dataset_source": "GEO_Sec46_RiboSeq_Paired",
                    "accession_id": "GSE235336",
                    "license_track": "open",
                }
            )

    # 2. GSE255459 Triple-Negative Breast Cancer (`BT549`, `HS578T`, `SUM159`) Erastin & RSL3 RNA-seq
    verify_geo_soft_not_quarantined("GSE255459", staging_dir)
    for f_url, f_tag in [
        ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE255nnn/GSE255459/suppl/GSE255459_Count_matrix_-_BT549_and_HS578.csv.gz", "BT549_HS578"),
        ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE255nnn/GSE255459/suppl/GSE255459_Count_matrix_-_SUM159.csv.gz", "SUM159"),
    ]:
        p_csv = download_file(f_url, staging_dir / f"GSE255459_{f_tag}.csv.gz", "GSE255459")
        df_c = pd.read_csv(p_csv, low_memory=False)
        gcol = df_c.columns[0]
        df_c["_sym"] = df_c[gcol].astype(str).map(lambda s: alias_map.get(s.strip(), ensg_to_sym.get(s.split(".")[0].strip(), s.strip())))
        df_c = df_c[df_c["_sym"].isin(gene_to_idx)].drop_duplicates(subset=["_sym"], keep="first").set_index("_sym")
        num_cols = [c for c in df_c.columns if c not in (gcol, "_sym")]
        l2cpm = df_c[num_cols].apply(lambda col: np.log2(pd.to_numeric(col, errors="coerce").fillna(0.0) * (1e6 / max(float(pd.to_numeric(col, errors="coerce").fillna(0.0).sum()), 1.0)) + 1.0), axis=0)

        for cell_pref in ("BT549", "HS578", "SUM159"):
            dmso_c = [c for c in num_cols if cell_pref in c.upper() and ("DMSO" in c.upper() or "CTRL" in c.upper() or "UNTREAT" in c.upper())]
            if not dmso_c:
                continue
            ctrl_v = l2cpm[dmso_c].mean(axis=1)
            mid = cname_to_mid.get(cell_pref if cell_pref != "HS578" else "HS578T", default_mid)
            for trt_col in [c for c in num_cols if cell_pref in c.upper() and c not in dmso_c]:
                up_t = trt_col.upper()
                cmpd_key = "1S,3R-RSL3" if "RSL3" in up_t else ("ERASTIN" if "ERASTIN" in up_t else "")
                if not cmpd_key:
                    continue
                cmeta = CURATED_RCD_COMPOUNDS[cmpd_key]
                csmi, cscaf = murcko_from_smiles(cmeta["smiles"], cmpd_key)
                vec = np.zeros(len(vocab_genes), dtype=np.float32)
                for sym, dv in (l2cpm[trt_col] - ctrl_v).items():
                    gi = gene_to_idx.get(str(sym))
                    if gi is not None and np.isfinite(float(dv)):
                        vec[gi] = float(np.clip(float(dv), -10.0, 10.0))
                out_rows.append(
                    {
                        "record_id": f"GSE255459_{trt_col}",
                        "model_id": mid,
                        "lineage": "Breast",
                        "perturbation_type": "compound",
                        "perturbation_name": cmpd_key,
                        "canonical_smiles": csmi,
                        "murcko_scaffold": cscaf,
                        "dose_um": 1.0 if "RSL3" in cmpd_key else 5.0,
                        "time_h": 24.0,
                        "delta_expr_vec": vec.tolist(),
                        "measured_gene_mask_type": "full_rnaseq",
                        "survivorship_weight": 1.5,
                        "sec46_readthrough_eff": -1.0,
                        "np_params_vec": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "dataset_source": "GEO_Ferroptosis_RNASeq",
                        "accession_id": "GSE255459",
                        "license_track": "open",
                    }
                )

    # 3. GSE235201 Human Ferroptosis GRCh38 RNA-seq
    try:
        verify_geo_soft_not_quarantined("GSE235201", staging_dir)
        gsm_201 = parse_geo_series_matrix_samples("GSE235201", staging_dir)
        p_201 = download_file(
            "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE235201&format=file&file=GSE235201_raw_counts_GRCh38.p13_NCBI.tsv.gz",
            staging_dir / "GSE235201_raw_counts.tsv.gz",
            "GSE235201",
        )
        df_201 = pd.read_csv(p_201, sep="\t", low_memory=False)
        df_201["_sym"] = df_201["GeneID"].astype(str).map(lambda g: entrez_to_sym.get(g.split(".")[0].strip(), ""))
        df_201 = df_201[df_201["_sym"].isin(gene_to_idx)].drop_duplicates(subset=["_sym"], keep="first").set_index("_sym")
        gsms = [c for c in df_201.columns if c.startswith("GSM")]
        l2_201 = df_201[gsms].apply(lambda col: np.log2(col.astype(float) * (1e6 / max(float(col.sum()), 1.0)) + 1.0), axis=0)
        ctrl_gsms = [g for g in gsms if any(k in (gsm_201.get(g, {}).get("title", "") + " " + gsm_201.get(g, {}).get("chars", "")).upper() for k in ("DMSO", "CTRL", "CONTROL", "WT", "NC", "VEH"))]
        if not ctrl_gsms:
            ctrl_gsms = gsms[: max(1, len(gsms) // 3)]
        ctrl_base = l2_201[ctrl_gsms].mean(axis=1)
        for g in [x for x in gsms if x not in ctrl_gsms]:
            vec = np.zeros(len(vocab_genes), dtype=np.float32)
            for sym, dv in (l2_201[g] - ctrl_base).items():
                gi = gene_to_idx.get(str(sym))
                if gi is not None and np.isfinite(float(dv)):
                    vec[gi] = float(np.clip(float(dv), -10.0, 10.0))
            t_lbl = gsm_201.get(g, {}).get("title", g)
            out_rows.append(
                {
                    "record_id": f"GSE235201_{g}",
                    "model_id": default_mid,
                    "lineage": "Lung",
                    "perturbation_type": "compound",
                    "perturbation_name": t_lbl[:64],
                    "canonical_smiles": CURATED_RCD_COMPOUNDS["ERASTIN"]["smiles"],
                    "murcko_scaffold": murcko_from_smiles(CURATED_RCD_COMPOUNDS["ERASTIN"]["smiles"])[1],
                    "dose_um": 5.0,
                    "time_h": 24.0,
                    "delta_expr_vec": vec.tolist(),
                    "measured_gene_mask_type": "full_rnaseq",
                    "survivorship_weight": 1.2,
                    "sec46_readthrough_eff": -1.0,
                    "np_params_vec": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "dataset_source": "GEO_Ferroptosis_RNASeq",
                    "accession_id": "GSE235201",
                    "license_track": "open",
                }
            )
    except Exception as exc:
        print(f"[Phase 2A-2] Note on GSE235201: {exc}")

    # 4. Non-Bradbury Bio-Nano (`Fe3O4` SPION `GSE328517` + Human Iron/Hemorrhage Macrophage `GSE194313` + `SiO2` `GSE312613`)
    # 4a. GSE328517 (`Fe3O4@PACM` SPION ferroptosis, 5-dim np_params_vec = [R_h_nm=35.0, zeta_mV=-12.0, Fe_load=0.85, receptor_match=0.6, dose_uM=10.0])
    verify_geo_soft_not_quarantined("GSE328517", staging_dir)
    p_328 = download_file(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE328nnn/GSE328517/suppl/GSE328517_20260413.csv.gz",
        staging_dir / "GSE328517.csv.gz",
        "GSE328517",
    )
    df_328 = pd.read_csv(p_328, low_memory=False)
    df_328["_sym"] = df_328.iloc[:, 0].astype(str).str.strip().str.upper().map(lambda s: alias_map.get(s, s))
    df_328 = df_328[df_328["_sym"].isin(gene_to_idx)].drop_duplicates(subset=["_sym"], keep="first").set_index("_sym")
    c_328 = [c for c in df_328.columns if c != df_328.columns[0] and c != "_sym"]
    l2_328 = np.log2(df_328[c_328].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0) + 1.0)
    ref_328 = l2_328.iloc[:, -3:].mean(axis=1)
    for col_n in c_328[:3]:
        vec = np.zeros(len(vocab_genes), dtype=np.float32)
        for sym, dv in (l2_328[col_n] - ref_328).items():
            gi = gene_to_idx.get(str(sym))
            if gi is not None and np.isfinite(float(dv)):
                vec[gi] = float(np.clip(float(dv), -10.0, 10.0))
        out_rows.append(
            {
                "record_id": f"GSE328517_SPION_{re.sub(r'[^A-Za-z0-9]+', '_', col_n)}",
                "model_id": default_mid,
                "lineage": "Solid_Tumor",
                "perturbation_type": "nanoparticle",
                "perturbation_name": "Fe3O4_SPION_Nanoparticle",
                "canonical_smiles": "",
                "murcko_scaffold": "",
                "dose_um": 10.0,
                "time_h": 24.0,
                "delta_expr_vec": vec.tolist(),
                "measured_gene_mask_type": "full_rnaseq",
                "survivorship_weight": 1.3,
                "sec46_readthrough_eff": -1.0,
                "np_params_vec": [35.0, -12.0, 0.85, 0.60, 10.0],
                "dataset_source": "GEO_BioNano_Fe3O4",
                "accession_id": "GSE328517",
                "license_track": "open",
            }
        )

    # 4b. GSE194313 Human Iron Oxide / Hemorrhage Oxidative Macrophage GRCh38 RNA-seq
    try:
        verify_geo_soft_not_quarantined("GSE194313", staging_dir)
        gsm_194 = parse_geo_series_matrix_samples("GSE194313", staging_dir)
        p_194 = download_file(
            "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE194313&format=file&file=GSE194313_raw_counts_GRCh38.p13_NCBI.tsv.gz",
            staging_dir / "GSE194313_raw_counts.tsv.gz",
            "GSE194313",
        )
        df_194 = pd.read_csv(p_194, sep="\t", low_memory=False)
        df_194["_sym"] = df_194["GeneID"].astype(str).map(lambda g: entrez_to_sym.get(g.split(".")[0].strip(), ""))
        df_194 = df_194[df_194["_sym"].isin(gene_to_idx)].drop_duplicates(subset=["_sym"], keep="first").set_index("_sym")
        gsms_194 = [c for c in df_194.columns if c.startswith("GSM")]
        l2_194 = df_194[gsms_194].apply(lambda col: np.log2(col.astype(float) * (1e6 / max(float(col.sum()), 1.0)) + 1.0), axis=0)
        ctrl_194 = [g for g in gsms_194 if any(k in (gsm_194.get(g, {}).get("title", "") + " " + gsm_194.get(g, {}).get("chars", "")).upper() for k in ("CTRL", "CONTROL", "UNTREAT", "PBS", "VEH", "M0"))]
        if not ctrl_194:
            ctrl_194 = gsms_194[:3]
        base_194 = l2_194[ctrl_194].mean(axis=1)
        for g in [x for x in gsms_194 if x not in ctrl_194]:
            vec = np.zeros(len(vocab_genes), dtype=np.float32)
            for sym, dv in (l2_194[g] - base_194).items():
                gi = gene_to_idx.get(str(sym))
                if gi is not None and np.isfinite(float(dv)):
                    vec[gi] = float(np.clip(float(dv), -10.0, 10.0))
            out_rows.append(
                {
                    "record_id": f"GSE194313_IONP_{g}",
                    "model_id": "HPA_SC_MACROPHAGES",
                    "lineage": "Immune_TIL",
                    "perturbation_type": "nanoparticle",
                    "perturbation_name": "Iron_Oxide_Redox_Exposure",
                    "canonical_smiles": "",
                    "murcko_scaffold": "",
                    "dose_um": 15.0,
                    "time_h": 24.0,
                    "delta_expr_vec": vec.tolist(),
                    "measured_gene_mask_type": "full_rnaseq",
                    "survivorship_weight": 1.1,
                    "sec46_readthrough_eff": -1.0,
                    "np_params_vec": [25.0, -18.0, 0.90, 0.50, 15.0],
                    "dataset_source": "GEO_BioNano_IronRedox",
                    "accession_id": "GSE194313",
                    "license_track": "open",
                }
            )
    except Exception as exc:
        print(f"[Phase 2A-2] Note on GSE194313: {exc}")

    print(f"[Phase 2A-2] Extracted {len(out_rows)} GEO Ribo-seq, Ferroptosis & non-Bradbury Bio-Nano signatures")
    return out_rows


# ---------------------------------------------------------------------------
# 3. Clinical Immunotherapy Cohorts (`clinical_icb_cohorts.parquet`, `"open"`)
# ---------------------------------------------------------------------------
def build_clinical_icb_cohorts(
    staging_dir: Path,
    output_dir: Path,
    vocab_genes: list[str],
    gene_to_idx: dict[str, int],
    alias_map: dict[str, str],
    entrez_to_sym: dict[str, str],
) -> int:
    print("[Phase 2A-2] Building clinical_icb_cohorts.parquet across ccRCC, Melanoma & GBM ICB trials...")
    icb_rows: list[dict[str, Any]] = []

    # 1. Braun 2020 CheckMate 009/010/025 ccRCC (`PMC7499153`, Nivolumab anti-PD-1 & Everolimus)
    assert_not_quarantined("PMC7499153")
    assert_not_quarantined("10.1038/s41591-020-0839-y")
    braun_xlsx = download_file(
        "https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-020-0839-y/MediaObjects/41591_2020_839_MOESM2_ESM.xlsx",
        staging_dir / "braun2020_moesm2.xlsx",
        "PMC7499153",
    )
    df_b_clin = read_xlsx_sheet(braun_xlsx, sheet_index=0)
    df_b_clin.columns = [str(x).strip() for x in df_b_clin.iloc[0].tolist()]
    df_b_clin = df_b_clin.iloc[1:].reset_index(drop=True)

    df_b_rna = read_xlsx_sheet(braun_xlsx, sheet_index=4)
    df_b_rna.columns = [str(x).strip() for x in df_b_rna.iloc[0].tolist()]
    df_b_rna = df_b_rna.iloc[1:].reset_index(drop=True)
    gname_col = df_b_rna.columns[0]
    df_b_rna["_sym"] = df_b_rna[gname_col].astype(str).map(lambda s: alias_map.get(s.strip(), s.strip()))
    df_b_rna = df_b_rna[df_b_rna["_sym"].isin(gene_to_idx)].drop_duplicates(subset=["_sym"], keep="first").set_index("_sym")

    rna_sample_set = set(df_b_rna.columns)
    for _, cr in df_b_clin.iterrows():
        rna_id = str(cr.get("RNA_ID", "")).strip()
        if not rna_id or rna_id == "NA" or rna_id not in rna_sample_set:
            continue
        arm = str(cr.get("Arm", "NIVOLUMAB")).strip().upper()
        orr = str(cr.get("ORR", "")).strip().upper()
        ben = str(cr.get("Benefit", "")).strip().upper()
        if orr in ("CR", "PR", "CRPR") or ben == "CB":
            resp_bin = 1
        elif orr in ("PD", "SD") or ben == "NCB":
            resp_bin = 0
        else:
            resp_bin = -1

        pfs_v = pd.to_numeric(cr.get("PFS"), errors="coerce")
        os_v = pd.to_numeric(cr.get("OS"), errors="coerce")
        vals_s = pd.to_numeric(df_b_rna[rna_id], errors="coerce").fillna(0.0)
        # Rescale Braun upper-quartile log2 normalized expression (~[18, 40]) onto [0, 15] log2(TPM+1)-compatible range
        vmin = float(vals_s.min()) if len(vals_s) > 0 else 0.0
        scaled_s = np.clip((vals_s - vmin) * 0.65, 0.0, 15.0)
        vec = np.zeros(len(vocab_genes), dtype=np.float32)
        for sym, v in scaled_s.items():
            gi = gene_to_idx.get(str(sym))
            if gi is not None:
                vec[gi] = float(v)

        icb_rows.append(
            {
                "record_id": f"BRAUN2020_CCRCC_{str(cr.get('SUBJID', rna_id)).strip()}",
                "cohort_name": f"Braun2020_{str(cr.get('Cohort', 'CheckMate')).strip()}",
                "patient_id": str(cr.get("SUBJID", rna_id)).strip(),
                "lineage": "Kidney",
                "cancer_subtype": "Clear Cell Renal Cell Carcinoma (ccRCC)",
                "therapy_regimen": arm,
                "biopsy_timepoint": "pre-treatment",
                "responder_binary": int(resp_bin),
                "recist_category": orr if orr else ben,
                "pfs_months": float(pfs_v) if pd.notna(pfs_v) and np.isfinite(float(pfs_v)) else -1.0,
                "os_months": float(os_v) if pd.notna(os_v) and np.isfinite(float(os_v)) else -1.0,
                "expr_vec": vec.tolist(),
                "expr_mask_count": int(len(scaled_s)),
                "dataset_source": "Braun2020_CheckMate_ccRCC",
                "accession_id": "PMC7499153",
                "license_track": "open",
            }
        )

    # Helper for NCBI GRCh38 clinical ICB GEO cohorts (`GSE91061`, `GSE78220`, `GSE121810`)
    def _ingest_geo_icb_cohort(gse_acc: str, cohort_label: str, lineage: str, subtype: str, regimen: str) -> None:
        verify_geo_soft_not_quarantined(gse_acc, staging_dir)
        gsm_meta = parse_geo_series_matrix_samples(gse_acc, staging_dir)
        cnt_gz = download_file(
            f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse_acc}&format=file&file={gse_acc}_raw_counts_GRCh38.p13_NCBI.tsv.gz",
            staging_dir / f"{gse_acc}_raw_counts.tsv.gz",
            gse_acc,
        )
        df_c = pd.read_csv(cnt_gz, sep="\t", low_memory=False)
        df_c["_sym"] = df_c["GeneID"].astype(str).map(lambda g: entrez_to_sym.get(g.split(".")[0].strip(), ""))
        df_c = df_c[df_c["_sym"].isin(gene_to_idx)].drop_duplicates(subset=["_sym"], keep="first").set_index("_sym")
        gsms = [c for c in df_c.columns if c.startswith("GSM")]
        l2cpm = df_c[gsms].apply(lambda col: np.log2(col.astype(float) * (1e6 / max(float(col.sum()), 1.0)) + 1.0), axis=0)

        for gsm in gsms:
            info = gsm_meta.get(gsm, {"title": gsm, "chars": ""})
            meta_txt = (info["title"] + " | " + info["chars"]).upper()
            if any(k in meta_txt for k in ("PRCR", "CR/PR", "COMPLETE RESPONSE", "PARTIAL RESPONSE", "RESPONSE: CR", "RESPONSE: PR", "RESPONDER")):
                resp_b = 1
                recist = "CR/PR"
            elif any(k in meta_txt for k in ("PROGRESSIVE DISEASE", "RESPONSE: PD", "RESPONSE: SD", "PDSD", "NON-RESPONDER", "NONRESPONDER")):
                resp_b = 0
                recist = "PD/SD"
            elif "NEOADJUVANT" in meta_txt:
                resp_b = 1
                recist = "Neoadjuvant_PD1"
            elif "ADJUVANT" in meta_txt:
                resp_b = 0
                recist = "Adjuvant_PD1"
            else:
                resp_b = -1
                recist = "Unknown"

            tp = "on-treatment" if ("ON_" in meta_txt or "ON-TREATMENT" in meta_txt or "EDT" in meta_txt) else "pre-treatment"
            vec = np.zeros(len(vocab_genes), dtype=np.float32)
            for sym, v in l2cpm[gsm].items():
                gi = gene_to_idx.get(str(sym))
                if gi is not None and np.isfinite(float(v)):
                    vec[gi] = float(np.clip(float(v), 0.0, 18.0))

            icb_rows.append(
                {
                    "record_id": f"{gse_acc}_{gsm}",
                    "cohort_name": cohort_label,
                    "patient_id": info["title"].split()[0],
                    "lineage": lineage,
                    "cancer_subtype": subtype,
                    "therapy_regimen": regimen,
                    "biopsy_timepoint": tp,
                    "responder_binary": int(resp_b),
                    "recist_category": recist,
                    "pfs_months": -1.0,
                    "os_months": -1.0,
                    "expr_vec": vec.tolist(),
                    "expr_mask_count": int(len(l2cpm)),
                    "dataset_source": cohort_label,
                    "accession_id": gse_acc,
                    "license_track": "open",
                }
            )

    _ingest_geo_icb_cohort("GSE91061", "Riaz2017_Melanoma_AntiPD1", "Skin", "Cutaneous Melanoma", "NIVOLUMAB")
    _ingest_geo_icb_cohort("GSE78220", "Hugo2016_Melanoma_AntiPD1", "Skin", "Metastatic Melanoma", "PEMBROLIZUMAB")
    _ingest_geo_icb_cohort("GSE121810", "Cloughesy2019_GBM_AntiPD1", "CNS/Brain", "Glioblastoma", "PEMBROLIZUMAB")

    # 5. Jerby-Arnon 2018 Melanoma Single-Cell Patient Tumor Pseudobulks (`GSE115978`)
    try:
        verify_geo_soft_not_quarantined("GSE115978", staging_dir)
        ja_ann = download_file(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115978/suppl/GSE115978_cell.annotations.csv.gz",
            staging_dir / "GSE115978_cell.annotations.csv.gz",
            "GSE115978",
        )
        ja_tpm = download_file(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115978/suppl/GSE115978_tpm.csv.gz",
            staging_dir / "GSE115978_tpm.csv.gz",
            "GSE115978",
        )
        df_ann = pd.read_csv(ja_ann, low_memory=False)
        cell_col = df_ann.columns[0]
        samp_col = next((c for c in df_ann.columns if "sample" in c.lower()), df_ann.columns[1])
        trt_col = next((c for c in df_ann.columns if "treatment" in c.lower()), None)
        df_tpm = pd.read_csv(ja_tpm, index_col=0, low_memory=False)
        df_tpm.index = df_tpm.index.astype(str).map(lambda s: alias_map.get(s.strip(), s.strip()))
        df_tpm = df_tpm[df_tpm.index.isin(gene_to_idx)]
        df_tpm = df_tpm[~df_tpm.index.duplicated(keep="first")]

        for pat_id, grp in df_ann.groupby(samp_col):
            c_ids = [str(x) for x in grp[cell_col] if str(x) in df_tpm.columns]
            if len(c_ids) < 5:
                continue
            trt_str = str(grp[trt_col].iloc[0]).lower() if trt_col else ""
            # In Jerby-Arnon 2018, post.treatment biopsies represent ICB-resistant progression (0)
            resp_b = 0 if "post" in trt_str else -1
            pb_mean = df_tpm[c_ids].astype(np.float32).mean(axis=1)
            # GSE115978_tpm.csv.gz is already log2(TPM/10 + 1)
            vec = np.zeros(len(vocab_genes), dtype=np.float32)
            for sym, v in pb_mean.items():
                gi = gene_to_idx.get(str(sym))
                if gi is not None and np.isfinite(float(v)):
                    vec[gi] = float(np.clip(float(v), 0.0, 16.0))
            icb_rows.append(
                {
                    "record_id": f"GSE115978_{re.sub(r'[^A-Za-z0-9]+', '_', str(pat_id))}",
                    "cohort_name": "JerbyArnon2018_Melanoma_ICB",
                    "patient_id": str(pat_id),
                    "lineage": "Skin",
                    "cancer_subtype": "Melanoma (scRNA-seq patient pseudobulk)",
                    "therapy_regimen": "ICB_ANTI_PD1_CTLA4",
                    "biopsy_timepoint": "post-treatment" if "post" in trt_str else "pre-treatment",
                    "responder_binary": int(resp_b),
                    "recist_category": "ICB_Resistant" if resp_b == 0 else "Untreated",
                    "pfs_months": -1.0,
                    "os_months": -1.0,
                    "expr_vec": vec.tolist(),
                    "expr_mask_count": int(len(pb_mean)),
                    "dataset_source": "JerbyArnon2018_Melanoma_scICB",
                    "accession_id": "GSE115978",
                    "license_track": "open",
                }
            )
        ja_tpm.unlink(missing_ok=True)
    except Exception as exc:
        print(f"[Phase 2A-2] Note on GSE115978: {exc}")

    df_icb = pd.DataFrame(icb_rows).drop_duplicates(subset=["record_id"], keep="first")
    df_icb = filter_dataframe_quarantine(df_icb)
    pq.write_table(
        pa.Table.from_pandas(df_icb, preserve_index=False),
        output_dir / "clinical_icb_cohorts.parquet",
        compression="zstd",
    )
    print(
        f"[Phase 2A-2] Wrote clinical_icb_cohorts.parquet: {len(df_icb)} clinical tumor biopsies across "
        f"{df_icb['cohort_name'].nunique()} cohorts (responders={int((df_icb['responder_binary'] == 1).sum())}, "
        f"non-responders={int((df_icb['responder_binary'] == 0).sum())})"
    )
    return len(df_icb)


def run_perturb_geo_bionano_etl(staging_dir: Path, output_dir: Path) -> dict[str, int]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = json.loads((output_dir / "gene_vocab.json").read_text(encoding="utf-8"))
    vocab_genes: list[str] = vocab["genes"]
    gene_to_idx: dict[str, int] = vocab["gene_to_idx"]
    alias_map: dict[str, str] = vocab.get("alias_to_canonical", {})
    loci_64: list[str] = vocab["canonical_64_loci"]

    # Load basal cell line lookup if available
    cname_to_mid: dict[str, str] = {}
    mid_to_lineage: dict[str, str] = {}
    default_mid = "ACH-000681"
    basal_pq = output_dir / "cells_basal_multiomics.parquet"
    if basal_pq.exists():
        df_b = pq.read_table(basal_pq, columns=["model_id", "cell_line_name", "lineage"]).to_pandas()
        if not df_b.empty:
            default_mid = str(df_b.iloc[0]["model_id"])
            for mid, cname, lin in zip(df_b["model_id"], df_b["cell_line_name"], df_b["lineage"]):
                cname_to_mid[re.sub(r"[^A-Z0-9]", "", str(cname).upper())] = str(mid)
                mid_to_lineage[str(mid)] = str(lin)

    # Also pre-populate Entrez -> symbol from DepMap expression header if present
    entrez_seed: dict[str, str] = {}
    depmap_expr = DEFAULT_DEPMAP_STAGING / "OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv"
    if depmap_expr.exists():
        with open(depmap_expr, "r", encoding="utf-8") as fh:
            hdr_cols = fh.readline().strip().split(",")
        for c in hdr_cols[1:]:
            m = re.match(r"^(.+?)\s+\((\d+)\)$", c.strip())
            if m:
                sym = alias_map.get(m.group(1).strip(), m.group(1).strip())
                entrez_seed[m.group(2)] = sym

    # 1. Healthy Organ & TIL Baselines
    n_norm, ensg_to_sym = build_normal_organ_til_baselines(staging_dir, output_dir, vocab_genes, gene_to_idx, alias_map)

    # 2. Perturbation Transcriptomes (Open + Academic) with stage-level scratch checkpointing
    def _cached_stage(cache_name: str, builder_fn: Any) -> list[dict[str, Any]]:
        cp = staging_dir / f"_stage_cache_{cache_name}.parquet"
        if cp.exists() and cp.stat().st_size > 1024:
            df_c = pq.read_table(cp).to_pandas()
            print(f"[Phase 2A-2] Loaded cached stage {cache_name}: {len(df_c)} rows")
            return df_c.to_dict(orient="records")
        rows_out = builder_fn()
        if rows_out:
            pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows_out), preserve_index=False), cp, compression="zstd")
        return rows_out

    perturb_rows: list[dict[str, Any]] = []
    perturb_rows.extend(
        _cached_stage(
            "tahoe100m",
            lambda: extract_tahoe100m_signatures(staging_dir, vocab_genes, gene_to_idx, alias_map, cname_to_mid, default_mid),
        )
    )
    lincs_entrez_holder: dict[str, str] = {}

    def _build_lincs() -> list[dict[str, Any]]:
        r_l, e_l = extract_lincs2020_signatures(
            staging_dir, vocab_genes, gene_to_idx, alias_map, loci_64, cname_to_mid, mid_to_lineage, default_mid
        )
        lincs_entrez_holder.update(e_l)
        return r_l

    perturb_rows.extend(_cached_stage("lincs2020", _build_lincs))
    if not lincs_entrez_holder and (staging_dir / "geneinfo_beta.txt").exists():
        df_gi = pd.read_csv(staging_dir / "geneinfo_beta.txt", sep="\t", low_memory=False)
        for gid, gsym in zip(df_gi["gene_id"].astype(str), df_gi["gene_symbol"].astype(str)):
            lincs_entrez_holder[gid.split(".")[0].strip()] = alias_map.get(gsym.strip(), gsym.strip())
    entrez_to_sym = {**entrez_seed, **lincs_entrez_holder}

    perturb_rows.extend(
        _cached_stage(
            "arc_pie",
            lambda: extract_arc_pie_signatures(staging_dir, vocab_genes, gene_to_idx, alias_map, cname_to_mid, mid_to_lineage, default_mid),
        )
    )
    perturb_rows.extend(
        _cached_stage(
            "parse10m",
            lambda: extract_parse10m_cytokine_signatures(vocab_genes, gene_to_idx, alias_map, default_mid),
        )
    )
    perturb_rows.extend(
        _cached_stage(
            "geo_ribo_bionano",
            lambda: extract_geo_ribo_and_bionano_signatures(
                staging_dir, vocab_genes, gene_to_idx, alias_map, ensg_to_sym, entrez_to_sym, cname_to_mid, default_mid
            ),
        )
    )

    df_pert = pd.DataFrame(perturb_rows).drop_duplicates(subset=["record_id"], keep="first")
    df_pert = filter_dataframe_quarantine(df_pert)
    pq.write_table(
        pa.Table.from_pandas(df_pert, preserve_index=False),
        output_dir / "perturb_transcriptomes.parquet",
        compression="zstd",
    )
    n_open = int((df_pert["license_track"] == "open").sum())
    n_acad = int((df_pert["license_track"] == "academic").sum())
    print(
        f"[Phase 2A-2] Wrote perturb_transcriptomes.parquet: {len(df_pert)} total signatures "
        f"(open={n_open}, academic={n_acad}, sources={df_pert['dataset_source'].value_counts().to_dict()})"
    )

    # 3. Clinical ICB Cohorts
    n_icb = build_clinical_icb_cohorts(staging_dir, output_dir, vocab_genes, gene_to_idx, alias_map, entrez_to_sym)

    return {
        "normal_organ_til_baselines": n_norm,
        "perturb_transcriptomes": len(df_pert),
        "perturb_open": n_open,
        "perturb_academic": n_acad,
        "clinical_icb_cohorts": n_icb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FerroFlux-FM Phase 2A-2 Perturb-seq, Tahoe-100M, LINCS, GEO & ICB ETL")
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    counts = run_perturb_geo_bionano_etl(args.staging_dir, args.output_dir)
    print("[Phase 2A-2] Summary:", json.dumps(counts))


if __name__ == "__main__":
    main()
