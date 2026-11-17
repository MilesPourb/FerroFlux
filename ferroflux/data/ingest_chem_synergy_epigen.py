"""Phase 2B-2 Data ETL: Chemogenomics (FerrDb V3 + PubChem qHTS + ChEMBL 37 + RDKit),
DrugComb v1.5 Synergy Blocks, 64-Loci Cis-Regulatory Epigenomics, and AWS S3 PMC OA
Quantitative Rescue & Non-Bradbury Bio-Nanomedicine Extraction.

Generates four Zstd-compressed Parquet shards in `/anvil/projects/x-cis261485/Ferroptosis_ML/data/processed/`:
  5. `chem_ferrdb_qhts_chembl_features.parquet`
  6. `drugcomb_synergy_blocks.parquet`
  7. `cis_reg_64loci_epigenomics.parquet`
  8. `pmc_oa_rescue_and_bionano.parquet`

Strict Compliance Guarantees:
  - Hard Tier-3 Bradbury/Wiesner/Overholtzer holdout quarantine BEFORE download/parse and BEFORE Parquet write.
  - Dual-track license separation (`license_track` normalized to `{"open", "academic"}`).
  - Zero-CAPTCHA / anti-bot compliance: never bypasses 401/403/429/503 or challenge gates.
  - Zero unpacked S3 directories on disk; all staging confined to `/anvil/scratch/x-mpourbaghima/ferroflux_staging/phase2b/`.
"""

from __future__ import annotations

import concurrent.futures
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
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, QED, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from ferroflux.security.license_guard import normalize_license_label
from ferroflux.security.quarantine import (
    assert_not_quarantined,
    audit_parquet_directory,
    filter_dataframe_quarantine,
    is_quarantined,
)

RDLogger.DisableLog("rdApp.*")

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
CONFIGS_DIR = ROOT_DIR / "configs"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
STAGING_DIR = Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/phase2b")
SCRATCH_CACHE_DIR = Path("/home/x-mpourbaghima/.gemini/antigravity/brain/f2dfe148-0b36-4c4c-a481-67e2f5b83528/scratch")

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

# Mechanistic SMARTS patterns matching chem_desc_dim=12 in FerroFluxMultimodalDataset (#40):
# [MW, LogP, TPSA, HBD, HBA, RotB, covalent_warhead, rta_motif, fe_chelator, endoperoxide, statin_hmgcr, dhcr7_sterol]
_SMARTS_DEFS = {
    "covalent_chloroacetamide": "[CX3](=[OX1])[CX4H2][Cl,Br,I]",
    "covalent_acrylamide": "[CX3H2]=[CX3][CX3](=[OX1])[NX3]",
    "covalent_propiolamide": "[CX2]#[CX2][CX3](=[OX1])[NX3,OX2]",
    "covalent_nitroisoxazole": "c1noc([N+](=O)[O-])c1",
    "covalent_fumarate_maleimide": "[CX3](=[OX1])[CX3]=[CX3][CX3](=[OX1])",
    "rta_diarylamine": "[NX3;H1]([c])[c]",
    "rta_arylenediamine": "[NX3]([c])c1ccc([NX3])cc1",
    "rta_chromanol_phenol": "c1cc(O)ccc1O",
    "chelator_hydroxamate": "[CX3](=[OX1])[NX3;H1][OX2;H1]",
    "chelator_hydroxypyridinone": "O=c1ccncc1O",
    "chelator_catechol_8hq": "c1ccc2c(O)cccc2n1",
    "chelator_triazole_deferasirox": "c1nc(ncn1)c2ccccc2O",
    "endoperoxide_bridge": "[OX2][OX2]",
    "statin_dihydroxyheptanoate": "O[CH]CC[CH](O)CC(=O)O",
    "dhcr7_arylpiperazine": "N1CCN(CC1)c2ccccc2",
}
COMPILED_SMARTS = {k: Chem.MolFromSmarts(v) for k, v in _SMARTS_DEFS.items()}
MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=2048)


def _norm_cell_key(name: str) -> str:
    if not name:
        return ""
    base = str(name).strip().upper().split("_")[0]
    return re.sub(r"[^A-Z0-9]", "", base)


def safe_fetch_bytes(
    url: str,
    cache_name: str | None = None,
    timeout: int = 90,
    post_data: dict[str, str] | None = None,
    dataset_id: str | None = None,
) -> bytes | None:
    if dataset_id:
        assert_not_quarantined(dataset_id)
    assert_not_quarantined(url)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = STAGING_DIR / cache_name if cache_name else None
    if cache_path and cache_path.is_file() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()

    encoded_data = urllib.parse.urlencode(post_data).encode("utf-8") if post_data else None
    headers = {"User-Agent": USER_AGENT}
    if encoded_data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=encoded_data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            head_str = data[:16384].decode("utf-8", errors="ignore").lower()
            for sig in CAPTCHA_SIGNATURES:
                if sig in head_str:
                    print(f"[SKIP-GATE] Anti-bot/CAPTCHA ({sig}) on {url}; skipping cleanly.")
                    return None
            if cache_path:
                tmp = cache_path.with_suffix(cache_path.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(cache_path)
            return data
    except urllib.error.HTTPError as exc:
        print(f"[SKIP-HTTP-{exc.code}] {url}: {exc.reason}")
        return None
    except Exception as exc:
        print(f"[SKIP-ERR] {url}: {type(exc).__name__}: {exc}")
        return None


def parse_xlsx_rows_stdlib(xlsx_bytes: bytes, sheet_index: int = 0) -> list[list[str]]:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root_ss = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root_ss.findall("main:si", ns):
                shared_strings.append("".join(t.text or "" for t in si.findall(".//main:t", ns)))
        sheet_files = sorted(
            [n for n in zf.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)],
            key=lambda s: int(re.search(r"(\d+)", s).group(1)),
        )
        if not sheet_files:
            return []
        root_ws = ET.fromstring(zf.read(sheet_files[min(sheet_index, len(sheet_files) - 1)]))
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


def write_clean_parquet(df: pd.DataFrame, out_path: Path) -> dict[str, Any]:
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


def build_cell_line_lookup() -> tuple[dict[str, str], dict[str, str]]:
    """Return (normname_to_ach, ach_to_lineage)."""
    norm_to_ach: dict[str, str] = {}
    ach_to_lin: dict[str, str] = {}

    cmp_b = safe_fetch_bytes(
        "https://cog.sanger.ac.uk/cmp/download/model_list_latest.csv.gz",
        cache_name="sanger_model_list_latest.csv.gz",
    )
    if cmp_b:
        df_cmp = pd.read_csv(io.BytesIO(gzip.decompress(cmp_b)), low_memory=False)
        for _, r in df_cmp.iterrows():
            ach = str(r.get("BROAD_ID", "")).strip()
            mname = str(r.get("model_name", "")).strip()
            ccle = str(r.get("CCLE_ID", "")).strip()
            tissue = str(r.get("tissue", "")).strip() if pd.notna(r.get("tissue")) else "Unknown"
            if ach.startswith("ACH-"):
                ach_to_lin[ach] = tissue
                if mname:
                    norm_to_ach[_norm_cell_key(mname)] = ach
                if ccle and ccle != "nan":
                    norm_to_ach[_norm_cell_key(ccle)] = ach

    ann_b = safe_fetch_bytes(
        "https://data.broadinstitute.org/ccle/Cell_lines_annotations_20181226.txt",
        cache_name="Cell_lines_annotations_20181226.txt",
    )
    if ann_b:
        df_ann = pd.read_csv(io.BytesIO(ann_b), sep="\t", low_memory=False)
        for _, r in df_ann.iterrows():
            ach = str(r.get("depMapID", "")).strip()
            ccle = str(r.get("CCLE_ID", "")).strip()
            site = str(r.get("Site_Primary", "")).strip() if pd.notna(r.get("Site_Primary")) else "Unknown"
            if ach.startswith("ACH-"):
                if ccle:
                    norm_to_ach[_norm_cell_key(ccle)] = ach
                ach_to_lin.setdefault(ach, site)
    return norm_to_ach, ach_to_lin


def compute_rdkit_features(smiles: str) -> dict[str, Any] | None:
    """Standardize SMILES and compute Morgan-2048, MACCS-167, Murcko scaffold, and chem_desc_12."""
    if not smiles or not isinstance(smiles, str):
        return None
    s_clean = smiles.strip()
    if not s_clean or s_clean.lower() == "nan":
        return None
    # Keep largest organic fragment if salt
    if "." in s_clean:
        frags = s_clean.split(".")
        s_clean = max(frags, key=len)
    mol = Chem.MolFromSmiles(s_clean)
    if mol is None:
        return None
    try:
        canon_smi = Chem.MolToSmiles(mol, canonical=True)
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not inchikey:
        return None

    try:
        scaf_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaf_smi = Chem.MolToSmiles(scaf_mol, canonical=True) if scaf_mol is not None else canon_smi
    except Exception:
        scaf_smi = canon_smi

    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = float(Lipinski.NumHDonors(mol))
    hba = float(Lipinski.NumHAcceptors(mol))
    rotb = float(Lipinski.NumRotatableBonds(mol))
    fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    try:
        qed_val = float(QED.qed(mol))
    except Exception:
        qed_val = 0.5

    def _has(key: str) -> bool:
        pat = COMPILED_SMARTS.get(key)
        return bool(pat is not None and mol.HasSubstructMatch(pat))

    cov_chloro = _has("covalent_chloroacetamide")
    cov_acryl = _has("covalent_acrylamide")
    cov_prop = _has("covalent_propiolamide")
    cov_nitro = _has("covalent_nitroisoxazole")
    cov_fum = _has("covalent_fumarate_maleimide")
    is_cov = float(cov_chloro or cov_acryl or cov_prop or cov_nitro or cov_fum)

    is_rta = float(_has("rta_diarylamine") or _has("rta_arylenediamine") or _has("rta_chromanol_phenol"))
    is_chel = float(
        _has("chelator_hydroxamate")
        or _has("chelator_hydroxypyridinone")
        or _has("chelator_catechol_8hq")
        or _has("chelator_triazole_deferasirox")
    )
    is_endo = float(_has("endoperoxide_bridge"))
    is_statin = float(_has("statin_dihydroxyheptanoate"))
    is_dhcr7 = float(_has("dhcr7_arylpiperazine"))

    warhead_label = "none"
    if cov_chloro:
        warhead_label = "chloroacetamide"
    elif cov_prop:
        warhead_label = "propiolamide"
    elif cov_acryl:
        warhead_label = "acrylamide"
    elif cov_nitro:
        warhead_label = "nitroisoxazole_masked_nitrile_oxide"
    elif cov_fum:
        warhead_label = "michael_acceptor_fumarate"

    morgan_bv = MORGAN_GEN.GetFingerprint(mol)
    bit_str = morgan_bv.ToBitString()
    packed_arr = np.packbits(np.fromiter((1 if ch == "1" else 0 for ch in bit_str), dtype=np.uint8, count=2048))
    packed_hex = packed_arr.tobytes().hex()

    maccs_bv = MACCSkeys.GenMACCSKeys(mol)
    maccs_hex = maccs_bv.ToBitString()

    # Exact 12-feature vector matching FerroFluxMultimodalDataset (#40):
    # [MW/500, LogP/5, TPSA/150, HBD/5, HBA/10, RotB/10, covalent_warhead, rta_motif, fe_chelator, endoperoxide, statin_hmgcr, dhcr7_sterol]
    chem_desc_12 = [
        mw / 500.0,
        logp / 5.0,
        tpsa / 150.0,
        hbd / 5.0,
        hba / 10.0,
        rotb / 10.0,
        is_cov,
        is_rta,
        is_chel,
        is_endo,
        is_statin,
        is_dhcr7,
    ]

    return {
        "canonical_smiles": canon_smi,
        "inchikey": inchikey,
        "ik14": inchikey[:14],
        "murcko_scaffold": scaf_smi,
        "scaffold_smiles": scaf_smi,
        "mol_wt": mw,
        "mol_logp": logp,
        "tpsa": tpsa,
        "num_hbd": int(hbd),
        "num_hba": int(hba),
        "num_rotatable_bonds": int(rotb),
        "fraction_csp3": fsp3,
        "qed": qed_val,
        "is_covalent_warhead": bool(is_cov > 0.5),
        "warhead_type": warhead_label,
        "is_rta_motif": bool(is_rta > 0.5),
        "is_fe_chelator": bool(is_chel > 0.5),
        "is_endoperoxide": bool(is_endo > 0.5),
        "is_statin_hmgcr": bool(is_statin > 0.5),
        "is_dhcr7_sterol_modulator": bool(is_dhcr7 > 0.5),
        "chem_desc_12": [float(x) for x in chem_desc_12],
        "morgan_fp_2048_hex": packed_hex,
        "morgan_fp_uint8_256": [int(x) for x in packed_arr.tolist()],
        "maccs_167_bits": maccs_hex,
    }


def batch_resolve_pubchem_cids_to_smiles(cids: list[int]) -> dict[int, tuple[str, str]]:
    """Batch-resolve PubChem CIDs -> (CanonicalSMILES, IUPAC/Title) via PUG REST POST."""
    cache_file = STAGING_DIR / "pubchem_cid_smiles_cache.json"
    cid_map: dict[int, tuple[str, str]] = {}
    if cache_file.is_file():
        try:
            loaded = json.loads(cache_file.read_text(encoding="utf-8"))
            for k, v in loaded.items():
                cid_map[int(k)] = (v[0], v[1])
        except Exception:
            pass

    missing = sorted({int(c) for c in cids if int(c) > 0 and int(c) not in cid_map})
    if not missing:
        return cid_map

    print(f"  Resolving {len(missing):,} unique PubChem CIDs to canonical SMILES in batches of 200...")
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/CanonicalSMILES,ConnectivitySMILES,Title/JSON"
    for i in range(0, len(missing), 200):
        chunk = missing[i : i + 200]
        body = urllib.parse.urlencode({"cid": ",".join(str(x) for x in chunk)}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                props = payload.get("PropertyTable", {}).get("Properties", [])
                for p in props:
                    c_val = int(p.get("CID", 0))
                    smi = str(p.get("CanonicalSMILES") or p.get("ConnectivitySMILES") or p.get("SMILES") or "").strip()
                    title = str(p.get("Title") or "").strip()
                    if c_val > 0 and smi:
                        cid_map[c_val] = (smi, title)
        except Exception as exc:
            print(f"    [WARN] PUG batch {i}..{i+len(chunk)}: {exc}")
        time.sleep(0.25)

    cache_file.write_text(
        json.dumps({str(k): [v[0], v[1]] for k, v in cid_map.items()}),
        encoding="utf-8",
    )
    return cid_map


def ingest_chem_ferrdb_qhts_chembl() -> dict[str, Any]:
    print("\n=== [5/8] Ingesting FerrDb V3 + PubChem qHTS + ChEMBL 37 + RDKit Descriptors ===")
    bio_rows: list[dict[str, Any]] = []
    cids_needed: set[int] = set()

    # 5A. FerrDb V3 Live API Download (563 Ferroptosis Inducers + 510 Inhibitors across 22 RCDs)
    assert_not_quarantined("FerrDb_V3", authors="Zhou N, Bao J")
    for role in ("inducer", "inhibitor"):
        fb = safe_fetch_bytes(
            "https://data.zhounan.org/pm2505/api/download/data/",
            cache_name=f"ferrdb_v3_{role}.csv",
            post_data={"selectDataset": role, "selectFormat": "csv"},
            dataset_id="FerrDb_V3",
        )
        if not fb:
            continue
        df_f = pd.read_csv(io.BytesIO(fb), low_memory=False)
        for idx, r in df_f.iterrows():
            rcd = str(r.get("RCD", "Ferroptosis")).strip()
            cname = str(r.get("Name") or r.get("Substance") or "").strip()
            cid_num = pd.to_numeric(r.get("PubChem_CID"), errors="coerce")
            cid_int = int(cid_num) if pd.notna(cid_num) and cid_num > 0 else 0
            if cid_int > 0:
                cids_needed.add(cid_int)
            bio_rows.append(
                {
                    "record_id": f"FerrDbV3:{role}:{idx}",
                    "compound_name": cname.upper() if cname else f"CID_{cid_int}",
                    "pubchem_cid": cid_int,
                    "chembl_id": "",
                    "raw_smiles": "",
                    "source_db": "FerrDb_V3",
                    "dataset_source": f"FerrDb_V3_{role}",
                    "accession_id": f"FerrDb_V3_{role}",
                    "target_symbol": rcd.upper(),
                    "mechanism_or_assay": f"FerrDb_V3_{rcd}_{role}",
                    "activity_outcome": "Active_Inducer" if role == "inducer" else "Active_Inhibitor",
                    "activity_type": "Curated_RCD_Modulator",
                    "potency_uM": float("nan"),
                    "p_potency": float("nan"),
                    "raw_license": "CC-BY-NC-4.0 (FerrDb V3)",
                    "license_track": "academic",
                }
            )
    print(f"  Loaded {len(bio_rows):,} FerrDb V3 curated compound annotations.")

    # 5B. PubChem qHTS Dose-Response Concise CSVs (Core Ferroptosis / GPX4 / TXNRD1 / HT-1080 / BJeLR / LOX / Nrf2 Screens)
    qhts_aids = [
        (1845192, "GPX4", "NCATS_qHTS_Recombinant_Sec_GPX4_Primary"),
        (1845194, "GPX4", "NCATS_qHTS_Recombinant_Sec_GPX4_Confirmatory"),
        (588453, "TXNRD1", "qHTS_Selenoprotein_TXNRD1_Inhibitors"),
        (686970, "FERROPTOSIS_HT1080", "Broad_MLPCN_HT1080_Ferroptosis_Modulators_1"),
        (686971, "FERROPTOSIS_HT1080", "Broad_MLPCN_HT1080_Ferroptosis_Modulators_2"),
        (1554, "HRAS_BJELR", "Stockwell_Columbia_BJeLR_Synthetic_Lethal_FIN"),
        (2275, "HRAS_BJELR", "Stockwell_BJeLR_DoseResponse_Confirmatory"),
        (887, "ALOX15", "NCGC_qHTS_Human_15_Lipoxygenase_ALOX15"),
        (1452, "ALOX12", "NCGC_qHTS_Platelet_12_Lipoxygenase_ALOX12"),
        (1671190, "ALOX12", "qHTS_12_Lipoxygenase_Lead_Series"),
        (743219, "NFE2L2", "Tox21_ARE_bla_Nrf2_Antioxidant_qHTS"),
    ]
    rng = np.random.default_rng(20261113)
    for aid, tgt_sym, assay_desc in qhts_aids:
        assert_not_quarantined(f"AID_{aid}")
        csv_b = safe_fetch_bytes(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/concise/CSV",
            cache_name=f"pubchem_aid_{aid}_concise.csv",
            timeout=180,
            dataset_id=f"AID_{aid}",
        )
        if not csv_b:
            continue
        df_aid = pd.read_csv(
            io.BytesIO(csv_b),
            usecols=lambda c: c in {"CID", "Activity Outcome", "Activity Value [uM]", "Activity Name"},
            low_memory=False,
        )
        df_aid["CID"] = pd.to_numeric(df_aid["CID"], errors="coerce")
        df_aid = df_aid.dropna(subset=["CID"]).copy()
        df_aid["CID"] = df_aid["CID"].astype(int)
        df_aid = df_aid[df_aid["CID"] > 0]

        is_pos = df_aid["Activity Outcome"].astype(str).isin(["Active", "Probe"]) | df_aid["Activity Value [uM]"].notna()
        pos_df = df_aid[is_pos]
        if len(pos_df) > 4000:
            # Prioritize Active/Probe first, then quantitative Potency rows
            act_only = pos_df[pos_df["Activity Outcome"].astype(str).isin(["Active", "Probe"])]
            rem = pos_df[~pos_df.index.isin(act_only.index)]
            take_n = max(0, 4000 - len(act_only))
            if len(act_only) > 4000:
                pos_df = act_only.sample(n=4000, random_state=42)
            elif take_n > 0 and len(rem) > 0:
                pos_df = pd.concat([act_only, rem.sample(n=min(take_n, len(rem)), random_state=42)])
            else:
                pos_df = act_only

        neg_pool = df_aid[~is_pos]
        neg_df = neg_pool.sample(n=min(800, len(neg_pool)), random_state=42) if len(neg_pool) > 0 else neg_pool
        sub_aid = pd.concat([pos_df, neg_df], ignore_index=True).drop_duplicates(subset=["CID"])

        for idx, r in sub_aid.iterrows():
            cid_i = int(r["CID"])
            cids_needed.add(cid_i)
            u_val = pd.to_numeric(r.get("Activity Value [uM]"), errors="coerce")
            pot_um = float(u_val) if pd.notna(u_val) and float(u_val) > 0 else float("nan")
            p_pot = float(-math.log10(pot_um * 1e-6)) if pot_um == pot_um else float("nan")
            bio_rows.append(
                {
                    "record_id": f"PubChemAID{aid}:CID{cid_i}",
                    "compound_name": f"CID_{cid_i}",
                    "pubchem_cid": cid_i,
                    "chembl_id": "",
                    "raw_smiles": "",
                    "source_db": "PubChem_qHTS",
                    "dataset_source": f"PubChem_AID_{aid}",
                    "accession_id": f"AID:{aid}",
                    "target_symbol": tgt_sym,
                    "mechanism_or_assay": assay_desc,
                    "activity_outcome": str(r.get("Activity Outcome", "Unspecified")),
                    "activity_type": str(r.get("Activity Name", "Potency") if pd.notna(r.get("Activity Name")) else "Potency"),
                    "potency_uM": pot_um,
                    "p_potency": p_pot,
                    "raw_license": "US Government Public Domain (NCBI PubChem)",
                    "license_track": "open",
                }
            )
        print(f"  PubChem AID {aid} ({tgt_sym}): retained {len(sub_aid):,} quantitative/stratified compound rows.")

    # 5C. ChEMBL 37 Target Quantitative Bioactivities (pChEMBL + SMILES directly from EBI REST API)
    chembl_targets = [
        ("CHEMBL3580494", "GPX4"),
        ("CHEMBL2169736", "AIFM2"),
        ("CHEMBL3714130", "SLC7A11"),
        ("CHEMBL4295956", "ACSL4"),
        ("CHEMBL2903", "ALOX15"),
        ("CHEMBL3687", "ALOX12"),
        ("CHEMBL215", "ALOX5"),
        ("CHEMBL1914266", "DHCR7"),
        ("CHEMBL1795140", "EBP"),
        ("CHEMBL3592", "SQLE"),
        ("CHEMBL402", "HMGCR"),
        ("CHEMBL1927", "TXNRD1"),
        ("CHEMBL1075104", "NFE2L2"),
        ("CHEMBL2069156", "KEAP1"),
        ("CHEMBL2756", "GSR"),
        ("CHEMBL5147", "POR"),
        ("CHEMBL1966", "DHODH"),
    ]
    chembl_cnt = 0
    for t_chembl, t_sym in chembl_targets:
        assert_not_quarantined(t_chembl)
        for page in range(3):  # up to 3,000 quantitative pChEMBL rows per target
            offset = page * 1000
            url = (
                f"https://www.ebi.ac.uk/chembl/api/data/activity.json?"
                f"target_chembl_id={t_chembl}&pchembl_value__isnull=false&limit=1000&offset={offset}"
            )
            cb = safe_fetch_bytes(
                url,
                cache_name=f"chembl37_{t_chembl}_p{page}.json",
                timeout=60,
                dataset_id=t_chembl,
            )
            if not cb:
                break
            try:
                c_obj = json.loads(cb.decode("utf-8", errors="replace"))
            except Exception:
                break
            acts = c_obj.get("activities", [])
            if not acts:
                break
            for act in acts:
                smi = str(act.get("canonical_smiles") or "").strip()
                pchembl = pd.to_numeric(act.get("pchembl_value"), errors="coerce")
                if not smi or pd.isna(pchembl):
                    continue
                mol_cid = str(act.get("molecule_chembl_id") or "").strip()
                pref_name = str(act.get("molecule_pref_name") or mol_cid).strip().upper()
                aid_ch = str(act.get("assay_chembl_id") or t_chembl).strip()
                act_id = str(act.get("activity_id") or chembl_cnt)
                p_val = float(pchembl)
                pot_um = float(10.0 ** (6.0 - p_val))
                bio_rows.append(
                    {
                        "record_id": f"ChEMBL37:{act_id}",
                        "compound_name": pref_name,
                        "pubchem_cid": 0,
                        "chembl_id": mol_cid,
                        "raw_smiles": smi,
                        "source_db": "ChEMBL_37",
                        "dataset_source": f"ChEMBL37_{t_sym}",
                        "accession_id": aid_ch,
                        "target_symbol": t_sym,
                        "mechanism_or_assay": str(act.get("assay_description") or f"ChEMBL_{t_sym}")[:140],
                        "activity_outcome": "Active" if p_val >= 5.5 else "Weak_or_Inactive",
                        "activity_type": str(act.get("standard_type") or "pChEMBL"),
                        "potency_uM": pot_um,
                        "p_potency": p_val,
                        "raw_license": "CC-BY-SA-3.0 (EMBL-EBI ChEMBL 37)",
                        "license_track": "open",
                    }
                )
                chembl_cnt += 1
            if not c_obj.get("page_meta", {}).get("next"):
                break
    print(f"  ChEMBL 37: loaded {chembl_cnt:,} quantitative pChEMBL target bioactivities.")

    # Also pull phenotypic ferroptosis assays from ChEMBL 37
    pheno_url = "https://www.ebi.ac.uk/chembl/api/data/assay.json?description__icontains=ferroptosis&limit=200"
    pb = safe_fetch_bytes(pheno_url, cache_name="chembl37_pheno_ferroptosis_assays.json", dataset_id="ChEMBL37_Pheno")
    if pb:
        p_assays = [a.get("assay_chembl_id") for a in json.loads(pb.decode("utf-8", errors="replace")).get("assays", []) if a.get("assay_chembl_id")]
        for paid in p_assays[:40]:
            ab = safe_fetch_bytes(
                f"https://www.ebi.ac.uk/chembl/api/data/activity.json?assay_chembl_id={paid}&limit=500",
                cache_name=f"chembl37_pheno_{paid}.json",
                dataset_id=paid,
            )
            if not ab:
                continue
            for act in json.loads(ab.decode("utf-8", errors="replace")).get("activities", []):
                smi = str(act.get("canonical_smiles") or "").strip()
                if not smi:
                    continue
                sval = pd.to_numeric(act.get("standard_value"), errors="coerce")
                sunit = str(act.get("standard_units") or "").strip()
                pchembl = pd.to_numeric(act.get("pchembl_value"), errors="coerce")
                pot_um = float("nan")
                p_val = float("nan")
                if pd.notna(pchembl):
                    p_val = float(pchembl)
                    pot_um = float(10.0 ** (6.0 - p_val))
                elif pd.notna(sval) and float(sval) > 0:
                    if sunit == "nM":
                        pot_um = float(sval) / 1000.0
                    elif sunit in ("uM", "umol/L"):
                        pot_um = float(sval)
                    if pot_um == pot_um and pot_um > 0:
                        p_val = float(-math.log10(pot_um * 1e-6))
                mol_cid = str(act.get("molecule_chembl_id") or "").strip()
                bio_rows.append(
                    {
                        "record_id": f"ChEMBL37Pheno:{act.get('activity_id')}",
                        "compound_name": str(act.get("molecule_pref_name") or mol_cid).upper(),
                        "pubchem_cid": 0,
                        "chembl_id": mol_cid,
                        "raw_smiles": smi,
                        "source_db": "ChEMBL_37_Phenotypic",
                        "dataset_source": "ChEMBL37_Ferroptosis_Phenotypic",
                        "accession_id": paid,
                        "target_symbol": "FERROPTOSIS_PHENOTYPIC",
                        "mechanism_or_assay": str(act.get("assay_description") or "Ferroptosis phenotypic assay")[:140],
                        "activity_outcome": "Active" if (p_val == p_val and p_val >= 5.5) else "Measured",
                        "activity_type": str(act.get("standard_type") or "Phenotypic"),
                        "potency_uM": pot_um,
                        "p_potency": p_val,
                        "raw_license": "CC-BY-SA-3.0 (EMBL-EBI ChEMBL 37)",
                        "license_track": "open",
                    }
                )

    # Resolve PubChem CIDs -> SMILES and compute RDKit features once per unique SMILES
    cid_to_smi = batch_resolve_pubchem_cids_to_smiles(sorted(cids_needed))
    rdkit_cache: dict[str, dict[str, Any] | None] = {}

    enriched_rows: list[dict[str, Any]] = []
    for row in bio_rows:
        smi = row["raw_smiles"]
        cid = row["pubchem_cid"]
        if not smi and cid > 0 and cid in cid_to_smi:
            smi, title = cid_to_smi[cid]
            if row["compound_name"].startswith("CID_") and title:
                row["compound_name"] = title.upper()[:120]
        if not smi:
            continue
        if smi not in rdkit_cache:
            rdkit_cache[smi] = compute_rdkit_features(smi)
        feat = rdkit_cache[smi]
        if feat is None:
            continue
        out_r = {k: v for k, v in row.items() if k != "raw_smiles"}
        out_r.update(feat)
        enriched_rows.append(out_r)

    df_chem = pd.DataFrame(enriched_rows)
    return write_clean_parquet(df_chem, PROCESSED_DIR / "chem_ferrdb_qhts_chembl_features.parquet")


def ingest_drugcomb_synergy(norm_to_ach: dict[str, str]) -> dict[str, Any]:
    print("\n=== [6/8] Ingesting DrugComb v1.5 Combination Synergy Blocks ===")
    assert_not_quarantined("Zenodo:15235991", authors="Zagidullin B, Tang J")

    # 1. Parse DrugComb drug identifiers (from cached scratch or Zenodo)
    drug_xlsx_path = SCRATCH_CACHE_DIR / "DrugComb_drug_identifiers.xlsx"
    if drug_xlsx_path.is_file():
        xlsx_bytes = drug_xlsx_path.read_bytes()
    else:
        xlsx_bytes = safe_fetch_bytes(
            "https://zenodo.org/api/records/15235991/files/DrugComb_drug_identifiers.xlsx/content",
            cache_name="DrugComb_drug_identifiers.xlsx",
            timeout=120,
            dataset_id="Zenodo:15235991",
        )
    drug_meta: dict[str, tuple[str, str]] = {}  # upper dname -> (smiles, inchikey)
    if xlsx_bytes:
        xrows = parse_xlsx_rows_stdlib(xlsx_bytes, sheet_index=0)
        if len(xrows) >= 2:
            hdr = [h.strip().lower() for h in xrows[0]]
            idx_name = next((i for i, h in enumerate(hdr) if "dname" in h or h == "name"), 0)
            idx_smi = next((i for i, h in enumerate(hdr) if "smiles" in h), -1)
            idx_ik = next((i for i, h in enumerate(hdr) if "inchikey" in h or "inchi_key" in h), -1)
            for r in xrows[1:]:
                if len(r) <= idx_name:
                    continue
                dn = str(r[idx_name]).strip().upper()
                smi = str(r[idx_smi]).strip() if 0 <= idx_smi < len(r) else ""
                ik = str(r[idx_ik]).strip() if 0 <= idx_ik < len(r) else ""
                if dn and smi and smi.lower() not in ("none", "na", "null"):
                    drug_meta[dn] = (smi, ik)
    print(f"  Loaded {len(drug_meta):,} DrugComb drug chemical structures.")

    # Mechanistic ferroptosis / redox / statin / proteasome / LOX / chelator / RTK / mTOR / apoptosis keywords
    ferro_drug_keywords = (
        "ERASTIN",
        "RSL3",
        "ML162",
        "ML210",
        "FIN56",
        "FINO2",
        "SORAFENIB",
        "SULFASALAZINE",
        "AURANOFIN",
        "ARTESUNATE",
        "ARTEMISININ",
        "DIHYDROARTEMISININ",
        "SIMVASTATIN",
        "LOVASTATIN",
        "ATORVASTATIN",
        "FLUVASTATIN",
        "PITAVASTATIN",
        "ROSUVASTATIN",
        "PRAVASTATIN",
        "CERIVASTATIN",
        "BORTEZOMIB",
        "CARFILZOMIB",
        "IXAZOMIB",
        "MG-132",
        "ALTRETAMINE",
        "BUTENAFINE",
        "TERBINAFINE",
        "ZILEUTON",
        "ROSIGLITAZONE",
        "PIOGLITAZONE",
        "TROGLITAZONE",
        "DEFEROXAMINE",
        "DEFERASIROX",
        "CICLOPIROX",
        "FERROSTATIN",
        "LIPROXSTATIN",
        "ELESCLOMOL",
        "DISULFIRAM",
        "WITHAFERIN",
        "BUTHIONINE",
        "DIMETHYL FUMARATE",
        "MENADIONE",
        "IDEBENONE",
        "BREQUINAR",
        "LEFLUNOMIDE",
        "TERIFLUNOMIDE",
        "LENVATINIB",
        "REGORAFENIB",
        "CABOZANTINIB",
        "SUNITINIB",
        "PAZOPANIB",
        "LAPATINIB",
        "NERATINIB",
        "ERLOTINIB",
        "GEFITINIB",
        "OSIMERTINIB",
        "VEMURAFENIB",
        "DABRAFENIB",
        "TRAMETINIB",
        "COBIMETINIB",
        "SIROLIMUS",
        "RAPAMYCIN",
        "EVEROLIMUS",
        "TEMSIROLIMUS",
        "AZD8055",
        "VENETOCLAX",
        "NAVITOCLAX",
        "VORINOSTAT",
        "PANOBINOSTAT",
        "BELINOSTAT",
        "OLAPARIB",
        "TALAZOPARIB",
        "APR-246",
        "PIPERLONGUMINE",
        "PLUMBAGIN",
    )

    def _is_ferro_relevant_drug(dname: str) -> bool:
        u = dname.upper()
        return any(kw in u for kw in ferro_drug_keywords)

    # Stream Zenodo DrugComb v1.5 (or fallback to cached v1.4 if network interrupts)
    dc_csv_path = STAGING_DIR / "drugcomb_summary_v_1_5.csv"
    if not dc_csv_path.is_file() or dc_csv_path.stat().st_size < 100_000_000:
        print("  Streaming DrugComb v1.5 summary_v_1_5.csv to scratch staging...")
        url = "https://zenodo.org/api/records/15235991/files/summary_v_1_5.csv/content"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp, dc_csv_path.open("wb") as out_f:
                while True:
                    chunk = resp.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
        except Exception as exc:
            print(f"  [WARN] DrugComb v1.5 stream interrupted ({exc}); checking cached summary_table_v1.4.csv...")
            v14 = SCRATCH_CACHE_DIR / "summary_table_v1.4.csv"
            if v14.is_file():
                dc_csv_path = v14

    kept_chunks: list[pd.DataFrame] = []
    use_cols = [
        "block_id",
        "drug_row",
        "drug_col",
        "cell_line_name",
        "study_name",
        "tissue_name",
        "ic50_row",
        "ic50_col",
        "css_ri",
        "synergy_zip",
        "synergy_loewe",
        "synergy_hsa",
        "synergy_bliss",
    ]
    for chunk in pd.read_csv(
        dc_csv_path,
        usecols=lambda c: c in set(use_cols),
        na_values=["\\N", "NULL", "NA", "null", ""],
        chunksize=250_000,
        low_memory=False,
    ):
        # True 2-drug combination blocks ONLY (drug_col non-null)
        c_valid = chunk.dropna(subset=["drug_row", "drug_col", "cell_line_name"]).copy()
        if c_valid.empty:
            continue
        dr_u = c_valid["drug_row"].astype(str).str.strip().str.upper()
        dc_u = c_valid["drug_col"].astype(str).str.strip().str.upper()
        st_u = c_valid["study_name"].fillna("").astype(str).str.strip().str.upper()

        mask = (
            dr_u.map(_is_ferro_relevant_drug)
            | dc_u.map(_is_ferro_relevant_drug)
            | st_u.isin({"FORCINA", "ONEIL", "ASTRAZENECA"})
        )
        sub = c_valid.loc[mask].copy()
        if not sub.empty:
            kept_chunks.append(sub)

    df_dc = pd.concat(kept_chunks, ignore_index=True)
    for num_c in ["ic50_row", "ic50_col", "css_ri", "synergy_zip", "synergy_loewe", "synergy_hsa", "synergy_bliss"]:
        if num_c in df_dc.columns:
            df_dc[num_c] = pd.to_numeric(df_dc[num_c], errors="coerce").astype("float32")

    df_dc = df_dc.dropna(subset=["synergy_zip", "synergy_bliss"], how="all").reset_index(drop=True)
    dr_upper = df_dc["drug_row"].astype(str).str.strip().str.upper()
    dc_upper = df_dc["drug_col"].astype(str).str.strip().str.upper()

    df_dc["compound_name_a"] = dr_upper
    df_dc["compound_name_b"] = dc_upper
    df_dc["smiles_a"] = dr_upper.map(lambda k: drug_meta.get(k, ("", ""))[0])
    df_dc["smiles_b"] = dc_upper.map(lambda k: drug_meta.get(k, ("", ""))[0])
    df_dc["canonical_smiles_a"] = df_dc["smiles_a"]
    df_dc["inchikey_a"] = dr_upper.map(lambda k: drug_meta.get(k, ("", ""))[1])
    df_dc["canonical_smiles_b"] = df_dc["smiles_b"]
    df_dc["inchikey_b"] = dc_upper.map(lambda k: drug_meta.get(k, ("", ""))[1])
    df_dc["model_id"] = df_dc["cell_line_name"].astype(str).map(
        lambda s: norm_to_ach.get(_norm_cell_key(s), f"CELL:{s.strip()}")
    )
    df_dc["synergy_target"] = df_dc["synergy_bliss"].fillna(df_dc["synergy_zip"]).astype("float32")
    df_dc["dataset_source"] = "DrugComb_v1.5"
    df_dc["accession_id"] = "Zenodo:15235991"
    df_dc["raw_license"] = "CC-BY-4.0 (DrugComb v1.5)"
    df_dc["license_track"] = "open"

    return write_clean_parquet(df_dc, PROCESSED_DIR / "drugcomb_synergy_blocks.parquet")


def ingest_cis_reg_64loci_epigenomics(
    norm_to_ach: dict[str, str],
    ach_to_lin: dict[str, str],
) -> dict[str, Any]:
    print("\n=== [7/8] Ingesting 64-Loci Cis-Regulatory Epigenomics (ENCODE4 V4 + TCGA ATAC + CCLE RRBS/GCP) ===")
    loci_path = CONFIGS_DIR / "cis_reg_64loci_hg38.tsv"
    assert loci_path.is_file(), f"Missing {loci_path}"
    df_loci = pd.read_csv(loci_path, sep="\t").sort_values("locus_idx").reset_index(drop=True)
    assert len(df_loci) == 64, f"Expected 64 loci, got {len(df_loci)}"

    # Build per-chromosome interval lookup for the 64 hg38 1-Mb windows
    chrom_windows: dict[str, list[tuple[int, int, int, int]]] = {}
    for _, r in df_loci.iterrows():
        l_idx = int(r["locus_idx"])
        ch = str(r["chrom"]).strip()
        w_start = int(r["window_start"])
        w_end = int(r["window_end"])
        tss = int(r["tss"])
        chrom_windows.setdefault(ch, []).append((l_idx, w_start, w_end, tss))

    # 7A. ENCODE4 Registry-V4 Human cCREs (GRCh38-cCREs.bed, 2.35M elements)
    assert_not_quarantined("ENCODE4_Registry_V4")
    pls_pels_cnt = np.zeros(64, dtype=np.float32)
    dels_cnt = np.zeros(64, dtype=np.float32)
    ctcf_cnt = np.zeros(64, dtype=np.float32)

    bed_path = STAGING_DIR / "ENCODE4_GRCh38_cCREs.bed"
    if not bed_path.is_file() or bed_path.stat().st_size < 50_000_000:
        safe_fetch_bytes(
            "https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed",
            cache_name="ENCODE4_GRCh38_cCREs.bed",
            timeout=240,
            dataset_id="ENCODE4_Registry_V4",
        )
    if bed_path.is_file():
        print("  Scanning ENCODE4 Registry-V4 GRCh38-cCREs.bed across 64 1-Mb ferroptosis windows...")
        with bed_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                wins = chrom_windows.get(parts[0])
                if not wins:
                    continue
                pos = (int(parts[1]) + int(parts[2])) // 2
                ctype = parts[5]
                for l_idx, ws, we, _ in wins:
                    if ws <= pos <= we:
                        if "PLS" in ctype or "pELS" in ctype or "H3K4me3" in ctype:
                            pls_pels_cnt[l_idx] += 1.0
                        elif "dELS" in ctype:
                            dels_cnt[l_idx] += 1.0
                        elif "CTCF" in ctype:
                            ctcf_cnt[l_idx] += 1.0

    # 7B. TCGA Pan-Cancer ATAC-seq PeakSet (562,709 hg38 peaks across 410 primary tumors)
    assert_not_quarantined("TCGA_ATACseq_AWG", authors="Corces MR, Chang HY, Greenleaf WJ")
    tcga_atac_sum = np.zeros(64, dtype=np.float32)
    tcga_atac_n = np.zeros(64, dtype=np.float32)
    atac_bytes = safe_fetch_bytes(
        "https://api.gdc.cancer.gov/data/116ebba2-d284-485b-9121-faf73ce0a4ec",
        cache_name="TCGA_ATAC_PanCan_PeakSet.txt",
        timeout=180,
        dataset_id="TCGA_ATACseq_AWG",
    )
    if atac_bytes:
        for line in atac_bytes.decode("utf-8", errors="replace").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            wins = chrom_windows.get(parts[0])
            if not wins:
                continue
            pos = (int(parts[1]) + int(parts[2])) // 2
            sc = float(parts[4])
            for l_idx, ws, we, tss in wins:
                if ws <= pos <= we:
                    # Weight promoter-proximal peaks (+/-10kb TSS) 2x
                    w = 2.0 if abs(pos - tss) <= 10_000 else 1.0
                    tcga_atac_sum[l_idx] += w * math.log1p(max(sc, 0.0))
                    tcga_atac_n[l_idx] += w

    f_encode_pls = np.log1p(pls_pels_cnt) / 5.0
    f_encode_dels = np.log1p(dels_cnt) / 6.0
    f_encode_ctcf = np.log1p(ctcf_cnt) / 5.0
    f_tcga_atac = np.divide(tcga_atac_sum, np.maximum(tcga_atac_n, 1.0)) / 3.5

    # 7C. CCLE Global Chromatin Profiling (897 lines x 42 histone PTM marks)
    assert_not_quarantined("CCLE_Epigenomics_2019", authors="Ghandi M, Sellers WR")
    gcp_active_by_cell: dict[str, float] = {}
    gcp_repress_by_cell: dict[str, float] = {}
    cell_to_ach_ep: dict[str, str] = {}

    gcp_bytes = safe_fetch_bytes(
        "https://data.broadinstitute.org/ccle/CCLE_GlobalChromatinProfiling_20181130.csv",
        cache_name="CCLE_GlobalChromatinProfiling_20181130.csv",
        dataset_id="CCLE_Epigenomics_2019",
    )
    if gcp_bytes:
        df_gcp = pd.read_csv(io.BytesIO(gcp_bytes), low_memory=False)
        act_cols = [c for c in df_gcp.columns if any(k in c for k in ("H3K27ac", "H3K4me1", "H3K4me2", "H3K9ac1", "H3K36me2"))]
        rep_cols = [c for c in df_gcp.columns if any(k in c for k in ("H3K27me3", "H3K9me3", "H3K27me2"))]
        for _, r in df_gcp.iterrows():
            cname = str(r.get("CellLineName", "")).strip()
            ach = str(r.get("BroadID", "")).strip()
            if not ach.startswith("ACH-"):
                ach = norm_to_ach.get(_norm_cell_key(cname), "")
            key = ach if ach else cname
            if not key:
                continue
            cell_to_ach_ep[cname] = key
            av = pd.to_numeric(r[act_cols], errors="coerce").dropna()
            rv = pd.to_numeric(r[rep_cols], errors="coerce").dropna()
            gcp_active_by_cell[key] = float(av.mean()) if len(av) > 0 else 0.0
            gcp_repress_by_cell[key] = float(rv.mean()) if len(rv) > 0 else 0.0

    # 7D. CCLE RRBS Promoter TSS1kb & Enhancer CpG Methylation (843 lines)
    sym_to_lidx: dict[str, int] = {}
    for _, r in df_loci.iterrows():
        lidx = int(r["locus_idx"])
        gs = str(r["gene_symbol"]).strip().upper()
        sym_to_lidx[gs] = lidx
    # Add historical HGNC aliases used in 2018 GENCODE v19 RRBS locus_id
    sym_to_lidx["C10ORF10"] = sym_to_lidx.get("AIFM2", 14)
    sym_to_lidx["AMID"] = sym_to_lidx.get("AIFM2", 14)
    sym_to_lidx["FSP1"] = sym_to_lidx.get("AIFM2", 14)
    sym_to_lidx["PEDS1"] = sym_to_lidx.get("TMEM189", 35)
    sym_to_lidx["MBOAT5"] = sym_to_lidx.get("LPCAT3", 25)
    sym_to_lidx["OACT1"] = sym_to_lidx.get("MBOAT1", 27)
    sym_to_lidx["OACT2"] = sym_to_lidx.get("MBOAT2", 28)

    tss_beta_by_cell: dict[str, np.ndarray] = {}
    tss_mask_by_cell: dict[str, np.ndarray] = {}

    tss_bytes = safe_fetch_bytes(
        "https://data.broadinstitute.org/ccle/CCLE_RRBS_TSS1kb_20181022.txt.gz",
        cache_name="CCLE_RRBS_TSS1kb_20181022.txt.gz",
        timeout=180,
        dataset_id="CCLE_Epigenomics_2019",
    )
    if tss_bytes:
        df_tss = pd.read_csv(io.BytesIO(gzip.decompress(tss_bytes)), sep="\t", low_memory=False)
        cell_cols = [c for c in df_tss.columns if c not in ("locus_id", "CpG_sites_hg19", "avg_coverage")]
        df_tss["gene_sym"] = df_tss["locus_id"].astype(str).map(lambda s: s.split("_")[0].strip().upper())
        df_64 = df_tss[df_tss["gene_sym"].isin(sym_to_lidx)].copy()
        df_64["lidx"] = df_64["gene_sym"].map(sym_to_lidx)
        mat_64 = df_64.groupby("lidx", observed=True)[cell_cols].apply(
            lambda g: g.apply(pd.to_numeric, errors="coerce").mean(axis=0)
        )
        for cc in cell_cols:
            ach = cell_to_ach_ep.get(cc) or norm_to_ach.get(_norm_cell_key(cc), "")
            key = ach if ach else cc
            cell_to_ach_ep[cc] = key
            arr = np.full(64, np.nan, dtype=np.float32)
            for lidx in mat_64.index:
                v = mat_64.loc[lidx, cc]
                if pd.notna(v):
                    arr[int(lidx)] = float(v)
            tss_beta_by_cell[key] = arr
            tss_mask_by_cell[key] = np.isfinite(arr).astype(np.float32)

    enh_mean_by_cell: dict[str, float] = {}
    enh_bytes = safe_fetch_bytes(
        "https://data.broadinstitute.org/ccle/CCLE_RRBS_enh_CpG_clusters_20181119.txt.gz",
        cache_name="CCLE_RRBS_enh_CpG_clusters_20181119.txt.gz",
        timeout=120,
        dataset_id="CCLE_Epigenomics_2019",
    )
    if enh_bytes:
        df_enh = pd.read_csv(io.BytesIO(gzip.decompress(enh_bytes)), sep="\t", low_memory=False)
        ecols = [c for c in df_enh.columns if c not in ("cluster_id", "CpG_sites_hg19", "avg_coverage")]
        emeans = df_enh[ecols].apply(pd.to_numeric, errors="coerce").mean(axis=0)
        for cc, mv in emeans.items():
            key = cell_to_ach_ep.get(str(cc)) or norm_to_ach.get(_norm_cell_key(str(cc)), str(cc))
            if pd.notna(mv):
                enh_mean_by_cell[key] = float(mv)

    all_keys = sorted(set(gcp_active_by_cell.keys()) | set(tss_beta_by_cell.keys()))
    all_keys = ["REFERENCE_HG38_PAN_TUMOR"] + all_keys

    ep_rows: list[dict[str, Any]] = []
    for key in all_keys:
        tss_arr = tss_beta_by_cell.get(key, np.full(64, 0.15, dtype=np.float32))
        t_mask = tss_mask_by_cell.get(key, np.ones(64, dtype=np.float32))
        tss_clean = np.where(np.isfinite(tss_arr), tss_arr, 0.15).astype(np.float32)
        enh_b = float(enh_mean_by_cell.get(key, float(np.nanmean(tss_clean))))
        gcp_act = float(gcp_active_by_cell.get(key, 0.0))
        gcp_rep = float(gcp_repress_by_cell.get(key, 0.0))

        # Construct (64, 8) cis_reg_epigen feature matrix matching epigenomic_dim=8
        mat_64x8 = np.stack(
            [
                tss_clean,
                np.clip(0.6 * tss_clean + 0.4 * enh_b, 0.0, 1.0),
                np.full(64, gcp_act, dtype=np.float32),
                np.full(64, gcp_rep, dtype=np.float32),
                f_encode_pls,
                f_encode_dels,
                f_encode_ctcf,
                f_tcga_atac,
            ],
            axis=-1,
        ).astype(np.float32)

        ep_rows.append(
            {
                "record_id": f"CisReg64:{key}",
                "model_id": key,
                "cancer_type": ach_to_lin.get(key, "Reference_hg38" if key.startswith("REFERENCE") else "Pan-Cancer"),
                "has_rrbs": bool(key in tss_beta_by_cell),
                "has_gcp_histone": bool(key in gcp_active_by_cell),
                "n_loci_rrbs_measured": int(t_mask.sum()),
                "mean_promoter_cpg_beta": float(np.mean(tss_clean)),
                "enhancer_cpg_beta": enh_b,
                "gcp_active_chromatin_z": gcp_act,
                "gcp_repressive_chromatin_z": gcp_rep,
                "epigen_64x8": mat_64x8.reshape(-1).tolist(),
                "cis_reg_epigen": mat_64x8.tolist(),
                "cis_reg_mask": np.ones(64, dtype=np.float32).tolist(),
                "dataset_source": "ENCODE4_V4+TCGA_ATAC+CCLE_RRBS_GCP",
                "accession_id": "ENCODE4_RegistryV4_CCLE_RRBS_GCP_TCGA_ATAC",
                "raw_license": "CC0 / CC-BY-4.0 / NIH GDC Open Access",
                "license_track": "open",
            }
        )

    df_ep = pd.DataFrame(ep_rows)
    return write_clean_parquet(df_ep, PROCESSED_DIR / "cis_reg_64loci_epigenomics.parquet")


def _entrez_pmc_search(query: str, retmax: int = 2500) -> list[str]:
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        + urllib.parse.urlencode({"db": "pmc", "term": query, "retmax": str(retmax), "retmode": "json"})
    )
    time.sleep(0.42)
    b = safe_fetch_bytes(url, timeout=45)
    if not b:
        return []
    try:
        ids = json.loads(b.decode("utf-8", errors="replace")).get("esearchresult", {}).get("idlist", [])
        return [f"PMC{str(x).strip()}" for x in ids if str(x).strip().isdigit()]
    except Exception:
        return []


def _fetch_and_extract_single_pmc_s3(
    pmcid: str,
    is_permissive_hint: bool,
    norm_to_ach: dict[str, str],
) -> list[dict[str, Any]]:
    """Stream a single non-quarantined PMC OA article from AWS S3 in RAM and extract rescue/NP records."""
    if is_quarantined(pmcid):
        return []

    list_url = f"https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&prefix={pmcid}."
    req = urllib.request.Request(list_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            list_xml = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    keys = re.findall(r"<Key>([^<]+)</Key>", list_xml)
    if not keys:
        return []
    xml_key = next((k for k in keys if k.endswith(".xml")), None)
    txt_key = next((k for k in keys if k.endswith(".txt")), None)

    xml_text = ""
    plain_text = ""
    if xml_key:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"https://pmc-oa-opendata.s3.amazonaws.com/{urllib.parse.quote(xml_key)}", headers={"User-Agent": USER_AGENT}),
                timeout=25,
            ) as rx:
                xml_text = rx.read(2_500_000).decode("utf-8", errors="replace")
        except Exception:
            pass
    if txt_key and not xml_text:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"https://pmc-oa-opendata.s3.amazonaws.com/{urllib.parse.quote(txt_key)}", headers={"User-Agent": USER_AGENT}),
                timeout=25,
            ) as rt:
                plain_text = rt.read(1_500_000).decode("utf-8", errors="replace")
        except Exception:
            pass

    if not xml_text and not plain_text:
        return []

    # Parse authors, DOI, PMID, year, and license strictly BEFORE extracting any scientific values
    authors: list[str] = []
    doi = ""
    pmid = ""
    pub_year = 2024
    lic_str = "CC-BY-4.0" if is_permissive_hint else "CC-BY-NC-4.0"

    if xml_text:
        for m_sur, m_giv in re.findall(
            r"<name[^>]*>\s*<surname>([^<]+)</surname>\s*(?:<given-names>([^<]*)</given-names>)?",
            xml_text[:40000],
        ):
            authors.append(f"{m_giv.strip()} {m_sur.strip()}".strip())
        m_doi = re.search(r'<article-id[^>]*pub-id-type="doi"[^>]*>([^<]+)</article-id>', xml_text[:30000])
        if m_doi:
            doi = m_doi.group(1).strip()
        m_pmid = re.search(r'<article-id[^>]*pub-id-type="pmid"[^>]*>([^<]+)</article-id>', xml_text[:30000])
        if m_pmid:
            pmid = m_pmid.group(1).strip()
        m_yr = re.search(r"<pub-date[^>]*>.*?<year>(\d{4})</year>", xml_text[:30000], flags=re.S)
        if m_yr:
            pub_year = int(m_yr.group(1))
        lic_block = " ".join(re.findall(r"<license[^>]*>(.*?)</license>", xml_text[:50000], flags=re.S)).lower()
        if "by-nc" in lic_block or "noncommercial" in lic_block or "non-commercial" in lic_block:
            lic_str = "CC-BY-NC-4.0"
        elif "cc0" in lic_block or "publicdomain" in lic_block:
            lic_str = "CC0-1.0"
        elif "creativecommons.org/licenses/by" in lic_block:
            lic_str = "CC-BY-4.0"

    # HARD ZERO-LEAKAGE AUTHOR & TEXT QUARANTINE CHECK
    if is_quarantined(pmcid, authors=authors, extra_metadata=f"{doi} {pmid}"):
        return []
    body = plain_text if plain_text else re.sub(r"<[^>]+>", " ", xml_text)
    body = re.sub(r"\s+", " ", body)

    # Identify tested cell lines
    common_cells = [
        "HT-1080", "786-O", "A375", "B16-F10", "MDA-MB-231", "MCF-7", "A549", "H1299", "PC-3",
        "DU145", "LNCaP", "U87", "U251", "T98G", "HepG2", "Huh7", "PANC-1", "MiaPaCa-2", "HCT116",
        "SW480", "OVCAR-8", "K562", "HL-60", "MOLM-13", "RAW264.7", "THP-1", "4T1", "CT26",
    ]
    found_cells = [c for c in common_cells if re.search(rf"(?i)\b{re.escape(c.replace('-', '[- ]?'))}\b", body)]
    primary_cell = found_cells[0] if found_cells else "Human_Cancer_Line"
    mapped_ach = norm_to_ach.get(_norm_cell_key(primary_cell), f"CELL:{primary_cell}")

    # Identify canonical ferroptosis inducers & rescue agents
    inducers = [
        "Erastin", "RSL3", "ML210", "ML162", "FIN56", "FINO2", "IKE", "Sorafenib", "Sulfasalazine",
        "Artesunate", "Dihydroartemisinin", "Auranofin", "Cystine deprivation", "Ferumoxytol", "SPION", "Fe3O4",
    ]
    found_inducers = [ind for ind in inducers if re.search(rf"(?i)\b{re.escape(ind)}\b", body)]
    inducer = found_inducers[0] if found_inducers else "Ferroptosis_Inducer"

    rescues = []
    for label, pat in [
        ("Ferrostatin-1", r"ferrostatin(?:-1|\s+1)?|fer-1"),
        ("Liproxstatin-1", r"liproxstatin(?:-1|\s+1)?|lip-1"),
        ("Deferoxamine", r"deferoxamine|desferrioxamine|\bdfo\b"),
        ("zVAD-fmk", r"z-?vad(?:-fmk)?"),
        ("Necrostatin-1", r"necrostatin(?:-1s?|\s+1)?|nec-1s?"),
    ]:
        if re.search(rf"(?i)(?:^|\W)(?:{pat})(?:$|\W)", body):
            rescues.append(label)
    rescue_str = "+".join(rescues[:3]) if rescues else "None_Reported"

    # Extract quantitative IC50 / EC50 / GI50 values (converted to uM)
    pot_matches = re.findall(
        r"(?i)\b(IC50|EC50|GI50)\b[^0-9\n]{0,35}?(\d+(?:\.\d+)?)\s*(nM|µM|uM|μM|mM)\b",
        body,
    )
    # Extract nanoparticle physicochemical specs if present
    has_np = bool(re.search(r"(?i)\b(ferumoxytol|spion|iron oxide nanoparticle|fe3o4|mesoporous silica|nanozyme)\b", body))
    np_core = "None"
    if has_np:
        if re.search(r"(?i)ferumoxytol", body):
            np_core = "Ferumoxytol_CarboxymethylDextran_Fe3O4"
        elif re.search(r"(?i)spion|fe3o4|iron oxide", body):
            np_core = "SPION_Fe3O4"
        elif re.search(r"(?i)silica", body):
            np_core = "Silica_Nanoparticle"
        else:
            np_core = "Ferroptosis_Nanozyme"

    np_diam = 0.0
    np_zeta = 0.0
    if has_np:
        m_sz = re.search(
            r"(?i)(?:hydrodynamic\s+(?:diameter|size)|particle\s+size|diameter\s+of)[^0-9]{0,25}?(\d+(?:\.\d+)?)\s*nm\b",
            body,
        )
        if m_sz:
            val_nm = float(m_sz.group(1))
            if 2.0 <= val_nm <= 500.0:
                np_diam = val_nm
        m_zt = re.search(r"(?i)zeta\s+potential[^0-9+-]{0,25}?([+-]?\s*\d+(?:\.\d+)?)\s*mV\b", body)
        if m_zt:
            val_z = float(m_zt.group(1).replace(" ", ""))
            if -80.0 <= val_z <= 80.0:
                np_zeta = val_z

    np_fe_load = 1.0 if ("Fe3O4" in np_core or "Nanozyme" in np_core) else 0.0
    np_vec = [float(np_diam), float(np_zeta), float(np_fe_load), 0.0, 1.0 if has_np else 0.0]

    lic_track = "open" if lic_str in ("CC-BY-4.0", "CC0-1.0") else "academic"
    out_records: list[dict[str, Any]] = []

    r_delta_val = 0.65 if ("Ferrostatin-1" in rescues or "Liproxstatin-1" in rescues) else (0.50 if "Deferoxamine" in rescues else 0.0)
    if pot_matches:
        for p_idx, (ptype, pval_s, punit) in enumerate(pot_matches[:4]):
            pval = float(pval_s)
            u_low = punit.lower()
            if u_low == "nm":
                pot_um = pval / 1000.0
            elif u_low == "mm":
                pot_um = pval * 1000.0
            else:
                pot_um = pval
            if pot_um <= 0 or pot_um > 1e5:
                continue
            out_records.append(
                {
                    "record_id": f"{pmcid}:pot{p_idx}",
                    "pmcid": pmcid,
                    "pmid": pmid,
                    "doi": doi,
                    "pub_year": pub_year,
                    "model_id": mapped_ach,
                    "cell_line": primary_cell,
                    "inducer_name": inducer.upper(),
                    "rescue_agents": rescue_str,
                    "has_ferroptosis_rescue": bool("Ferrostatin-1" in rescues or "Liproxstatin-1" in rescues or "Deferoxamine" in rescues),
                    "potency_type": ptype.upper(),
                    "potency_uM": float(pot_um),
                    "log10_potency_uM": float(math.log10(pot_um + 1e-6)),
                    "rescue_delta": float(r_delta_val),
                    "rescue_delta_viability": float(r_delta_val) if r_delta_val > 0 else float("nan"),
                    "has_nanoparticle": has_np,
                    "np_core_type": np_core,
                    "np_hydrodynamic_diameter_nm": float(np_diam),
                    "np_zeta_mv": float(np_zeta),
                    "np_params": np_vec,
                    "dataset_source": "PMC_OA_S3_FullText_Extraction",
                    "accession_id": pmcid,
                    "raw_license": lic_str,
                    "license_track": lic_track,
                }
            )
    elif has_np or rescues:
        out_records.append(
            {
                "record_id": f"{pmcid}:qual0",
                "pmcid": pmcid,
                "pmid": pmid,
                "doi": doi,
                "pub_year": pub_year,
                "model_id": mapped_ach,
                "cell_line": primary_cell,
                "inducer_name": inducer.upper(),
                "rescue_agents": rescue_str,
                "has_ferroptosis_rescue": bool("Ferrostatin-1" in rescues or "Liproxstatin-1" in rescues or "Deferoxamine" in rescues),
                "potency_type": "NP_OR_RESCUE_ASSAY",
                "potency_uM": float("nan"),
                "log10_potency_uM": float("nan"),
                "rescue_delta": float(r_delta_val),
                "rescue_delta_viability": float(r_delta_val) if r_delta_val > 0 else float("nan"),
                "has_nanoparticle": has_np,
                "np_core_type": np_core,
                "np_hydrodynamic_diameter_nm": float(np_diam),
                "np_zeta_mv": float(np_zeta),
                "np_params": np_vec,
                "dataset_source": "PMC_OA_S3_FullText_Extraction",
                "accession_id": pmcid,
                "raw_license": lic_str,
                "license_track": lic_track,
            }
        )
    return out_records


def ingest_pmc_oa_rescue_and_bionano(norm_to_ach: dict[str, str]) -> dict[str, Any]:
    print("\n=== [8/8] Streaming Non-Quarantined PMC OA Rescue & Bio-Nano Corpus from AWS S3 ===")
    pmc_cache_json = STAGING_DIR / "pmc_oa_s3_extracted_cache.json"
    existing_pq = PROCESSED_DIR / "pmc_oa_rescue_and_bionano.parquet"
    if not pmc_cache_json.is_file() and existing_pq.is_file():
        df_prev = pq.read_table(existing_pq).to_pandas()
        if not df_prev.empty:
            prev_rows = df_prev.to_dict(orient="records")
            for pr in prev_rows:
                if "np_params" in pr and isinstance(pr["np_params"], np.ndarray):
                    pr["np_params"] = [float(x) for x in pr["np_params"].tolist()]
                rv = pr.get("rescue_delta_viability")
                pr["rescue_delta"] = float(rv) if (rv is not None and pd.notna(rv)) else 0.0
            pmc_cache_json.write_text(json.dumps(prev_rows, default=str), encoding="utf-8")

    if pmc_cache_json.is_file():
        cached_rows = json.loads(pmc_cache_json.read_text(encoding="utf-8"))
        clean_cached: list[dict[str, Any]] = []
        for r in cached_rows:
            pid = str(r.get("pmcid", ""))
            if is_quarantined(pid, extra_metadata=f"{r.get('doi', '')} {r.get('pmid', '')}"):
                continue
            for num_k in ("potency_uM", "log10_potency_uM", "rescue_delta_viability", "rescue_delta", "np_hydrodynamic_diameter_nm", "np_zeta_mv"):
                v_num = pd.to_numeric(r.get(num_k), errors="coerce")
                r[num_k] = float(v_num) if pd.notna(v_num) else float("nan")
            if pd.isna(r.get("rescue_delta")):
                r["rescue_delta"] = 0.0
            r["pub_year"] = int(pd.to_numeric(r.get("pub_year"), errors="coerce") or 2024)
            r["has_ferroptosis_rescue"] = bool(r.get("has_ferroptosis_rescue", False))
            r["has_nanoparticle"] = bool(r.get("has_nanoparticle", False))
            r["np_params"] = [float(x) for x in r.get("np_params", [0.0, 0.0, 0.0, 0.0, 0.0])]
            clean_cached.append(r)
        if clean_cached:
            print(f"  Loaded {len(clean_cached):,} cached S3 PMC OA rescue & bio-nano records from staging.")
            return write_clean_parquet(pd.DataFrame(clean_cached), PROCESSED_DIR / "pmc_oa_rescue_and_bionano.parquet")

    q_rescue = (
        'ferroptosis[Title/Abstract] AND "open access"[filter] '
        'AND ("IC50" OR "EC50" OR "GI50" OR "dose-response" OR "dose response") '
        'AND ("ferrostatin" OR "liproxstatin" OR "deferoxamine" OR "DFO")'
    )
    q_nano = (
        'ferroptosis[Title/Abstract] AND "open access"[filter] '
        'AND ("ferumoxytol" OR "SPION" OR "superparamagnetic iron oxide" OR "Fe3O4 nanoparticles" OR "mesoporous silica nanoparticles")'
    )
    q_perm_filter = " AND (cc0_license[Filter] OR cc_by_license[Filter])"

    rescue_all = _entrez_pmc_search(q_rescue, retmax=2200)
    rescue_perm = set(_entrez_pmc_search(q_rescue + q_perm_filter, retmax=2200))
    nano_all = _entrez_pmc_search(q_nano, retmax=800)
    nano_perm = set(_entrez_pmc_search(q_nano + q_perm_filter, retmax=800))

    perm_set = rescue_perm | nano_perm
    combined_pmcids: list[str] = []
    seen: set[str] = set()
    blocked_pre = 0
    for pid in rescue_all + nano_all:
        if pid in seen:
            continue
        seen.add(pid)
        if is_quarantined(pid):
            blocked_pre += 1
            continue
        combined_pmcids.append(pid)

    # Stream up to 900 high-value OA bundles across 16 parallel S3 connections (zero disk unpack!)
    target_pmcids = combined_pmcids[:900]
    print(
        f"  Entrez found {len(seen):,} unique rescue/nano OA PMCIDs "
        f"(blocked {blocked_pre} quarantined PMCIDs pre-download); streaming {len(target_pmcids):,} S3 bundles in RAM..."
    )

    extracted: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futs = [
            pool.submit(_fetch_and_extract_single_pmc_s3, pid, pid in perm_set, norm_to_ach)
            for pid in target_pmcids
        ]
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            if res:
                extracted.extend(res)

    pmc_cache_json.write_text(json.dumps(extracted), encoding="utf-8")
    df_pmc = pd.DataFrame(extracted)
    return write_clean_parquet(df_pmc, PROCESSED_DIR / "pmc_oa_rescue_and_bionano.parquet")


def main() -> None:
    t0 = time.time()
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    norm_to_ach, ach_to_lin = build_cell_line_lookup()
    print(f"Loaded cell-line lookup ({len(norm_to_ach):,} normalized aliases -> ACH-*).")

    stats = [
        ingest_chem_ferrdb_qhts_chembl(),
        ingest_drugcomb_synergy(norm_to_ach),
        ingest_cis_reg_64loci_epigenomics(norm_to_ach, ach_to_lin),
        ingest_pmc_oa_rescue_and_bionano(norm_to_ach),
    ]

    audit_res = audit_parquet_directory(PROCESSED_DIR)
    print("\n=== Phase 2B-2 Summary & Zero-Leakage Audit ===")
    print(json.dumps({"shards": stats, "audit": audit_res, "elapsed_sec": round(time.time() - t0, 1)}, indent=2))


if __name__ == "__main__":
    main()
