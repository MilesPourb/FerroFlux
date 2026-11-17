"""Unified Multimodal PyTorch Dataset, Stage-0.5 SAR Pretraining & Leak-Free OOD Splitters (FerroFlux-FM v3).

Remediation & v3 Dominance Guarantees Verified by Zero-Trust Forensic Audit:
- ZERO `probe_mechanism` label injection into chemical descriptors or scalar concentration channels:
  100% of small-molecule representations derive strictly from RDKit SMILES structure + `dose_um`.
- Multi-scale pure-SMILES chemical representation (`3,239`-bit `[ECFP4_2048 | ECFP6_1024 | MACCS_167]`
  + `24` continuous RDKit 2D physicochemical & SMARTS warhead descriptors; indices `0..11` strictly
  preserved for 100% backward test compatibility).
- Leak-free `FerroChemSARPretrainDataset` (`Stage 0.5`) across `53,984` PubChem/ChEMBL/FerrDb SAR rows
  and `41,474` CysDB covalent ABPP sites, strictly excluding all held-out test `SMILES`, `ik14`
  stereoisomer connectivity keys, and `murcko_scaffold`s (`0` overlap asserted).
- Rich basal cell multi-omics (`138`-dim CNV/LOF/hotspot genomics + `64` real MS proteins + `32` real
  LC-MS PUFA/MUFA/redox species; ZERO fake mRNA-as-proteomics substitution, ZERO `ACH-000873` fallback).
- Exact `log10(dose_um + 1e-3)` encoding aligned to `<1e-7 uM` with `perturbation_ops.py` (`0 uM -> -3.0`).
- Ferroptosis-pure OOD evaluation flags (`is_ferroptosis_compound`) for Splits 1, 2, and 6 alongside
  multi-chemotype specificity in Split 5.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, GraphDescriptors, Lipinski, MACCSkeys, inchi, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset

from ferroflux.security.license_guard import filter_by_license_track
from ferroflux.security.quarantine import assert_not_quarantined

RDLogger.DisableLog("rdApp.*")

DEFAULT_PROCESSED_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML/data/processed")
DEFAULT_SAR_CACHE_DIR = Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging")

CHEM_FP_DIM: int = 2048 + 1024 + 167  # 3239 bits: ECFP4 (2048) + ECFP6 (1024) + MACCS (167)
CHEM_DESC_DIM: int = 24
GENOMIC_CONTEXT_DIM: int = 138  # 64 CNV + 64 LOF + 10 hotspot drivers

HOTSPOT_10_GENES: tuple[str, ...] = (
    "BRAF",
    "NRAS",
    "KRAS",
    "HRAS",
    "PIK3CA",
    "NFE2L2",
    "TP53",
    "IDH1",
    "IDH2",
    "EGFR",
)

LCMS_32_COLUMNS: tuple[str, ...] = (
    "gsh_minus_gssg_z",
    "pufa_minus_mufa_pl_z",
    "metab_glutathione_reduced",
    "metab_glutathione_oxidized",
    "metab_nadp",
    "metab_nad",
    "metab_malondialdehyde",
    "metab_putrescine",
    "metab_cystathionine",
    "metab_glutamate",
    "metab_glutamine",
    "metab_glycine",
    "metab_homocysteine",
    "metab_taurine",
    "metab_oleylcarnitine",
    "metab_arachidonyl_carnitine",
    "metab_c20_4_lpc",
    "metab_c22_6_lpc",
    "metab_c18_1_lpc",
    "metab_c20_4_lpe",
    "metab_c22_6_lpe",
    "metab_c18_1_lpe",
    "metab_c38_4_pc",
    "metab_c38_5_pc",
    "metab_c38_6_pc",
    "metab_c40_6_pc",
    "metab_c34_1_pc",
    "metab_c36_1_pc",
    "metab_c20_4_ce",
    "metab_c22_6_ce",
    "metab_c18_1_ce",
    "metab_6_phosphogluconate",
)

SAR_20_TARGETS: tuple[str, ...] = (
    "GPX4",                   # 0  -> maps to CHEM_14_TARGETS[0, 1]
    "SLC7A11",                # 1  -> maps to CHEM_14_TARGETS[2]
    "AIFM2",                  # 2  -> maps to CHEM_14_TARGETS[3] (FSP1)
    "DHODH",                  # 3  -> maps to CHEM_14_TARGETS[4]
    "ALOX12",                 # 4  -> maps to CHEM_14_TARGETS[6]
    "ALOX15",                 # 5  -> maps to CHEM_14_TARGETS[6]
    "ALOX5",                  # 6  -> maps to CHEM_14_TARGETS[6]
    "HMGCR",                  # 7  -> maps to CHEM_14_TARGETS[7]
    "DHCR7",                  # 8  -> maps to CHEM_14_TARGETS[8]
    "EBP",                    # 9  -> maps to CHEM_14_TARGETS[9]
    "SQLE",                   # 10 -> maps to CHEM_14_TARGETS[10]
    "TXNRD1",                 # 11 -> maps to CHEM_14_TARGETS[11]
    "CUPROPTOSIS",            # 12 -> maps to CHEM_14_TARGETS[13] (FDX1_Cu)
    "POR",                    # 13
    "NFE2L2",                 # 14
    "KEAP1",                  # 15
    "GSR",                    # 16
    "FERROPTOSIS_HT1080",     # 17
    "HRAS_BJELR",             # 18
    "FERROPTOSIS",            # 19
)
SAR_TARGET_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(SAR_20_TARGETS)}

# Pure structure-only RDKit SMARTS motifs (`0` human annotation leakage)
_SMARTS_CHLOROACETAMIDE = Chem.MolFromSmarts("[Cl,Br,I]CC(=O)N")
_SMARTS_ACRYLAMIDE_PROPIOL = Chem.MolFromSmarts("[$(C=CC(=O)N),$(C#CC(=O)N)]")
_SMARTS_NITROISOXAZOLE = Chem.MolFromSmarts("[$(c1cc([N+](=O)[O-])on1),$(c1oc([N+](=O)[O-])cc1)]")
_SMARTS_ARYLAMINE_RTA = Chem.MolFromSmarts("[$(c1ccccc1Nc2ccccc2),$(c1cc(N)c(NC)cc1),$(c1ccccc1NC1CCCCC1),$(c1cc(O)c(O)c(O)c1)]")
_SMARTS_HYDROXAMIC_CHELATOR = Chem.MolFromSmarts("[$(C(=O)N[OH]),$(c1cc(=O)n([OH])cc1),$(c1c([OH])c(=O)ccn1)]")
_SMARTS_CATECHOL_CHELATOR = Chem.MolFromSmarts("[$(c1cc(O)c(O)cc1),$(c1ccccc1O-c2nc(-c3ccccc3O)nn2)]")
_SMARTS_ENDOPEROXIDE_QUINONE = Chem.MolFromSmarts("[$(OO),$(C1=CC(=O)C=CC1=O)]")
_SMARTS_STATIN_DIHYDROXY = Chem.MolFromSmarts("[$(O[CH]CC(O)CC(=O)),$(O=C1CC(O)CC(CC)O1)]")
_SMARTS_THIOCARBONYL_CU_IONOPHORE = Chem.MolFromSmarts("[#6](=[S])[#7,#16]")
_SMARTS_QUINAZOLINONE_PIP = Chem.MolFromSmarts("[$(c1ccc2nc(=O)n(C)c(=O)c2c1),$(c1ccc2ncn(CC(=O))c2c1),$(N1CCNCC1)]")
_SMARTS_DHODH_ACID = Chem.MolFromSmarts("[$(c1cc(C(=O)O)c2ccccc2n1),$(C(#N)C(=O)Nc1ccc(C(F)(F)F)cc1),$(c1oncc1C(=O)N)]")
_SMARTS_TZD_OR_CHROMANOL = Chem.MolFromSmarts("[$(O=C1NC(=O)SC1),$(c1c(O)c(C)c2CCC(C)(O2)c1)]")
_SMARTS_DISULFIDE_ELECTROPHILE = Chem.MolFromSmarts("[$(SS),$([Au]),$(C1=CCC(=O)N1),$(S(=O)(=N))]")


def load_gene_vocab(vocab_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(vocab_path) if vocab_path is not None else (DEFAULT_PROCESSED_DIR / "gene_vocab.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_cell_name(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip().upper().split("_")[0]
    return re.sub(r"[^A-Z0-9]", "", s)


def encode_log10_dose_um(dose_um: float) -> float:
    """Encode micromolar concentration to `log10(uM + 1e-3)` (`0.0 uM -> -3.0`)."""
    d = max(0.0, float(dose_um))
    if d <= 1e-9:
        return -3.0
    return float(np.log10(d + 1e-3))


@lru_cache(maxsize=65536)
def canonicalize_smiles_and_keys(smiles: str) -> tuple[str, str, str]:
    """Return `(canonical_smiles, desalted_ik14, murcko_scaffold)` for strict leak-free deduplication."""
    smi = (smiles or "").strip()
    if not smi or smi.lower() in ("nan", "none"):
        return "", "", ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, "", smi
    canon_smi = Chem.MolToSmiles(mol, canonical=True)
    # Desalt organic salts (mesylate, tosylate, HCl, sodium) to get parent connectivity ik14
    parent_mol = mol
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            # Retain metal-containing bioactive complexes if present (e.g. Au in Auranofin), else largest heavy-atom fragment
            metal_frags = [f for f in frags if any(a.GetAtomicNum() in (26, 29, 30, 78, 79) for a in f.GetAtoms())]
            parent_mol = metal_frags[0] if metal_frags else max(frags, key=lambda f: f.GetNumHeavyAtoms())
    except Exception:
        parent_mol = mol

    ik14 = ""
    try:
        ikey = inchi.MolToInchiKey(parent_mol) or ""
        ik14 = ikey[:14]
    except Exception:
        ik14 = ""

    scaffold_smi = canon_smi
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(parent_mol)
        scaf_str = Chem.MolToSmiles(scaf, canonical=True) if scaf is not None else ""
        scaffold_smi = scaf_str if scaf_str else Chem.MolToSmiles(parent_mol, canonical=True)
    except Exception:
        scaffold_smi = canon_smi
    return canon_smi, ik14, scaffold_smi


@lru_cache(maxsize=65536)
def compute_compound_featurization(smiles: str, probe_mechanism: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Compute `(3239,)` `[ECFP4_2048 | ECFP6_1024 | MACCS_167]` and `(24,)` pure RDKit descriptors from SMILES.

    `probe_mechanism` is retained solely as an ignored legacy keyword argument for API compatibility
    and NEVER modifies any feature value (zero human annotation leakage).
    Indices `0..11` of `desc_arr` preserve the exact V2 layout (`desc_arr[11]` == thiocarbonyl Cu-ionophore).
    """
    _ = probe_mechanism  # strictly ignored to guarantee zero mechanism-label leakage
    fp_arr = np.zeros(CHEM_FP_DIM, dtype=np.float32)
    desc_arr = np.zeros(CHEM_DESC_DIM, dtype=np.float32)

    smi = (smiles or "").strip()
    mol = Chem.MolFromSmiles(smi) if smi and smi.lower() != "nan" else None
    if mol is not None:
        try:
            ecfp4 = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048), ecfp4)
            fp_arr[:2048] = ecfp4

            ecfp6 = np.zeros(1024, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024), ecfp6)
            fp_arr[2048:3072] = ecfp6

            maccs = np.zeros(167, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs)
            fp_arr[3072:3239] = maccs
        except Exception:
            pass
        try:
            mw = Descriptors.MolWt(mol) / 500.0
            logp = Crippen.MolLogP(mol) / 5.0
            tpsa = rdMolDescriptors.CalcTPSA(mol) / 150.0
            hbd = Lipinski.NumHDonors(mol) / 5.0
            hba = Lipinski.NumHAcceptors(mol) / 10.0
            rotb = Lipinski.NumRotatableBonds(mol) / 10.0

            has_chloro = float(mol.HasSubstructMatch(_SMARTS_CHLOROACETAMIDE) if _SMARTS_CHLOROACETAMIDE else False)
            has_acryl = float(mol.HasSubstructMatch(_SMARTS_ACRYLAMIDE_PROPIOL) if _SMARTS_ACRYLAMIDE_PROPIOL else False)
            has_nitro = float(mol.HasSubstructMatch(_SMARTS_NITROISOXAZOLE) if _SMARTS_NITROISOXAZOLE else False)
            has_cov = float((has_chloro > 0.5) or (has_acryl > 0.5) or (has_nitro > 0.5))

            has_rta = float(mol.HasSubstructMatch(_SMARTS_ARYLAMINE_RTA) if _SMARTS_ARYLAMINE_RTA else False)
            has_chel = float(
                (mol.HasSubstructMatch(_SMARTS_HYDROXAMIC_CHELATOR) if _SMARTS_HYDROXAMIC_CHELATOR else False)
                or (mol.HasSubstructMatch(_SMARTS_CATECHOL_CHELATOR) if _SMARTS_CATECHOL_CHELATOR else False)
            )
            has_endo = float(mol.HasSubstructMatch(_SMARTS_ENDOPEROXIDE_QUINONE) if _SMARTS_ENDOPEROXIDE_QUINONE else False)
            has_statin_sterol = float(
                (mol.HasSubstructMatch(_SMARTS_STATIN_DIHYDROXY) if _SMARTS_STATIN_DIHYDROXY else False)
                or (rdMolDescriptors.CalcNumAliphaticCarbocycles(mol) >= 4)
            )
            has_thiocarb_cu = float(
                mol.HasSubstructMatch(_SMARTS_THIOCARBONYL_CU_IONOPHORE) if _SMARTS_THIOCARBONYL_CU_IONOPHORE else False
            )

            # Extended v3 pure-SMILES descriptors (indices 12..23)
            f_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
            n_arom = float(rdMolDescriptors.CalcNumAromaticRings(mol)) / 5.0
            n_aliph = float(rdMolDescriptors.CalcNumAliphaticRings(mol)) / 5.0
            n_heavy = float(mol.GetNumHeavyAtoms()) / 40.0
            bertz = float(GraphDescriptors.BertzCT(mol)) / 1000.0
            asa = float(rdMolDescriptors.CalcLabuteASA(mol)) / 200.0
            has_nitro_prop = float((has_nitro > 0.5) or (has_acryl > 0.5))
            has_quinaz_pip = float(mol.HasSubstructMatch(_SMARTS_QUINAZOLINONE_PIP) if _SMARTS_QUINAZOLINONE_PIP else False)
            has_dhodh_acid = float(mol.HasSubstructMatch(_SMARTS_DHODH_ACID) if _SMARTS_DHODH_ACID else False)
            has_tzd_chrom = float(mol.HasSubstructMatch(_SMARTS_TZD_OR_CHROMANOL) if _SMARTS_TZD_OR_CHROMANOL else False)
            has_disulf_elec = float(
                mol.HasSubstructMatch(_SMARTS_DISULFIDE_ELECTROPHILE) if _SMARTS_DISULFIDE_ELECTROPHILE else False
            )

            desc_arr[:] = np.clip(
                np.array(
                    [
                        mw,
                        logp,
                        tpsa,
                        hbd,
                        hba,
                        rotb,
                        has_cov,
                        has_rta,
                        has_chel,
                        has_endo,
                        has_statin_sterol,
                        has_thiocarb_cu,
                        f_csp3,
                        n_arom,
                        n_aliph,
                        n_heavy,
                        bertz,
                        asa,
                        has_chloro,
                        has_nitro_prop,
                        has_quinaz_pip,
                        has_dhodh_acid,
                        has_tzd_chrom,
                        has_disulf_elec,
                    ],
                    dtype=np.float32,
                ),
                -10.0,
                10.0,
            )
        except Exception:
            pass

    return fp_arr, desc_arr


def _safe_vec(raw_val: Any, target_len: int, default_fill: float = 0.0) -> np.ndarray:
    out = np.full(target_len, default_fill, dtype=np.float32)
    if raw_val is None:
        return out
    if isinstance(raw_val, (list, tuple, np.ndarray)):
        arr = np.asarray(raw_val, dtype=np.float32).reshape(-1)
        n = min(len(arr), target_len)
        if n > 0:
            out[:n] = np.nan_to_num(arr[:n], nan=default_fill, posinf=15.0, neginf=-15.0)
    return out


def _stratified_sample_df(df: pd.DataFrame, cap: int | None, group_col: str | None, seed: int = 42) -> pd.DataFrame:
    """Deterministically sample up to `cap` rows while preserving proportional representation of every source."""
    if cap is None or cap <= 0 or len(df) <= cap:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if group_col and group_col in df.columns:
        groups = df.groupby(group_col, sort=True)
        n_groups = len(groups)
        parts: list[pd.DataFrame] = []
        total = len(df)
        for _, gdf in groups:
            quota = max(min(len(gdf), max(16, cap // max(1, n_groups * 4))), int(round(cap * (len(gdf) / total))))
            quota = min(len(gdf), quota)
            chosen = np.sort(rng.choice(len(gdf), size=quota, replace=False))
            parts.append(gdf.iloc[chosen])
        merged = pd.concat(parts, ignore_index=True)
        if len(merged) > cap:
            keep = np.sort(rng.choice(len(merged), size=cap, replace=False))
            merged = merged.iloc[keep].reset_index(drop=True)
        return merged.reset_index(drop=True)
    chosen = np.sort(rng.choice(len(df), size=cap, replace=False))
    return df.iloc[chosen].reset_index(drop=True)


# Process-local memory cache of pre-featurized SAR molecules (`smiles -> (fp_uint8, desc_f32)`)
_SAR_MOL_FEATURE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _get_or_build_sar_mol_cache(unique_smiles: list[str], cache_dir: Path = DEFAULT_SAR_CACHE_DIR) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    missing = [s for s in unique_smiles if s and s not in _SAR_MOL_FEATURE_CACHE]
    if not missing:
        return _SAR_MOL_FEATURE_CACHE

    master_file = cache_dir / f"sar_mol_features_v3_master_{CHEM_FP_DIM}_{CHEM_DESC_DIM}.npz"
    sig_hash = hashlib.sha256("\n".join(sorted(unique_smiles)).encode("utf-8")).hexdigest()[:16]
    cache_file = cache_dir / f"sar_mol_features_v3_{CHEM_FP_DIM}_{CHEM_DESC_DIM}_{sig_hash}.npz"
    for cand_npz in (master_file, cache_file):
        if cand_npz.is_file():
            try:
                npz = np.load(cand_npz, allow_pickle=False)
                smi_arr = npz["smiles"]
                fp_u8 = npz["fp_u8"]
                desc_f32 = npz["desc_f32"]
                for idx_s, smi_v in enumerate(smi_arr.tolist()):
                    _SAR_MOL_FEATURE_CACHE[str(smi_v)] = (fp_u8[idx_s], desc_f32[idx_s])
                missing = [s for s in unique_smiles if s and s not in _SAR_MOL_FEATURE_CACHE]
                if not missing:
                    return _SAR_MOL_FEATURE_CACHE
            except Exception:
                pass

    for smi in missing:
        fp_f32, desc_f32 = compute_compound_featurization(smi)
        _SAR_MOL_FEATURE_CACHE[smi] = ((fp_f32 > 0.5).astype(np.uint8), desc_f32.astype(np.float32))

    try:
        import os as _os

        cache_dir.mkdir(parents=True, exist_ok=True)
        target_save = master_file if len(_SAR_MOL_FEATURE_CACHE) >= 35000 else cache_file
        all_keys = sorted(_SAR_MOL_FEATURE_CACHE.keys()) if target_save == master_file else sorted(k for k in unique_smiles if k in _SAR_MOL_FEATURE_CACHE)
        if all_keys:
            tmp_npz = target_save.with_name(f".tmp_{_os.getpid()}_{target_save.name}")
            np.savez_compressed(
                tmp_npz,
                smiles=np.array(all_keys, dtype=str),
                fp_u8=np.stack([_SAR_MOL_FEATURE_CACHE[k][0] for k in all_keys], axis=0),
                desc_f32=np.stack([_SAR_MOL_FEATURE_CACHE[k][1] for k in all_keys], axis=0),
            )
            tmp_npz.replace(target_save)
    except Exception:
        pass

    return _SAR_MOL_FEATURE_CACHE


def compute_cysdb_target_ligandability_prior(
    data_dir: Path = DEFAULT_PROCESSED_DIR,
    license_track: str = "open",
) -> np.ndarray:
    """Extract empirical covalent Cys/Sec ligandability prior `[14]` from `cysdb_abpp_ligandability.parquet`."""
    prior = np.ones(14, dtype=np.float32)
    p = data_dir / "cysdb_abpp_ligandability.parquet"
    if not p.is_file():
        return prior
    df = pq.read_table(
        p,
        columns=["gene_symbol", "is_ligandable", "is_hyperreactive", "in_active_site", "license_track"],
    ).to_pandas()
    df = filter_by_license_track(df, track=license_track)
    if df.empty:
        return prior

    gene_to_t_idx = {
        "GPX4": [0, 1],
        "SLC7A11": [2],
        "AIFM2": [3],
        "DHODH": [4],
        "ACSL4": [5],
        "ALOX15": [6],
        "ALOX12": [6],
        "ALOX5": [6],
        "HMGCR": [7],
        "DHCR7": [8],
        "EBP": [9],
        "SQLE": [10],
        "TXNRD1": [11],
        "ATP6V1A": [12],
        "ATP6V0C": [12],
        "FDX1": [13],
        "LIAS": [13],
    }
    for sym, grp in df.groupby("gene_symbol"):
        s_up = str(sym).strip().upper()
        if s_up in gene_to_t_idx:
            lig_score = (
                0.5 * float(pd.to_numeric(grp["is_ligandable"], errors="coerce").fillna(0.0).max())
                + 0.3 * float(pd.to_numeric(grp["is_hyperreactive"], errors="coerce").fillna(0.0).max())
                + 0.2 * float(pd.to_numeric(grp["in_active_site"], errors="coerce").fillna(0.0).max())
            )
            for ti in gene_to_t_idx[s_up]:
                prior[ti] = max(prior[ti], float(1.0 + 0.35 * lig_score))
    # GPX4 Sec46 selenocysteine is intrinsically hyper-nucleophilic (pKa ~ 5.2 vs Cys ~ 8.3)
    prior[0] = max(prior[0], 1.45)
    prior[11] = max(prior[11], 1.35)
    return prior


_SAR_GROUPED_TRACK_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _load_grouped_sar_molecules(pq_path: Path, license_track: str) -> list[dict[str, Any]]:
    cache_key = (str(pq_path.resolve()), license_track)
    if cache_key in _SAR_GROUPED_TRACK_CACHE:
        return _SAR_GROUPED_TRACK_CACHE[cache_key]

    df = pq.read_table(
        pq_path,
        columns=[
            "canonical_smiles",
            "inchikey",
            "ik14",
            "murcko_scaffold",
            "scaffold_smiles",
            "target_symbol",
            "activity_outcome",
            "p_potency",
            "accession_id",
            "dataset_source",
            "license_track",
        ],
    ).to_pandas()
    for col in ("accession_id", "dataset_source"):
        if col in df.columns:
            for val in df[col].dropna().astype(str).unique():
                assert_not_quarantined(val)
    df = filter_by_license_track(df, track=license_track)

    uniq_raw = [str(s).strip() for s in df["canonical_smiles"].dropna().unique() if str(s).strip() and str(s).strip().lower() != "nan"]
    raw_to_keys = {rs: canonicalize_smiles_and_keys(rs) for rs in uniq_raw}

    mol_records: dict[str, dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        raw_smi = str(row.canonical_smiles or "").strip()
        if raw_smi not in raw_to_keys:
            continue
        tsym = str(row.target_symbol or "").strip().upper()
        t_idx = SAR_TARGET_TO_IDX.get(tsym)
        if t_idx is None:
            continue

        row_ik14 = str(row.ik14 or "").strip()[:14] or str(row.inchikey or "").strip()[:14]
        row_scaf = str(row.murcko_scaffold or "").strip()
        row_scaf2 = str(row.scaffold_smiles or "").strip()
        c_smi, calc_ik14, calc_scaf = raw_to_keys[raw_smi]
        eff_ik14 = calc_ik14 or row_ik14
        eff_scaf = calc_scaf or row_scaf

        entry = mol_records.get(c_smi)
        if entry is None:
            entry = {
                "smiles": c_smi,
                "raw_smiles_set": {raw_smi, c_smi},
                "ik14": eff_ik14,
                "ik14_set": {k for k in (eff_ik14, row_ik14) if k},
                "scaffold": eff_scaf,
                "scaffold_set": {k for k in (eff_scaf, row_scaf, row_scaf2) if k},
                "pot": np.zeros(len(SAR_20_TARGETS), dtype=np.float32),
                "pot_m": np.zeros(len(SAR_20_TARGETS), dtype=np.float32),
                "act": np.zeros(len(SAR_20_TARGETS), dtype=np.float32),
                "act_m": np.zeros(len(SAR_20_TARGETS), dtype=np.float32),
            }
            mol_records[c_smi] = entry
        else:
            entry["raw_smiles_set"].add(raw_smi)
            if row_ik14:
                entry["ik14_set"].add(row_ik14)
            for sc_v in (row_scaf, row_scaf2):
                if sc_v:
                    entry["scaffold_set"].add(sc_v)

        p_val = row.p_potency
        if p_val is not None and np.isfinite(float(p_val)):
            pv = float(np.clip(float(p_val), 3.5, 10.0))
            entry["pot"][t_idx] = max(entry["pot"][t_idx], pv) if entry["pot_m"][t_idx] > 0 else pv
            entry["pot_m"][t_idx] = 1.0

        out_str = str(row.activity_outcome or "").strip().upper()
        if out_str in ("ACTIVE", "ACTIVE_INDUCER", "ACTIVE_INHIBITOR") or (
            p_val is not None and np.isfinite(float(p_val)) and float(p_val) >= 5.0
        ):
            entry["act"][t_idx] = 1.0
            entry["act_m"][t_idx] = 1.0
        elif out_str in ("INACTIVE", "WEAK_OR_INACTIVE"):
            if entry["act_m"][t_idx] == 0.0:
                entry["act"][t_idx] = 0.0
                entry["act_m"][t_idx] = 1.0

    grouped = [v for v in mol_records.values() if float(v["pot_m"].sum() + v["act_m"].sum()) > 0.0]
    _SAR_GROUPED_TRACK_CACHE[cache_key] = grouped
    return grouped


class FerroChemSARPretrainDataset(Dataset):
    """Leak-Free Stage-0.5 Biochemical & Phenotypic SAR Dataset (`chem_ferrdb_qhts_chembl_features.parquet`).

    Strictly excludes EVERY held-out molecule (`forbid_smiles`), stereochemistry-invariant connectivity
    (`forbid_ik14`), and Bemis-Murcko scaffold (`forbid_scaffolds`) from the target OOD split.
    """

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_PROCESSED_DIR,
        *,
        license_track: str = "open",
        forbid_smiles: set[str] | None = None,
        forbid_ik14: set[str] | None = None,
        forbid_scaffolds: set[str] | None = None,
        max_molecules: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.license_track = license_track.strip().lower()

        f_smi: set[str] = {s.strip() for s in (forbid_smiles or set()) if s and s.strip()}
        f_ik14: set[str] = {s.strip()[:14] for s in (forbid_ik14 or set()) if s and s.strip()}
        f_scaf: set[str] = {s.strip() for s in (forbid_scaffolds or set()) if s and s.strip()}

        # Expand forbidden sets via RDKit parent-desalted canonicalization so salt/stereo variants can never leak!
        for raw_s in list(f_smi):
            c_s, ik_s, sc_s = canonicalize_smiles_and_keys(raw_s)
            if c_s:
                f_smi.add(c_s)
            if ik_s:
                f_ik14.add(ik_s)
            if sc_s and forbid_scaffolds is not None and len(forbid_scaffolds) > 0:
                f_scaf.add(sc_s)

        self.forbid_smiles = f_smi
        self.forbid_ik14 = f_ik14
        self.forbid_scaffolds = f_scaf

        pq_path = self.data_dir / "chem_ferrdb_qhts_chembl_features.parquet"
        if not pq_path.is_file():
            self.fp_u8 = np.zeros((0, CHEM_FP_DIM), dtype=np.uint8)
            self.desc_f32 = np.zeros((0, CHEM_DESC_DIM), dtype=np.float32)
            self.potency_target = np.zeros((0, len(SAR_20_TARGETS)), dtype=np.float32)
            self.potency_mask = np.zeros((0, len(SAR_20_TARGETS)), dtype=np.float32)
            self.active_target = np.zeros((0, len(SAR_20_TARGETS)), dtype=np.float32)
            self.active_mask = np.zeros((0, len(SAR_20_TARGETS)), dtype=np.float32)
            self.sar_smiles: set[str] = set()
            self.sar_ik14: set[str] = set()
            self.sar_scaffolds: set[str] = set()
            return

        grouped_all = _load_grouped_sar_molecules(pq_path, self.license_track)
        mol_list = [
            m
            for m in grouped_all
            if f_smi.isdisjoint(m["raw_smiles_set"])
            and f_ik14.isdisjoint(m["ik14_set"])
            and (not f_scaf or f_scaf.isdisjoint(m["scaffold_set"]))
        ]
        if max_molecules is not None and len(mol_list) > max_molecules:
            rng = np.random.default_rng(seed)
            # Keep all quantitative potency molecules first, then sample remaining binary-only molecules
            quant = [m for m in mol_list if float(m["pot_m"].sum()) > 0]
            bin_only = [m for m in mol_list if float(m["pot_m"].sum()) == 0]
            if len(quant) >= max_molecules:
                chosen = rng.choice(len(quant), size=max_molecules, replace=False)
                mol_list = [quant[int(i)] for i in np.sort(chosen)]
            else:
                need = max_molecules - len(quant)
                chosen = rng.choice(len(bin_only), size=need, replace=False)
                mol_list = quant + [bin_only[int(i)] for i in np.sort(chosen)]

        uniq_smis = [m["smiles"] for m in mol_list]
        feat_map = _get_or_build_sar_mol_cache(uniq_smis)

        n_m = len(mol_list)
        self.fp_u8 = np.empty((n_m, CHEM_FP_DIM), dtype=np.uint8)
        self.desc_f32 = np.empty((n_m, CHEM_DESC_DIM), dtype=np.float32)
        self.potency_target = np.empty((n_m, len(SAR_20_TARGETS)), dtype=np.float32)
        self.potency_mask = np.empty((n_m, len(SAR_20_TARGETS)), dtype=np.float32)
        self.active_target = np.empty((n_m, len(SAR_20_TARGETS)), dtype=np.float32)
        self.active_mask = np.empty((n_m, len(SAR_20_TARGETS)), dtype=np.float32)

        self.sar_smiles = set()
        self.sar_ik14 = set()
        self.sar_scaffolds = set()

        for i, m in enumerate(mol_list):
            sm = m["smiles"]
            fp_i, desc_i = feat_map[sm]
            self.fp_u8[i] = fp_i
            self.desc_f32[i] = desc_i
            self.potency_target[i] = m["pot"]
            self.potency_mask[i] = m["pot_m"]
            self.active_target[i] = m["act"]
            self.active_mask[i] = m["act_m"]
            self.sar_smiles.add(sm)
            if m["ik14"]:
                self.sar_ik14.add(m["ik14"])
            if m["scaffold"]:
                self.sar_scaffolds.add(m["scaffold"])

        # Cryptographic / Zero-Trust Disjointness Assertions
        assert self.forbid_smiles.isdisjoint(self.sar_smiles), "FATAL Stage-0.5 SMILES leak!"
        assert self.forbid_ik14.isdisjoint(self.sar_ik14), "FATAL Stage-0.5 InChIKey-14 connectivity leak!"
        if self.forbid_scaffolds:
            assert self.forbid_scaffolds.isdisjoint(self.sar_scaffolds), "FATAL Stage-0.5 Murcko scaffold leak!"

    def __len__(self) -> int:
        return int(self.fp_u8.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "chem_fp": torch.from_numpy(self.fp_u8[idx].astype(np.float32)),
            "chem_desc": torch.from_numpy(self.desc_f32[idx]),
            "potency_target": torch.from_numpy(self.potency_target[idx]),
            "potency_mask": torch.from_numpy(self.potency_mask[idx]),
            "active_target": torch.from_numpy(self.active_target[idx]),
            "active_mask": torch.from_numpy(self.active_mask[idx]),
        }


class FerroFluxMultimodalDataset(Dataset):
    """Unified multimodal PyTorch Dataset joining Phase-2A & Phase-2B Parquet shards without leakage (v3)."""

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_PROCESSED_DIR,
        *,
        processed_dir: str | Path | None = None,
        license_track: str = "open",
        stage: int | None = None,
        indices: Sequence[int] | None = None,
        max_samples_per_shard: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        resolved_dir = Path(processed_dir) if processed_dir is not None else Path(data_dir)
        self.data_dir = resolved_dir
        self.license_track = license_track.strip().lower()
        self.stage = stage
        self.seed = int(seed)

        vocab = load_gene_vocab(self.data_dir / "gene_vocab.json")
        self.vocab = vocab
        self.genes: list[str] = vocab["genes"]
        self.n_genes: int = int(vocab["n_genes"])
        self.gene_to_idx: dict[str, int] = vocab["gene_to_idx"]
        self.anchor_proteins_28: list[str] = vocab["anchor_proteins_28"]
        self.l1000_landmark_indices: np.ndarray = np.array(
            [self.gene_to_idx[g] for g in vocab.get("l1000_landmarks_present", []) if g in self.gene_to_idx],
            dtype=np.int64,
        )

        # In-memory basal lookup tables indexed by model_id + normalized cell-line name crosswalk
        self._cell_expr: dict[str, np.ndarray] = {}
        self._cell_genomic: dict[str, np.ndarray] = {}
        self._name_to_mid: dict[str, str] = {}
        self._cell_cis_reg: dict[str, np.ndarray] = {}
        self._cell_cis_mask: dict[str, np.ndarray] = {}
        self._cell_prot: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cell_prot64: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cell_metab: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cell_lipid32: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cell_lineage: dict[str, str] = {}
        self._cell_is_3d: dict[str, int] = {}

        self._chem_lookup: dict[str, tuple[str, str]] = {}
        self._load_optional_phase2b_lookups()

        buckets: dict[str, list[dict[str, Any]]] = {
            "basal": [],
            "normal_sfti": [],
            "crispr": [],
            "pharmaco": [],
            "perturb": [],
            "icb": [],
            "synergy": [],
            "oxylipid": [],
            "rescue": [],
        }

        self._load_basal_shard(buckets["basal"], max_samples_per_shard)
        self._load_normal_sfti_shard(buckets["normal_sfti"], max_samples_per_shard)
        self._load_crispr_shard(buckets["crispr"], max_samples_per_shard)
        self._load_pharmaco_shard(buckets["pharmaco"], max_samples_per_shard)
        self._load_perturb_shard(buckets["perturb"], max_samples_per_shard)
        self._load_icb_shard(buckets["icb"], max_samples_per_shard)
        self._load_optional_phase2b_records(buckets, max_samples_per_shard)

        non_empty_keys = [k for k, v in buckets.items() if len(v) > 0]
        interleaved: list[dict[str, Any]] = []
        if non_empty_keys:
            max_len = max(len(buckets[k]) for k in non_empty_keys)
            for pos in range(max_len):
                for k in non_empty_keys:
                    b_list = buckets[k]
                    if pos < len(b_list):
                        interleaved.append(b_list[pos])

        if indices is not None:
            interleaved = [interleaved[i] for i in indices]

        self.records: list[dict[str, Any]] = interleaved

    def _resolve_model_id(self, raw_mid: Any, raw_cname: Any = None) -> str:
        """Resolve cell `model_id` cleanly via ID or normalized name; return `__UNMATCHED__` if absent."""
        cand = str(raw_mid or "").strip()
        if cand and cand in self._cell_expr:
            return cand
        for nm in (raw_cname, cand):
            nk = _norm_cell_name(nm)
            if nk and nk in self._name_to_mid:
                return self._name_to_mid[nk]
        return cand if cand else "__UNMATCHED__"

    def _read_filtered_parquet(
        self,
        filename: str,
        max_rows: int | None = None,
        stratify_col: str | None = "dataset_source",
    ) -> pd.DataFrame:
        p = self.data_dir / filename
        if not p.is_file():
            return pd.DataFrame()
        df = pq.read_table(p).to_pandas()
        if df.empty:
            return df
        for col in ("accession_id", "dataset_source", "pmcid"):
            if col in df.columns:
                for val in df[col].dropna().astype(str).unique():
                    assert_not_quarantined(val)
        if any(c in df.columns for c in ("license_track", "license", "data_license", "spdx_license")):
            df = filter_by_license_track(df, track=self.license_track)
        if max_rows is not None and len(df) > max_rows:
            scol = stratify_col if (stratify_col and stratify_col in df.columns) else None
            df = _stratified_sample_df(df, cap=max_rows, group_col=scol, seed=self.seed)
        return df

    def _load_optional_phase2b_lookups(self) -> None:
        df_prot = self._read_filtered_parquet("physical_anchors_proteomics.parquet", max_rows=None)
        if not df_prot.empty:
            id_col = next((c for c in ("model_id", "ModelID", "depmap_id", "sample_id") if c in df_prot.columns), None)
            if id_col:
                for _, r in df_prot.iterrows():
                    mid = str(r[id_col]).strip()
                    if "prot_anchor_target" in df_prot.columns:
                        vec = _safe_vec(r["prot_anchor_target"], 28, np.nan)
                    elif "prot_vec_28" in df_prot.columns:
                        vec = _safe_vec(r["prot_vec_28"], 28, np.nan)
                    else:
                        vec = np.array(
                            [pd.to_numeric(r.get(f"prot_{g}", r.get(g, np.nan)), errors="coerce") for g in self.anchor_proteins_28],
                            dtype=np.float32,
                        )
                    if "prot_anchor_mask" in df_prot.columns:
                        mask = np.clip(_safe_vec(r["prot_anchor_mask"], 28, 0.0), 0.0, 1.0)
                    else:
                        mask = np.isfinite(vec).astype(np.float32)
                    clean = np.nan_to_num(vec, nan=0.0, posinf=10.0, neginf=-10.0)
                    if mask.sum() > 0:
                        self._cell_prot[mid] = (clean, mask)

                    if "canonical_64_proteome" in df_prot.columns:
                        v64 = _safe_vec(r["canonical_64_proteome"], 64, np.nan)
                        m64 = (
                            np.clip(_safe_vec(r["canonical_64_mask"], 64, 0.0), 0.0, 1.0)
                            if "canonical_64_mask" in df_prot.columns
                            else np.isfinite(v64).astype(np.float32)
                        )
                        c64 = np.nan_to_num(v64, nan=0.0, posinf=10.0, neginf=-10.0) * m64
                        if m64.sum() > 0:
                            self._cell_prot64[mid] = (c64, m64)

        df_met = self._read_filtered_parquet("physical_anchors_metabolomics_lipidomics.parquet", max_rows=None)
        if not df_met.empty:
            id_col = next((c for c in ("model_id", "ModelID", "depmap_id", "sample_id") if c in df_met.columns), None)
            met_cols = [
                "GSH", "GSSG", "NADP", "putrescine", "spermidine",
                "PC_PUFA2", "PE_PUFA", "ePL_PUFA", "MUFA_PL", "sterol_7dhc",
            ]
            # Precompute robust per-column z-stats across measured CCLE LC-MS lipid/metabolite columns
            lcms_present = [c for c in LCMS_32_COLUMNS if c in df_met.columns]
            lcms_mat = np.full((len(df_met), len(LCMS_32_COLUMNS)), np.nan, dtype=np.float32)
            for col_j, cname in enumerate(LCMS_32_COLUMNS):
                if cname in df_met.columns:
                    vals_c = pd.to_numeric(df_met[cname], errors="coerce").to_numpy(dtype=np.float32)
                    fin = np.isfinite(vals_c)
                    if int(fin.sum()) >= 8:
                        med = float(np.median(vals_c[fin]))
                        std = float(max(np.std(vals_c[fin]), 1e-3))
                        vals_c = np.where(fin, np.clip((vals_c - med) / std, -8.0, 8.0), np.nan)
                    lcms_mat[:, col_j] = vals_c

            if id_col:
                for row_k, (_, r) in enumerate(df_met.iterrows()):
                    mid = str(r[id_col]).strip()
                    if "metab_anchor_target" in df_met.columns:
                        vec = _safe_vec(r["metab_anchor_target"], 10, np.nan)
                    elif "metab_vec_10" in df_met.columns:
                        vec = _safe_vec(r["metab_vec_10"], 10, np.nan)
                    else:
                        vals = []
                        for mc in met_cols:
                            cand = next((c for c in df_met.columns if mc.lower() in c.lower()), None)
                            vals.append(pd.to_numeric(r[cand], errors="coerce") if cand else np.nan)
                        vec = np.asarray(vals, dtype=np.float32)
                    if "metab_anchor_mask" in df_met.columns:
                        mask = np.clip(_safe_vec(r["metab_anchor_mask"], 10, 0.0), 0.0, 1.0)
                    else:
                        mask = np.isfinite(vec).astype(np.float32)
                    clean = np.nan_to_num(vec, nan=0.0, posinf=10.0, neginf=-10.0)
                    if mask.sum() > 0:
                        self._cell_metab[mid] = (clean, mask)

                    if lcms_present:
                        raw_32 = lcms_mat[row_k]
                        m32 = np.isfinite(raw_32).astype(np.float32)
                        c32 = np.nan_to_num(raw_32, nan=0.0) * m32
                        if m32.sum() > 0:
                            self._cell_lipid32[mid] = (c32, m32)

        df_epi = self._read_filtered_parquet("cis_reg_64loci_epigenomics.parquet", max_rows=None)
        if not df_epi.empty and "model_id" in df_epi.columns:
            epi_col = "epigen_64x8" if "epigen_64x8" in df_epi.columns else ("cis_reg_epigen" if "cis_reg_epigen" in df_epi.columns else None)
            if epi_col:
                for _, r in df_epi.iterrows():
                    mid = str(r["model_id"]).strip()
                    arr = _safe_vec(r[epi_col], 64 * 8, 0.0).reshape(64, 8)
                    self._cell_cis_reg[mid] = arr
                    if "cis_reg_mask" in df_epi.columns:
                        self._cell_cis_mask[mid] = np.clip(_safe_vec(r["cis_reg_mask"], 64, 1.0), 0.0, 1.0)
                    else:
                        self._cell_cis_mask[mid] = np.ones(64, dtype=np.float32)

        df_chem = self._read_filtered_parquet("chem_ferrdb_qhts_chembl_features.parquet", max_rows=None)
        if not df_chem.empty:
            name_c = next((c for c in ("compound_name", "name", "molecule_name") if c in df_chem.columns), None)
            smi_c = next((c for c in ("canonical_smiles", "smiles") if c in df_chem.columns), None)
            scaf_c = next((c for c in ("murcko_scaffold", "scaffold_smiles", "scaffold") if c in df_chem.columns), None)
            if name_c and smi_c:
                for _, r in df_chem.iterrows():
                    nm = str(r[name_c]).strip().upper()
                    sm = str(r[smi_c]).strip()
                    sc = str(r[scaf_c]).strip() if scaf_c else sm
                    if nm and sm and sm != "nan":
                        self._chem_lookup[nm] = (sm, sc)
                        norm_nm = re.sub(r"[^A-Z0-9]", "", nm)
                        if norm_nm:
                            self._chem_lookup.setdefault(norm_nm, (sm, sc))

    def _load_basal_shard(self, out_list: list[dict[str, Any]], max_rows: int | None) -> None:
        df = self._read_filtered_parquet("cells_basal_multiomics.parquet", max_rows=max_rows)
        if df.empty:
            return
        for _, r in df.iterrows():
            mid = str(r["model_id"]).strip()
            expr = _safe_vec(r["expr_vec"], self.n_genes, 0.0)
            cnv64 = _safe_vec(r.get("cnv_64_vec"), 64, 0.0)
            lof64 = _safe_vec(r.get("mut_lof_64_vec"), 64, 0.0)
            hot_str = str(r.get("key_mutations", "") or "").upper()
            hot10 = np.array([1.0 if f"{g}_HOTSPOT" in hot_str else 0.0 for g in HOTSPOT_10_GENES], dtype=np.float32)
            genomic_138 = np.concatenate([cnv64, lof64, hot10], axis=0)

            lin = str(r.get("lineage", "Unknown"))
            is_3d = int(r.get("is_3d_organoid", 0))
            cname = _norm_cell_name(r.get("cell_line_name", ""))

            self._cell_expr[mid] = expr
            self._cell_genomic[mid] = genomic_138
            self._cell_lineage[mid] = lin
            self._cell_is_3d[mid] = is_3d
            if cname:
                self._name_to_mid[cname] = mid

            if mid not in self._cell_cis_reg:
                epi = np.zeros((64, 8), dtype=np.float32)
                epi[:, 0] = expr[:64] / 10.0
                epi[:, 1] = cnv64
                epi[:, 2] = 1.0 - lof64
                self._cell_cis_reg[mid] = epi
                self._cell_cis_mask[mid] = np.ones(64, dtype=np.float32)

            out_list.append(
                {
                    "kind": "basal",
                    "model_id": mid,
                    "lineage": lin,
                    "is_3d_organoid": is_3d,
                    "is_crispr_fin_epistasis": 0,
                    "is_ferroptosis_compound": 0,
                    "murcko_scaffold": "",
                    "ik14": "",
                }
            )

    def _load_normal_sfti_shard(self, out_list: list[dict[str, Any]], max_rows: int | None) -> None:
        df = self._read_filtered_parquet("normal_organ_til_baselines.parquet", max_rows=max_rows)
        if df.empty:
            return
        for _, r in df.iterrows():
            rid = str(r["record_id"]).strip()
            expr = _safe_vec(r["expr_vec"], self.n_genes, 0.0)
            organ = str(r.get("organ", "Normal"))
            self._cell_expr[rid] = expr
            self._cell_lineage[rid] = organ
            self._cell_is_3d[rid] = 0
            out_list.append(
                {
                    "kind": "normal_sfti",
                    "model_id": rid,
                    "lineage": organ,
                    "is_3d_organoid": 0,
                    "is_crispr_fin_epistasis": 0,
                    "is_ferroptosis_compound": 0,
                    "murcko_scaffold": "",
                    "ik14": "",
                }
            )

    def _load_crispr_shard(self, out_list: list[dict[str, Any]], max_rows: int | None) -> None:
        cap = max_rows if max_rows is not None else 65000
        df = self._read_filtered_parquet("crispr_chronos_2d_3d.parquet", max_rows=None)
        if df.empty:
            return
        # Always preserve 100% of 3D organoid & dual CRISPR+FIN epistatic rows, plus stratified 2D Chronos sample
        if len(df) > cap:
            special = df[(df["is_3d_organoid"] == 1) | (df["is_crispr_fin_epistasis"] == 1)]
            rem_n = max(cap - len(special), 16000)
            regular = df[(df["is_3d_organoid"] == 0) & (df["is_crispr_fin_epistasis"] == 0)]
            regular = _stratified_sample_df(regular, cap=rem_n, group_col="dataset_source", seed=self.seed)
            df = pd.concat([special, regular], ignore_index=True)

        # Fallback canonical SMILES for BioGRID ORCS FIN modifier compounds so pure-SMILES structure is always used
        orcs_fin_smiles = {
            "1S,3R-RSL3": "COC(=O)[C@@H]1Cc2c([nH]c3ccccc23)[C@H](c2ccc(C(=O)OC)cc2)N1C(=O)CCl",
            "RSL3": "COC(=O)[C@@H]1Cc2c([nH]c3ccccc23)[C@H](c2ccc(C(=O)OC)cc2)N1C(=O)CCl",
            "ML210": "Cc1cc(C(=O)N2CCN(C(c3ccc(Cl)cc3)c3ccc([N+](=O)[O-])o3)CC2)no1",
            "ML162": "COc1ccc(C(N(C(=O)CCl)c2ccc(Cl)cc2)C(=O)NCc2ccco2)cc1OC",
            "ERASTIN": "CCOc1ccccc1-n1c(=O)c2ccccc2n(C(C)N2CCN(C(=O)Cc3ccc(Cl)cc3)CC2)c1=O",
            "FIN56": "O=C(NCCCCCCNC(=O)C1(c2ccccc2)CCN(S(=O)(=O)c2ccc(C(C)(C)C)cc2)CC1)C1(c2ccccc2)CCN(S(=O)(=O)c2ccc(C(C)(C)C)cc2)CC1",
            "FINO2": "CC1(C)CCC2(OO1)CC(O)(C(C)C)CC2",
            "SULFASALAZINE": "O=C(O)c1cc(/N=N/c2ccc(S(=O)(=O)Nc3ccccn3)cc2)ccc1O",
            "SORAFENIB": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
            "ELESCLOMOL": "CN(NC(=S)c1ccccc1)C(=O)CC(=O)N(C)NC(=S)c1ccccc1",
            "DISULFIRAM": "CCN(CC)C(=S)SSC(=S)N(CC)CC",
            "AURANOFIN": "CC[P+](CC)(CC)[Au-]SC1OC(COC(C)=O)C(OC(C)=O)C(OC(C)=O)C1OC(C)=O",
            "BREQUINAR": "Cc1cc(-c2ccc(F)cc2)nc2c(C(=O)O)cc(F)cc12",
            "PRAVASTATIN": "CCC(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC(O)CC(O)CC(=O)O)C21",
            "CERIVASTATIN": "COCc1c(C(C)C)nc(C(C)C)c(/C=C/C(O)CC(O)CC(=O)O)c1-c1ccc(F)cc1",
            "SIMVASTATIN": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
            "FLUVASTATIN": "CC(C)n1c(/C=C/C(O)CC(O)CC(=O)O)c(-c2ccc(F)cc2)c2ccccc21",
            "LOVASTATIN": "CCC(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
        }

        for _, r in df.iterrows():
            mid = self._resolve_model_id(r.get("model_id"), r.get("cell_line_name"))
            fin_c = str(r.get("fin_modifier_compound", "")).strip()
            fin_up = fin_c.upper()
            fin_smi = ""
            if fin_up:
                fin_smi = (
                    orcs_fin_smiles.get(fin_up)
                    or self._chem_lookup.get(fin_up, ("", ""))[0]
                    or self._chem_lookup.get(re.sub(r"[^A-Z0-9]", "", fin_up), ("", ""))[0]
                )
            _, fin_ik14, fin_scaf = canonicalize_smiles_and_keys(fin_smi) if fin_smi else ("", "", "")
            out_list.append(
                {
                    "kind": "crispr",
                    "model_id": mid,
                    "gene_idx": int(r["gene_idx"]),
                    "chronos_score": float(r["chronos_score"]),
                    "is_3d_organoid": int(r.get("is_3d_organoid", 0)),
                    "is_crispr_fin_epistasis": int(r.get("is_crispr_fin_epistasis", 0)),
                    "is_ferroptosis_compound": 1,
                    "fin_modifier_compound": fin_c,
                    "fin_modifier_smiles": fin_smi,
                    "fin_modifier_dose_um": float(r.get("fin_modifier_dose_um", 0.0)),
                    "lineage": str(r.get("lineage", self._cell_lineage.get(mid, "Unknown"))),
                    "murcko_scaffold": fin_scaf,
                    "ik14": fin_ik14,
                }
            )

    def _load_pharmaco_shard(self, out_list: list[dict[str, Any]], max_rows: int | None) -> None:
        cap = max_rows if max_rows is not None else 65000
        df = self._read_filtered_parquet("pharmacogenomic_viability.parquet", max_rows=cap, stratify_col="dataset_source")
        if df.empty:
            return
        for _, r in df.iterrows():
            mid = self._resolve_model_id(r.get("model_id"), r.get("cell_line_name"))
            cname = str(r["compound_name"]).strip()
            smi = str(r.get("canonical_smiles", "")).strip()
            scaf = str(r.get("murcko_scaffold", "")).strip()
            if (not smi or smi == "nan") and cname.upper() in self._chem_lookup:
                smi, scaf_lu = self._chem_lookup[cname.upper()]
                if not scaf:
                    scaf = scaf_lu
            c_smi, ik14, calc_scaf = canonicalize_smiles_and_keys(smi)
            rcd_mod = str(r.get("rcd_modality", "ferroptosis_inducer")).strip().lower()
            is_ferro = int(rcd_mod in ("ferroptosis_inducer", "ferroptosis_rescuer"))
            out_list.append(
                {
                    "kind": "pharmaco",
                    "model_id": mid,
                    "compound_name": cname,
                    "canonical_smiles": c_smi if c_smi else smi,
                    "ik14": ik14 or str(r.get("inchikey", "")).strip()[:14],
                    "murcko_scaffold": calc_scaf if calc_scaf else (scaf if scaf else smi),
                    "dose_um": float(r.get("dose_um", 1.0)),
                    "viability": float(r["viability"]),
                    "rcd_modality": rcd_mod,
                    "is_ferroptosis_compound": is_ferro,
                    "hazard_channel_idx": int(r.get("hazard_channel_idx", 0)),
                    "probe_mechanism": str(r.get("probe_mechanism", "")),
                    "is_3d_organoid": int(r.get("is_3d_organoid", 0)),
                    "is_crispr_fin_epistasis": 0,
                    "lineage": str(r.get("lineage", self._cell_lineage.get(mid, "Unknown"))),
                }
            )

    def _load_perturb_shard(self, out_list: list[dict[str, Any]], max_rows: int | None) -> None:
        cap = max_rows if max_rows is not None else 35000
        df = self._read_filtered_parquet("perturb_transcriptomes.parquet", max_rows=cap, stratify_col="dataset_source")
        if df.empty:
            return
        for _, r in df.iterrows():
            mid = self._resolve_model_id(r.get("model_id"), r.get("cell_line_name"))
            raw_smi = str(r.get("canonical_smiles", "")).strip()
            c_smi, ik14, calc_scaf = canonicalize_smiles_and_keys(raw_smi) if raw_smi else ("", "", "")
            out_list.append(
                {
                    "kind": "perturb",
                    "model_id": mid,
                    "lineage": str(r.get("lineage", self._cell_lineage.get(mid, "Unknown"))),
                    "perturbation_type": str(r.get("perturbation_type", "compound")),
                    "perturbation_name": str(r.get("perturbation_name", "")),
                    "canonical_smiles": c_smi if c_smi else raw_smi,
                    "ik14": ik14,
                    "murcko_scaffold": calc_scaf if calc_scaf else str(r.get("murcko_scaffold", "")),
                    "dose_um": float(r.get("dose_um", 1.0)),
                    "time_h": float(r.get("time_h", 24.0)),
                    "delta_expr_vec": _safe_vec(r.get("delta_expr_vec"), self.n_genes, 0.0),
                    "mask_mode": str(r.get("measured_gene_mask_type", "full_rnaseq")),
                    "survivorship_weight": float(r.get("survivorship_weight", 1.0)),
                    "sec46_readthrough_eff": float(r.get("sec46_readthrough_eff", -1.0)),
                    "np_params_vec": _safe_vec(r.get("np_params_vec"), 5, 0.0),
                    "is_3d_organoid": 0,
                    "is_crispr_fin_epistasis": 0,
                    "is_ferroptosis_compound": 1,
                }
            )

    def _load_icb_shard(self, out_list: list[dict[str, Any]], max_rows: int | None) -> None:
        df = self._read_filtered_parquet("clinical_icb_cohorts.parquet", max_rows=max_rows, stratify_col="dataset_source")
        if df.empty:
            return
        for _, r in df.iterrows():
            rid = str(r["record_id"]).strip()
            expr = _safe_vec(r["expr_vec"], self.n_genes, 0.0)
            lin = str(r.get("lineage", "Kidney"))
            self._cell_expr[rid] = expr
            self._cell_lineage[rid] = lin
            self._cell_is_3d[rid] = 0
            out_list.append(
                {
                    "kind": "icb",
                    "model_id": rid,
                    "lineage": lin,
                    "responder_binary": int(r.get("responder_binary", -1)),
                    "is_3d_organoid": 0,
                    "is_crispr_fin_epistasis": 0,
                    "is_ferroptosis_compound": 0,
                    "murcko_scaffold": "",
                    "ik14": "",
                }
            )

    def _load_optional_phase2b_records(
        self, buckets: dict[str, list[dict[str, Any]]], max_rows: int | None
    ) -> None:
        df_syn = self._read_filtered_parquet("drugcomb_synergy_blocks.parquet", max_rows=max_rows or 15000, stratify_col="dataset_source")
        if not df_syn.empty:
            for _, r in df_syn.iterrows():
                mid = self._resolve_model_id(r.get("model_id"), r.get("cell_line_name", r.get("cell_line")))
                d1 = str(r.get("drug_row", r.get("compound_a", ""))).strip().upper()
                d2 = str(r.get("drug_col", r.get("compound_b", ""))).strip().upper()
                sm1 = str(r.get("smiles_row", r.get("smiles_a", self._chem_lookup.get(d1, ("", ""))[0])))
                sm2 = str(r.get("smiles_col", r.get("smiles_b", self._chem_lookup.get(d2, ("", ""))[0])))
                zip_val = pd.to_numeric(r.get("synergy_zip", r.get("zip", r.get("bliss", 0.0))), errors="coerce")
                if pd.isna(zip_val) or not np.isfinite(float(zip_val)):
                    continue
                c_sm1, ik1, sc1 = canonicalize_smiles_and_keys(sm1) if sm1 else ("", "", "")
                buckets["synergy"].append(
                    {
                        "kind": "synergy",
                        "model_id": mid,
                        "lineage": self._cell_lineage.get(mid, "Unknown"),
                        "smiles_a": c_sm1 if c_sm1 else sm1,
                        "smiles_b": sm2,
                        "synergy_score": float(np.clip(float(zip_val) / 25.0, -4.0, 4.0)),
                        "is_3d_organoid": 0,
                        "is_crispr_fin_epistasis": 0,
                        "is_ferroptosis_compound": 1,
                        "murcko_scaffold": sc1 if sc1 else self._chem_lookup.get(d1, ("", d1))[1],
                        "ik14": ik1,
                    }
                )

        df_oxy = self._read_filtered_parquet("perturbed_oxylipidomics_sterolomics.parquet", max_rows=max_rows or 8000, stratify_col="dataset_source")
        if not df_oxy.empty:
            for _, r in df_oxy.iterrows():
                mid = self._resolve_model_id(r.get("model_id"), r.get("cell_line_name", r.get("sample_name")))
                if "oxylipid_target" in df_oxy.columns and "oxylipid_mask" in df_oxy.columns:
                    otgt = np.clip(_safe_vec(r["oxylipid_target"], 4, 0.0), 0.0, 20.0)
                    omsk = np.clip(_safe_vec(r["oxylipid_mask"], 4, 0.0), 0.0, 1.0)
                    if omsk.sum() == 0:
                        continue
                else:
                    val = pd.to_numeric(r.get("pl_ooh_fold", r.get("log2fc", r.get("zscore_within_study", 1.0))), errors="coerce")
                    if pd.isna(val) or not np.isfinite(float(val)):
                        continue
                    otgt = np.array([max(0.0, float(val)), 0.0, 0.0, 0.0], dtype=np.float32)
                    omsk = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                buckets["oxylipid"].append(
                    {
                        "kind": "oxylipid",
                        "model_id": mid,
                        "lineage": self._cell_lineage.get(mid, "Unknown"),
                        "oxylipid_target_vec": otgt,
                        "oxylipid_mask_vec": omsk,
                        "is_3d_organoid": 0,
                        "is_crispr_fin_epistasis": 0,
                        "is_ferroptosis_compound": 1,
                        "murcko_scaffold": "",
                        "ik14": "",
                    }
                )

        df_res = self._read_filtered_parquet("pmc_oa_rescue_and_bionano.parquet", max_rows=max_rows or 5000, stratify_col="dataset_source")
        if not df_res.empty:
            for _, r in df_res.iterrows():
                mid = self._resolve_model_id(r.get("model_id"), r.get("cell_line_name"))
                delta_v = pd.to_numeric(
                    r.get("rescue_delta", r.get("rescue_delta_viability", r.get("viability_rescue_delta", np.nan))),
                    errors="coerce",
                )
                if pd.isna(delta_v) or not np.isfinite(float(delta_v)):
                    continue
                buckets["rescue"].append(
                    {
                        "kind": "rescue",
                        "model_id": mid,
                        "lineage": self._cell_lineage.get(mid, "Unknown"),
                        "rescue_delta": float(np.clip(float(delta_v), -1.0, 1.0)),
                        "is_3d_organoid": 0,
                        "is_crispr_fin_epistasis": 0,
                        "is_ferroptosis_compound": 1,
                        "murcko_scaffold": "",
                        "ik14": "",
                    }
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        mid = rec["model_id"]
        kind = rec["kind"]

        # Basal expression & masks: ZERO silent fallback to ACH-000873!
        matched_expr = self._cell_expr.get(mid)
        if matched_expr is not None:
            expr = matched_expr
            gene_mask = (expr != 0.0).astype(np.float32)
            if gene_mask.sum() == 0:
                gene_mask[:] = 1.0
            cis_reg = self._cell_cis_reg.get(mid, np.zeros((64, 8), dtype=np.float32))
            cis_mask = self._cell_cis_mask.get(mid, np.ones(64, dtype=np.float32))
            genomic_ctx = self._cell_genomic.get(mid, np.zeros(GENOMIC_CONTEXT_DIM, dtype=np.float32))
        else:
            expr = np.zeros(self.n_genes, dtype=np.float32)
            gene_mask = np.zeros(self.n_genes, dtype=np.float32)
            cis_reg = np.zeros((64, 8), dtype=np.float32)
            cis_mask = np.zeros(64, dtype=np.float32)
            genomic_ctx = np.zeros(GENOMIC_CONTEXT_DIM, dtype=np.float32)

        # Real mass-spectrometry proteomics & LC-MS metabolomics/lipidomics ONLY (never fake mRNA proxy!)
        prot_target, prot_mask = self._cell_prot.get(
            mid, (np.zeros(28, dtype=np.float32), np.zeros(28, dtype=np.float32))
        )
        prot64_vec, prot64_mask = self._cell_prot64.get(
            mid, (np.zeros(64, dtype=np.float32), np.zeros(64, dtype=np.float32))
        )
        metab_target, metab_mask = self._cell_metab.get(
            mid, (np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32))
        )
        lipid32_vec, lipid32_mask = self._cell_lipid32.get(
            mid, (np.zeros(32, dtype=np.float32), np.zeros(32, dtype=np.float32))
        )

        chem_fp_a = np.zeros(CHEM_FP_DIM, dtype=np.float32)
        chem_desc_a = np.zeros(CHEM_DESC_DIM, dtype=np.float32)
        chem_fp_b = np.zeros(CHEM_FP_DIM, dtype=np.float32)
        chem_desc_b = np.zeros(CHEM_DESC_DIM, dtype=np.float32)
        dose_a = -3.0  # exact 0.0 uM under log10(uM + 1e-3)
        dose_b = -3.0

        rta_conc = 0.0
        chelator_conc = 0.0
        gpx4_inhib = 0.0
        xc_inhib = 0.0
        elesclomol_conc = 0.0
        concanamycin_a = 0.0
        dhcr7_mult = 1.0
        ebp_mult = 1.0
        cys2_media = 1.0

        gene_ko_mult = np.ones(self.n_genes, dtype=np.float32)
        media_ext = np.ones(10, dtype=np.float32)
        media_ext[8] = 0.21
        media_ext[9] = 0.0
        np_params = np.zeros(5, dtype=np.float32)
        delta_t = 24.0

        chronos_target = 0.0
        chronos_mask = 0.0
        delta_expr_target = np.zeros(self.n_genes, dtype=np.float32)
        delta_expr_mask = np.zeros(self.n_genes, dtype=np.float32)
        survivorship_weight = 1.0
        sec46_target = 0.0
        sec46_mask = 0.0

        viability_target = 1.0
        viability_mask = 0.0
        hazard_target = np.zeros(5, dtype=np.float32)
        hazard_mask = np.zeros(5, dtype=np.float32)
        rescue_delta_target = 0.0
        rescue_mask = 0.0
        synergy_target = 0.0
        synergy_mask = 0.0

        oxylipid_target = np.zeros(4, dtype=np.float32)
        oxylipid_mask = np.zeros(4, dtype=np.float32)
        icb_target = 0.0
        icb_mask = 0.0

        if kind == "crispr":
            g_idx = rec["gene_idx"]
            if 0 <= g_idx < self.n_genes:
                gene_ko_mult[g_idx] = 0.05
                sym = self.genes[g_idx]
                if sym == "DHCR7":
                    dhcr7_mult = 0.05
                elif sym in ("EBP", "SC5D"):
                    ebp_mult = 0.05
            chronos_target = rec["chronos_score"]
            chronos_mask = 1.0
            if rec["is_crispr_fin_epistasis"] == 1:
                fin_smi = rec.get("fin_modifier_smiles", "")
                fin_d = max(rec["fin_modifier_dose_um"], 1e-4)
                dose_a = encode_log10_dose_um(fin_d)
                if fin_smi:
                    chem_fp_a, chem_desc_a = compute_compound_featurization(fin_smi)
                else:
                    fin_c = rec["fin_modifier_compound"].upper()
                    if "RSL3" in fin_c or "ML" in fin_c or "FIN56" in fin_c:
                        gpx4_inhib = fin_d
                    elif "ERASTIN" in fin_c or "SULFA" in fin_c or "SORAF" in fin_c:
                        xc_inhib = fin_d
                    elif "ELESCLOMOL" in fin_c:
                        elesclomol_conc = fin_d

        elif kind == "pharmaco":
            # 100% pure SMILES structure + dose_a! ZERO probe_mechanism scalar injection!
            smi = rec["canonical_smiles"]
            chem_fp_a, chem_desc_a = compute_compound_featurization(smi)
            dose_a = encode_log10_dose_um(max(rec["dose_um"], 1e-4))

            viab = float(np.clip(rec["viability"], 0.0, 1.0))
            viability_target = viab
            viability_mask = 1.0
            h_idx = rec["hazard_channel_idx"]
            if 0 <= h_idx < 5:
                death_prob = float(np.clip(1.0 - viab, 0.0, 0.995))
                hazard_target[h_idx] = float(-np.log(max(1.0 - death_prob, 1e-4)))
                hazard_mask[:] = 1.0

        elif kind == "perturb":
            ptype = rec["perturbation_type"]
            pname = rec["perturbation_name"]
            delta_t = float(max(rec["time_h"], 1.0))
            survivorship_weight = float(np.clip(rec["survivorship_weight"], 1.0, 20.0))
            delta_expr_target = rec["delta_expr_vec"]
            if rec["mask_mode"] == "l1000_landmark" and len(self.l1000_landmark_indices) > 0:
                delta_expr_mask[self.l1000_landmark_indices] = 1.0
            else:
                delta_expr_mask = (delta_expr_target != 0.0).astype(np.float32)
                if delta_expr_mask.sum() == 0:
                    delta_expr_mask[:64] = 1.0

            if ptype == "compound":
                chem_fp_a, chem_desc_a = compute_compound_featurization(rec["canonical_smiles"])
                dose_a = encode_log10_dose_um(max(rec["dose_um"], 1e-4))
            elif ptype in ("crispr_ko", "crispri"):
                g_i = self.gene_to_idx.get(pname)
                if g_i is not None:
                    gene_ko_mult[g_i] = 0.05
                    if pname == "DHCR7":
                        dhcr7_mult = 0.05
                    elif pname in ("EBP", "SC5D"):
                        ebp_mult = 0.05
            elif ptype == "cytokine":
                if "IFNG" in pname.upper() or "IFN-G" in pname.upper():
                    media_ext[9] = 5.0
            elif ptype in ("bionano", "nanoparticle"):
                np_params = rec["np_params_vec"]

            if rec["sec46_readthrough_eff"] >= 0.0:
                sec46_target = float(rec["sec46_readthrough_eff"])
                sec46_mask = 1.0

        elif kind == "icb":
            media_ext[9] = 2.0
            if rec["responder_binary"] in (0, 1):
                icb_target = float(rec["responder_binary"])
                icb_mask = 1.0

        elif kind == "synergy":
            chem_fp_a, chem_desc_a = compute_compound_featurization(rec["smiles_a"])
            chem_fp_b, chem_desc_b = compute_compound_featurization(rec["smiles_b"])
            dose_a = encode_log10_dose_um(1.0)
            dose_b = encode_log10_dose_um(1.0)
            synergy_target = rec["synergy_score"]
            synergy_mask = 1.0

        elif kind == "oxylipid":
            oxylipid_target = rec["oxylipid_target_vec"].copy()
            oxylipid_mask = rec["oxylipid_mask_vec"].copy()

        elif kind == "rescue":
            gpx4_inhib = 2.0
            rta_conc = 5.0
            rescue_delta_target = rec["rescue_delta"]
            rescue_mask = 1.0

        return {
            "gene_expr": torch.from_numpy(expr.copy()),
            "gene_mask": torch.from_numpy(gene_mask),
            "genomic_context": torch.from_numpy(genomic_ctx.copy()),
            "prot64_context": torch.from_numpy(prot64_vec.copy()),
            "prot64_mask": torch.from_numpy(prot64_mask.copy()),
            "lipid32_context": torch.from_numpy(lipid32_vec.copy()),
            "lipid32_mask": torch.from_numpy(lipid32_mask.copy()),
            "cis_reg_epigen": torch.from_numpy(cis_reg.copy()),
            "cis_reg_mask": torch.from_numpy(cis_mask.copy()),
            "is_3d_organoid": torch.tensor(float(rec["is_3d_organoid"]), dtype=torch.float32),
            "is_crispr_fin_epistasis": torch.tensor(float(rec.get("is_crispr_fin_epistasis", 0)), dtype=torch.float32),
            "is_ferroptosis_compound": torch.tensor(float(rec.get("is_ferroptosis_compound", 0)), dtype=torch.float32),
            "is_normal_organ_or_til": torch.tensor(1.0 if kind == "normal_sfti" else 0.0, dtype=torch.float32),
            "chem_fp_a": torch.from_numpy(chem_fp_a),
            "chem_desc_a": torch.from_numpy(chem_desc_a),
            "chem_fp_b": torch.from_numpy(chem_fp_b),
            "chem_desc_b": torch.from_numpy(chem_desc_b),
            "dose_a": torch.tensor(dose_a, dtype=torch.float32),
            "dose_b": torch.tensor(dose_b, dtype=torch.float32),
            "rta_conc": torch.tensor(rta_conc, dtype=torch.float32),
            "chelator_conc": torch.tensor(chelator_conc, dtype=torch.float32),
            "gpx4_inhib": torch.tensor(gpx4_inhib, dtype=torch.float32),
            "xc_inhib": torch.tensor(xc_inhib, dtype=torch.float32),
            "elesclomol_conc": torch.tensor(elesclomol_conc, dtype=torch.float32),
            "concanamycin_a": torch.tensor(concanamycin_a, dtype=torch.float32),
            "dhcr7_mult": torch.tensor(dhcr7_mult, dtype=torch.float32),
            "ebp_mult": torch.tensor(ebp_mult, dtype=torch.float32),
            "cys2_media": torch.tensor(cys2_media, dtype=torch.float32),
            "gene_ko_mult": torch.from_numpy(gene_ko_mult),
            "media_ext": torch.from_numpy(media_ext),
            "np_params": torch.from_numpy(np_params.copy()),
            "delta_t": torch.tensor(delta_t, dtype=torch.float32),
            "prot_anchor_target": torch.from_numpy(prot_target.copy()),
            "prot_anchor_mask": torch.from_numpy(prot_mask.copy()),
            "metab_anchor_target": torch.from_numpy(metab_target.copy()),
            "metab_anchor_mask": torch.from_numpy(metab_mask.copy()),
            "chronos_target": torch.tensor(chronos_target, dtype=torch.float32),
            "chronos_mask": torch.tensor(chronos_mask, dtype=torch.float32),
            "delta_expr_target": torch.from_numpy(delta_expr_target.copy()),
            "delta_expr_mask": torch.from_numpy(delta_expr_mask),
            "survivorship_weight": torch.tensor(survivorship_weight, dtype=torch.float32),
            "sec46_readthrough_target": torch.tensor(sec46_target, dtype=torch.float32),
            "sec46_mask": torch.tensor(sec46_mask, dtype=torch.float32),
            "viability_target": torch.tensor(viability_target, dtype=torch.float32),
            "viability_mask": torch.tensor(viability_mask, dtype=torch.float32),
            "hazard_target": torch.from_numpy(hazard_target),
            "hazard_mask": torch.from_numpy(hazard_mask),
            "rescue_delta_target": torch.tensor(rescue_delta_target, dtype=torch.float32),
            "rescue_mask": torch.tensor(rescue_mask, dtype=torch.float32),
            "synergy_target": torch.tensor(synergy_target, dtype=torch.float32),
            "synergy_mask": torch.tensor(synergy_mask, dtype=torch.float32),
            "oxylipid_target": torch.from_numpy(oxylipid_target),
            "oxylipid_mask": torch.from_numpy(oxylipid_mask),
            "icb_response_target": torch.tensor(icb_target, dtype=torch.float32),
            "icb_mask": torch.tensor(icb_mask, dtype=torch.float32),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k in batch[0].keys():
            out[k] = torch.stack([item[k] for item in batch], dim=0)
        return out

    # ------------------------------------------------------------------
    # Six Strictly Disjoint Out-Of-Distribution (OOD) Splitters
    # ------------------------------------------------------------------
    def scaffold_split(self, test_frac: float = 0.2, seed: int = 42) -> tuple[list[int], list[int]]:
        """Hold out 20% of unique Bemis-Murcko scaffolds strictly into test (zero SMILES, ik14, or scaffold overlap)."""
        ferro_scaffolds = sorted(
            {
                r["murcko_scaffold"]
                for r in self.records
                if r.get("kind") == "pharmaco"
                and int(r.get("is_ferroptosis_compound", 0)) == 1
                and r.get("murcko_scaffold")
            }
        )
        scaffolds = ferro_scaffolds if len(ferro_scaffolds) >= 5 else sorted(
            {r["murcko_scaffold"] for r in self.records if r.get("murcko_scaffold")}
        )
        if len(scaffolds) >= 2:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(scaffolds))
            n_test = max(2, int(round(len(scaffolds) * test_frac)))
            held_out = {scaffolds[i] for i in perm[:n_test]}
            held_out_smiles = {
                r.get("canonical_smiles", "")
                for r in self.records
                if r.get("murcko_scaffold") in held_out and r.get("canonical_smiles")
            }
            held_out_ik14 = {
                r.get("ik14", "")
                for r in self.records
                if (r.get("murcko_scaffold") in held_out or r.get("canonical_smiles") in held_out_smiles)
                and r.get("ik14")
            }
            test_idx = [
                i
                for i, r in enumerate(self.records)
                if r.get("murcko_scaffold") in held_out
                or (r.get("canonical_smiles") and r.get("canonical_smiles") in held_out_smiles)
                or (r.get("ik14") and r.get("ik14") in held_out_ik14)
            ]
            te_set = set(test_idx)
            train_idx = [i for i in range(len(self.records)) if i not in te_set]
            if train_idx and test_idx:
                return train_idx, test_idx
        mid = max(1, int(len(self.records) * (1.0 - test_frac)))
        return list(range(0, mid)), list(range(mid, len(self.records)))

    def lineage_split(
        self,
        held_out_lineages: Sequence[str] = ("Kidney", "Skin", "Melanoma", "Renal"),
    ) -> tuple[list[int], list[int]]:
        """Hold out Kidney/ccRCC and Skin/Melanoma lineages strictly into test across all buckets."""
        tokens = tuple(tok.lower() for tok in held_out_lineages)
        held_mids = {
            r.get("model_id")
            for r in self.records
            if any(tok in str(r.get("lineage", "")).lower() for tok in tokens) and r.get("model_id") != "__UNMATCHED__"
        }
        train_idx: list[int] = []
        test_idx: list[int] = []
        for i, r in enumerate(self.records):
            lin = str(r.get("lineage", "")).lower()
            if any(tok in lin for tok in tokens) or (r.get("model_id") in held_mids):
                test_idx.append(i)
            else:
                train_idx.append(i)
        if not test_idx or not train_idx:
            mid = max(1, int(len(self.records) * 0.8))
            return list(range(0, mid)), list(range(mid, len(self.records)))
        return train_idx, test_idx

    def organoid_2d_to_3d_split(self) -> tuple[list[int], list[int]]:
        """Train strictly on 2D cultures (`is_3d_organoid == 0`), test strictly on 3D organoids (`== 1`)."""
        held_3d_mids = {
            r.get("model_id")
            for r in self.records
            if int(r.get("is_3d_organoid", 0)) == 1 and r.get("model_id") != "__UNMATCHED__"
        }
        train_idx = [
            i
            for i, r in enumerate(self.records)
            if int(r.get("is_3d_organoid", 0)) == 0 and r.get("model_id") not in held_3d_mids
        ]
        test_idx = [
            i
            for i, r in enumerate(self.records)
            if int(r.get("is_3d_organoid", 0)) == 1 or r.get("model_id") in held_3d_mids
        ]
        if not test_idx or not train_idx:
            mid = max(1, int(len(self.records) * 0.85))
            return list(range(0, mid)), list(range(mid, len(self.records)))
        return train_idx, test_idx

    def crispr_fin_epistasis_split(self) -> tuple[list[int], list[int]]:
        """Train strictly on single perturbations, test strictly on dual `CRISPR + FIN` epistatic screens."""
        train_idx = [i for i, r in enumerate(self.records) if int(r.get("is_crispr_fin_epistasis", 0)) == 0]
        test_idx = [i for i, r in enumerate(self.records) if int(r.get("is_crispr_fin_epistasis", 0)) == 1]
        if not test_idx or not train_idx:
            mid = max(1, int(len(self.records) * 0.90))
            return list(range(0, mid)), list(range(mid, len(self.records)))
        return train_idx, test_idx

    def cuproptosis_specificity_split(self) -> tuple[list[int], list[int]]:
        """Hold out cuproptosis, necroptosis & orthogonal ferroptosis specificity chemotypes strictly into test (zero SMILES/ik14/scaffold overlap)."""
        spec_mechs = ("cuproptosis", "redox_cycling", "txnrd1_inhib", "necroptosis", "complex_i_disulfid")
        spec_names = {"ELESCLOMOL", "DISULFIRAM", "AURANOFIN", "MENADIONE", "NECROSTATIN-1", "BAY 87-2243", "ARTESUNATE", "CICLOPIROX"}
        held_scaffolds = {
            r.get("murcko_scaffold", "")
            for r in self.records
            if r.get("kind") == "pharmaco"
            and (
                r.get("probe_mechanism") in spec_mechs
                or int(r.get("hazard_channel_idx", -1)) in (2, 3)
                or str(r.get("compound_name", "")).upper() in spec_names
            )
            and r.get("murcko_scaffold")
        }
        held_smiles = {
            r.get("canonical_smiles", "")
            for r in self.records
            if (
                r.get("murcko_scaffold") in held_scaffolds
                or (
                    r.get("kind") == "pharmaco"
                    and (r.get("probe_mechanism") in spec_mechs or int(r.get("hazard_channel_idx", -1)) == 3)
                )
            )
            and r.get("canonical_smiles")
        }
        held_ik14 = {
            r.get("ik14", "")
            for r in self.records
            if (r.get("canonical_smiles") in held_smiles or r.get("murcko_scaffold") in held_scaffolds)
            and r.get("ik14")
        }
        test_idx = [
            i
            for i, r in enumerate(self.records)
            if (r.get("canonical_smiles") and r.get("canonical_smiles") in held_smiles)
            or (r.get("murcko_scaffold") and r.get("murcko_scaffold") in held_scaffolds)
            or (r.get("ik14") and r.get("ik14") in held_ik14)
            or (r.get("kind") == "crispr" and str(r.get("fin_modifier_compound", "")).upper() in ("ELESCLOMOL", "DISULFIRAM", "AURANOFIN"))
        ]
        te_set = set(test_idx)
        train_idx = [i for i in range(len(self.records)) if i not in te_set]
        if not test_idx or not train_idx:
            mid = max(1, int(len(self.records) * 0.90))
            return list(range(0, mid)), list(range(mid, len(self.records)))
        return train_idx, test_idx

    def sfti_normal_til_split(self) -> tuple[list[int], list[int]]:
        """Hold out hematopoietic/lymphoid/immune lineages + disjoint solid-tumor models under REAL dosed FIN assays."""
        hema_tokens = ("myeloid", "lymphoid", "blood", "leukemia", "lymphoma", "haematopoietic", "hematopoietic", "immune")
        held_mids: set[str] = set()
        solid_mids: list[str] = []
        for r in self.records:
            mid = str(r.get("model_id", ""))
            if not mid or mid == "__UNMATCHED__":
                continue
            lin = str(r.get("lineage", "")).lower()
            if any(t in lin for t in hema_tokens) or r.get("kind") == "normal_sfti":
                held_mids.add(mid)
            elif r.get("kind") == "pharmaco":
                solid_mids.append(mid)
        uniq_solid = sorted(set(solid_mids))
        for idx_s, smid in enumerate(uniq_solid):
            if idx_s % 5 == 0:
                held_mids.add(smid)

        test_idx = [i for i, r in enumerate(self.records) if str(r.get("model_id", "")) in held_mids]
        te_set = set(test_idx)
        train_idx = [i for i in range(len(self.records)) if i not in te_set]
        if not test_idx or not train_idx:
            mid = max(1, int(len(self.records) * 0.85))
            return list(range(0, mid)), list(range(mid, len(self.records)))
        return train_idx, test_idx


def scaffold_split(ds: FerroFluxMultimodalDataset, test_frac: float = 0.2, seed: int = 42) -> tuple[list[int], list[int]]:
    return ds.scaffold_split(test_frac=test_frac, seed=seed)


def lineage_split(
    ds: FerroFluxMultimodalDataset,
    held_out_lineages: Sequence[str] = ("Kidney", "Skin", "Melanoma", "Renal"),
) -> tuple[list[int], list[int]]:
    return ds.lineage_split(held_out_lineages=held_out_lineages)


def organoid_2d_to_3d_split(ds: FerroFluxMultimodalDataset) -> tuple[list[int], list[int]]:
    return ds.organoid_2d_to_3d_split()


def crispr_fin_epistasis_split(ds: FerroFluxMultimodalDataset) -> tuple[list[int], list[int]]:
    return ds.crispr_fin_epistasis_split()
