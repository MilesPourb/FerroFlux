"""Phase 2A-1 Data ETL: Basal Multi-Omics, 2D/3D CRISPR Chronos & Epistatic Screens, and Pharmacogenomics.

Produces three Zstd-compressed Parquet shards in `data/processed/`:
1. `cells_basal_multiomics.parquet`
2. `crispr_chronos_2d_3d.parquet`
3. `pharmacogenomic_viability.parquet`

Enforces zero-leakage quarantine (`ferroflux.security.quarantine`) on every source and record,
stages raw downloads strictly in `/anvil/scratch/x-mpourbaghima/ferroflux_staging/phase2a/depmap_pharmaco/`,
and aligns all gene vectors to `data/processed/gene_vocab.json` (`n_genes = 2048`).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.Scaffolds import MurckoScaffold

from ferroflux.security.quarantine import (
    assert_not_quarantined,
    filter_dataframe_quarantine,
)

RDLogger.DisableLog("rdApp.*")

ROOT_DIR = Path("/anvil/projects/x-cis261485/Ferroptosis_ML")
DEFAULT_STAGING_DIR = Path("/anvil/scratch/x-mpourbaghima/ferroflux_staging/phase2a/depmap_pharmaco")
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed"
RSCRIPT_BIN = "/apps/spack/anvil/apps/r/4.1.0-gcc-11.2.0-yaooqbd/bin/Rscript"

# Canonical reference SMILES & mechanistic annotations for ferroptosis and 22-RCD compounds
CURATED_RCD_COMPOUNDS: dict[str, dict[str, Any]] = {
    "ERASTIN": {
        "smiles": "CCOc1ccccc1-n1c(=O)c2ccccc2n(C(C)N2CCN(C(=O)Cc3ccc(Cl)cc3)CC2)c1=O",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "xc_inhib",
    },
    "IMIDAZOLE KETONE ERASTIN": {
        "smiles": "CC(C)Oc1ccc(-c2nc3ccccc3n2CC(=O)c2ccc(CN3CCN(c4ccc(C(F)(F)F)cc4)CC3)cc2)cc1",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "xc_inhib",
    },
    "1S,3R-RSL3": {
        "smiles": "COC(=O)[C@@H]1Cc2c([nH]c3ccccc23)[C@H](c2ccc(C(=O)OC)cc2)N1C(=O)CCl",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gpx4_inhib",
    },
    "RSL3": {
        "smiles": "COC(=O)[C@@H]1Cc2c([nH]c3ccccc23)[C@H](c2ccc(C(=O)OC)cc2)N1C(=O)CCl",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gpx4_inhib",
    },
    "ML210": {
        "smiles": "Cc1cc(C(=O)N2CCN(C(c3ccc(Cl)cc3)c3ccc([N+](=O)[O-])o3)CC2)no1",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gpx4_inhib",
    },
    "ML162": {
        "smiles": "COc1ccc(C(N(C(=O)CCl)c2ccc(Cl)cc2)C(=O)NCc2ccco2)cc1OC",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gpx4_inhib",
    },
    "FIN56": {
        "smiles": "O=C(NCCCCCCNC(=O)C1(c2ccccc2)CCN(S(=O)(=O)c2ccc(C(C)(C)C)cc2)CC1)C1(c2ccccc2)CCN(S(=O)(=O)c2ccc(C(C)(C)C)cc2)CC1",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gpx4_inhib",
    },
    "CIL56": {
        "smiles": "CCN(CC)C(=O)c1ccc(C2c3ccccc3C(=O)N2NC(=O)c2cccs2)cc1",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gpx4_inhib",
    },
    "FINO2": {
        "smiles": "CC1(C)CCC2(OO1)CC(O)(C(C)C)CC2",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "endoperoxide",
    },
    "SULFASALAZINE": {
        "smiles": "O=C(O)c1cc(/N=N/c2ccc(S(=O)(=O)Nc3ccccn3)cc2)ccc1O",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "xc_inhib",
    },
    "SORAFENIB": {
        "smiles": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "xc_inhib",
    },
    "ARTESUNATE": {
        "smiles": "CC1CCC2C(C)C(OC(=O)CCC(=O)O)OC3OC4(C)CCC1C23OO4",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "endoperoxide",
    },
    "ARTEMISININ": {
        "smiles": "CC1CCC2C(C)C(=O)OC3OC4(C)CCC1C23OO4",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "endoperoxide",
    },
    "DIHYDROARTEMISININ": {
        "smiles": "CC1CCC2C(C)C(O)OC3OC4(C)CCC1C23OO4",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "endoperoxide",
    },
    "AURANOFIN": {
        "smiles": "CC[P+](CC)(CC)[Au-]SC1OC(COC(C)=O)C(OC(C)=O)C(OC(C)=O)C1OC(C)=O",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "txnrd1_inhib",
    },
    "ALTRETAMINE": {
        "smiles": "CN(C)c1nc(N(C)C)nc(N(C)C)n1",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gpx4_inhib",
    },
    "BUTHIONINE SULFOXIMINE": {
        "smiles": "CCCCS(=O)(=N)CCC(N)C(=O)O",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gsh_depletion",
    },
    "BREQUINAR": {
        "smiles": "Cc1cc(-c2ccc(F)cc2)nc2c(C(=O)O)cc(F)cc12",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "dhodh_inhib",
    },
    "SIMVASTATIN": {
        "smiles": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "statin_hmgcr",
    },
    "LOVASTATIN": {
        "smiles": "CCC(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "statin_hmgcr",
    },
    "FLUVASTATIN": {
        "smiles": "CC(C)n1c(/C=C/C(O)CC(O)CC(=O)O)c(-c2ccc(F)cc2)c2ccccc21",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "statin_hmgcr",
    },
    "ATORVASTATIN": {
        "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "statin_hmgcr",
    },
    "PITAVASTATIN": {
        "smiles": "O=C(O)CC(O)CC(O)/C=C/c1c(C2CC2)nc2ccccc2c1-c1ccc(F)cc1",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "statin_hmgcr",
    },
    "MENADIONE": {
        "smiles": "Cc1cc(=O)c2ccccc2c1=O",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "redox_cycling",
    },
    "APR-246": {
        "smiles": "COC1(CO)CN2CCC1CC2=O",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gsh_depletion",
    },
    "PRIMA-1MET": {
        "smiles": "COC1(CO)CN2CCC1CC2=O",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gsh_depletion",
    },
    "PIPERLONGUMINE": {
        "smiles": "COc1cc(/C=C/C(=O)N2CCC=CC2=O)cc(OC)c1OC",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gsh_depletion",
    },
    "WITHAFERIN A": {
        "smiles": "CC1=C(CO)C(=O)OC(C1)C(C)C1CCC2C3CC4OC45C(O)C=CC(=O)C5(C)C3CCC12C",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gpx4_inhib",
    },
    "FERROSTATIN-1": {
        "smiles": "CCOC(=O)c1cc(NC2CCCCC2)c(N)cc1",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "rta",
    },
    "LIPROXSTATIN-1": {
        "smiles": "Clc1ccc(CNc2ccccc2N2CCNCC2)cc1",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "rta",
    },
    "DEFEROXAMINE": {
        "smiles": "CC(=O)N(O)CCCCCNC(=O)CCC(=O)N(O)CCCCCNC(=O)CCC(=O)N(O)CCCCCN",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "fe_chelator",
    },
    "DEFERASIROX": {
        "smiles": "O=C(O)c1ccc(-n2nc(-c3ccccc3O)nc2-c2ccccc2O)cc1",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "fe_chelator",
    },
    "CICLOPIROX": {
        "smiles": "Cc1cc(C2CCCCC2)n(O)c(=O)c1",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "fe_chelator",
    },
    "ROSIGLITAZONE": {
        "smiles": "CN(CCOc1ccc(CC2SC(=O)NC2=O)cc1)c1ccccn1",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "acsl4_inhib",
    },
    "PIOGLITAZONE": {
        "smiles": "CCc1ccc(CCOc2ccc(CC3SC(=O)NC3=O)cc2)nc1",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "acsl4_inhib",
    },
    "ZILEUTON": {
        "smiles": "CC(c1cc2ccccc2s1)N(O)C(N)=O",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "alox_inhib",
    },
    "GLUTATHIONE": {
        "smiles": "NC(CCC(=O)NC(CS)C(=O)NCC(=O)O)C(=O)O",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "thiol_antioxidant",
    },
    "N-ACETYL CYSTEINE": {
        "smiles": "CC(=O)NC(CS)C(=O)O",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "thiol_antioxidant",
    },
    "ALPHA-LIPOIC ACID": {
        "smiles": "O=C(O)CCCCC1CCSS1",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "thiol_antioxidant",
    },
    "ELESCLOMOL": {
        "smiles": "CN(NC(=S)c1ccccc1)C(=O)CC(=O)N(C)NC(=S)c1ccccc1",
        "rcd_modality": "cuproptosis",
        "hazard_channel_idx": 3,
        "probe_mechanism": "cuproptosis",
    },
    "DISULFIRAM": {
        "smiles": "CCN(CC)C(=S)SSC(=S)N(CC)CC",
        "rcd_modality": "cuproptosis",
        "hazard_channel_idx": 3,
        "probe_mechanism": "cuproptosis",
    },
    "VENETOCLAX": {
        "smiles": "CC1(C)CCC(CN2CCN(c3ccc(C(=O)NS(=O)(=O)c4ccc(NCC5CCOCC5)c([N+](=O)[O-])c4)c(Oc4cnc5[nH]ccc5c4)c3)CC2)=C(c2ccc(Cl)cc2)C1",
        "rcd_modality": "apoptosis",
        "hazard_channel_idx": 1,
        "probe_mechanism": "apoptosis",
    },
    "ABT-199": {
        "smiles": "CC1(C)CCC(CN2CCN(c3ccc(C(=O)NS(=O)(=O)c4ccc(NCC5CCOCC5)c([N+](=O)[O-])c4)c(Oc4cnc5[nH]ccc5c4)c3)CC2)=C(c2ccc(Cl)cc2)C1",
        "rcd_modality": "apoptosis",
        "hazard_channel_idx": 1,
        "probe_mechanism": "apoptosis",
    },
    "NAVITOCLAX": {
        "smiles": "CC1(C)CCC(CN2CCN(c3ccc(C(=O)NS(=O)(=O)c4ccc(NC(CCN5CCOCC5)CSc5ccccc5)c(S(=O)(=O)C(F)(F)F)c4)cc3)CC2)=C(c2ccc(Cl)cc2)C1",
        "rcd_modality": "apoptosis",
        "hazard_channel_idx": 1,
        "probe_mechanism": "apoptosis",
    },
    "BORTEZOMIB": {
        "smiles": "CC(C)CC(NC(=O)C(Cc1ccccc1)NC(=O)c1cnccn1)B(O)O",
        "rcd_modality": "apoptosis",
        "hazard_channel_idx": 1,
        "probe_mechanism": "proteasome_inhib",
    },
    "STAUROSPORINE": {
        "smiles": "CNC1CC2OC(C)(C1OC)n1c3ccccc3c3c4c(c5c6ccccc6n2c5c31)C(=O)NC4",
        "rcd_modality": "apoptosis",
        "hazard_channel_idx": 1,
        "probe_mechanism": "apoptosis",
    },
    "PACLITAXEL": {
        "smiles": "CC(=O)OC1C(=O)C2(C)C(O)CC3OCC3(OC(C)=O)C2C(OC(=O)c2ccccc2)C2(O)CC(OC(=O)C(O)C(NC(=O)c3ccccc3)c3ccccc3)C(C)=C1C2(C)C",
        "rcd_modality": "apoptosis",
        "hazard_channel_idx": 1,
        "probe_mechanism": "apoptosis",
    },
    "DOXORUBICIN": {
        "smiles": "COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)CC(O)(C(=O)CO)CC3OC1CC(N)C(O)C(C)O1",
        "rcd_modality": "apoptosis",
        "hazard_channel_idx": 1,
        "probe_mechanism": "apoptosis",
    },
    "VORINOSTAT": {
        "smiles": "O=C(CCCCCCC(=O)Nc1ccccc1)NO",
        "rcd_modality": "apoptosis",
        "hazard_channel_idx": 1,
        "probe_mechanism": "hdac_inhib",
    },
    "NECROSTATIN-1": {
        "smiles": "CN1C(=O)NC(Cc2c[nH]c3ccccc23)C1=S",
        "rcd_modality": "necroptosis",
        "hazard_channel_idx": 2,
        "probe_mechanism": "necroptosis",
    },
    "RAPAMYCIN": {
        "smiles": "COC1CCC2CC(C)C(=CC=CC=CC(C)CC(C)C(=O)C(OC)C(O)C(C)CC3CCC(O)(C(=O)C(=O)N4CCCCC4C(=O)OC(C(C)CC1O)C(O)(C)C2=O)O3)OC",
        "rcd_modality": "autophagy",
        "hazard_channel_idx": 1,
        "probe_mechanism": "mtor_inhib",
    },
    "DASATINIB": {
        "smiles": "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
        "rcd_modality": "apoptosis",
        "hazard_channel_idx": 1,
        "probe_mechanism": "kinase_inhib",
    },
    "LAPATINIB": {
        "smiles": "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
        "rcd_modality": "apoptosis",
        "hazard_channel_idx": 1,
        "probe_mechanism": "kinase_inhib",
    },
    "BAY 87-2243": {
        "smiles": "Cc1ccc(-c2nc(-c3ccc(OC(F)(F)F)cc3)no2)cc1CN1CCC(c2nc(-c3ccccn3)no2)CC1",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "complex_i_disulfid",
    },
    "IFSP1": {
        "smiles": "Cc1ccc(-c2nc3cc(C#N)ccc3n2N)cc1",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "fsp1_inhib",
    },
    "TERIFLUNOMIDE": {
        "smiles": "CC(=C(C#N)C(=O)Nc1ccc(C(F)(F)F)cc1)O",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "dhodh_inhib",
    },
    "LEFLUNOMIDE": {
        "smiles": "Cc1oncc1C(=O)Nc1ccc(C(F)(F)F)cc1",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "dhodh_inhib",
    },
    "BAY-2402234": {
        "smiles": "CCC1=NN(C(=O)N1c1ccc(Oc2ccc(C(F)(F)F)c(Cl)c2)nc1)c1ccc(F)c(CO)c1",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "dhodh_inhib",
    },
    "MEVASTATIN": {
        "smiles": "CCC(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "statin_hmgcr",
    },
    "PRAVASTATIN": {
        "smiles": "CCC(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC(O)CC(O)CC(=O)O)C21",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "statin_hmgcr",
    },
    "ROSUVASTATIN": {
        "smiles": "CC(C)c1nc(N(C)S(=O)(=O)C)nc(-c2ccc(F)cc2)c1/C=C/C(O)CC(O)CC(=O)O",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "statin_hmgcr",
    },
    "CERIVASTATIN": {
        "smiles": "COCc1c(C(C)C)nc(C(C)C)c(/C=C/C(O)CC(O)CC(=O)O)c1-c1ccc(F)cc1",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "statin_hmgcr",
    },
    "TERBINAFINE": {
        "smiles": "CN(C/C=C/C#CC(C)(C)C)Cc1cccc2ccccc12",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "sqle_inhib",
    },
    "PRIMA-1": {
        "smiles": "OC1(CO)CN2CCC1CC2=O",
        "rcd_modality": "ferroptosis_inducer",
        "hazard_channel_idx": 0,
        "probe_mechanism": "gsh_depletion",
    },
    "DEFERIPRONE": {
        "smiles": "Cc1c(O)c(=O)ccn1C",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "fe_chelator",
    },
    "DEXRAZOXANE": {
        "smiles": "CC(CN1CC(=O)NC(=O)C1)N1CC(=O)NC(=O)C1",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "fe_chelator",
    },
    "TROGLITAZONE": {
        "smiles": "Cc1c(C)c2c(c(C)c1O)CCC(C)(COc1ccc(CC3SC(=O)NC3=O)cc1)O2",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "acsl4_inhib",
    },
    "BAICALEIN": {
        "smiles": "O=c1cc(-c2ccccc2)oc2cc(O)c(O)c(O)c12",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "alox_inhib",
    },
    "IDEBENONE": {
        "smiles": "COc1c(OC)c(=O)c(CCCCCCCCCCO)c(C)c1=O",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "rta",
    },
    "SAPROPTERIN": {
        "smiles": "CC(O)C(O)C1CNc2nc(N)[nH]c(=O)c2N1",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "rta",
    },
    "D-ALPHA-TOCOPHERYL-SUCCINATE": {
        "smiles": "CC1=C(C)C(OC(=O)CCC(=O)O)=C(C)C2=C1OC(C)(CCCC(C)CCCC(C)CCCC(C)C)CC2",
        "rcd_modality": "ferroptosis_rescuer",
        "hazard_channel_idx": -1,
        "probe_mechanism": "rta",
    },
}

_NORM_CURATED_LOOKUP: dict[str, tuple[str, dict[str, Any]]] = {
    re.sub(r"[^A-Z0-9]", "", k.upper()): (k, v) for k, v in CURATED_RCD_COMPOUNDS.items()
}
for _alias, _canon in (
    ("1S3RRSL3", "1S,3R-RSL3"),
    ("RSL3", "1S,3R-RSL3"),
    ("BSO", "BUTHIONINE SULFOXIMINE"),
    ("BUTHIONINESULFOXIMINE", "BUTHIONINE SULFOXIMINE"),
    ("ACETYLCYSTEINE", "N-ACETYL CYSTEINE"),
    ("NAC", "N-ACETYL CYSTEINE"),
    ("NACETYLCYSTEINE", "N-ACETYL CYSTEINE"),
    ("NACETYLCYSTEINEAMIDE", "N-ACETYL CYSTEINE"),
    ("DEFEROXAMINEMESYLATE", "DEFEROXAMINE"),
    ("DFO", "DEFEROXAMINE"),
    ("IKE", "IMIDAZOLE KETONE ERASTIN"),
    ("PRIMA1MET", "PRIMA-1MET"),
    ("SORAFENIBTOSYLATE", "SORAFENIB"),
    ("ELESCLOMOLSTA4783", "ELESCLOMOL"),
    ("PRAVASTATINSODIUM", "PRAVASTATIN"),
    ("FLUVASTATINSODIUM", "FLUVASTATIN"),
    ("ATORVASTATINCALCIUM", "ATORVASTATIN"),
):
    if _canon in CURATED_RCD_COMPOUNDS:
        _NORM_CURATED_LOOKUP[_alias] = (_canon, CURATED_RCD_COMPOUNDS[_canon])

# Focused panel of ~130 core ferroptosis, Sec46, sterol, iron, immune & 22-RCD genes for Chronos extraction
CORE_CHRONOS_GENES: list[str] = [
    "GPX4", "SLC7A11", "SLC3A2", "GCLC", "GCLM", "GSS", "GSR",
    "PSTK", "SEPSECS", "SEPHS2", "EEFSEC", "SECISBP2", "LRP8", "PRDX6",
    "AIFM2", "VKORC1L1", "DHODH", "GCH1", "PTS", "SPR", "TXNRD1", "TXN", "NQO1", "MGST1",
    "ACSL4", "LPCAT3", "ACSL3", "MBOAT1", "MBOAT2", "SCD", "FADS1", "FADS2",
    "FAR1", "AGPS", "GNPAT", "TMEM189",
    "DHCR7", "EBP", "SC5D", "SQLE", "HMGCR", "FDPS", "POR", "CYB5R1", "ALOX15", "ALOX12",
    "TFRC", "FTH1", "FTL", "NCOA4", "SLC40A1", "STEAP3", "SLC11A2", "HMOX1", "ODC1", "SMS",
    "NFE2L2", "KEAP1", "BACH1", "TP53", "YAP1", "NF2", "FDX1", "CD274",
    "SRM", "SAT1", "TRIT1", "CBS", "CTH", "G6PD", "PGD", "IDH1", "ME1", "GLS",
    "MVK", "PMVK", "MVD", "FDFT1", "LSS", "CYP51A1", "MSMO1", "NSDHL", "DHCR24",
    "PDSS1", "PDSS2", "COQ2", "UBIAD1", "PLA2G6", "MBOAT7", "ACACA", "FASN",
    "ATP6V0A1", "ATP6V0C", "ATP6V1A", "ATP6V1B2", "MCOLN1", "LGALS3", "LAMP1", "CTSB",
    "IREB2", "CISD1", "NFS1", "MC1R", "ITGAV", "ITGB3", "FOLH1", "ERBB2", "EGFR",
    "TLR2", "TLR7", "CGAS", "STING1", "IFNGR1", "JAK1", "STAT1",
    "LIAS", "DLAT", "NCKAP1", "RPN1", "SLC31A1", "BCL2", "BCL2L1", "MCL1", "BAX", "CASP8",
    "RIPK1", "MLKL", "GSDMD", "PARP1", "ATG5", "ATG7", "BECN1", "MTOR", "ATF4",
    "EPAS1", "VHL", "BAP1", "MITF", "SOX10", "AXL", "ZEB1", "WWTR1", "BRAF", "NRAS", "KRAS", "PTEN",
]


import xml.etree.ElementTree as ET


def read_xlsx_sheet(xlsx_path: Path, sheet_index: int = 0, max_rows: int | None = None) -> pd.DataFrame:
    """Fast stdlib XLSX reader (zipfile + ElementTree iterparse) requiring no openpyxl."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            with zf.open("xl/sharedStrings.xml") as sf:
                for event, elem in ET.iterparse(sf, events=("end",)):
                    if elem.tag == f"{ns}si":
                        txt = "".join(t.text or "" for t in elem.iter(f"{ns}t"))
                        shared_strings.append(txt)
                        elem.clear()

        sheet_files = sorted(
            [n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")],
            key=lambda s: int(re.sub(r"\D", "", s) or "0"),
        )
        target_sheet = sheet_files[min(sheet_index, len(sheet_files) - 1)]

        def _col_idx(cell_ref: str) -> int:
            letters = ""
            for ch in cell_ref:
                if ch.isalpha():
                    letters += ch
                else:
                    break
            idx = 0
            for ch in letters.upper():
                idx = idx * 26 + (ord(ch) - ord("A") + 1)
            return max(idx - 1, 0)

        rows: list[list[str]] = []
        max_cols = 0
        with zf.open(target_sheet) as wf:
            row_dict: dict[int, str] = {}
            cell_type = ""
            cell_col = 0
            for event, elem in ET.iterparse(wf, events=("start", "end")):
                if event == "start" and elem.tag == f"{ns}c":
                    cell_type = elem.attrib.get("t", "")
                    cell_col = _col_idx(elem.attrib.get("r", "A1"))
                elif event == "end" and elem.tag == f"{ns}v":
                    raw_v = elem.text or ""
                    if cell_type == "s" and raw_v.isdigit():
                        s_i = int(raw_v)
                        val = shared_strings[s_i] if s_i < len(shared_strings) else raw_v
                    else:
                        val = raw_v
                    row_dict[cell_col] = val
                elif event == "end" and elem.tag == f"{ns}is":
                    val = "".join(t.text or "" for t in elem.iter(f"{ns}t"))
                    row_dict[cell_col] = val
                elif event == "end" and elem.tag == f"{ns}row":
                    if row_dict:
                        cmax = max(row_dict.keys()) + 1
                        max_cols = max(max_cols, cmax)
                        row_list = [row_dict.get(i, "") for i in range(cmax)]
                        rows.append(row_list)
                        row_dict = {}
                    elem.clear()
                    if max_rows is not None and len(rows) >= max_rows + 1:
                        break
        if not rows:
            return pd.DataFrame()
        header = [h if h else f"col_{i}" for i, h in enumerate(rows[0])]
        if len(header) < max_cols:
            header.extend(f"col_{i}" for i in range(len(header), max_cols))
        norm_rows = [r + [""] * (len(header) - len(r)) if len(r) < len(header) else r[: len(header)] for r in rows[1:]]
        return pd.DataFrame(norm_rows, columns=header)


def download_file(url: str, dest: Path, accession_check: str) -> Path:
    """Download file via curl with automatic retries, quarantine verification, and JSON/HTML error detection."""
    assert_not_quarantined(accession_check)
    assert_not_quarantined(url)
    if dest.exists() and dest.stat().st_size > 512:
        with open(dest, "rb") as fh:
            head = fh.read(256)
        if not (head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or b'"type":"invalid-json"' in head):
            return dest
        dest.unlink(missing_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        "curl",
        "-fsSL",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "-A",
        "FerroFlux-ETL/1.0",
        "-o",
        str(tmp),
        url,
    ]
    subprocess.run(cmd, check=True)
    with open(tmp, "rb") as fh:
        head = fh.read(512)
    if head.lstrip().startswith(b"{") and b'"url"' in head:
        redirect_obj = json.loads(tmp.read_text(encoding="utf-8"))
        s3_url = str(redirect_obj.get("url", "")).strip()
        if s3_url:
            assert_not_quarantined(s3_url)
            subprocess.run(
                ["curl", "-fsSL", "--retry", "5", "--retry-delay", "2", "-A", "FerroFlux-ETL/1.0", "-o", str(tmp), s3_url],
                check=True,
            )
            with open(tmp, "rb") as fh2:
                head = fh2.read(256)
    if head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or b'"type":"invalid-json"' in head:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Endpoint {url} returned an HTML/error payload rather than data.")
    tmp.replace(dest)
    return dest


def canonicalize_compound(name: str, raw_smiles: str = "") -> dict[str, Any] | None:
    """Map compound name / SMILES to curated RCD annotation or RDKit canonical descriptors."""
    up = str(name).strip().upper()
    if not up or up == "NAN" or ":" in up:
        return None

    norm_up = re.sub(r"[^A-Z0-9]", "", up)
    if not norm_up:
        return None

    matched_meta: dict[str, Any] | None = None
    if up in CURATED_RCD_COMPOUNDS:
        matched_meta = CURATED_RCD_COMPOUNDS[up]
    elif norm_up in _NORM_CURATED_LOOKUP:
        up, matched_meta = _NORM_CURATED_LOOKUP[norm_up]
    else:
        # Strip common salt/ester suffixes (`MESYLATE`, `TOSYLATE`, `HCL`, `HYDROCHLORIDE`, `SODIUM`, `CALCIUM`)
        base_token = re.split(r"[\s_\(\[\-]+(?:MESYLATE|TOSYLATE|HYDROCHLORIDE|HCL|SODIUM|CALCIUM|MALEATE|CITRATE|SULFATE|TARTRATE)\b", up)[0]
        norm_base = re.sub(r"[^A-Z0-9]", "", base_token)
        if norm_base and norm_base in _NORM_CURATED_LOOKUP:
            up, matched_meta = _NORM_CURATED_LOOKUP[norm_base]
        else:
            for key, meta in CURATED_RCD_COMPOUNDS.items():
                if len(key) >= 6 and (up.startswith(key + " ") or up.startswith(key + "-") or up.startswith(key + "(")):
                    matched_meta = meta
                    up = key
                    break

    smi = str(raw_smiles).strip() if raw_smiles and str(raw_smiles).strip().lower() not in ("nan", "none", "") else ""
    if not smi and matched_meta:
        smi = matched_meta["smiles"]

    if matched_meta is None:
        return None

    canon_smi = smi
    ikey = ""
    scaffold_smi = up
    if smi:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canon_smi = Chem.MolToSmiles(mol, canonical=True)
            try:
                ikey = inchi.MolToInchiKey(mol) or ""
            except Exception:
                ikey = ""
            try:
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                scaf_str = Chem.MolToSmiles(scaf, canonical=True) if scaf is not None else ""
                scaffold_smi = scaf_str if scaf_str else canon_smi
            except Exception:
                scaffold_smi = canon_smi

    return {
        "compound_name": up,
        "canonical_smiles": canon_smi,
        "inchikey": ikey,
        "murcko_scaffold": scaffold_smi,
        "rcd_modality": matched_meta["rcd_modality"],
        "hazard_channel_idx": int(matched_meta["hazard_channel_idx"]),
        "probe_mechanism": matched_meta["probe_mechanism"],
    }


def parse_depmap_gene_col(col: str, alias_map: dict[str, str]) -> str:
    sym = col.split(" (")[0].strip()
    return alias_map.get(sym, sym)


def align_expression_dataframe(
    df_expr: pd.DataFrame,
    vocab_genes: list[str],
    gene_to_idx: dict[str, int],
    alias_map: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Project an arbitrary samples x genes expression DataFrame onto [N, 2048] float32."""
    n_samples = len(df_expr)
    n_genes = len(vocab_genes)
    mat = np.zeros((n_samples, n_genes), dtype=np.float32)
    observed = np.zeros(n_genes, dtype=bool)

    col_indices: list[int] = []
    target_indices: list[int] = []
    for c_i, col in enumerate(df_expr.columns):
        sym = parse_depmap_gene_col(str(col), alias_map)
        idx = gene_to_idx.get(sym)
        if idx is not None and not observed[idx]:
            observed[idx] = True
            col_indices.append(c_i)
            target_indices.append(idx)

    if col_indices:
        sub = df_expr.iloc[:, col_indices].to_numpy(dtype=np.float32, copy=True)
        valid_counts = np.zeros(n_samples, dtype=np.int32)
        finite_mask = np.isfinite(sub)
        valid_counts = finite_mask.sum(axis=1).astype(np.int32)
        sub = np.nan_to_num(sub, nan=0.0, posinf=15.0, neginf=0.0)
        mat[:, target_indices] = sub
    else:
        valid_counts = np.zeros(n_samples, dtype=np.int32)

    return mat, valid_counts


def run_depmap_pharmaco_etl(staging_dir: Path, output_dir: Path) -> dict[str, int]:
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "gene_vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab_genes: list[str] = vocab["genes"]
    gene_to_idx: dict[str, int] = vocab["gene_to_idx"]
    alias_map: dict[str, str] = vocab.get("alias_to_canonical", {})
    loci_64: list[str] = vocab["canonical_64_loci"]

    print("[Phase 2A-1] Downloading DepMap 24Q4, NextGen 3D, Sanger CMP/GDSC, PRISM, BeatAML, CCMA V3 & BioGRID ORCS...")

    # 1. Core DepMap 24Q4 & NextGen 3D Organoids
    model_csv = download_file("https://ndownloader.figshare.com/files/51065297", staging_dir / "Model.csv", "figshare.27993248")
    expr_csv = download_file(
        "https://ndownloader.figshare.com/files/51065492",
        staging_dir / "OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv",
        "figshare.27993248",
    )
    expr_uncorr_csv = download_file(
        "https://ndownloader.figshare.com/files/51065489",
        staging_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "figshare.27993248",
    )
    crispr_csv = download_file(
        "https://ndownloader.figshare.com/files/51064667",
        staging_dir / "CRISPRGeneEffect.csv",
        "figshare.27993248",
    )
    cn_csv = download_file(
        "https://ndownloader.figshare.com/files/51065303",
        staging_dir / "OmicsAbsoluteCNGene.csv",
        "figshare.27993248",
    )
    mut_hot_csv = download_file(
        "https://ndownloader.figshare.com/files/51065750",
        staging_dir / "OmicsSomaticMutationsMatrixHotspot.csv",
        "figshare.27993248",
    )
    mut_dam_csv = download_file(
        "https://ndownloader.figshare.com/files/51065747",
        staging_dir / "OmicsSomaticMutationsMatrixDamaging.csv",
        "figshare.27993248",
    )

    ng3d_meta_csv = download_file(
        "https://ndownloader.figshare.com/files/64494438",
        staging_dir / "nextgen3d_model_metadata.csv",
        "figshare.29472362",
    )
    ng3d_chronos_csv = download_file(
        "https://ndownloader.figshare.com/files/64494480",
        staging_dir / "minipool_chronos_gene_effects.csv",
        "figshare.29472362",
    )

    # 2. Sanger CMP & Organoids CRISPR
    cmp_models_gz = download_file(
        "https://cog.sanger.ac.uk/cmp/download/model_list_latest.csv.gz",
        staging_dir / "model_list_latest.csv.gz",
        "Sanger_CMP_2026",
    )
    cmp_org_zip = download_file(
        "https://cog.sanger.ac.uk/cmp/download/Organoids_fitness_scores_20260723.zip",
        staging_dir / "Organoids_fitness_scores_20260723.zip",
        "Sanger_Organoids_20260723",
    )

    # 3. BeatAML Waves 1-4
    beataml_probit_txt = download_file(
        "https://media.githubusercontent.com/media/biodev/beataml2.0_data/main/beataml_probit_curve_fits_v4_dbgap.txt",
        staging_dir / "beataml_probit_curve_fits_v4_dbgap.txt",
        "BeatAML_Waves1to4",
    )
    beataml_exp_txt = download_file(
        "https://media.githubusercontent.com/media/biodev/beataml2.0_data/main/beataml_waves1to4_norm_exp_dbgap.txt",
        staging_dir / "beataml_waves1to4_norm_exp_dbgap.txt",
        "BeatAML_Waves1to4",
    )

    # 4. Optional CCMA V3 Pediatric 2D/3D Neurospheres (graceful skip if Mendeley CDN returns 502)
    ccma_available = False
    ccma_info_csv = staging_dir / "CCMA_sampleinfo_042024.csv"
    ccma_rna_csv = staging_dir / "CCMA_rnaseq_logcpm1_042024.csv"
    ccma_crispr_csv = staging_dir / "CCMA_crispr_genedependency_042024.csv"
    ccma_drug_csv = staging_dir / "CCMA_drugscreen_zscores_042024.csv"
    try:
        download_file(
            "https://data.mendeley.com/public-api/datasets/rnfs539pfw/files/37e65525-0240-461b-8ed0-cfcf8f8f2017/file_downloaded",
            ccma_info_csv,
            "10.17632/rnfs539pfw.3",
        )
        download_file(
            "https://data.mendeley.com/public-api/datasets/rnfs539pfw/files/e5db45a1-8883-459d-975d-9bb14492e2ee/file_downloaded",
            ccma_rna_csv,
            "10.17632/rnfs539pfw.3",
        )
        download_file(
            "https://data.mendeley.com/public-api/datasets/rnfs539pfw/files/39eeac25-546c-4f43-85e0-0e4fb0a25653/file_downloaded",
            ccma_crispr_csv,
            "10.17632/rnfs539pfw.3",
        )
        download_file(
            "https://data.mendeley.com/public-api/datasets/rnfs539pfw/files/d9ef2772-95a8-4449-98f7-225fdcab2e55/file_downloaded",
            ccma_drug_csv,
            "10.17632/rnfs539pfw.3",
        )
        ccma_available = True
    except Exception as exc:
        print(f"[Phase 2A-1] Note: Mendeley CCMA CDN unavailable ({exc}); relying on DepMap NextGen 3D + Sanger 3D Organoids.")

    # ------------------------------------------------------------------
    # PART A: Harmonize Basal Multi-Omics (`cells_basal_multiomics.parquet`)
    # ------------------------------------------------------------------
    print("[Phase 2A-1] Building cells_basal_multiomics.parquet...")
    df_models = pd.read_csv(model_csv, low_memory=False)
    df_ng3d = pd.read_csv(ng3d_meta_csv, low_memory=False)
    ng3d_ids: set[str] = set()
    if "ModelID" in df_ng3d.columns:
        mask_3d = pd.Series(False, index=df_ng3d.index)
        if "IsNextGen" in df_ng3d.columns:
            mask_3d |= df_ng3d["IsNextGen"].astype(str).str.lower().isin(["true", "1"])
        if "CRISPRScreenType" in df_ng3d.columns:
            mask_3d |= df_ng3d["CRISPRScreenType"].astype(str).str.upper().str.startswith("3D")
        if "GrowthPattern" in df_ng3d.columns:
            mask_3d |= df_ng3d["GrowthPattern"].astype(str).str.lower().isin(["dome", "spheroid", "organoid"])
        ng3d_ids.update(df_ng3d.loc[mask_3d, "ModelID"].dropna().astype(str).tolist())

    df_cmp = pd.read_csv(cmp_models_gz, low_memory=False)
    sidm_to_ach: dict[str, str] = {}
    cosmic_to_ach: dict[str, str] = {}
    cmp_3d_sidms: set[str] = set()
    cmp_3d_meta: dict[str, dict[str, Any]] = {}
    for _, r in df_cmp.iterrows():
        sidm = str(r.get("model_id", "")).strip()
        broad = str(r.get("BROAD_ID", "")).strip()
        cosmic = str(r.get("COSMIC_ID", "")).split(".")[0].strip()
        mtype = str(r.get("model_type", "")).lower()
        growth = str(r.get("growth_properties", "")).lower()
        if sidm and broad.startswith("ACH-"):
            sidm_to_ach[sidm] = broad
        if cosmic and cosmic != "nan" and broad.startswith("ACH-"):
            cosmic_to_ach[cosmic] = broad
        if "organoid" in mtype or "3d" in growth:
            cmp_3d_sidms.add(sidm)
            tiss = str(r.get("tissue", "Bowel")).strip()
            lin_map = {
                "Large Intestine": "Bowel",
                "Esophagus": "Esophagus/Stomach",
                "Stomach": "Esophagus/Stomach",
                "Pancreas": "Pancreas",
                "Lung": "Lung",
                "Breast": "Breast",
                "Biliary Tract": "Biliary Tract",
                "Liver": "Liver",
                "Ovary": "Ovary/Fallopian Tube",
                "Kidney": "Kidney",
                "Skin": "Skin",
                "Central Nervous System": "CNS/Brain",
            }
            cmp_3d_meta[sidm] = {
                "cell_line_name": str(r.get("model_name", sidm)).strip(),
                "cosmic_id": cosmic if cosmic != "nan" else "",
                "lineage": lin_map.get(tiss, tiss if tiss and tiss != "nan" else "Bowel"),
                "sublineage": str(r.get("cancer_type", "Patient-Derived 3D Organoid")).strip(),
            }

    model_meta_map: dict[str, dict[str, Any]] = {}
    for _, r in df_models.iterrows():
        mid = str(r.get("ModelID", "")).strip()
        if not mid.startswith("ACH-"):
            continue
        sidm = str(r.get("SangerModelID", "")).strip()
        if sidm == "nan":
            sidm = ""
        cosmic = str(r.get("COSMICID", "")).split(".")[0].strip()
        if cosmic == "nan":
            cosmic = ""
        if sidm:
            sidm_to_ach[sidm] = mid
        if cosmic:
            cosmic_to_ach[cosmic] = mid
        gp = str(r.get("GrowthPattern", "")).lower()
        mtype = str(r.get("DepmapModelType", "")).lower()
        model_type_col = str(r.get("ModelType", "")).lower()
        is_3d = int(
            mid in ng3d_ids
            or sidm in cmp_3d_sidms
            or "organoid" in model_type_col
            or any(k in gp for k in ("organoid", "sphere", "spheroid", "dome", "3d"))
        )
        lineage = str(r.get("OncotreeLineage", "Unknown")).strip()
        if not lineage or lineage == "nan":
            lineage = "Unknown"
        sublineage = str(r.get("OncotreePrimaryDisease", mtype)).strip()
        if not sublineage or sublineage == "nan":
            sublineage = "Unknown"
        cname = str(r.get("CellLineName", r.get("StrippedCellLineName", mid))).strip()
        model_meta_map[mid] = {
            "cell_line_name": cname if cname and cname != "nan" else mid,
            "sanger_model_id": sidm,
            "cosmic_id": cosmic,
            "lineage": lineage,
            "sublineage": sublineage,
            "is_3d_organoid": is_3d,
        }

    # Read DepMap Expression (first col is ModelID), supplemented by uncorrected TPM for 3D Dome/Spheroid models
    df_expr_raw = pd.read_csv(expr_csv, index_col=0, low_memory=False)
    df_expr_raw.index = df_expr_raw.index.astype(str)
    df_expr_raw = df_expr_raw[~df_expr_raw.index.duplicated(keep="first")]

    df_expr_uncorr = pd.read_csv(expr_uncorr_csv, index_col=0, low_memory=False)
    df_expr_uncorr.index = df_expr_uncorr.index.astype(str)
    df_expr_uncorr = df_expr_uncorr[~df_expr_uncorr.index.duplicated(keep="first")]
    missing_expr_ids = [m for m in df_expr_uncorr.index if m.startswith("ACH-") and m not in df_expr_raw.index]
    if missing_expr_ids:
        common_cols = [c for c in df_expr_raw.columns if c in df_expr_uncorr.columns]
        df_expr_raw = pd.concat([df_expr_raw[common_cols], df_expr_uncorr.loc[missing_expr_ids, common_cols]], axis=0)
    del df_expr_uncorr
    depmap_expr_mat, depmap_valid_cnt = align_expression_dataframe(df_expr_raw, vocab_genes, gene_to_idx, alias_map)

    # Read CN & Damaging/Hotspot mutations across 64 loci
    df_cn_raw = pd.read_csv(cn_csv, index_col=0, low_memory=False)
    df_cn_raw.index = df_cn_raw.index.astype(str)
    cn_col_for_locus: dict[int, str] = {}
    for col in df_cn_raw.columns:
        sym = parse_depmap_gene_col(str(col), alias_map)
        if sym in loci_64:
            cn_col_for_locus[loci_64.index(sym)] = col

    df_dam_raw = pd.read_csv(mut_dam_csv, index_col=0, low_memory=False)
    df_dam_raw.index = df_dam_raw.index.astype(str)
    dam_col_for_locus: dict[int, str] = {}
    for col in df_dam_raw.columns:
        sym = parse_depmap_gene_col(str(col), alias_map)
        if sym in loci_64:
            dam_col_for_locus[loci_64.index(sym)] = col

    df_hot_raw = pd.read_csv(mut_hot_csv, index_col=0, low_memory=False)
    df_hot_raw.index = df_hot_raw.index.astype(str)
    driver_hot_cols: dict[str, str] = {}
    for col in df_hot_raw.columns:
        sym = parse_depmap_gene_col(str(col), alias_map)
        if sym in {"BRAF", "NRAS", "KRAS", "HRAS", "PIK3CA", "NFE2L2", "TP53", "IDH1", "IDH2", "EGFR"}:
            driver_hot_cols[sym] = col

    basal_rows: list[dict[str, Any]] = []
    zero_64 = [0.0] * 64

    for row_i, mid in enumerate(df_expr_raw.index):
        if not mid.startswith("ACH-"):
            continue
        meta = model_meta_map.get(
            mid,
            {
                "cell_line_name": mid,
                "sanger_model_id": "",
                "cosmic_id": "",
                "lineage": "Unknown",
                "sublineage": "Unknown",
                "is_3d_organoid": int(mid in ng3d_ids),
            },
        )
        cn_vec = list(zero_64)
        if mid in df_cn_raw.index:
            cn_row = df_cn_raw.loc[mid]
            if isinstance(cn_row, pd.DataFrame):
                cn_row = cn_row.iloc[0]
            for l_i, cname in cn_col_for_locus.items():
                val = float(cn_row[cname])
                if np.isfinite(val) and val > 0:
                    cn_vec[l_i] = float(np.log2(max(val, 0.125) / 2.0))

        lof_vec = list(zero_64)
        mut_tags: list[str] = []
        if mid in df_dam_raw.index:
            dam_row = df_dam_raw.loc[mid]
            if isinstance(dam_row, pd.DataFrame):
                dam_row = dam_row.iloc[0]
            for l_i, cname in dam_col_for_locus.items():
                val = float(dam_row[cname])
                if np.isfinite(val) and val > 0:
                    lof_vec[l_i] = 1.0
                    mut_tags.append(f"{loci_64[l_i]}_LOF")

        if mid in df_hot_raw.index:
            hot_row = df_hot_raw.loc[mid]
            if isinstance(hot_row, pd.DataFrame):
                hot_row = hot_row.iloc[0]
            for sym, cname in driver_hot_cols.items():
                val = float(hot_row[cname])
                if np.isfinite(val) and val > 0:
                    mut_tags.append(f"{sym}_HOTSPOT")

        basal_rows.append(
            {
                "model_id": mid,
                "cell_line_name": meta["cell_line_name"],
                "sanger_model_id": meta["sanger_model_id"],
                "cosmic_id": meta["cosmic_id"],
                "lineage": meta["lineage"],
                "sublineage": meta["sublineage"],
                "is_3d_organoid": int(meta["is_3d_organoid"]),
                "expr_vec": depmap_expr_mat[row_i].tolist(),
                "expr_mask_count": int(depmap_valid_cnt[row_i]),
                "cnv_64_vec": cn_vec,
                "mut_lof_64_vec": lof_vec,
                "key_mutations": ";".join(mut_tags[:12]),
                "dataset_source": "DepMap_NextGen_3D" if meta["is_3d_organoid"] else "DepMap_24Q4",
                "accession_id": "10.6084/m9.figshare.29472362" if meta["is_3d_organoid"] else "10.25452/figshare.plus.27993248",
                "license_track": "open",
            }
        )

    existing_model_ids = {r["model_id"] for r in basal_rows}

    # Add Sanger CMP 3D Patient-Derived Organoids RNA-seq (`rnaseq_latest.csv.gz` streamed & filtered to 3D organoids)
    sanger_3d_cache = staging_dir / "sanger_3d_organoids_expr.json"
    if not sanger_3d_cache.exists():
        import csv as _csv
        assert_not_quarantined("https://cog.sanger.ac.uk/cmp/download/rnaseq_latest.csv.gz")
        sanger_vecs: dict[str, dict[str, float]] = {}
        proc = subprocess.Popen(
            ["curl", "-fsSL", "--retry", "4", "-A", "FerroFlux-ETL/1.0", "https://cog.sanger.ac.uk/cmp/download/rnaseq_latest.csv.gz"],
            stdout=subprocess.PIPE,
        )
        assert proc.stdout is not None
        with gzip.open(proc.stdout, mode="rt", encoding="utf-8", errors="replace") as gz_fh:
            reader = _csv.reader(gz_fh)
            hdr = next(reader, [])
            m_idx = hdr.index("model_id") if "model_id" in hdr else 1
            g_idx_col = hdr.index("gene_symbol") if "gene_symbol" in hdr else 3
            tpm_idx = hdr.index("rsem_tpm") if "rsem_tpm" in hdr else (len(hdr) - 1)
            for row_vals in reader:
                if len(row_vals) <= tpm_idx:
                    continue
                sidm = row_vals[m_idx]
                if sidm not in cmp_3d_sidms:
                    continue
                sym_raw = row_vals[g_idx_col].strip()
                sym = alias_map.get(sym_raw, sym_raw)
                if sym not in gene_to_idx:
                    continue
                tpm_str = row_vals[tpm_idx]
                if not tpm_str:
                    continue
                try:
                    tpm_v = float(tpm_str)
                except ValueError:
                    continue
                if np.isfinite(tpm_v) and tpm_v >= 0.0:
                    sanger_vecs.setdefault(sidm, {})[sym] = float(np.log2(tpm_v + 1.0))
        proc.wait()
        sanger_3d_cache.write_text(json.dumps(sanger_vecs), encoding="utf-8")
    else:
        sanger_vecs = json.loads(sanger_3d_cache.read_text(encoding="utf-8"))

    for sidm in sorted(cmp_3d_sidms):
        if sidm in sidm_to_ach:
            continue
        sidm_to_ach[sidm] = sidm
        cmeta = cmp_3d_meta.get(
            sidm,
            {"cell_line_name": sidm, "cosmic_id": "", "lineage": "Bowel", "sublineage": "Patient-Derived 3D Organoid"},
        )
        if cmeta["cosmic_id"]:
            cosmic_to_ach.setdefault(cmeta["cosmic_id"], sidm)
        gdict = sanger_vecs.get(sidm, {})
        vec = np.zeros(len(vocab_genes), dtype=np.float32)
        for sym, l2v in gdict.items():
            gi = gene_to_idx.get(sym)
            if gi is not None:
                vec[gi] = float(l2v)
        # If an organoid lacks RNA-seq in CMP, skip adding empty basal unless >100 genes measured
        if len(gdict) < 100:
            continue
        existing_model_ids.add(sidm)
        basal_rows.append(
            {
                "model_id": sidm,
                "cell_line_name": cmeta["cell_line_name"],
                "sanger_model_id": sidm,
                "cosmic_id": cmeta["cosmic_id"],
                "lineage": cmeta["lineage"],
                "sublineage": cmeta["sublineage"],
                "is_3d_organoid": 1,
                "expr_vec": vec.tolist(),
                "expr_mask_count": int(len(gdict)),
                "cnv_64_vec": list(zero_64),
                "mut_lof_64_vec": list(zero_64),
                "key_mutations": "",
                "dataset_source": "Sanger_CMP_3D_Organoids",
                "accession_id": "Sanger_CMP_2026",
                "license_track": "open",
            }
        )

    cname_to_mid = {
        re.sub(r"[^A-Z0-9]", "", r["cell_line_name"].upper()): r["model_id"]
        for r in basal_rows
    }

    # Add BeatAML Waves 1-4 primary patient RNA-seq profiles
    df_beat_exp = pd.read_csv(beataml_exp_txt, sep="\t", low_memory=False)
    gene_sym_col = "display_label" if "display_label" in df_beat_exp.columns else df_beat_exp.columns[1]
    meta_cols_beat = [c for c in ["stable_id", "display_label", "description", "biotype"] if c in df_beat_exp.columns]
    sample_cols_beat = [c for c in df_beat_exp.columns if c not in meta_cols_beat]
    df_beat_sub = df_beat_exp.set_index(gene_sym_col)[sample_cols_beat]
    df_beat_sub = df_beat_sub[~df_beat_sub.index.duplicated(keep="first")].T
    # Shift to non-negative log2 scale if centered/CPM
    beat_mat, beat_cnt = align_expression_dataframe(df_beat_sub, vocab_genes, gene_to_idx, alias_map)
    beat_mat = np.clip(beat_mat - np.minimum(beat_mat.min(axis=1, keepdims=True), 0.0), 0.0, 20.0)
    beat_id_map: dict[str, str] = {}
    for row_i, raw_sid in enumerate(df_beat_sub.index):
        sid = str(raw_sid).strip()
        mid = f"BEATAML_{sid}"
        beat_id_map[sid] = mid
        if mid not in existing_model_ids:
            existing_model_ids.add(mid)
            basal_rows.append(
                {
                    "model_id": mid,
                    "cell_line_name": sid,
                    "sanger_model_id": "",
                    "cosmic_id": "",
                    "lineage": "Myeloid",
                    "sublineage": "Acute Myeloid Leukemia (Primary Ex Vivo)",
                    "is_3d_organoid": 0,
                    "expr_vec": beat_mat[row_i].tolist(),
                    "expr_mask_count": int(beat_cnt[row_i]),
                    "cnv_64_vec": list(zero_64),
                    "mut_lof_64_vec": list(zero_64),
                    "key_mutations": "",
                    "dataset_source": "BeatAML_v4",
                    "accession_id": "10.1016/j.ccell.2022.07.002",
                    "license_track": "open",
                }
            )

    ccma_id_map: dict[str, str] = {}
    if ccma_available:
        df_ccma_info = pd.read_csv(ccma_info_csv, low_memory=False)
        ccma_meta: dict[str, dict[str, Any]] = {}
        for _, r in df_ccma_info.iterrows():
            row_vals = [str(v).strip() for v in r.values if pd.notna(v)]
            sid = str(r.iloc[0]).strip()
            cname = str(r.iloc[1]).strip() if len(r) > 1 else sid
            txt = " ".join(row_vals).lower()
            is_3d = int(any(k in txt for k in ["sphere", "suspension", "organoid", "3d", "semi-adherent"]))
            lin = "CNS/Brain" if any(k in txt for k in ["glioma", "medullo", "atrt", "ependym", "dipg", "brain", "cns", "neuro"]) else "Pediatric Solid"
            for key in {sid, cname}:
                if key and key != "nan":
                    ccma_meta[key] = {"cname": cname, "is_3d": is_3d, "lineage": lin}

        df_ccma_rna_raw = pd.read_csv(ccma_rna_csv, low_memory=False)
        if "gene" in df_ccma_rna_raw.columns:
            meta_drop = [c for c in ("geneid", "chr", "length", "gene", "start", "end", "strand") if c in df_ccma_rna_raw.columns]
            sample_cols_ccma = [c for c in df_ccma_rna_raw.columns if c not in meta_drop]
            df_ccma_rna = df_ccma_rna_raw.set_index("gene")[sample_cols_ccma]
            df_ccma_rna = df_ccma_rna[~df_ccma_rna.index.duplicated(keep="first")].T
        else:
            df_ccma_rna = df_ccma_rna_raw.set_index(df_ccma_rna_raw.columns[0])
            sample_genes_in_idx = sum(1 for x in df_ccma_rna.index[:200].astype(str) if x in gene_to_idx)
            if sample_genes_in_idx > 20:
                df_ccma_rna = df_ccma_rna[~df_ccma_rna.index.duplicated(keep="first")].T
        ccma_mat, ccma_cnt = align_expression_dataframe(df_ccma_rna, vocab_genes, gene_to_idx, alias_map)
        ccma_mat = np.clip(ccma_mat, 0.0, 20.0)
        for row_i, raw_sid in enumerate(df_ccma_rna.index):
            sid = str(raw_sid).strip()
            default_is_3d = int(bool(re.search(r"_(?:S\d*|O\d*)$", sid)) or ("sphere" in sid.lower()))
            cmeta = ccma_meta.get(sid, {"cname": sid, "is_3d": default_is_3d, "lineage": "CNS/Brain"})
            mid = f"CCMA_{re.sub(r'[^A-Za-z0-9_-]', '_', sid)}"
            ccma_id_map[sid] = mid
            ccma_id_map[cmeta["cname"]] = mid
            if sid.endswith("_S"):
                for rep_k in (f"{sid}1", f"{sid}2", f"{sid}3"):
                    ccma_id_map.setdefault(rep_k, mid)
            elif re.search(r"_S\d+$", sid):
                ccma_id_map.setdefault(re.sub(r"_S\d+$", "_S", sid), mid)
            if mid not in existing_model_ids and int(ccma_cnt[row_i]) >= 100:
                existing_model_ids.add(mid)
                basal_rows.append(
                    {
                        "model_id": mid,
                        "cell_line_name": cmeta["cname"],
                        "sanger_model_id": "",
                        "cosmic_id": "",
                        "lineage": cmeta["lineage"],
                        "sublineage": "Pediatric Cancer Model",
                        "is_3d_organoid": int(cmeta["is_3d"]),
                        "expr_vec": ccma_mat[row_i].tolist(),
                        "expr_mask_count": int(ccma_cnt[row_i]),
                        "cnv_64_vec": list(zero_64),
                        "mut_lof_64_vec": list(zero_64),
                        "key_mutations": "",
                        "dataset_source": "CCMA_v3",
                        "accession_id": "10.17632/rnfs539pfw.3",
                        "license_track": "open",
                    }
                )

    df_basal = pd.DataFrame(basal_rows)
    df_basal = filter_dataframe_quarantine(df_basal)
    pq.write_table(
        pa.Table.from_pandas(df_basal, preserve_index=False),
        output_dir / "cells_basal_multiomics.parquet",
        compression="zstd",
    )
    print(f"[Phase 2A-1] Wrote cells_basal_multiomics.parquet: {len(df_basal)} models (3D={int(df_basal['is_3d_organoid'].sum())})")

    mid_to_lineage = dict(zip(df_basal["model_id"], df_basal["lineage"]))
    mid_to_3d = dict(zip(df_basal["model_id"], df_basal["is_3d_organoid"]))
    default_ref_mid = str(df_basal.iloc[0]["model_id"])

    # ------------------------------------------------------------------
    # PART B: Harmonize 2D/3D CRISPR & BioGRID ORCS Epistasis (`crispr_chronos_2d_3d.parquet`)
    # ------------------------------------------------------------------
    print("[Phase 2A-1] Building crispr_chronos_2d_3d.parquet...")
    chronos_rows: list[dict[str, Any]] = []
    panel_set = set(CORE_CHRONOS_GENES)

    # 1. DepMap 24Q4 CRISPRGeneEffect
    df_crispr = pd.read_csv(crispr_csv, index_col=0, low_memory=False)
    df_crispr.index = df_crispr.index.astype(str)
    selected_cols = []
    for col in df_crispr.columns:
        sym = parse_depmap_gene_col(str(col), alias_map)
        if sym in panel_set and sym in gene_to_idx:
            selected_cols.append((col, sym, gene_to_idx[sym]))

    for mid, row in df_crispr.iterrows():
        if not mid.startswith("ACH-"):
            continue
        lin = mid_to_lineage.get(mid, model_meta_map.get(mid, {}).get("lineage", "Unknown"))
        is_3d = int(mid_to_3d.get(mid, model_meta_map.get(mid, {}).get("is_3d_organoid", 0)))
        for col, sym, g_idx in selected_cols:
            val = float(row[col])
            if np.isfinite(val):
                chronos_rows.append(
                    {
                        "record_id": f"DEPMAP_CHRONOS_{mid}_{sym}",
                        "model_id": mid,
                        "gene_symbol": sym,
                        "gene_idx": int(g_idx),
                        "chronos_score": float(val),
                        "is_3d_organoid": is_3d,
                        "is_crispr_fin_epistasis": 0,
                        "fin_modifier_compound": "",
                        "fin_modifier_dose_um": 0.0,
                        "lineage": lin,
                        "dataset_source": "DepMap_24Q4_Chronos",
                        "accession_id": "10.25452/figshare.plus.27993248",
                        "license_track": "open",
                    }
                )

    # 2. DepMap NextGen 3D Minipool Chronos (`0048_Adh_Alt_SF_Dome` -> `ACH-000048`, paired Dome vs Plastic; ZERO ACH-000873 fallback!)
    df_mp = pd.read_csv(ng3d_chronos_csv, index_col=0, low_memory=False)
    mp_genes_in_idx = sum(1 for x in df_mp.index[:50].astype(str) if parse_depmap_gene_col(x, alias_map) in gene_to_idx)
    if mp_genes_in_idx > 10:
        df_mp = df_mp.T
    for raw_id, row in df_mp.iterrows():
        mid = str(raw_id).strip()
        ach_match = re.search(r"ACH-\d+", mid)
        if ach_match:
            resolved_mid = ach_match.group(0)
        else:
            lead_num = re.match(r"^(\d{4,6})_", mid)
            cand_ach = f"ACH-{int(lead_num.group(1)):06d}" if lead_num else ""
            resolved_mid = cand_ach if cand_ach in existing_model_ids else ""
        if not resolved_mid or resolved_mid not in existing_model_ids:
            continue
        is_mp_3d = int(any(k in mid.lower() for k in ("dome", "spheroid", "organoid", "3d")) or ("plastic" not in mid.lower()))
        lin = mid_to_lineage.get(resolved_mid, "Bowel")
        for col in df_mp.columns:
            sym = parse_depmap_gene_col(str(col), alias_map)
            if sym in gene_to_idx:
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(val) and np.isfinite(float(val)):
                    chronos_rows.append(
                        {
                            "record_id": f"NEXTGEN3D_MP_{mid}_{sym}",
                            "model_id": resolved_mid,
                            "gene_symbol": sym,
                            "gene_idx": int(gene_to_idx[sym]),
                            "chronos_score": float(val),
                            "is_3d_organoid": is_mp_3d,
                            "is_crispr_fin_epistasis": 0,
                            "fin_modifier_compound": "",
                            "fin_modifier_dose_um": 0.0,
                            "lineage": lin,
                            "dataset_source": "DepMap_NextGen_3D_Minipool",
                            "accession_id": "10.6084/m9.figshare.29472362",
                            "license_track": "open",
                        }
                    )

    # 3. Sanger 3D Patient-Derived Organoids CRISPR (`Organoids_fitness_scores_20260723.zip`; ZERO 2D ACH-000873 fallback!)
    with zipfile.ZipFile(cmp_org_zip, "r") as zf:
        target_z = next(
            (n for n in zf.namelist() if "fold_change" in n.lower() and n.endswith(".tsv")),
            zf.namelist()[0],
        )
        with zf.open(target_z) as fh:
            df_org_raw = pd.read_csv(fh, sep="\t", header=None, low_memory=False)
        # Row 0: model_name in cols 3:, Row 1: model_id (SIDM*) in cols 3:, Rows 5+: col 1 is symbol
        mnames_row = df_org_raw.iloc[0, 3:].astype(str).tolist()
        sidms_row = df_org_raw.iloc[1, 3:].astype(str).tolist()
        df_org_genes = df_org_raw.iloc[5:].copy()
        df_org_genes["_sym"] = df_org_genes.iloc[:, 1].astype(str).map(lambda s: alias_map.get(s.strip(), s.strip()))
        df_org_genes = df_org_genes[df_org_genes["_sym"].isin(panel_set) & df_org_genes["_sym"].isin(gene_to_idx)]
        for c_offset, (mname_str, sidm_str) in enumerate(zip(mnames_row, sidms_row)):
            col_i = 3 + c_offset
            sidm_clean = sidm_str.strip()
            resolved_mid = sidm_to_ach.get(sidm_clean, "")
            if not resolved_mid or resolved_mid not in existing_model_ids:
                clean_c = re.sub(r"[^A-Z0-9]", "", mname_str.upper())
                resolved_mid = cname_to_mid.get(clean_c, "")
            # Retain ONLY 3D organoids with genuine measured 3D basal RNA-seq (never fallback to ACH-000873!)
            if not resolved_mid or resolved_mid not in existing_model_ids or int(mid_to_3d.get(resolved_mid, 0)) != 1:
                continue
            lin = mid_to_lineage.get(resolved_mid, cmp_3d_meta.get(sidm_clean, {}).get("lineage", "Bowel"))
            vals_col = pd.to_numeric(df_org_genes.iloc[:, col_i], errors="coerce")
            for (_, grow), v_num in zip(df_org_genes.iterrows(), vals_col):
                if pd.notna(v_num) and np.isfinite(float(v_num)):
                    sym = str(grow["_sym"])
                    chronos_rows.append(
                        {
                            "record_id": f"SANGER_ORG3D_{sidm_clean}_{sym}",
                            "model_id": resolved_mid,
                            "gene_symbol": sym,
                            "gene_idx": int(gene_to_idx[sym]),
                            "chronos_score": float(np.clip(float(v_num), -4.0, 4.0)),
                            "is_3d_organoid": 1,
                            "is_crispr_fin_epistasis": 0,
                            "fin_modifier_compound": "",
                            "fin_modifier_dose_um": 0.0,
                            "lineage": lin,
                            "dataset_source": "Sanger_Organoids_20260723",
                            "accession_id": "Sanger_Organoids_20260723",
                            "license_track": "open",
                        }
                    )

    # 4. CCMA V3 Pediatric 2D/3D CRISPR (`gene`, `sample`, `beta` long format; ZERO ACH-000873 fallback!)
    if ccma_available and ccma_crispr_csv.exists():
        df_ccma_cr = pd.read_csv(ccma_crispr_csv, low_memory=False)
        if {"gene", "sample", "beta"}.issubset(df_ccma_cr.columns):
            df_ccma_cr["_sym"] = df_ccma_cr["gene"].astype(str).map(lambda s: alias_map.get(s.strip(), s.strip()))
            df_ccma_cr = df_ccma_cr[df_ccma_cr["_sym"].isin(panel_set) & df_ccma_cr["_sym"].isin(gene_to_idx)].copy()
            df_ccma_cr["_beta"] = pd.to_numeric(df_ccma_cr["beta"], errors="coerce")
            df_ccma_cr = df_ccma_cr[df_ccma_cr["_beta"].notna()]
            grouped_cr = df_ccma_cr.groupby(["sample", "_sym"], as_index=False)["_beta"].mean()
            for _, grow in grouped_cr.iterrows():
                sid = str(grow["sample"]).strip()
                resolved_mid = ccma_id_map.get(sid) or ccma_id_map.get(re.sub(r"_S\d+$", "_S", sid))
                if not resolved_mid or resolved_mid not in existing_model_ids:
                    continue
                sym = str(grow["_sym"])
                val = float(np.clip(float(grow["_beta"]), -4.0, 4.0))
                lin = mid_to_lineage.get(resolved_mid, "CNS/Brain")
                is_3d = int(mid_to_3d.get(resolved_mid, 1))
                chronos_rows.append(
                    {
                        "record_id": f"CCMA_CRISPR_{resolved_mid}_{sym}",
                        "model_id": resolved_mid,
                        "gene_symbol": sym,
                        "gene_idx": int(gene_to_idx[sym]),
                        "chronos_score": val,
                        "is_3d_organoid": is_3d,
                        "is_crispr_fin_epistasis": 0,
                        "fin_modifier_compound": "",
                        "fin_modifier_dose_um": 0.0,
                        "lineage": lin,
                        "dataset_source": "CCMA_v3_CRISPR",
                        "accession_id": "10.17632/rnfs539pfw.3",
                        "license_track": "open",
                    }
                )

    # 5. BioGRID ORCS v2.0.18 Chemogenetic `Cas9 + FIN` Epistatic Screens (Signed Effect Sizes ONLY!)
    orcs_tgz = download_file(
        "https://downloads.thebiogrid.org/Download/BioGRID-ORCS/Release-Archive/BIOGRID-ORCS-2.0.18/BIOGRID-ORCS-ALL-homo_sapiens-2.0.18.screens.tar.gz",
        staging_dir / "BIOGRID-ORCS-ALL-homo_sapiens-2.0.18.screens.tar.gz",
        "BIOGRID-ORCS-2.0.18",
    )
    orcs_panel_set = set(CORE_CHRONOS_GENES) | set(vocab_genes[:256])
    SIGNED_ORCS_TYPES = (
        "LOG2FC",
        "CRISPR SCORE",
        "Z-SCORE",
        "NORMZ",
        "RHO",
        "RANKS",
        "CASTLE EFFECT",
        "NES",
        "LFC",
        "BETA",
    )
    orcs_cond_patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\b(?:1S,3R-RSL-?3|RSL-?3)\b", re.I), "1S,3R-RSL3"),
        (re.compile(r"\bML-?210\b", re.I), "ML210"),
        (re.compile(r"\bML-?162\b", re.I), "ML162"),
        (re.compile(r"\bERASTIN\b", re.I), "ERASTIN"),
        (re.compile(r"\bFIN-?56\b", re.I), "FIN56"),
        (re.compile(r"\bFINO-?2\b", re.I), "FINO2"),
        (re.compile(r"\bSULFASALAZINE\b", re.I), "SULFASALAZINE"),
        (re.compile(r"\bSORAFENIB\b", re.I), "SORAFENIB"),
        (re.compile(r"\bELESCLOMOL\b", re.I), "ELESCLOMOL"),
        (re.compile(r"\bDISULFIRAM\b", re.I), "DISULFIRAM"),
        (re.compile(r"\bAURANOFIN\b", re.I), "AURANOFIN"),
        (re.compile(r"\bBREQUINAR\b", re.I), "BREQUINAR"),
        (re.compile(r"\bPRAVASTATIN\b", re.I), "PRAVASTATIN"),
        (re.compile(r"\bCERIVASTATIN\b", re.I), "CERIVASTATIN"),
        (re.compile(r"\bSIMVASTATIN\b", re.I), "SIMVASTATIN"),
        (re.compile(r"\bFLUVASTATIN\b", re.I), "FLUVASTATIN"),
        (re.compile(r"\bLOVASTATIN\b", re.I), "LOVASTATIN"),
    ]

    def _parse_orcs_dose_um(raw_dose: Any) -> float:
        s = str(raw_dose or "").strip()
        m_num = re.search(r"(\d+(?:\.\d+)?)\s*(nM|uM|µM|mM)", s, flags=re.I)
        if not m_num:
            return 1.0
        val = float(m_num.group(1))
        unit = m_num.group(2).lower()
        if unit == "nm":
            return max(val * 1e-3, 1e-4)
        if unit == "mm":
            return min(val * 1e3, 1e3)
        return max(val, 1e-4)

    with tarfile.open(orcs_tgz, "r:gz") as tf:
        idx_members = [m for m in tf.getmembers() if "INDEX" in m.name.upper()]
        if idx_members:
            with tf.extractfile(idx_members[0]) as fh:
                df_orcs_idx = pd.read_csv(fh, sep="\t", low_memory=False)
            # Match strictly on CONDITION_NAME and loss-of-function Cas9/CRISPRi enzymes (never SARS-CoV-2 titles or CRISPRa SAM!)
            matched_screens: list[tuple[str, str, str, str, float, str]] = []
            for _, r in df_orcs_idx.iterrows():
                cond_name = str(r.get("CONDITION_NAME", "")).strip()
                if not cond_name or cond_name.lower() == "nan":
                    continue
                enzyme_up = str(r.get("ENZYME", "")).upper()
                setup_up = str(r.get("EXPERIMENTAL_SETUP", "")).upper()
                if any(tok in enzyme_up or tok in setup_up for tok in ("VP64", "SAM", "CRISPRA", "ACTIVATION", "P300", "VPR")):
                    continue
                matched_fin = ""
                for pat_rx, canon_fin in orcs_cond_patterns:
                    if pat_rx.search(cond_name):
                        matched_fin = canon_fin
                        break
                if not matched_fin:
                    continue
                # Select ONLY signed effect-size column (never unsigned p-Value, FDR, or MaGeCK RRA!)
                score_col = None
                for k in range(1, 6):
                    stype = str(r.get(f"SCORE.{k}_TYPE", "")).strip().upper()
                    if (
                        any(tok in stype for tok in SIGNED_ORCS_TYPES)
                        and "P-VAL" not in stype
                        and "P_VAL" not in stype
                        and "FDR" not in stype
                        and "POS SCORE" not in stype
                        and "NEG SCORE" not in stype
                        and stype != "MAGECK SCORE"
                    ):
                        score_col = f"SCORE.{k}"
                        break
                if score_col is None:
                    continue
                try:
                    assert_not_quarantined(str(r.get("SOURCE_ID", "")), authors=str(r.get("AUTHOR", "")))
                except Exception:
                    continue
                sid = str(r.get("#SCREEN_ID", r.iloc[0])).strip()
                cline = str(r.get("CELL_LINE", "HT-1080")).strip()
                dose_um = _parse_orcs_dose_um(r.get("CONDITION_DOSAGE", "1 uM"))
                sig_ind = str(r.get("SIGNIFICANCE_INDICATOR", "")).strip()
                matched_screens.append((sid, cline, matched_fin, score_col, dose_um, sig_ind))

            member_by_sid: dict[str, tarfile.TarInfo] = {}
            want_sids = {s[0] for s in matched_screens}
            for m in tf.getmembers():
                m_base = Path(m.name).name
                for sid in want_sids:
                    if f"SCREEN_{sid}-" in m_base or f"_{sid}-" in m_base or f"-{sid}." in m_base or m_base.startswith(f"{sid}."):
                        member_by_sid[sid] = m
                        break

            for sid, cline, fin_cmpd, score_col, dose_um, sig_ind in matched_screens:
                m = member_by_sid.get(sid)
                if m is None:
                    continue
                clean_c = re.sub(r"[^A-Z0-9]", "", cline.upper())
                resolved_mid = cname_to_mid.get(clean_c, "")
                if not resolved_mid:
                    # Try stripping subclone suffixes (`HUH7SR` -> `HUH7`, `HAP1` -> `HAP1`)
                    for fallback_key in (clean_c.replace("SR", ""), "786O" if "786" in clean_c else "", "OVCAR8" if "OVCAR8" in clean_c else ""):
                        if fallback_key and fallback_key in cname_to_mid:
                            resolved_mid = cname_to_mid[fallback_key]
                            break
                if not resolved_mid:
                    resolved_mid = f"ORCS_CELL_{clean_c}"
                lin = mid_to_lineage.get(
                    resolved_mid,
                    "Kidney" if "786" in clean_c else ("Ovary/Fallopian Tube" if "OVCAR" in clean_c else ("Myeloid" if any(x in clean_c for x in ("MOLM", "HAP", "K562")) else ("Lymphoid" if "NALM" in clean_c else "Soft Tissue"))),
                )
                with tf.extractfile(m) as sfh:
                    df_sc = pd.read_csv(sfh, sep="\t", low_memory=False)
                if score_col not in df_sc.columns:
                    continue
                all_s = pd.to_numeric(df_sc[score_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
                all_s = all_s[np.isfinite(all_s)]
                if all_s.size < 5:
                    continue
                # Smooth per-screen robust Z-standardization (median / MAD; zero cliff!)
                if "ALL SIGNIFICANT" in sig_ind.upper() or all_s.size < 500:
                    center = 0.0
                    scale = float(max(np.sqrt(np.mean(all_s**2)), 0.5))
                else:
                    center = float(np.median(all_s))
                    mad = float(1.4826 * np.median(np.abs(all_s - center)))
                    scale = max(mad, float(np.std(all_s)), 0.35)

                sym_col = next((c for c in df_sc.columns if "OFFICIAL_SYMBOL" in c.upper() or "SYMBOL" in c.upper()), df_sc.columns[1])
                df_sc["_sym"] = df_sc[sym_col].astype(str).map(lambda s: alias_map.get(s.strip(), s.strip()))
                df_sc = df_sc[df_sc["_sym"].isin(orcs_panel_set) & df_sc["_sym"].isin(gene_to_idx)]
                for _, sr in df_sc.iterrows():
                    sym = str(sr["_sym"])
                    val = pd.to_numeric(sr[score_col], errors="coerce")
                    if pd.notna(val) and np.isfinite(float(val)):
                        norm_val = float(np.clip((float(val) - center) / scale, -4.0, 4.0))
                        chronos_rows.append(
                            {
                                "record_id": f"ORCS_FIN_{sid}_{sym}",
                                "model_id": resolved_mid,
                                "gene_symbol": sym,
                                "gene_idx": int(gene_to_idx[sym]),
                                "chronos_score": norm_val,
                                "is_3d_organoid": 0,
                                "is_crispr_fin_epistasis": 1,
                                "fin_modifier_compound": fin_cmpd,
                                "fin_modifier_dose_um": float(dose_um),
                                "lineage": lin,
                                "dataset_source": "BioGRID_ORCS_v2.0.18",
                                "accession_id": f"ORCS_SCREEN_{sid}",
                                "license_track": "open",
                            }
                        )

    df_chronos = pd.DataFrame(chronos_rows).drop_duplicates(subset=["record_id"], keep="first")
    df_chronos = filter_dataframe_quarantine(df_chronos)
    pq.write_table(
        pa.Table.from_pandas(df_chronos, preserve_index=False),
        output_dir / "crispr_chronos_2d_3d.parquet",
        compression="zstd",
    )
    print(
        f"[Phase 2A-1] Wrote crispr_chronos_2d_3d.parquet: {len(df_chronos)} rows "
        f"(3D={int(df_chronos['is_3d_organoid'].sum())}, CRISPR+FIN={int(df_chronos['is_crispr_fin_epistasis'].sum())})"
    )

    # ------------------------------------------------------------------
    # PART C: Harmonize Pharmacogenomic Viability (`pharmacogenomic_viability.parquet`)
    # ------------------------------------------------------------------
    print("[Phase 2A-1] Building pharmacogenomic_viability.parquet...")
    pharm_rows: list[dict[str, Any]] = []

    # 1. CTRPv2 (`PSet_CTRPv2.rds` via base Rscript)
    ctrp_rds = download_file(
        "https://zenodo.org/records/7826870/files/PSet_CTRPv2.rds?download=1",
        staging_dir / "PSet_CTRPv2.rds",
        "10.5281/zenodo.7826870",
    )
    ctrp_curves_csv = staging_dir / "ctrpv2_curves.csv"
    ctrp_drug_csv = staging_dir / "ctrpv2_drug.csv"
    ctrp_cell_csv = staging_dir / "ctrpv2_cell.csv"
    if not (ctrp_curves_csv.exists() and ctrp_drug_csv.exists() and ctrp_cell_csv.exists()):
        r_cmd = f"""
        x <- readRDS("{ctrp_rds}")
        write.csv(attr(x, "drug"), "{ctrp_drug_csv}", row.names=TRUE)
        write.csv(attr(x, "cell"), "{ctrp_cell_csv}", row.names=TRUE)
        sens <- attr(x, "sensitivity")
        write.csv(cbind(sens$info, sens$profiles), "{ctrp_curves_csv}", row.names=TRUE)
        """
        subprocess.run([RSCRIPT_BIN, "-e", r_cmd], check=True)

    df_ctrp_drug = pd.read_csv(ctrp_drug_csv, low_memory=False)
    ctrp_drug_map: dict[str, dict[str, Any]] = {}
    for _, r in df_ctrp_drug.iterrows():
        did = str(r.iloc[0]).strip()
        dname = str(r.get("drugid", r.get("cpd_name", did))).strip()
        smi = str(r.get("smiles", r.get("cpd_smiles", ""))).strip()
        cinfo = canonicalize_compound(dname, smi) or canonicalize_compound(did, smi)
        if cinfo is not None:
            ctrp_drug_map[did] = cinfo
            ctrp_drug_map[dname] = cinfo

    df_ctrp_curves = pd.read_csv(ctrp_curves_csv, low_memory=False)
    df_ctrp_curves = df_ctrp_curves[df_ctrp_curves["drugid"].astype(str).isin(ctrp_drug_map)]
    for idx_i, r in df_ctrp_curves.iterrows():
        cinfo = ctrp_drug_map[str(r["drugid"])]
        raw_cell = str(r.get("cellid", "")).strip()
        clean_c = re.sub(r"[^A-Z0-9]", "", raw_cell.upper())
        mid = cname_to_mid.get(clean_c)
        if not mid:
            continue
        aac = pd.to_numeric(r.get("aac_recomputed", r.get("aac_published")), errors="coerce")
        if pd.isna(aac) or not np.isfinite(float(aac)):
            continue
        aac_f = float(aac)
        if aac_f > 1.5:
            aac_f /= 100.0
        viab = float(np.clip(1.0 - aac_f, 0.0, 1.0))
        ic50 = pd.to_numeric(r.get("ic50_recomputed", r.get("ec50_published")), errors="coerce")
        ic50_log = float(np.log10(np.clip(float(ic50), 1e-4, 1e4))) if pd.notna(ic50) and np.isfinite(float(ic50)) and float(ic50) > 0 else 0.0
        pharm_rows.append(
            {
                "record_id": f"CTRPV2_{idx_i}_{mid}_{cinfo['compound_name']}",
                "model_id": mid,
                "compound_name": cinfo["compound_name"],
                "canonical_smiles": cinfo["canonical_smiles"],
                "inchikey": cinfo["inchikey"],
                "murcko_scaffold": cinfo["murcko_scaffold"],
                "dose_um": 1.0,
                "viability": viab,
                "ic50_log10_um": ic50_log,
                "rcd_modality": cinfo["rcd_modality"],
                "hazard_channel_idx": cinfo["hazard_channel_idx"],
                "probe_mechanism": cinfo["probe_mechanism"],
                "is_3d_organoid": int(mid_to_3d.get(mid, 0)),
                "lineage": mid_to_lineage.get(mid, "Unknown"),
                "dataset_source": "CTRPv2",
                "accession_id": "10.5281/zenodo.7826870",
                "license_track": "open",
            }
        )

    # 2. PRISM 19Q4 Secondary Dose-Response (`20237739`)
    prism19_sec = download_file(
        "https://ndownloader.figshare.com/files/20237739",
        staging_dir / "secondary-screen-dose-response-curve-parameters.csv",
        "figshare.9393293",
    )
    for chunk_i, chunk in enumerate(pd.read_csv(prism19_sec, chunksize=80000, low_memory=False)):
        chunk_names = chunk["name"].fillna("").astype(str).str.upper()
        # Fast pre-filter before row loop
        keep_mask = chunk_names.isin(CURATED_RCD_COMPOUNDS.keys()) | chunk_names.str.contains(
            "ERASTIN|RSL|ML-?210|ML-?162|FIN-?56|FINO|CIL-?56|FERROSTATIN|LIPROXSTATIN|DEFER|CICLOPIROX|DEXRAZOXANE|SULFASALAZINE|SORAFENIB|ARTESUNATE|ARTEMISININ|AURANOFIN|VASTATIN|GLITAZONE|ZILEUTON|BAICALEIN|IDEBENONE|SAPROPTERIN|TOCOPHER|ELESCLOMOL|DISULFIRAM|BUTHIONINE|BSO|BREQUINAR|TERIFLUNOMIDE|LEFLUNOMIDE|TERBINAFINE|BAY-?2402234|BAY-?87-?2243|IFSP1|PRIMA|APR-?246|PIPERLONGUMINE|WITHAFERIN|GLUTATHIONE|ACETYLCYSTEINE|LIPOIC|ALTRETAMINE|MENADIONE|BORTEZOMIB|VENETOCLAX|STAUROSPORINE|PACLITAXEL|DOXORUBICIN|VORINOSTAT|NECROSTATIN|RAPAMYCIN|DASATINIB|LAPATINIB",
            regex=True,
            na=False,
        )
        sub = chunk[keep_mask]
        for r_i, r in sub.iterrows():
            mid = str(r.get("depmap_id", "")).strip()
            if not mid.startswith("ACH-") or mid not in existing_model_ids:
                continue
            cinfo = canonicalize_compound(str(r.get("name", "")), str(r.get("smiles", "")))
            if cinfo is None:
                continue
            auc = pd.to_numeric(r.get("auc"), errors="coerce")
            if pd.isna(auc) or not np.isfinite(float(auc)):
                continue
            viab = float(np.clip(float(auc), 0.0, 1.0))
            ic50 = pd.to_numeric(r.get("ic50"), errors="coerce")
            ic50_log = float(np.log10(np.clip(float(ic50), 1e-4, 1e4))) if pd.notna(ic50) and np.isfinite(float(ic50)) and float(ic50) > 0 else 0.0
            pharm_rows.append(
                {
                    "record_id": f"PRISM19SEC_{chunk_i}_{r_i}_{mid}",
                    "model_id": mid,
                    "compound_name": cinfo["compound_name"],
                    "canonical_smiles": cinfo["canonical_smiles"],
                    "inchikey": cinfo["inchikey"],
                    "murcko_scaffold": cinfo["murcko_scaffold"],
                    "dose_um": 2.5,
                    "viability": viab,
                    "ic50_log10_um": ic50_log,
                    "rcd_modality": cinfo["rcd_modality"],
                    "hazard_channel_idx": cinfo["hazard_channel_idx"],
                    "probe_mechanism": cinfo["probe_mechanism"],
                    "is_3d_organoid": int(mid_to_3d.get(mid, 0)),
                    "lineage": mid_to_lineage.get(mid, "Unknown"),
                    "dataset_source": "PRISM_19Q4_Secondary",
                    "accession_id": "10.6084/m9.figshare.9393293",
                    "license_track": "open",
                }
            )

    # 3. PRISM 24Q2 Extended Primary (`46630981` + `46630984`)
    p24_cmpd = download_file("https://ndownloader.figshare.com/files/46630981", staging_dir / "prism24_ext_cmpd.csv", "figshare.25917643")
    p24_mat = download_file("https://ndownloader.figshare.com/files/46630984", staging_dir / "prism24_ext_matrix.csv", "figshare.25917643")
    df_p24_c = pd.read_csv(p24_cmpd, low_memory=False)
    p24_id_map: dict[str, tuple[dict[str, Any], float]] = {}
    for _, r in df_p24_c.iterrows():
        cinfo = canonicalize_compound(str(r.get("Drug.Name", "")), "")
        if cinfo is not None:
            dose_val = pd.to_numeric(r.get("dose"), errors="coerce")
            d_um = float(dose_val) if pd.notna(dose_val) and np.isfinite(float(dose_val)) and float(dose_val) > 0 else 2.5
            for raw_id in str(r.get("IDs", "")).split(";"):
                rid = raw_id.strip()
                if rid:
                    p24_id_map[rid] = (cinfo, d_um)
            p24_id_map[str(r.get("Drug.Name", "")).strip()] = (cinfo, d_um)

    df_p24_m = pd.read_csv(p24_mat, index_col=0, low_memory=False)
    # Orient compounds x ACH-* cell lines
    ach_in_cols = sum(1 for c in df_p24_m.columns[:50] if str(c).startswith("ACH-"))
    if ach_in_cols < 10:
        df_p24_m = df_p24_m.T
    for row_key, row in df_p24_m.iterrows():
        rk = str(row_key).strip()
        match_entry = p24_id_map.get(rk)
        if match_entry is None:
            for k_sub, val_sub in p24_id_map.items():
                if k_sub and k_sub in rk:
                    match_entry = val_sub
                    break
        if match_entry is None:
            continue
        cinfo, d_um = match_entry
        for col_mid, lfc_val in row.items():
            mid = str(col_mid).strip()
            if not mid.startswith("ACH-") or mid not in existing_model_ids:
                continue
            v_num = pd.to_numeric(lfc_val, errors="coerce")
            if pd.isna(v_num) or not np.isfinite(float(v_num)):
                continue
            # Convert log2 fold-change to surviving cell fraction [0, 1]
            viab = float(np.clip(2.0 ** float(v_num), 0.0, 1.0))
            pharm_rows.append(
                {
                    "record_id": f"PRISM24_{rk}_{mid}",
                    "model_id": mid,
                    "compound_name": cinfo["compound_name"],
                    "canonical_smiles": cinfo["canonical_smiles"],
                    "inchikey": cinfo["inchikey"],
                    "murcko_scaffold": cinfo["murcko_scaffold"],
                    "dose_um": d_um,
                    "viability": viab,
                    "ic50_log10_um": 0.0,
                    "rcd_modality": cinfo["rcd_modality"],
                    "hazard_channel_idx": cinfo["hazard_channel_idx"],
                    "probe_mechanism": cinfo["probe_mechanism"],
                    "is_3d_organoid": int(mid_to_3d.get(mid, 0)),
                    "lineage": mid_to_lineage.get(mid, "Unknown"),
                    "dataset_source": "PRISM_24Q2_Primary",
                    "accession_id": "10.6084/m9.figshare.25917643",
                    "license_track": "open",
                }
            )

    # 4. BeatAML Waves 1-4 Ex Vivo Patient Drug Curves
    df_beat_prob = pd.read_csv(beataml_probit_txt, sep="\t", low_memory=False)
    for r_i, r in df_beat_prob.iterrows():
        inh = str(r.get("inhibitor", "")).strip()
        cinfo = canonicalize_compound(inh, "")
        if cinfo is None:
            continue
        sid = str(r.get("dbgap_rnaseq_sample", r.get("lab_id", ""))).strip()
        mid = beat_id_map.get(sid)
        if not mid:
            continue
        auc = pd.to_numeric(r.get("auc"), errors="coerce")
        ic50 = pd.to_numeric(r.get("ic50"), errors="coerce")
        if pd.isna(auc) or not np.isfinite(float(auc)):
            continue
        # BeatAML probit AUC spans roughly [0, 300] -> normalize onto [0, 1] viability
        viab = float(np.clip(float(auc) / 300.0, 0.0, 1.0))
        ic50_log = float(np.log10(np.clip(float(ic50), 1e-4, 1e4))) if pd.notna(ic50) and np.isfinite(float(ic50)) and float(ic50) > 0 else 0.0
        pharm_rows.append(
            {
                "record_id": f"BEATAML_DRUG_{r_i}_{mid}",
                "model_id": mid,
                "compound_name": cinfo["compound_name"],
                "canonical_smiles": cinfo["canonical_smiles"],
                "inchikey": cinfo["inchikey"],
                "murcko_scaffold": cinfo["murcko_scaffold"],
                "dose_um": 1.0,
                "viability": viab,
                "ic50_log10_um": ic50_log,
                "rcd_modality": cinfo["rcd_modality"],
                "hazard_channel_idx": cinfo["hazard_channel_idx"],
                "probe_mechanism": cinfo["probe_mechanism"],
                "is_3d_organoid": 0,
                "lineage": "Myeloid",
                "dataset_source": "BeatAML_v4",
                "accession_id": "10.1016/j.ccell.2022.07.002",
                "license_track": "open",
            }
        )

    # 5. Sanger GDSC2 v8.5 (`GDSC2_fitted_dose_response_27Oct23.xlsx`: APR-246, Elesclomol, Sorafenib, GSH, NAC, alpha-LA, 2D & 3D organoids)
    gdsc2_xlsx = download_file(
        "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx",
        staging_dir / "GDSC2_fitted_dose_response_27Oct23.xlsx",
        "Sanger_GDSC2_v8.5",
    )
    df_gdsc2 = read_xlsx_sheet(gdsc2_xlsx, sheet_index=0)
    if not df_gdsc2.empty and "DRUG_NAME" in df_gdsc2.columns:
        for r_i, r in df_gdsc2.iterrows():
            cinfo = canonicalize_compound(str(r.get("DRUG_NAME", "")), "")
            if cinfo is None:
                continue
            sidm = str(r.get("SANGER_MODEL_ID", "")).strip()
            cosmic = str(r.get("COSMIC_ID", "")).split(".")[0].strip()
            mid = sidm_to_ach.get(sidm) or cosmic_to_ach.get(cosmic)
            if not mid:
                clean_c = re.sub(r"[^A-Z0-9]", "", str(r.get("CELL_LINE_NAME", "")).upper())
                mid = cname_to_mid.get(clean_c)
            if not mid:
                continue
            auc = pd.to_numeric(r.get("AUC"), errors="coerce")
            ln_ic50 = pd.to_numeric(r.get("LN_IC50"), errors="coerce")
            if pd.isna(auc) or not np.isfinite(float(auc)):
                continue
            viab = float(np.clip(float(auc), 0.0, 1.0))
            ic50_log = float(float(ln_ic50) / np.log(10.0)) if pd.notna(ln_ic50) and np.isfinite(float(ln_ic50)) else 0.0
            is_3d = int(mid_to_3d.get(mid, 0) or (sidm in cmp_3d_sidms))
            pharm_rows.append(
                {
                    "record_id": f"GDSC2_{r_i}_{mid}_{cinfo['compound_name']}",
                    "model_id": mid,
                    "compound_name": cinfo["compound_name"],
                    "canonical_smiles": cinfo["canonical_smiles"],
                    "inchikey": cinfo["inchikey"],
                    "murcko_scaffold": cinfo["murcko_scaffold"],
                    "dose_um": 1.0,
                    "viability": viab,
                    "ic50_log10_um": ic50_log,
                    "rcd_modality": cinfo["rcd_modality"],
                    "hazard_channel_idx": cinfo["hazard_channel_idx"],
                    "probe_mechanism": cinfo["probe_mechanism"],
                    "is_3d_organoid": is_3d,
                    "lineage": mid_to_lineage.get(mid, "Unknown"),
                    "dataset_source": "GDSC2_v8.5",
                    "accession_id": "Sanger_GDSC2_v8.5",
                    "license_track": "open",
                }
            )

    # 6. Optional CCMA V3 Pediatric 2D/3D Neurosphere Drug Screen
    if ccma_available and ccma_drug_csv.exists():
        df_ccma_dr = pd.read_csv(ccma_drug_csv, low_memory=False)
        if len(df_ccma_dr.columns) <= 12:
            d_col = next((c for c in df_ccma_dr.columns if "drug_name" in c.lower() or "drug" in c.lower() or "compound" in c.lower()), df_ccma_dr.columns[1])
            s_col = next((c for c in df_ccma_dr.columns if "sample" in c.lower() or "model" in c.lower() or "cell" in c.lower()), df_ccma_dr.columns[0])
            v_col = next((c for c in df_ccma_dr.columns if "z_score" in c.lower() or "z" in c.lower() or "score" in c.lower()), df_ccma_dr.columns[-1])
            uniq_d_map = {
                raw_d: canonicalize_compound(str(raw_d), "")
                for raw_d in df_ccma_dr[d_col].dropna().unique()
            }
            valid_drugs = {k: v for k, v in uniq_d_map.items() if v is not None}
            df_ccma_sub = df_ccma_dr[df_ccma_dr[d_col].isin(valid_drugs)].copy()
            for r_i, r in df_ccma_sub.iterrows():
                cinfo = valid_drugs[r[d_col]]
                raw_s = str(r[s_col]).strip()
                mid = ccma_id_map.get(raw_s) or ccma_id_map.get(re.sub(r"_S\d+$", "_S", raw_s))
                if not mid or mid not in existing_model_ids:
                    continue
                zval = pd.to_numeric(r[v_col], errors="coerce")
                if pd.isna(zval) or not np.isfinite(float(zval)):
                    continue
                viab = float(1.0 / (1.0 + np.exp(-float(zval))))
                pharm_rows.append(
                    {
                        "record_id": f"CCMA_DRUG_{r_i}_{mid}_{cinfo['compound_name']}",
                        "model_id": mid,
                        "compound_name": cinfo["compound_name"],
                        "canonical_smiles": cinfo["canonical_smiles"],
                        "inchikey": cinfo["inchikey"],
                        "murcko_scaffold": cinfo["murcko_scaffold"],
                        "dose_um": 1.0,
                        "viability": viab,
                        "ic50_log10_um": 0.0,
                        "rcd_modality": cinfo["rcd_modality"],
                        "hazard_channel_idx": cinfo["hazard_channel_idx"],
                        "probe_mechanism": cinfo["probe_mechanism"],
                        "is_3d_organoid": int(mid_to_3d.get(mid, 1)),
                        "lineage": mid_to_lineage.get(mid, "CNS/Brain"),
                        "dataset_source": "CCMA_v3_Drug",
                        "accession_id": "10.17632/rnfs539pfw.3",
                        "license_track": "open",
                    }
                )

    df_pharm = pd.DataFrame(pharm_rows).drop_duplicates(subset=["record_id"], keep="first")
    df_pharm = filter_dataframe_quarantine(df_pharm)
    pq.write_table(
        pa.Table.from_pandas(df_pharm, preserve_index=False),
        output_dir / "pharmacogenomic_viability.parquet",
        compression="zstd",
    )
    print(
        f"[Phase 2A-1] Wrote pharmacogenomic_viability.parquet: {len(df_pharm)} rows across "
        f"{df_pharm['compound_name'].nunique()} RCD compounds ({df_pharm['murcko_scaffold'].nunique()} Murcko scaffolds)"
    )

    return {
        "cells_basal_multiomics": len(df_basal),
        "crispr_chronos_2d_3d": len(df_chronos),
        "pharmacogenomic_viability": len(df_pharm),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FerroFlux-FM Phase 2A-1 DepMap, CRISPR & Pharmacogenomics ETL")
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    counts = run_depmap_pharmaco_etl(args.staging_dir, args.output_dir)
    print("[Phase 2A-1] Summary:", json.dumps(counts))


if __name__ == "__main__":
    main()
