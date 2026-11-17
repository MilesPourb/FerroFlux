# FerroFlux-FM

**Ferroptosis Metabolic-Flux Foundation Model (`FerroFlux-FM`)** — a grey-box mechanistic and multi-omic foundation model for ferroptosis sensitivity, chemogenetic epistasis, causal radical-trapping/iron-chelation rescue, 2D-to-3D organoid transfer, and spatial lipid-peroxidation trigger waves.

## Architecture Overview

- **Multi-Omic & Cis-Regulatory Perceiver Trunk (`ferroflux/layers/perceiver_trunk.py`):** Encodes `2,048`-gene basal transcriptomes (`log2(TPM+1)`), `128` copy-number / loss-of-function somatic genomic features (`cnv_64_vec`, `mut_lof_64_vec`), `64` mass-spectrometry protein anchors (`GPX4` post-transcriptional `Sec46` translation decoupling), `32` LC-MS phospholipid/redox species (`PUFA-PL`, `MUFA-PL`, `GSH/GSSG`), and `64` cis-regulatory promoter/enhancer methylation loci.
- **Multi-Scale Chemical & Perturbation Operator (`ferroflux/layers/perturbation_ops.py`):** Pretrained in **Stage 0.5** on `53,984` quantitative ferroptosis SAR records (`FerrDb V3`, `PubChem qHTS`, `ChEMBL 37`) and `41,474` covalent cysteine/selenocysteine ABPP profiles (`CysDB v1.0`) using pure-SMILES `ECFP4` (`2,048`) + `ECFP6` (`1,024`) + `MACCS` (`167`) + `24` RDKit/warhead descriptors, modulated by cell-state `FiLM` bioactivation (`POR`/`CYB5R1`/`NQO1`).
- **Differentiable Stoichiometric & Kinetic Core (`ferroflux/layers/diff_ferro_gem.py`):** Projects steady-state metabolic fluxes onto the nullspace and irreversible cone ($S v = 0, v_{\text{irrev}} \ge 0$) of a curated `94`-reaction ferroptosis network (`configs/ferro_gem_94rxn.json`) via Douglas-Rachford splitting and integrates `[PL-OOH]` membrane peroxidation kinetics (`ACSL4`/`LPCAT3`, `System xc⁻`–`GSH`–`GPX4`, `FSP1`/`CoQ10H2`, `GCH1`/`BH4`, `DHODH`, `[7-DHC]` sterol shielding, and labile `Fe²⁺` Fenton initiation).
- **APHYNITY Grey-Box Modulation & 4-Corner Epistasis (`ferroflux/models/ferroflux_fm.py`):** Multiplicative norm-penalized grey-box residual (`APHYNITY`) that strictly preserves `100%` monotonic `Liproxstatin-1` / `Deferoxamine` rescue and `0.0` cuproptosis invariance error, paired with a counterfactual `(KO + Drug) - (WT + Drug)` epistasis head and differentiable ranking + correlation objective (`ferroflux/models/losses.py`).

## Repository Structure

```text
FerroFlux/
├── configs/                # Training curriculum, 94-rxn stoichiometry, cis-reg loci & quarantine manifests
├── data/processed/         # Gene vocabulary (Parquet shards excluded via .gitignore; SHA-256 tracked in data_manifests/)
├── data_manifests/         # Provenance & integrity manifests for all 14 training shards
├── ferroflux/
│   ├── audit/              # Zero-trust 8-pillar forensic audit & post-remediation verifiers
│   ├── baselines/          # Matched Ridge, ElasticNet, LightGBM, XGBoost, Chemprop-MLP & LinearPerturb baselines
│   ├── data/               # Multi-modal ETL pipelines, pure-SMILES featurization & strict OOD splitters
│   ├── layers/             # Perceiver trunk, perturbation operators & DiffFerroGEM stoichiometric/ODE solver
│   ├── models/             # FerroFlux-FM model & differentiable ranking/stoichiometric losses
│   ├── security/           # Tier-3 accession/PMCID quarantine & AlphaGenome 2 inference-only guards
│   ├── training/           # Multi-stage curriculum trainer & strict out-of-sample 6-split OOD / ablation evaluator
│   └── validation/         # Zero-circularity Stage-0 blind evaluation & Nature-style figure generator
├── results/                # Benchmark summary tables, publication figures (Fig1-Fig6) & forensic audit reports
├── scripts/                # Automated SLURM multi-node / highmem & verification pipelines
└── tests/                  # Pytest unit, biophysical invariant & zero-leakage forensic test suite
```

## Quick Start & Verification

```bash
# Run full unit, stoichiometric & zero-leakage forensic test suite
python -m pytest tests/ -q

# Run 7-check post-remediation zero-trust audit verifier
python -m ferroflux.audit.verify_v2_remediation

# Run strict out-of-sample 6-split OOD benchmark & ablations
python -m ferroflux.training.evaluate_ood_and_ablations --config configs/train_config.yaml --track open
```
