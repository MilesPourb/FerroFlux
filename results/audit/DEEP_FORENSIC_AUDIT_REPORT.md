# `FerroFlux-FM` V1 Zero-Trust Deep Forensic Audit Report

- **Execution Host:** `login06.anvil.rcac.purdue.edu` (Login-Node Safe Mode, Zero SLURM)
- **Elapsed Wall Time:** `2367.2s` | **Peak RSS:** `10578.9 MiB`
- **Severity Tallies:** `CRITICAL = 13` | `MAJOR = 11` | `MINOR = 5` | `VERIFIED-CLEAN = 4`

## 1. Master Severity-Ranked Forensic Ledger

| Severity | ID | Pillar | Finding |
| :--- | :--- | :---: | :--- |
| **CRITICAL** | `P1-SILENT-MID-FALLBACK` | P1 | Silent fallback assigns unmatched rows in 6 shards to ACH-000873 basal transcriptome |
| **CRITICAL** | `P1-UNSHUFFLED-TRUNCATION` | P1 | Unshuffled .iloc[:cap] prefix truncation drops entire downstream data sources sorted at the end of Parquet shards |
| **CRITICAL** | `P2-FULL-DATASET-LEAK` | P2 | train_curriculum.py pretrained all V1 BASE and XL checkpoints on 100% of FerroFluxMultimodalDataset (zero OOD test holdout) |
| **CRITICAL** | `P2-SCAFFOLD-SHORTCUT-AND-SIMILARITY` | P2 | Scaffold split SMILES disjointness, Tanimoto nearest-neighbor similarity, and probe_mechanism shortcut rate |
| **CRITICAL** | `P2-SPLIT6-SFTI-LABEL-AND-DOSE0-CONFOUND` | P2 | Split 6 mixes unperturbed dose=0 normal_sfti rows (hardcoded viability=1.0) with dosed tumor pharmaco rows AND passes is_normal_organ_or_til at test time |
| **CRITICAL** | `P2-TRUE-OOS-GAP` | P2 | Measured inflation gap between V1 in-sample/shortcut OOD table vs strictly out-of-sample shortcut-stripped V1 model |
| **CRITICAL** | `P3-TREATMENT-INPUT-CIRCULARITY` | P3 | Cohort 1 (AUROC=1.00) and Cohort 2 regimen ladder (rho=0.59) fed ground-truth treatment arms into np_params/media |
| **CRITICAL** | `P3-MRNA-STEROL-TAUTOLOGY` | P3 | Cohort 2 [7-DHC] validation compares model EBP/DHCR7 formula against input EBP/SC5D-DHCR7 mRNA; correlation collapses when those 3 genes are masked |
| **CRITICAL** | `P4-DEAD-FLUX-NET` | P4 | DiffFerroGEM 94-reaction flux_proposal_net has ZERO gradient from h_ferro, viability_pred, pl_ooh, and seven_dhc_ss |
| **CRITICAL** | `P4-CRISPR-2020-GENE-BLINDNESS` | P4 | Model is 100% blind to 2,020 of 2,048 (98.6%) single-gene CRISPR knockouts |
| **CRITICAL** | `P4-PROBE-MECH-INJECTION` | P4 | compute_compound_featurization injects covalent GPX4 warhead bit=1.0 onto ethanol ('CCO') when probe_mechanism='gpx4_inhib' |
| **CRITICAL** | `P5-TEST-SUBSET-SIZE-MISMATCH` | P5 | ood_benchmark_summary.csv evaluated FerroFlux-FM on n_test=512 subset while baselines were evaluated on full test splits |
| **CRITICAL** | `P7-FIG1B-HARDCODED-COUNTS` | P7 | Fig1b plots hardcoded upstream single-cell counts (9.5e7, 8.2e6) labeled 'Harmonized Perturbation / Cell Profiles' instead of the 734,267 rows on disk |
| **MAJOR** | `P1-PROT-PROXY` | P1 | Stage-1 basal records without MS proteomics substitute raw mRNA expression as prot_anchor_target |
| **MAJOR** | `P3-ZENODO-WAVE-DECOMPOSITION` | P3 | Decomposition of Cohort 3 (Zenodo 18271042) rho=0.9286 between internal PDE u_final/Moran's I vs external geometric paracrine_factor |
| **MAJOR** | `P4-DOSE-ENCODING-MISMATCH` | P4 | dataset.py encodes log10(uM + 1.0) while perturbation_ops.py decodes 10**d - 1e-3 (+0.999 uM shift; synergy 1.0 -> 9.999 uM) |
| **MAJOR** | `P4-UNMASKED-MGM` | P4 | Stage-1 mgm_loss computes Huber loss on uncorrupted gene_expr (identity autoencoding, zero random input masking) |
| **MAJOR** | `P4-ODE-ACCURACY` | P4 | 4-substep exponential-Rosenbrock lipid peroxidation ODE vs exact Radau stiff solver |
| **MAJOR** | `P5-ZERO-PVALUE-AND-NO-FDR` | P5 | Unpaired normal-approx z-tests rounded to exact p=0.0 without paired item alignment, cluster-bootstrap SE, or BH-FDR correction |
| **MAJOR** | `P5-STAGE0-SMALL-N-POWER-CEILING` | P5 | Public Stage-0 cohorts (n=8..21) hit combinatoric p-value floors (e.g. n=8 balanced min Mann-Whitney p=0.0286; cannot reach p<0.001) |
| **MAJOR** | `P6-THERMODYNAMIC-IRREV-VIOLATION` | P6 | Unconstrained linear nullspace projection v = r @ P_null violates v_r >= 0 on ~50% of irreversible reactions |
| **MAJOR** | `P6-CDOT-IRON-SCAVENGING-BIOPHYSICS` | P6 | C' dot biophysics multiplies by fe_load (predicting 0 iron flux if fe_load=0 for iron-free silica C' dots!) and hardcodes TFRC instead of MC1R/PSMA |
| **MAJOR** | `P7-FIG3D-VACUOUS-RESIDUAL` | P7 | Fig3d checks 'if mass_balance_residual in ol' which FerroFluxFM.forward() never emits, printing vacuous 0.00e+00 fallback |
| **MAJOR** | `P7-STALE-FIGURES-VS-TABLES` | P7 | Fig1-Fig6 on disk predate Saturday's FerroFlux-FM-XL OOD/Ablation CSV updates (figures show Friday BASE numbers while CSVs show Saturday XL numbers) |
| **MINOR** | `P1-VOCAB-IDX` | P1 | Hardcoded DEFAULT_ANCHOR_28_VOCAB_IDX & FDX1_VOCAB_IDX match current gene_vocab.json, but fragile if vocab changes |
| **MINOR** | `P1-LICENSE-AUDIT` | P1 | Pharmacogenomic source x license_track distribution audited (DepMap PRISM CC-BY-4.0, GDSC non-commercial terms noted) |
| **MINOR** | `P4-DISCONNECTED-OTHER17-RCD` | P4 | other_17_rcd_head is concatenated into rcd_22_hazards for L2 penalty only and never enters total_hazard / viability_pred |
| **MINOR** | `P6-PATHWAY-COMPLETENESS-AND-DHODH-NUANCE` | P6 | MBOAT1/2 (Liang et al., Cell 2023), SC5D, and VKORC1L1 are in 64 cis-loci but omitted from 28 protein anchors; k_dhodh_clear lacks DHODH enzyme scaling |
| **MINOR** | `P7-SYNTHETIC-PLOT-FALLBACKS` | P7 | generate_nature_figures.py contains synthetic fallback DataFrames when CSVs are missing (should raise FileNotFoundError in production) |
| **VERIFIED-CLEAN** | `P2-LINEAGE-MODEL-OVERLAP` | P2 | Lineage split cell/patient model_id overlap across all 9 modality buckets |
| **VERIFIED-CLEAN** | `P3-GEO-COL-ALIGNMENT` | P3 | Audit of GSM-to-expression-column alignment across GSE310954, GSE319397, and GSE319419 |
| **VERIFIED-CLEAN** | `P4-NUMERICAL-INVARIANTS` | P4 | Batch invariance, float64 nullspace projection residual ||S v||_2, and NaN/Inf input safety |
| **VERIFIED-CLEAN** | `P6-LITERATURE-CITATION-CHECK` | P6 | Peer-reviewed reference & retraction verification across foundational C' dot, 7-DHC, FSP1-VK, MBOAT1/2 & wave papers |

## 2. Detailed Evidence by Pillar

### Pillar 1: Data Provenance, Mis-Joins, Truncation & Vocabulary Alignment
- **[MINOR] `P1-VOCAB-IDX` — Hardcoded DEFAULT_ANCHOR_28_VOCAB_IDX & FDX1_VOCAB_IDX match current gene_vocab.json, but fragile if vocab changes**
  ```json
  {
    "n_checked": 29,
    "mismatches": 0
  }
  ```
- **[MAJOR] `P1-PROT-PROXY` — Stage-1 basal records without MS proteomics substitute raw mRNA expression as prot_anchor_target**
  ```json
  {
    "basal_models_total": 2454,
    "basal_with_real_ms_proteomics": 777,
    "basal_falling_back_to_mrna_proxy": 1677,
    "frac_mrna_proxy": 0.6834
  }
  ```
- **[CRITICAL] `P1-SILENT-MID-FALLBACK` — Silent fallback assigns unmatched rows in 6 shards to ACH-000873 basal transcriptome**
  ```json
  {
    "crispr_chronos_2d_3d.parquet": {
      "total_rows": 200617,
      "matched_to_basal_expr": 189792,
      "unmatched_falling_back_to_first_cell": 10825,
      "fallback_cell_id": "ACH-000873",
      "match_rate": 0.946,
      "recoverable_via_cell_line_name": 0
    },
    "drugcomb_synergy_blocks.parquet": {
      "total_rows": 224028,
      "matched_to_basal_expr": 160443,
      "unmatched_falling_back_to_first_cell": 63585,
      "fallback_cell_id": "ACH-000873",
      "match_rate": 0.7162,
      "recoverable_via_cell_line_name": 0
    },
    "perturb_transcriptomes.parquet": {
      "total_rows": 13517,
      "matched_to_basal_expr": 12938,
      "unmatched_falling_back_to_first_cell": 579,
      "fallback_cell_id": "ACH-000873",
      "match_rate": 0.9572,
      "recoverable_via_cell_line_name": 0
    },
    "perturbed_oxylipidomics_sterolomics.parquet": {
      "total_rows": 101341,
      "matched_to_basal_expr": 49289,
      "unmatched_falling_back_to_first_cell": 52052,
      "fallback_cell_id": "ACH-000873",
      "match_rate": 0.4864,
      "recoverable_via_cell_line_name": 0
    },
    "pharmacogenomic_viability.parquet": {
      "total_rows": 91800,
      "matched_to_basal_expr": 88361,
      "unmatched_falling_back_to_first_cell": 3439,
      "fallback_cell_id": "ACH-000873",
      "match_rate": 0.9625,
      "recoverable_via_cell_line_name": 0
    },
    "pmc_oa_rescue_and_bionano.parquet": {
      "total_rows": 955,
      "matched_to_basal_expr": 341,
      "unmatched_falling_back_to_first_cell": 614,
  ... (full evidence in pillar JSON)
  ```
- **[CRITICAL] `P1-UNSHUFFLED-TRUNCATION` — Unshuffled .iloc[:cap] prefix truncation drops entire downstream data sources sorted at the end of Parquet shards**
  ```json
  {
    "crispr_chronos_2d_3d.parquet": {
      "total_rows": 200617,
      "cap": 45000,
      "fraction_kept": 0.2243,
      "full_source_counts": {
        "DepMap_24Q4_Chronos": 171734,
        "Sanger_Organoids_20260723": 22032,
        "BioGRID_ORCS_v2.0.18": 4997,
        "DepMap_NextGen_3D_Minipool": 1854
      },
      "kept_prefix_source_counts": {
        "DepMap_24Q4_Chronos": 45000
      },
      "completely_dropped_sources": [
        "BioGRID_ORCS_v2.0.18",
        "DepMap_NextGen_3D_Minipool",
        "Sanger_Organoids_20260723"
      ]
    },
    "drugcomb_synergy_blocks.parquet": {
      "total_rows": 224028,
      "cap": 15000,
      "fraction_kept": 0.067,
      "full_source_counts": {
        "DrugComb_v1.5": 224028
      },
      "kept_prefix_source_counts": {
        "DrugComb_v1.5": 15000
      },
      "completely_dropped_sources": []
    },
    "perturbed_oxylipidomics_sterolomics.parquet": {
      "total_rows": 101341,
      "cap": 5000,
      "fraction_kept": 0.0493,
      "full_source_counts": {
        "MetaboLights_MTBLS8280": 44532,
        "MetaboLights_MTBLS15364": 21387,
        "MetaboLights_MTBLS1399": 9972,
        "MetabolomicsWorkbench_ST004203": 7510,
        "MetabolomicsWorkbench_ST001334": 6557,
        "MetaboLights_MTBLS774": 5033,
        "MetabolomicsWorkbench_ST003260": 2242,
        "MetabolomicsWorkbench_ST002115": 1092,
  ... (full evidence in pillar JSON)
  ```
- **[MINOR] `P1-LICENSE-AUDIT` — Pharmacogenomic source x license_track distribution audited (DepMap PRISM CC-BY-4.0, GDSC non-commercial terms noted)**
  ```json
  {
    "open": {
      "BeatAML_v4": 3597,
      "CTRPv2": 32028,
      "GDSC2_v8.5": 13149,
      "PRISM_19Q4_Secondary": 16120,
      "PRISM_24Q2_Primary": 26906
    }
  }
  ```

### Pillar 2: Train/Test Contamination, Shortcut Leaks & True Out-of-Sample Benchmark
- **[CRITICAL] `P2-FULL-DATASET-LEAK` — train_curriculum.py pretrained all V1 BASE and XL checkpoints on 100% of FerroFluxMultimodalDataset (zero OOD test holdout)**
  ```json
  {
    "confirmed_in_source": true,
    "dataset_records_loaded_open": 122139,
    "impact": "100% of test indices in ood_benchmark_summary.csv were included in the 22-epoch pretraining loop of BASE and XL checkpoints."
  }
  ```
- **[CRITICAL] `P2-SCAFFOLD-SHORTCUT-AND-SIMILARITY` — Scaffold split SMILES disjointness, Tanimoto nearest-neighbor similarity, and probe_mechanism shortcut rate**
  ```json
  {
    "n_train_unique_smiles": 379,
    "n_test_unique_smiles": 107,
    "exact_smiles_overlap_count": 0,
    "median_max_tanimoto_to_train": 0.268,
    "max_tanimoto_to_train": 1.0,
    "test_pharmaco_rows": 17092,
    "test_pharmaco_rows_leaking_probe_mechanism": 17092,
    "frac_leaking_probe_mechanism": 1.0
  }
  ```
- **[VERIFIED-CLEAN] `P2-LINEAGE-MODEL-OVERLAP` — Lineage split cell/patient model_id overlap across all 9 modality buckets**
  ```json
  {
    "n_train_models": 2502,
    "n_test_models": 662,
    "overlapping_model_ids_count": 0,
    "sample_overlapping_ids": []
  }
  ```
- **[CRITICAL] `P2-SPLIT6-SFTI-LABEL-AND-DOSE0-CONFOUND` — Split 6 mixes unperturbed dose=0 normal_sfti rows (hardcoded viability=1.0) with dosed tumor pharmaco rows AND passes is_normal_organ_or_til at test time**
  ```json
  {
    "n_test_normal_sfti_rows_dose0_viab1": 206,
    "n_test_tumor_pharmaco_rows_dosed": 1589,
    "trivial_dose0_baseline_exploit": "Any model predicting 1.0 when dose_a == 0 and <1.0 when dose_a > 0 achieves high AUROC without learning tissue selectivity."
  }
  ```
- **[CRITICAL] `P2-TRUE-OOS-GAP` — Measured inflation gap between V1 in-sample/shortcut OOD table vs strictly out-of-sample shortcut-stripped V1 model**
  ```json
  [
    {
      "ood_split": "1_leave_out_scaffold",
      "eval_target": "viability_dosed_only",
      "n_eval": 512,
      "reported_v1_leaked_xl_rho": 0.5818,
      "true_v1_oos_no_shortcut_rho": 0.2158,
      "inflation_delta_rho": 0.366,
      "reported_v1_leaked_xl_auroc": 0.8094,
      "true_v1_oos_no_shortcut_auroc": 0.6117
    },
    {
      "ood_split": "2_leave_out_lineage_ccrcc_melanoma",
      "eval_target": "viability_dosed_only",
      "n_eval": 504,
      "reported_v1_leaked_xl_rho": 0.4959,
      "true_v1_oos_no_shortcut_rho": 0.2815,
      "inflation_delta_rho": 0.2144,
      "reported_v1_leaked_xl_auroc": 0.7597,
      "true_v1_oos_no_shortcut_auroc": 0.6392
    },
    {
      "ood_split": "3_2d_to_3d_organoid_pdx",
      "eval_target": "chronos_fitness",
      "n_eval": 512,
      "reported_v1_leaked_xl_rho": 0.0211,
      "true_v1_oos_no_shortcut_rho": -0.0566,
      "inflation_delta_rho": 0.0777,
      "reported_v1_leaked_xl_auroc": 0.5034,
      "true_v1_oos_no_shortcut_auroc": 0.474
    },
    {
      "ood_split": "4_dual_crispr_fin_epistasis",
      "eval_target": "chronos_fitness",
      "n_eval": 512,
      "reported_v1_leaked_xl_rho": 0.2611,
      "true_v1_oos_no_shortcut_rho": -0.0542,
      "inflation_delta_rho": 0.3153,
      "reported_v1_leaked_xl_auroc": 0.5795,
      "true_v1_oos_no_shortcut_auroc": 0.495
    },
    {
      "ood_split": "5_cuproptosis_elesclomol_specificity",
      "eval_target": "viability_dosed_only",
      "n_eval": 512,
  ... (full evidence in pillar JSON)
  ```

### Pillar 3: Stage-0 Blind Validation Circularity, mRNA Tautology & Spatial Decomposition
- **[VERIFIED-CLEAN] `P3-GEO-COL-ALIGNMENT` — Audit of GSM-to-expression-column alignment across GSE310954, GSE319397, and GSE319419**
  ```json
  {
    "total_samples": 32,
    "positional_fallbacks_count": 0,
    "sample_alignments_gse319397": [
      {
        "gsm_id": "GSM9517314",
        "title": "Control (Saline treatment) mouse tumor 1",
        "matched_col": "S1",
        "method": "exact_token"
      },
      {
        "gsm_id": "GSM9517315",
        "title": "Control (Saline treatment) mouse tumor 2",
        "matched_col": "S2",
        "method": "exact_token"
      },
      {
        "gsm_id": "GSM9517316",
        "title": "Control (Saline treatment) mouse tumor 3",
        "matched_col": "S3",
        "method": "exact_token"
      },
      {
        "gsm_id": "GSM9517317",
        "title": "Dual Tx treatment mouse tumor 1",
        "matched_col": "DT1",
        "method": "exact_token"
      },
      {
        "gsm_id": "GSM9517318",
        "title": "Dual Tx treatment mouse tumor 2",
        "matched_col": "DT2",
        "method": "exact_token"
      }
    ],
    "fallback_rows": []
  }
  ```
- **[CRITICAL] `P3-TREATMENT-INPUT-CIRCULARITY` — Cohort 1 (AUROC=1.00) and Cohort 2 regimen ladder (rho=0.59) fed ground-truth treatment arms into np_params/media**
  ```json
  [
    {
      "ensemble": "BASE_academic",
      "gse310954_uncond_auroc": 0.0,
      "gse310954_uncond_mwu_p": 0.0043,
      "gse310954_uncond_sting_rho": -0.4727,
      "gse310954_fwd_ct_only_gm_gt_f10_match": false,
      "gse319397_sterol_rho_with_direct_mrna": 0.85,
      "gse319397_sterol_rho_when_ebp_sc5d_dhcr7_masked": -0.4667,
      "prostate_tiers_uncond_rho": 0.0101
    },
    {
      "ensemble": "XL_academic",
      "gse310954_uncond_auroc": 0.8333,
      "gse310954_uncond_mwu_p": 0.0823,
      "gse310954_uncond_sting_rho": -0.1,
      "gse310954_fwd_ct_only_gm_gt_f10_match": false,
      "gse319397_sterol_rho_with_direct_mrna": 0.7333,
      "gse319397_sterol_rho_when_ebp_sc5d_dhcr7_masked": 0.15,
      "prostate_tiers_uncond_rho": 0.2223
    },
    {
      "ensemble": "XL_open",
      "gse310954_uncond_auroc": 0.3,
      "gse310954_uncond_mwu_p": 0.329,
      "gse310954_uncond_sting_rho": -0.3727,
      "gse310954_fwd_ct_only_gm_gt_f10_match": false,
      "gse319397_sterol_rho_with_direct_mrna": 0.4167,
      "gse319397_sterol_rho_when_ebp_sc5d_dhcr7_masked": 0.0167,
      "prostate_tiers_uncond_rho": 0.0546
    }
  ]
  ```
- **[CRITICAL] `P3-MRNA-STEROL-TAUTOLOGY` — Cohort 2 [7-DHC] validation compares model EBP/DHCR7 formula against input EBP/SC5D-DHCR7 mRNA; correlation collapses when those 3 genes are masked**
  ```json
  {
    "BASE_academic": {
      "unmasked_rho": 0.85,
      "masked_rho": -0.4667
    },
    "XL_academic": {
      "unmasked_rho": 0.7333,
      "masked_rho": 0.15
    },
    "XL_open": {
      "unmasked_rho": 0.4167,
      "masked_rho": 0.0167
    }
  }
  ```
- **[MAJOR] `P3-ZENODO-WAVE-DECOMPOSITION` — Decomposition of Cohort 3 (Zenodo 18271042) rho=0.9286 between internal PDE u_final/Moran's I vs external geometric paracrine_factor**
  ```json
  {
    "n_fovs": 8,
    "rho_u_final_mean_alone": 0.7619,
    "rho_morans_i_alone": 0.3571,
    "rho_propagative_prob_alone": 0.0,
    "rho_external_paracrine_factor_alone": 0.9048,
    "rho_combined_product_reported": 0.9286
  }
  ```

### Pillar 4: Code, Autograd Jacobian, Dose/KO Bugs & Stiff ODE Numerical Forensics
- **[CRITICAL] `P4-DEAD-FLUX-NET` — DiffFerroGEM 94-reaction flux_proposal_net has ZERO gradient from h_ferro, viability_pred, pl_ooh, and seven_dhc_ss**
  ```json
  {
    "dead_flux_parameters": [
      "diff_ferro_gem.flux_proposal_net.0.weight",
      "diff_ferro_gem.flux_proposal_net.0.bias",
      "diff_ferro_gem.flux_proposal_net.2.weight",
      "diff_ferro_gem.flux_proposal_net.2.bias"
    ],
    "explanation": "fluxes = raw_flux @ P_null is returned in output dict and L2-shrunk toward zero by 1e-3*fluxes^2 in losses.py, never entering Compartment 1 or 2 ODEs."
  }
  ```
- **[MINOR] `P4-DISCONNECTED-OTHER17-RCD` — other_17_rcd_head is concatenated into rcd_22_hazards for L2 penalty only and never enters total_hazard / viability_pred**
  ```json
  {
    "parameters": [
      "other_17_rcd_head.weight",
      "other_17_rcd_head.bias"
    ]
  }
  ```
- **[CRITICAL] `P4-CRISPR-2020-GENE-BLINDNESS` — Model is 100% blind to 2,020 of 2,048 (98.6%) single-gene CRISPR knockouts**
  ```json
  {
    "n_genes_total": 2048,
    "n_genes_visible_to_ko_operator": 29,
    "n_genes_completely_ignored": 2019,
    "max_abs_chronos_diff_across_50_non_anchor_kos": 0.0,
    "max_abs_delta_expr_diff_across_50_non_anchor_kos": 1.7881393432617188e-07
  }
  ```
- **[MAJOR] `P4-DOSE-ENCODING-MISMATCH` — dataset.py encodes log10(uM + 1.0) while perturbation_ops.py decodes 10**d - 1e-3 (+0.999 uM shift; synergy 1.0 -> 9.999 uM)**
  ```json
  {
    "dose_table": [
      {
        "true_dose_uM": 0.0,
        "dataset_encoded_dose_a": 0.000434,
        "pertop_decoded_dose_uM": 1.0,
        "phantom_offset_uM": 1.0
      },
      {
        "true_dose_uM": 0.01,
        "dataset_encoded_dose_a": 0.004321,
        "pertop_decoded_dose_uM": 1.009,
        "phantom_offset_uM": 0.999
      },
      {
        "true_dose_uM": 0.1,
        "dataset_encoded_dose_a": 0.041393,
        "pertop_decoded_dose_uM": 1.099,
        "phantom_offset_uM": 0.999
      },
      {
        "true_dose_uM": 1.0,
        "dataset_encoded_dose_a": 0.30103,
        "pertop_decoded_dose_uM": 1.999,
        "phantom_offset_uM": 0.999
      },
      {
        "true_dose_uM": 10.0,
        "dataset_encoded_dose_a": 1.041393,
        "pertop_decoded_dose_uM": 10.999,
        "phantom_offset_uM": 0.999
      }
    ],
    "synergy_dose_1_decoded_as_uM": 9.999
  }
  ```
- **[MAJOR] `P4-UNMASKED-MGM` — Stage-1 mgm_loss computes Huber loss on uncorrupted gene_expr (identity autoencoding, zero random input masking)**
  ```json
  {
    "loss_fn_target": "batch['gene_expr'] (same unmasked tensor passed into perceiver_trunk)",
    "random_mask_corruption_rate_in_v1": 0.0
  }
  ```
- **[MAJOR] `P4-ODE-ACCURACY` — 4-substep exponential-Rosenbrock lipid peroxidation ODE vs exact Radau stiff solver**
  ```json
  {
    "n_regimes_tested": 200,
    "median_relative_error": 0.00071,
    "p95_relative_error": 0.058485,
    "max_relative_error": 0.564022,
    "always_strictly_positive": true
  }
  ```
- **[VERIFIED-CLEAN] `P4-NUMERICAL-INVARIANTS` — Batch invariance, float64 nullspace projection residual ||S v||_2, and NaN/Inf input safety**
  ```json
  {
    "batch_invariance_abs_err": 0.0,
    "max_sv_l2_residual_fp32_cast": 3.575163623766494e-08,
    "nan_inf_input_finite_outputs": true
  }
  ```
- **[CRITICAL] `P4-PROBE-MECH-INJECTION` — compute_compound_featurization injects covalent GPX4 warhead bit=1.0 onto ethanol ('CCO') when probe_mechanism='gpx4_inhib'**
  ```json
  {
    "ethanol_desc6_smiles_only": 0.0,
    "ethanol_desc6_with_probe_mechanism": 1.0
  }
  ```

### Pillar 5: Statistical Hypothesis Testing, Calibration & Power Forensics
- **[CRITICAL] `P5-TEST-SUBSET-SIZE-MISMATCH` — ood_benchmark_summary.csv evaluated FerroFlux-FM on n_test=512 subset while baselines were evaluated on full test splits**
  ```json
  [
    {
      "track": "open",
      "split": "1_leave_out_scaffold",
      "ferroflux_n_test": 512,
      "baseline_n_test": 17092
    },
    {
      "track": "open",
      "split": "2_leave_out_lineage_ccrcc_melanoma",
      "ferroflux_n_test": 504,
      "baseline_n_test": 4355
    },
    {
      "track": "open",
      "split": "3_2d_to_3d_organoid_pdx",
      "ferroflux_n_test": 512,
      "baseline_n_test": 23165
    },
    {
      "track": "open",
      "split": "4_dual_crispr_fin_epistasis",
      "ferroflux_n_test": 512,
      "baseline_n_test": 4997
    },
    {
      "track": "open",
      "split": "5_cuproptosis_elesclomol_specificity",
      "ferroflux_n_test": 512,
      "baseline_n_test": 1493
    },
    {
      "track": "open",
      "split": "6_sfti_tumor_vs_normal_til",
      "ferroflux_n_test": 512,
      "baseline_n_test": 1795
    },
    {
      "track": "academic",
      "split": "1_leave_out_scaffold",
      "ferroflux_n_test": 512,
      "baseline_n_test": 6508
    },
    {
      "track": "academic",
  ... (full evidence in pillar JSON)
  ```
- **[MAJOR] `P5-ZERO-PVALUE-AND-NO-FDR` — Unpaired normal-approx z-tests rounded to exact p=0.0 without paired item alignment, cluster-bootstrap SE, or BH-FDR correction**
  ```json
  {
    "rows_with_exact_p_zero": 5,
    "total_ood_rows": 12,
    "has_fdr_column": false,
    "st0_has_fdr_column": false
  }
  ```
- **[MAJOR] `P5-STAGE0-SMALL-N-POWER-CEILING` — Public Stage-0 cohorts (n=8..21) hit combinatoric p-value floors (e.g. n=8 balanced min Mann-Whitney p=0.0286; cannot reach p<0.001)**
  ```json
  {
    "stage0_power": [
      {
        "cohort": "Zenodo 18271042 Live-Cell Spatial FOVs",
        "n_samples": 8,
        "min_detectable_rho_80pct_power_alpha05": 0.8491,
        "min_possible_exact_mann_whitney_p": 0.028571,
        "can_ever_reach_p_lt_0_001_binary": false
      },
      {
        "cohort": "GSE319397 Myc-CaP Prostate C' Dot RNA-seq",
        "n_samples": 9,
        "min_detectable_rho_80pct_power_alpha05": 0.8157,
        "min_possible_exact_mann_whitney_p": 0.015873,
        "can_ever_reach_p_lt_0_001_binary": false
      },
      {
        "cohort": "GSE310954 Melanoma C' Dot RNA-seq",
        "n_samples": 11,
        "min_detectable_rho_80pct_power_alpha05": 0.7576,
        "min_possible_exact_mann_whitney_p": 0.004329,
        "can_ever_reach_p_lt_0_001_binary": false
      },
      {
        "cohort": "Pooled GSE319397 + GSE319419 Prostate RNA-seq",
        "n_samples": 21,
        "min_detectable_rho_80pct_power_alpha05": 0.5786,
        "min_possible_exact_mann_whitney_p": 6e-06,
        "can_ever_reach_p_lt_0_001_binary": true
      }
    ],
    "recommended_stage1_sample_sizes": {
      "for_rho_0.50_at_p001_power80": 60,
      "for_rho_0.60_at_p001_power80": 39,
      "for_rho_0.40_at_p005_power80": 78
    }
  }
  ```

### Pillar 6: Stoichiometry, Thermodynamic Flux Cone, C'-Dot Biophysics & Literature Audit
- **[MAJOR] `P6-THERMODYNAMIC-IRREV-VIOLATION` — Unconstrained linear nullspace projection v = r @ P_null violates v_r >= 0 on ~50% of irreversible reactions**
  ```json
  {
    "num_metabolites": 74,
    "num_reactions": 94,
    "nullspace_dim": 25,
    "sv_frobenius_residual": 5.80334466501274e-15,
    "orthonormality_error": 2.6972892251573516e-15,
    "blocked_reactions_count": 0,
    "irreversible_internal_reactions": 71,
    "fraction_negative_flux_under_gaussian_proposals": 0.4972,
    "implicit_boundary_species_reactions": [
      {
        "id": "v_LRP8_SE",
        "equation": "1 NADPH_c => 1 SELENIDE_c + 1 NADP_c",
        "note": "Extracellular selenite/SELENOP implicit"
      },
      {
        "id": "v_TFRC_IMPORT",
        "equation": "=> 1 FE3_LYSO_l",
        "note": "Extracellular holo-Tf Fe3+ implicit"
      },
      {
        "id": "v_HMOX1",
        "equation": "1 HEME_c + 1 NADPH_c => 1 FE2_LIP_c + 1 NADP_c",
        "note": "Extracellular holo-Tf Fe3+ implicit"
      }
    ]
  }
  ```
- **[MAJOR] `P6-CDOT-IRON-SCAVENGING-BIOPHYSICS` — C' dot biophysics multiplies by fe_load (predicting 0 iron flux if fe_load=0 for iron-free silica C' dots!) and hardcodes TFRC instead of MC1R/PSMA**
  ```json
  {
    "zeros_out_iron_free_silica_when_fe_load_0": true,
    "hardcodes_tfrc_instead_of_mc1r_folh1": true,
    "biological_truth": "Cornell prime dots (C' dots; Kim et al., Nat. Nanotechnol. 2016) are synthesized WITHOUT iron cores; their surface PEG/silanol groups adsorb/chelate ambient ferric iron (Fe3+) from media/TME and traffic it into endolysosomes under amino-acid starvation (alpha-MSH/MC1R in melanoma; PSMAi/FOLH1 in prostate)."
  }
  ```
- **[MINOR] `P6-PATHWAY-COMPLETENESS-AND-DHODH-NUANCE` — MBOAT1/2 (Liang et al., Cell 2023), SC5D, and VKORC1L1 are in 64 cis-loci but omitted from 28 protein anchors; k_dhodh_clear lacks DHODH enzyme scaling**
  ```json
  {
    "loci_genes_missing_from_anchor_28": [
      "MBOAT1",
      "MBOAT2",
      "SC5D",
      "VKORC1L1",
      "FADS2"
    ],
    "dhodh_note": "diff_ferro_gem.py line 388 computes k_dhodh_clear = k_dhodh_c * (1 - 0.90*occ_dhodh) without multiplying by DHODH expression, and Mishima et al. (Nature 2023) showed high-dose Brequinar ferroptosis sensitization partly acts via FSP1 inhibition."
  }
  ```
- **[VERIFIED-CLEAN] `P6-LITERATURE-CITATION-CHECK` — Peer-reviewed reference & retraction verification across foundational C' dot, 7-DHC, FSP1-VK, MBOAT1/2 & wave papers**
  ```json
  {
    "executed": true,
    "exit_code": 0,
    "results_count": 1
  }
  ```

### Pillar 7: Figure-to-Table Truthfulness, Hardcoded Constants & Claim Forensics
- **[CRITICAL] `P7-FIG1B-HARDCODED-COUNTS` — Fig1b plots hardcoded upstream single-cell counts (9.5e7, 8.2e6) labeled 'Harmonized Perturbation / Cell Profiles' instead of the 734,267 rows on disk**
  ```json
  {
    "hardcoded_open_sum": 105878000,
    "actual_parquet_rows_on_disk": 734267,
    "per_shard_rows": {
      "cells_basal_multiomics.parquet": 2454,
      "chem_ferrdb_qhts_chembl_features.parquet": 53984,
      "cis_reg_64loci_epigenomics.parquet": 936,
      "clinical_icb_cohorts.parquet": 502,
      "crispr_chronos_2d_3d.parquet": 200617,
      "cysdb_abpp_ligandability.parquet": 41474,
      "drugcomb_synergy_blocks.parquet": 224028,
      "normal_organ_til_baselines.parquet": 206,
      "perturb_transcriptomes.parquet": 13517,
      "perturbed_oxylipidomics_sterolomics.parquet": 101341,
      "pharmacogenomic_viability.parquet": 91800,
      "physical_anchors_metabolomics_lipidomics.parquet": 928,
      "physical_anchors_proteomics.parquet": 1525,
      "pmc_oa_rescue_and_bionano.parquet": 955
    }
  }
  ```
- **[MAJOR] `P7-FIG3D-VACUOUS-RESIDUAL` — Fig3d checks 'if mass_balance_residual in ol' which FerroFluxFM.forward() never emits, printing vacuous 0.00e+00 fallback**
  ```json
  {
    "emitted_by_model": false,
    "printed_fallback": 0.0
  }
  ```
- **[MAJOR] `P7-STALE-FIGURES-VS-TABLES` — Fig1-Fig6 on disk predate Saturday's FerroFlux-FM-XL OOD/Ablation CSV updates (figures show Friday BASE numbers while CSVs show Saturday XL numbers)**
  ```json
  {
    "ood_csv_mtime": 1794713350.190913,
    "fig2_png_mtime": 1794621161.124775
  }
  ```
- **[MINOR] `P7-SYNTHETIC-PLOT-FALLBACKS` — generate_nature_figures.py contains synthetic fallback DataFrames when CSVs are missing (should raise FileNotFoundError in production)**
  ```json
  {
    "has_synthetic_fallback_code": true
  }
  ```
