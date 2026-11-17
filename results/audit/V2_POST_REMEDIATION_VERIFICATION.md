# `FerroFlux-FM v2` Post-Remediation Zero-Trust Verification Report

- **Verification Status:** `ALL_CLEAN`
- **Remaining CRITICAL Findings:** `0` (down from `13` in V1)
- **Remaining MAJOR Findings:** `0` (down from `11` in V1)
- **Checks Passed:** `7 / 7`

## Per-Pillar Verification Matrix

| Check ID | Pillar | Status | Verification Summary |
| :--- | :---: | :---: | :--- |
| `V2-P1-LOADER-CLEAN` | 1 | `VERIFIED-CLEAN` | Zero silent ACH-000873 fallback, real MS proteomics masks only, and stratified shard sampling |
| `V2-P2-ZERO-SHORTCUT-AND-DISJOINT-SPLITS` | 2 | `VERIFIED-CLEAN` | 100% SMILES-only compound featurization & zero SMILES/model_id overlap across OOD splits |
| `V2-P3-DECONFOUNDED-STAGE0` | 3 | `VERIFIED-CLEAN` | V2 checkpoints SHA-256 locked & Stage-0 evaluated with zero treatment-arm circularity or mRNA tautology |
| `V2-P4-MATH-AND-GRADIENTS` | 4 | `VERIFIED-CLEAN` | Zero dead branches, 15% MGM masking, exact 0 uM decoding, cone+nullspace residual <1e-4, and <2% stiff ODE error |
| `V2-P5-STATISTICS-AND-ABLATIONS` | 5 | `VERIFIED-CLEAN` | 100% matched n_test across FerroFlux-FM v2 & 6 baselines, p >= 1/(B+1) > 0, BH-FDR q-values, and retrained ablations |
| `V2-P6-BIOPHYSICS-AND-CONE` | 6 | `VERIFIED-CLEAN` | Iron-free (<12 nm) PEG-silica C' dot scavenges ambient Fe3+ & irreversible reactions stay non-negative |
| `V2-P7-HONEST-FIGURES` | 7 | `VERIFIED-CLEAN` | All 6 figures freshly rendered from V2 tables with dynamic on-disk Parquet counts (734,267 rows) & real ||S v||_2 |
