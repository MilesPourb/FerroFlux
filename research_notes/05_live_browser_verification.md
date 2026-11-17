# 05 — Live Portal, API, Repository & Literature Verification Report

**Verification Timestamp:** 2026-11-13 (UTC-5 / EST)  
**Execution Node:** `a076.anvil.rcac.purdue.edu` (Purdue Anvil HPC)  
**Python Environment Used:** `/home/x-mpourbaghima/.conda/envs/2024.02-py311/darpinmpnn/bin/python` (Python `3.11.17`, `pandas`, `pyarrow`, `urllib.request`)

---

## 0. Remote Debugging Endpoint (`127.0.0.1:9222`) & Strict Anti-CAPTCHA Compliance Audit

### 0.1 Diagnostic of `127.0.0.1:9222`
1. **Endpoint Probe:**
   - `curl -i -m 3 http://127.0.0.1:9222/json/version` → `curl: (7) Failed to connect to 127.0.0.1 port 9222: Connection refused`.
   - `~/.config/google-chrome/DevToolsActivePort` and `~/.config/chromium/DevToolsActivePort` → **absent** on `a076.anvil.rcac.purdue.edu`.
   - Conda environments checked (`/home/x-mpourbaghima/.conda/envs/2024.02-py311/darpinmpnn` and `/usr/bin/python3`): neither `playwright`, `selenium`, `websockets`, nor a local Chrome binary is installed on the Anvil node, and `/usr/bin/node` + `/usr/bin/npx` are legacy system packages.
2. **Exact Technical Root Cause:**
   - All agent commands execute on the remote SSH host **`a076.anvil.rcac.purdue.edu`**, whereas the user's Google Chrome instance (`Server running at: 127.0.0.1:9222 allowed remote debugging` via Chrome's `chrome://inspect/#remote-debugging` toggle) is running on the user's **local client machine** without an active SSH reverse tunnel (`ssh -R 9222:127.0.0.1:9222 ...`) to `a076.anvil.rcac.purdue.edu`.
   - Note also that Chrome M144+ `chrome://inspect/#remote-debugging` binds a WebSocket endpoint (`ws://127.0.0.1:9222/devtools/browser`) authenticated via the local `DevToolsActivePort` token rather than the legacy unauthenticated `/json/version` HTTP JSON endpoint.
3. **Live Verification Resolution & Anti-Bot / CAPTCHA Policy Enforcement:**
   - Because `a076.anvil.rcac.purdue.edu` has direct outbound HTTPS connectivity, we verified all four target domains directly against their **live public web pages, GitHub/PyPI/HuggingFace/Zenodo REST APIs, FerrDb's live FastAPI backend, ClinicalTrials.gov API v2, and NCBI PubMed E-utilities**.
   - Every HTTP fetch ran through an explicit anti-bot / CAPTCHA guard (`safe_fetch`) that **immediately skipped without retry, header spoofing, or bypass** any URL returning `401/403/429/503` or any challenge marker (`cf-turnstile`, `g-recaptcha`, `hcaptcha`, `just a moment...`, `verify you are human`).

### 0.2 Summary of Skipped / Gated Endpoints (Strict Policy Compliance)
| URL Checked | Live Status / Gate | Disposition | Official Un-gated Primary Source Used Instead |
|---|---|---|---|
| `https://depmap.org/portal/api/download/files` | `HTTP 200` containing `cf-turnstile` | **Skipped immediately** | `https://depmap.org/portal/` (`200 OK`) + `https://forum.depmap.org/t/announcing-the-26q3-release/4692.json` (`200 OK`) |
| `https://depmap.org/portal/download/api/downloads` | `HTTP 200` containing `cf-turnstile` | **Skipped immediately** | DepMap Portal HTML (`200 OK`) + Official DepMap Discourse JSON API (`200 OK`) |
| `https://arcinstitute.org/tools/virtualcellatlas` | `HTTP 403 Forbidden` | **Skipped immediately** | Official HuggingFace APIs (`tahoebio/Tahoe-100M`, `arcinstitute/*`, `200 OK`) + Zenodo API (`200 OK`) |
| `https://www.mskcc.org/research/ski/labs/michelle-bradbury` | `HTTP 403 Forbidden` | **Skipped immediately** | Weill Cornell Scholars (`https://scholars.weill.cornell.edu/michelle-s-bradbury`, `200 OK`), ClinicalTrials.gov API v2 (`200 OK`), PubMed E-utilities (`200 OK`) |
| `https://www.mskcc.org/cancer-care/doctors/michelle-bradbury` | `HTTP 403 Forbidden` | **Skipped immediately** | Same official primary sources above |
| `https://radiology.weill.cornell.edu/faculty/michelle-s-bradbury-md-phd` | `HTTP 403 Forbidden` | **Skipped immediately** | Redirected official VIVO profile `https://scholars.weill.cornell.edu/michelle-s-bradbury` (`200 OK`) |
| `https://alphagenome.google/{api,terms,atlas}` | Redirects (`302→200`) to `accounts.google.com/ServiceLogin` for `deepmind.google.com/science/alphagenome/*` | **Not bypassed** (requires Google sign-in) | Official GitHub (`google-deepmind/alphagenome` `README.md`, `faqs.md`, `ag1_vs_ag2.md`, `200 OK`) + `alphagenomedocs.com` (`200 OK`) |

---

## 1. AlphaGenome & AlphaGenome 2 (Google DeepMind) — Live Verification

### 1.1 Verified Live URLs, Page Titles, & Repository Metadata
| Resource | Live Verified URL | Status | Live Page Title / Repo Metadata |
|---|---|---|---|
| **AlphaGenome 2 Announcement Blog** | `https://deepmind.google/blog/alphagenome-2-deciphering-how-cells-read-dna-to-better-understand-disease/` | `200 OK` | *AlphaGenome 2: Deciphering how cells read DNA to better understand disease — Google DeepMind* (Published November 2026) |
| **AlphaGenome 1 Blog** | `https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/` | `200 OK` | Redirects to `https://deepmind.google/blog/alphagenome-ai-for-better-understanding-the-genome/` |
| **Official Python SDK Repo** | `https://github.com/google-deepmind/alphagenome` (`api.github.com/repos/google-deepmind/alphagenome`) | `200 OK` | `2,237` stars, `314` forks, `Apache-2.0` license; latest push & release **`v2.0.0` on `2026-11-11T13:49:23Z`** |
| **Official JAX Research & Weights Repo** | `https://github.com/google-deepmind/alphagenome_research` | `200 OK` | `921` stars, `Apache-2.0` code license (created `2025-12-17`, pushed `2026-10-13`); JAX model + TFRecord loader; weights hosted on Kaggle (`kaggle.com/models/google/alphagenome`) & HuggingFace (`huggingface.co/collections/google/alphagenome`), recommended `>= NVIDIA H100` for local inference |
| **PyPI Package** | `https://pypi.org/pypi/alphagenome/json` | `200 OK` | Latest **`alphagenome==2.0.0`** (`2026-11-11`), `requires-python >= 3.11` (supports Python 3.11–3.14; Python 3.10 dropped in `v0.11.0`) |
| **Official Documentation Hub** | `https://www.alphagenomedocs.com/` (`getting_started.html`, `api/models_v2.html`, `faqs.html`) | `200 OK` | *AlphaGenome — AlphaGenome documentation* / *AlphaGenome 2 — AlphaGenome* |
| **AlphaGenome 1 Primary Paper** | `https://www.nature.com/articles/s41586-025-10014-0` | Verified in `refs.bib` | Avsec Ž et al. *"Advancing regulatory variant effect prediction with AlphaGenome."* **Nature** `649(8099):1206–1218` (2026). `DOI: 10.1038/s41586-025-10014-0` (bioRxiv `10.1101/2025.06.25.661532`) |
| **AlphaGenome Atlas Preprint** | `https://doi.org/10.64898/2026.09.16.26363192` | Verified in `refs.bib` | Cheng J et al. *"AlphaGenome Atlas: in silico mutagenesis of the entire human genome improves prioritization and interpretation of non-coding variants."* **medRxiv** (Sept 2026). Covers **9 billion human variants** + **AlphaGenome Variant Impact (AVI)** score |

### 1.2 Exact Version History (`api.github.com/repos/google-deepmind/alphagenome/releases`)
- **`v0.1.0` (`2025-06-25`) → `v0.5.0` (`2025-11-11`)**: Initial AlphaGenome 1 SDK, alternate-allele in-silico mutagenesis (`ISM`).
- **`v0.6.0` (`2026-02-17`)**: Indel/frameshift/structural-variant properties, citation updated to *Nature* (2026).
- **`v0.9.0` (`2026-09-08`) & `v0.10.0` (`2026-10-08`)**: Added **AlphaGenome Atlas API** (`alphagenome.atlas.AtlasClient`), AVI score plotting, moved legacy AlphaGenome 1 API into `alphagenome.models.v1`.
- **`v0.11.0` (`2026-11-04`)**: Dropped Python 3.10, added Python 3.14, finalized `v1` namespace migration.
- **`v2.0.0` (`2026-11-11T13:49:23Z` — CURRENT)**: Launched **AlphaGenome 2** (`alphagenome.models.v2`), supporting reference-condition (`condition=`) and zero-shot **single-cell expression-conditioned predictions (`cell_conditioning=`)**, cell state embeddings (`encode_cells`, `CELL_EMBEDDINGS`), `GeneExpressionData`, input gradients (`predict_interval_gradients`), and enhancer-to-gene (E2G) scoring (`alphagenome.interpretation.enhancer_scoring`).

### 1.3 Technical Specifications: AlphaGenome 1 (`v1`) vs. AlphaGenome 2 (`v2`)
Verified directly from `src/alphagenome/models/v2/dna_client.py`, `src/alphagenome/models/v2/dna_output.py`, `docs/source/faqs.md`, and `docs/source/user_guides/ag1_vs_ag2.md`:

- **Supported Genome Assemblies:** Human `hg38` (`GRCh38.p13.genome.fa`, `Organism.HOMO_SAPIENS`) and Mouse `mm10` (`GRCm38.p6.genome.fa`, `Organism.MUS_MUSCULUS`). Arbitrary `ACGTN` sequences of supported length are accepted by `predict_sequence` (unbenchmarked outside human/mouse).
- **Exact Supported Input Sequence Lengths (`dna_client.py` lines 707–717):**
  1. `SEQUENCE_LENGTH_16KB` = **`16,384 bp`** ($2^{14}$)
  2. `SEQUENCE_LENGTH_100KB` = **`131,072 bp`** ($2^{17}$)
  3. `SEQUENCE_LENGTH_500KB` = **`524,288 bp`** ($2^{19}$)
  4. `SEQUENCE_LENGTH_1MB` = **`1,048,576 bp`** ($2^{20}$, recommended for highest accuracy)
  - *Note:* `predict_interval_gradients` and `predict_interval_gradients_batch` accept **only `100KB` (`131,072 bp`) and `1MB` (`1,048,576 bp`)**.
- **Training Scale & Architecture Shift in AlphaGenome 2:**
  - Trained on **>100 million single-cell profiles across 68 tissues**.
  - Replaces v1's fixed multi-task bulk output head with a **conditional architecture**: each call takes one biological condition (`condition=` from `supported_conditions()` **or** `cell_conditioning=` via `gene_counts_condition.from_anndata(...)` accepting **up to `256` human cells (`MAX_CONDITIONING_CELLS = 256`) of raw scRNA-seq UMI counts indexed by Ensembl gene IDs**).
  - Every output track carries an `imputation_mode` metadata tag: `observed`, `imputed_supported`, or `imputed_exploratory`.
- **Exact Output Modalities (`OutputType` Enum):**
  - **Shared across `v1` and `v2` (10 modalities):** `ATAC`, `CAGE`, `DNASE`, `RNA_SEQ`, `CHIP_HISTONE`, `CHIP_TF`, `SPLICE_SITES` (upgraded in `v2` from 4 condition-agnostic channels to **5 condition-specific channels**: `donor +`, `acceptor +`, `donor -`, `acceptor -`, `padding .`), `SPLICE_SITE_USAGE`, `SPLICE_JUNCTIONS`, `CONTACT_MAPS` (un-coarse-grained observed-over-expected 3D contacts in `v2`).
  - **Removed in `v2` (1 modality):** `PROCAP` (replaced by `CAGE` for TSS activity; still available in `alphagenome.models.v1`).
  - **Added in `v2` (5 brand-new modalities → 15 total in `v2`):**
    1. `SC_GENE_EXPRESSION`: Protocol-specific scRNA-seq expression tracks (in CPM) under `condition=` or `cell_conditioning=`.
    2. `SC_GENE_EXPRESSION_DENOISED`: Denoised single-cell gene expression tracks (1 channel/strand in CPM) from `cell_conditioning=`.
    3. `SC_ATAC`: Single-cell chromatin accessibility tracks.
    4. `CELL_EMBEDDINGS`: Learned low-dimensional cell-state representations compressed from scRNA-seq profiles (`encode_cells()` chunks up to `MAX_ENCODE_CELLS_PER_REQUEST = 1,000` cells/request).
    5. `METHYLATION`: CpG DNA methylation probability tracks (paired with new `MethylationScorer`).
- **Variant, Interval, & Enhancer Endpoints (`v2.dna_client.DnaClient`):**
  - Prediction: `predict_sequence(s)`, `predict_interval(s)`, `predict_variant(s)`, `predict_interval_gradients(_batch)`, `encode_cells`.
  - Scoring: `score_variant(s)` (`MAX_VARIANT_SCORERS_PER_REQUEST = 23`; returns **`raw_score` only** in `v2`, whereas `v1` also returned empirical gnomAD `quantile_score`), `score_interval(s)`, `score_ism_variants` (`MAX_ISM_INTERVAL_WIDTH = 10` bp per chunk, auto-split client-side), plus `alphagenome.interpretation.enhancer_scoring` (`score_enhancers`, `get_active_enhancers`, `assign_consensus_enhancers`) and `alphagenome.atlas.AtlasClient` (`query_variant`, `query_variants`, `query_interval`).
  - Note: AlphaGenome 2 has **no chemical/drug/nanoparticle perturbation input**; its "perturbation" capability is strictly **genetic (`Variant` / `ISM`)** and **transcriptomic state conditioning (`cell_conditioning=` given post-perturbation scRNA-seq counts)**.

### 1.4 CRITICAL 2026 API Access & Licensing Red Flag for `@verifier`
> **MANDATORY LEGAL / ARCHITECTURAL CONSTRAINT (Verified verbatim from `README.md` lines 36–47 and `docs/source/faqs.md` lines 249–256):**  
> *"AlphaGenome is governed by the AlphaGenome Terms of Service (`https://alphagenome.google/terms`): **direct API outputs and AlphaGenome Atlas scores are for non-commercial research use only, must not be used to train machine learning models, and must not be used for clinical decision-making.** For commercial and enterprise workflows, access AlphaGenome on Google Cloud via Model Garden on Gemini Enterprise Agent Platform (`https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/open-models/alphagenome`)."*
> - **Query Rate Limits:** Free non-commercial API keys (`alphagenome.google/api`) have dynamic demand-based throttling: pre-computed `AtlasClient` lookups support higher query rates, whereas live `v2` model calls are explicitly documented as suited for *"small-scale analyses rather than large, genome-wide analyses."* (DeepMind also announced an application window closing November 30, 2026 for extra research quota/credits, and upcoming AlphaGenome 2 Skills in Google Antigravity.)
> - **Implication for Our Ferroptosis Foundation Model:** **We CANNOT pretrain, distill, or supervise a new ML model on AlphaGenome 2 API or Atlas outputs without violating Google DeepMind's Terms of Service!** Any valid architecture must either (a) pretrain strictly on open public datasets (ENCODE, Roadmap, DepMap, Tahoe-100M, scPerturb) and invoke AlphaGenome 2 **only as an inference-time zero-shot variant/enhancer module** (or check whether local weights under `deepmind.google.com/science/alphagenome/model-terms` permit frozen feature extraction, noting `alphagenome_research` currently hosts AlphaGenome 1 JAX code pushed `2026-10-13`), or (b) replace AlphaGenome 2 during pretraining with open Apache-2.0 DNA models such as Arc's `Evo 2` (`arcinstitute/evo2_7b`) or Borzoi.

---

## 2. FerrDb (V1, V2, and V3) — Live Database & Downloadable Table Verification

### 2.1 Live Portal Routing & Publication Lineage
- **Root Portal (`https://www.zhounan.org/ferrdb/` & `http://www.zhounan.org/ferrdb/`):** `HTTP 200 OK`, `<title>FerrDb</title>`. Links to:
  1. **FerrDb V3 (Current, 2025/2026):** `https://www.zhounan.org/ferrdb/v3/pages/index.html` (`200 OK`), powered by live REST API `https://data.zhounan.org/pm2505/api/`. Published in *Nucleic Acids Research* `54(D1):D572–D582` (`2026 Jan 6`; epub Oct 31, 2025), **`PMID: 41171133`**, **`DOI: 10.1093/nar/gkaf1119`**, license `CC BY-NC 4.0` (free for academic use). Expands FerrDb from ferroptosis alone to **22 Regulated Cell Death (RCD) modalities** curated from **>20,000 articles** (`18,371` ferroptosis articles curated through `2024-12-31`; other RCDs through `2025-05-31`), adding `ChatRCD`, `vennRCD`, `enrichRCD`, `omicsRCD`, `classRCD`, `locateRCD`, and `crossRCD`.
  2. **FerrDb V2 (Legacy/Discontinued Maintenance, 2022/2023):** `http://www.zhounan.org/ferrdb/current/` (`200 OK`, note `/ferrdb/v2/` returns `404` — actual path is `/ferrdb/current/`!). Published in *Nucleic Acids Research* `51(D1):D571–D582` (2023), **`PMID: 36305834`**, **`DOI: 10.1093/nar/gkac935`**. Also hosts static interim archives `http://www.zhounan.org/ferrdb/current/extdownload/ferroptosis_early_preview_upto20231231.zip` and `ext_20240730.xlsx`.
  3. **FerrDb V1 (Legacy, 2020):** `http://www.zhounan.org/ferrdb/legacy/` (`download.html` `200 OK`). Published in *Database (Oxford)* `2020:baaa021`, **`PMID: 32219413`**, **`DOI: 10.1093/database/baaa021`**.

### 2.2 Exact Verified Counts Across FerrDb V1, V2, and Live V3
Verified directly from `GET https://data.zhounan.org/pm2505/api/db/statistics/` (`200 OK`) **AND** by executing live `POST https://data.zhounan.org/pm2505/api/download/data/` downloads of all 7 CSV tables (`200 OK`):

| Category | FerrDb V1 (2020 Legacy Static) | FerrDb V2 (2022/2023 *NAR*) | **FerrDb V3 (Live 2026 API — Ferroptosis Subset)** | **FerrDb V3 Downloaded CSV Rows (Ferroptosis / All 22 RCDs)** | Live V3 CSV Size & Exact Columns |
|---|---:|---:|---:|---:|---|
| **Drivers** | 108 genes (150 annots) | 264 | **734** (`735` unique `Symbol`s in CSV) | **1,163** / 2,146 rows | `618,803 B` — `['RCD', 'Symbol', 'HGNC_ID', 'ENSG_stable', 'Dependence', 'Evidence', 'Test_organism', 'Pathway', 'Confidence', 'Experiments', 'Caution', 'UniProtAC', 'PMID', 'Remark', 'Last_update', 'Data_source', 'Gene_type', 'Category']` |
| **Suppressors** | 69 genes (109 annots) | 238 | **1,021** | **1,673** / 3,047 rows | `860,788 B` — Same 18 columns as `driver` |
| **Markers** | 111 genes (123 annots) | 9 | **59** | **108** / 114 rows | `37,954 B` — Same 18 columns as `driver` |
| **Unclassified Genes** | *(not in V1)* | 110 | **143** (`144` unique `Symbol`s in CSV) | **156** / 1,256 rows | `318,699 B` — Same 18 columns as `driver` |
| **Inducers (Substances)** | 35 (54 annots) | 201 | **563** | **834** / 990 rows | `92,324 B` — `['RCD', 'Substance', 'Name', 'PubChem_CID', 'PMID', 'Remark', 'Last_update', 'Data_source', 'Category']` |
| **Inhibitors (Substances)** | 41 (46 annots) | 179 | **510** | **720** / 760 rows | `73,524 B` — Same 9 columns as `inducer` |
| **Disease Associations** | 95 (135 annots) | 143 | **437** | **1,101** / 1,231 rows | `132,216 B` — `['RCD', 'Effect', 'Test_organism', 'Confidence', 'PMID', 'Remark', 'Last_update', 'Data_source']` |

- **Download Format Fact-Check for `@verifier`:**
  - **FerrDb V3** (`https://www.zhounan.org/ferrdb/v3/pages/download.html` → `POST https://data.zhounan.org/pm2505/api/download/data/` with form fields `selectDataset={driver|suppressor|marker|unclassified|inducer|inhibitor|disease}` and `selectFormat={csv|txt}`) provides **CSV and TXT only (NOT JSON or XLSX)**!
  - Static **JSON + CSV + XLSX** files (`1_info_*.xlsx`, `1_info_*.csv`, `3_info_*.json`) exist **only** on the 2020 **FerrDb V1 legacy** page (`http://www.zhounan.org/ferrdb/legacy/fordownload/`), which contains only 259 unique genes and 76 small molecules. Scripts claiming "FerrDb V3 JSON downloads" are inaccurate; use `POST https://data.zhounan.org/pm2505/api/download/data/` with `selectFormat=csv` and filter `RCD == 'Ferroptosis'`.

---

## 3. DepMap Portal & Arc Institute Virtual Cell Atlas (`Tahoe-100M` / `scPerturb`) — Live Verification

### 3.1 Broad Institute DepMap Portal (`26Q3` Current Release)
- **Live Portal URL:** `https://depmap.org/portal/` (`HTTP 200 OK`, `<title>DepMap: The Cancer Dependency Map Project at Broad Institute</title>`).
- **Official Release Notes JSON Verified:** `https://forum.depmap.org/t/announcing-the-26q3-release/4692.json` (`HTTP 200 OK`, published **`2026-10-05T13:06:00Z`**).
- **Verified Release Timeline (2024–2026):** `24Q2` (`2024-06-03`) → `24Q4` (`2024-09-10`) → `25Q2` (`2025-06-05`) → `25Q3` (`2025-09-30`) → `26Q1` (`2026-04-01`) → **`26Q3` (`2026-10-05`, CURRENT LIVE RELEASE)**.
- **Exact Highlights in DepMap `26Q3`:**
  1. **150 new genome-wide CRISPR screens and multi-omics** for adult and pediatric **next-generation / 3D organoid & spheroid models** integrated from Neiswiender, Maffa, Brenan et al. (*Nature* 2026, `https://www.nature.com/articles/s41586-026-10843-7`). This is directly relevant to ferroptosis because 3D spheroid/organoid cultures exhibit core hypoxia/nutrient-starvation gradients and lipid-peroxidation propagation dynamics missed in 2D monolayers!
  2. **HCR-seq dataset:** Simultaneous single-cell DNA repair capacity + 10x scRNA-seq across 19 cancer cell lines (`github.com/getzlab/HCRseq`, Chen et al., *Nature Communications* 2026).
  3. **CRISPR Chronos Pipeline Updates:** Library correction improvements, removal of TKOv3/Brunello-only genes, and **ChrY gene effect mean-centering to 0** prior to CN correction.
  4. **Updated Mutation / GIAB hg38 BAM patches & PAM50 subtypes.**
  5. **Download Access Note:** Automated requests to `/portal/api/download/files` hit Cloudflare Turnstile (`cf-turnstile`, skipped per policy); DepMap community forum thread `#4652` (`2026-07-14`, *"Provide an open endpoint for latest version retrieval"*) confirms Broad placed Turnstile in front of the portal file-listing API in mid-2026, so bulk files (`CRISPRGeneEffect.csv`, `OmicsExpressionProteinCodingGenesTPMLogp1.csv`, `CCLE_metabolomics_20190502.csv`, PRISM Repurposing, CTRPv2) should be downloaded once via an interactive browser session at `https://depmap.org/portal/data_page/?tab=allData` and staged to `/anvil/projects/x-cis261485/`.

### 3.2 Arc Institute Virtual Cell Atlas & `Tahoe-100M` — Live Metadata & Compound Audit
- **Official HuggingFace Dataset (`tahoebio/Tahoe-100M`, alias `vevotx/Tahoe-100M`):** `https://huggingface.co/api/datasets/tahoebio/Tahoe-100M` (`HTTP 200 OK`, **`gated: False`**, license **`CC0-1.0` public domain**, `131,576` downloads, last modified `2026-10-14`, preprint `https://www.biorxiv.org/content/10.1101/2025.02.20.639398v1`).
- **Exact File & Split Inventory (`README.md` + API):**
  - **Total repository files:** `4,427` files.
  - **Single-cell expression shards:** `3,388` Parquet files (`data/train-00000-of-03388.parquet` … `03387`), **`num_examples = 95,624,334` cells**, **`337.64 GB` compressed download** (`1,693.65 GB` uncompressed).
  - **Pseudobulk differential expression shards:** `1,026` Parquet files (`metadata/pseudobulk_differential_expression/train-00000-of-01026.parquet` … `01025`).
  - **Metadata tables (downloaded & inspected live with `pyarrow`/`pandas`):**
    - `metadata/drug_metadata.parquet`: shape **`(379, 9)`** — **379 unique small-molecule drugs** (`drug`, `targets`, `moa-broad`, `moa-fine`, `human-approved`, `clinical-trials`, `gpt-notes-approval`, `canonical_smiles`, `pubchem_cid`).
    - `metadata/sample_metadata.parquet`: shape **`(1344, 8)`** — **1,344 plate/drug-dose samples** across 380 drug/control entries (typically 3 concentrations per drug).
    - `metadata/cell_line_metadata.parquet`: shape **`(1000, 10)`** — **102 unique cancer cell lines** (`99` unique DepMap `ACH-*` IDs; ~50 lines pooled per Mosaic village set) spanning 15 organs, including **10 Skin/Melanoma lines** (`SK-MEL-2`, `SK-MEL-30`, `LOX-IMVI`, `C32`, `RPMI-7951`, `MM415`, `MM485`, `Hs 695T`, `Hs 852.T`, `Hs 936.T(C1)`), **1 ccRCC Kidney line** (`A498` / `ACH-000555`), and **4 CNS/Brain glioma lines** (`A-172`, `CCF-STTG1`, `H4`, `SW 1088`), with annotated driver zygosity/mutations (`KRAS`, `STK11`, `SMARCA4`, etc.).
- **Arc Institute Companion Datasets & Models Verified Live (`huggingface.co/api/{datasets,models}?author=arcinstitute`, `200 OK`):**
  - Datasets: `arcinstitute/PIE_tahoe100m` (`62` files, per-cell-line DE tables `de/CVCL_*.parquet`, updated `2026-10-22`), `arcinstitute/State-Tahoe-Filtered`, `arcinstitute/State-Replogle-Filtered`, `arcinstitute/State-Parse-Filtered`, `arcinstitute/PIE_replogle_nadig_essential`, `arcinstitute/PIE_x_atlas_orion`, `arcinstitute/PIE_arc_vcc_25`, `arcinstitute/Stack-scBaseCount189M` (189M cells), `arcinstitute/Stack-CellxGene45M`, `arcinstitute/Perturb-Sapiens`, `arcinstitute/opengenome2`.
  - Pretrained Models: `arcinstitute/ST-SE-Tahoe`, `arcinstitute/ST-HVG-Tahoe`, `arcinstitute/ST-SE-Replogle`, `arcinstitute/SE-600M`, `arcinstitute/SE-100M`, `arcinstitute/Stack-Large`, `arcinstitute/evo2_1b_base`, `arcinstitute/evo2_7b`, `arcinstitute/evo2_20b`, `arcinstitute/evo2_40b`.

#### CRITICAL FACT-CHECK FOR `@verifier`: Ground-Truth Ferroptosis Compound Audit of `Tahoe-100M`
By inspecting every row of `tahoebio/Tahoe-100M/resolve/main/metadata/drug_metadata.parquet` (`379` drugs), we proved a critical ground-truth distinction that generic web summaries get wrong:
- **CANONICAL EXPERIMENTAL TOOL FERROPTOSIS INDUCERS & RADICAL-TRAPPING INHIBITORS ARE ABSENT (`0` rows in `Tahoe-100M`):**
  - **Absent Inducers:** `Erastin`, `Imidazole ketone erastin (IKE)`, `RSL3`, `ML210`, `ML162`, `FIN56`, `FINO2`, `Sulfasalazine`, `Sorafenib`, `Lenvatinib`, `Brequinar`, `Buthionine sulfoximine (BSO)`, `Elesclomol`, `Disulfiram`, `Withaferin A`, `APR-246 / Eprenetapopt`.
  - **Absent Radical-Trapping / Chelator Rescue Agents:** `Ferrostatin-1`, `Liproxstatin-1`, `Deferoxamine (DFO)`, `Deferasirox`, `Deferiprone`, `alpha-Tocopherol / Vitamin E`, `Idebenone`.
- **CLINICAL & MECHANISTIC FERROPTOSIS / REDOX / LIPID-REMODELING DRUGS GENUINELY PRESENT IN `Tahoe-100M` (`22` verified compounds):**
  1. **Endoperoxide Fe²⁺-Dependent Ferroptosis Inducers:** `Artesunate`, `Dihydroartemisinin`.
  2. **Selenoprotein / Thioredoxin Reductase Inhibitor:** `Auranofin` (`targets: TXNRD1, TXNRD2`).
  3. **NRF2 / GSH Covalent Electrophile:** `Dimethyl fumarate` (`targets: NFE2L2`).
  4. **Mevalonate / HMGCR Inhibitors (deplete CoQ10 & isopentenyl-Sec-tRNA):** `Pravastatin (sodium)` (`HMGCR`), `Pitavastatin (Calcium)` (`HMGCR`).
  5. **Quinones / Vitamin K Cycle Substrates & Redox Cyclers (FSP1/VKORC1L1 axis):** `Vitamin K4`, `Menadione`.
  6. **Intracellular Iron Chelator:** `Ciclopirox`.
  7. **5-Lipoxygenase (`ALOX5`) Lipid Peroxidation Inhibitors:** `Zileuton` (`ALOX5`), `Malotilate` (`ALOX5`).
  8. **Thiazolidinediones (PPARG agonists & established off-target `ACSL4` inhibitors blocking PUFA-PL incorporation):** `Rosiglitazone` (`PPARG, TRPC5, TRPM3`), `Pioglitazone` (`PPARG`).
  9. **Additional Validated Ferroptosis Modulators / Sensitizers:** `Altretamine`, `Dexamethasone` (`NR3C1`), `Regorafenib`, `Bortezomib` & `Ixazomib` (`PSMB5`), `Panobinostat` & `Belinostat` (`HDAC`), `Lapatinib ditosylate` & `Neratinib` (`EGFR, ERBB2`), and **8 `MTOR` inhibitors** (`Rapamycin`, `Everolimus`, `Temsirolimus`, `AZD-8055`, `Sapanisertib`, `Torkinib`, `Bimiralisib`, `GSK1059615` — governing mTORC1-dependent GPX4 synthesis/autophagic ferritinophagy).
- **Architectural Consequence:** `Tahoe-100M` provides massive single-cell transcriptional state-transition pretraining across 95.6M cells and 22 redox/lipid/iron-axis clinical drugs, **but canonical GPX4i/System-xc⁻i potency (`Erastin`, `RSL3`, `ML210`, `ML162`, `FIN56`) MUST come from CTRPv2, PRISM Repurposing, LINCS L1000, and curated CRISPR/drug screens!**

### 3.3 `scPerturb` Live Verification (`Zenodo` Open Records)
- **RNA & Protein Perturb-seq Archive:** `https://zenodo.org/records/13350497` (`api/records/13350497` `200 OK`, `DOI: 10.5281/zenodo.13350497`) — **54 open `.h5ad` files** directly downloadable via HTTPS without login:
  - Genome-wide & essential Perturb-seq: `ReplogleWeissman2022_K562_gwps.h5ad` (`8,805.5 MB`), `ReplogleWeissman2022_K562_essential.h5ad` (`1,546.7 MB`), `ReplogleWeissman2022_rpe1.h5ad` (`1,236.9 MB`), `NadigOConner2024_jurkat.h5ad` (`1,293.7 MB`), `NadigOConner2024_hepg2.h5ad` (`850.6 MB`), `NormanWeissman2019_filtered.h5ad` (`698.7 MB`).
  - Chemical sci-Plex screens: `SrivatsanTrapnell2020_sciplex3.h5ad` (`2,526.6 MB`), `sciplex4.h5ad` (`253.3 MB`), `sciplex2.h5ad` (`145.2 MB`).
  - Neuronal / iPSC oxidative stress & ferroptosis-relevant CRISPRi/a screens (Kampmann Lab): `TianKampmann2019_day7neuron.h5ad` (`268.9 MB`), `TianKampmann2019_iPSC.h5ad` (`350.8 MB`), `TianKampmann2021_CRISPRi.h5ad` (`289.4 MB`), `TianKampmann2021_CRISPRa.h5ad` (`154.8 MB`).
  - Immune / melanoma co-culture CRISPR screens: `FrangiehIzar2021_RNA.h5ad` (`1,458.9 MB`) + `protein.h5ad` (`24.7 MB`), `PapalexiSatija2021_eccite_RNA.h5ad` (`147.2 MB`).
- **scATAC Perturb-seq Archive:** `https://zenodo.org/records/7058382` (`200 OK`, `DOI: 10.5281/zenodo.7058382`, 6 `.zip` archives including `PierceGreenleaf2021` Spear-ATAC and `Liscovitch-BrauerSanjana2021` CRISPR-sciATAC).

---

## 4. PI Michelle S. Bradbury Lab (Weill Cornell Medicine / MSKCC) — Live Verification

### 4.1 Live Verified Academic Titles, Focus Areas, & Active Ferroptosis Grants (2026)
- **Primary Live Profile (`HTTP 200 OK`):** `https://scholars.weill.cornell.edu/michelle-s-bradbury` (redirected from `https://vivo.med.cornell.edu/display/cwid-msb2006`), `<title>Michelle S. Bradbury, PhD, MD — Professor of Pharmacology — Scholars @ Weill Cornell Medicine</title>`.
- **Verified Current Appointments (as of November 2026):**
  - **Professor of Pharmacology** (Primary appointment, Department of Pharmacology, Weill Cornell Medicine, **2026–present**)
  - **Professor of Neuroscience** (Feil Family Brain and Mind Research Institute, **2024–present**)
  - **Professor of Radiology in Radiation Oncology** (**2024–present**)
  - **Member**, Sandra and Edward Meyer Cancer Center (*Cancer Therapeutics* program) & Clinical and Translational Science Center; Physician, NewYork-Presbyterian Hospital; Clinician-Scientist & Director of Intraoperative Imaging, Memorial Sloan Kettering Cancer Center (previously Endowed Professor of Imaging Research in Radiology, 2024–2026, and Director of MI3).
- **Verified Active Research Grants (`5` active grants on live profile):**
  1. **`NCI R01 CA253658-05` (2023–2027, PI: Michelle S. Bradbury):** *"Ultrasmall particle-based solutions for inducing ferroptosis and improving anti-tumor immune responses in cancer."* (Directly funds the ferroptosis + antitumor immunity C' dot program!)
  2. **`NCI R01 CA243085-05` (2023–2026, PI: Michelle S. Bradbury):** *"Molecular Phenotyping and Image-Guidance for Surgical Treatment of High-Risk Prostate Cancer Using Ultrasmall Silica Nanoparticles."*
  3. **`DoD HT9425-23-1-0631` (2023–2027, PI: Michelle S. Bradbury):** *"Ultrasmall PSMA-Targeted Particle Radiotherapy for Regulating the Immunosuppressive Prostate Cancer Tumor Microenvironment and Enhancing Treatment Efficacy."*
  4. **`DoD HT9425-25-1-0084` (2025–2028, PI: Michelle S. Bradbury):** *"Targeting Inflammatory Pathways in Breast Cancer Brain Metastases for the Treatment of Edema and Radiation Necrosis Using Anti-VEGF Ultrasmall Particle Therapy."*
  5. **`MSKCC CC 254279-02` (2025–2027, PI: Michelle S. Bradbury):** *"Reprogramming of IL-1 Driven Inflammation and Immune Suppression in Breast Cancer and Clonal Hematopoiesis Models Using Targeted Ultrasmall Particle Therapies."*

### 4.2 Live Verified Clinical Trials of Cornell Dots (`C dots` / `C' dots`) — `ClinicalTrials.gov` API v2
| NCT ID | Status (Nov 2026) | Phase | Sponsor / PI | Nanoparticle Intervention | Indication & Title |
|---|---|---|---|---|---|
| **`NCT04167969`** | `RECRUITING` | Phase 1 | MSKCC (PI: **Michelle S. Bradbury, MD, PhD**) | `(64Cu)-NOTA-PSMAi-PEG-Cy5.5-C' dot` & `89Zr-DFO-PSMAi-PEG-Cy5.5-C' dots` | *The Use of Nanoparticles to Guide the Surgical Treatment of Prostate Cancer* (PET/MRI + intraoperative fluorescence) |
| **`NCT02106598`** | `ACTIVE_NOT_RECRUITING` | Phase 1/2 | MSKCC | `Fluorescent cRGDY-PEG-Cy5.5-C dots` | *Targeted Silica Nanoparticles for Real-Time Image-Guided Intraoperative Mapping of Nodal Metastases* (Head & Neck Melanoma; results in *JAMA Netw Open* 2021, `PMID: 33734415`) |
| **`NCT03465618`** | `ACTIVE_NOT_RECRUITING` | Phase 1 | MSKCC | `89Zr-DFO-cRGDY-PEG-Cy5-C' dots` | *A First in Human Study Using 89Zr-cRGDY Ultrasmall Silica Particle Tracers for Malignant Brain Tumors* |
| **`NCT01266096`** | `ACTIVE_NOT_RECRUITING` | Early Phase 1 (Microdosing) | MSKCC | `124I-cRGDY-PEG-dots` | *PET Imaging of Patients With Melanoma and Malignant Brain Tumors Using an 124I-labeled cRGDY Silica Nanomolecular Particle Tracer* (Landmark first-in-human trial; *Sci Transl Med* 2014 `PMID: 25355699`) |
| **`NCT05001282`** | `TERMINATED` | Phase 1/2 | Elucida Oncology (commercial C'-dot spinout) | `ELU001` (FRα-targeted C'-dot exatecan conjugate) | *A Study to Evaluate ELU001 in Patients With Solid Tumors That Overexpress Folate Receptor Alpha (FRα)* |
| **`NCT05622591`** | `WITHDRAWN` | Phase 1 | Elucida Oncology | `ELU001` | *ELU001 in Pediatric Subjects Who Have Relapsed and/or Refractory CBFA2T3-GLIS2-positive AML* |

### 4.3 Live Verified Bradbury Lab Publications (2020–2026 + Foundational Ferroptosis Lineage)
Verified directly via NCBI PubMed E-utilities XML (`efetch.fcgi`, `HTTP 200 OK`) with exact article-level `PMID`, `PMCID`, and `DOI` (zero reference-list DOI contamination):

#### A. Core Ferroptosis, Immunometabolic Reprogramming, & Collective Death Propagation Papers
1. **Siddiqui NA, Zhang L, DeLeon G, Naguib N, ..., Bradbury MS (2026).** *"Reprogramming of TLR-Ferroptosis Signaling and Immunometabolic Pathways Overcomes Myeloid Suppression to Improve Checkpoint Blockade in Prostate Cancer."* **Cancer Research** (2026; highlighted in WCM News June 15, 2026 & *Oncology Times* Sept 1, 2026).  
   - **Verified IDs:** `PMID: 42290280` | `PMCID: PMC13314319` | `DOI: 10.1158/0008-5472.CAN-25-4954`  
   - **Key Experimental Readouts for ML Alignment:** Ultrasmall PSMA-targeting amorphous silica C' dots act as single-agent immunometabolic ferroptosis sensitizers in *MYC*-driven prostate cancer — suppressing lipid/steroid biosynthesis, disrupting lipid peroxidation control and nutrient flux, reprogramming the **TLR–ferroptosis axis**, reversing myeloid suppression, driving CD8⁺ T-cell infiltration, and synergizing with CSF-1R + immune checkpoint blockade.
2. **Das J, Hombalkar SK, Klein AD, Nsasra E, ..., Bradbury MS, ..., Overholtzer M (2026).** *"Ferroptosis induces heterogeneous death profiles that are controlled by lysosome rupture."* **Developmental Cell** (2026).  
   - **Verified IDs:** `PMID: 41709461` | `PMCID: PMC13014609` | `DOI: 10.1016/j.devcel.2026.01.014`  
   - **Key Mechanistic Discovery:** Resolves **"single-cell" (cell-autonomous) vs. "propagative" (collective wave) ferroptosis** — showing that **lysosome rupture** governs collective propagation and that **glutathione (GSH) depletion is sufficient to convert GPX4 inhibition from an individual-cell death response into a collective wave response**!
3. **De Leon G, Zhang L, Siddiqui NA, Naguib N, ..., Socciarelli F, Pourbaghi M, ..., Merghoub T, Wolchok JD, Bradbury MS (2026; epub Dec 2025).** *"An ultrasmall core-shell silica nanoparticle improves antitumour immunity and survival by remodelling suppressive melanoma microenvironments."* **Nature Nanotechnology** (2025/2026).  
   - **Verified IDs:** `PMID: 41461940` | `PMCID: PMC12916478` | `DOI: 10.1038/s41565-025-02083-z`  
   - **Key Experimental Readouts:** Self-therapeutic C' dots remodel checkpoint-resistant melanoma TMEs via the **STING / IL-6 / PD-L1** and **type I/II interferon** axes, driving CD8⁺ T-cell-dependent survival synergy with anti-IL-6 + anti-PD-L1.
4. **Riegman M, Sagie L, Galed C, Levin T, ..., Bradbury MS, ..., Overholtzer M (2020).** *"Ferroptosis occurs through an osmotic mechanism and propagates independently of cell rupture."* **Nature Cell Biology** `22(9):1042–1048` (2020).  
   - **Verified IDs:** `PMID: 32868903` | `PMCID: PMC7644276` | `DOI: 10.1038/s41556-020-0565-1`  
   - **Key Mechanistic Discovery:** Demonstrates plasma-membrane nanopore osmotic swelling and shows that **intercellular ferroptosis trigger waves** occur under system $x_c^-$ inhibition / C' dots / erastin independently of lytic rupture, whereas direct GPX4 inhibition (`RSL3`) kills stochastically without wave propagation unless GSH is co-depleted (completed in `PMID: 41709461`!).
5. **Foundational Pre-2020 Ferroptosis Papers (for Lineage Completeness):**
   - **Riegman M, Bradbury MS, Overholtzer M (2019).** *"Population Dynamics in Cell Death: Mechanisms of Propagation."* **Trends in Cancer** `5(9):558–568`. `PMID: 31474361` | `PMCID: PMC7310667` | `DOI: 10.1016/j.trecan.2019.07.008`.
   - **Kim SE, Zhang L, Ma K, Riegman M, Chen F, Ingold I, Conrad M, Turker MZ, Gao M, Jiang X, Monette S, Pauliah M, Gonen M, Zanzonico P, Quinn T, Rodriguez E, Zanoni M, Wiesner U, Bradbury MS, Overholtzer M (2016).** *"Ultrasmall nanoparticles induce ferroptosis in nutrient-deprived cancer cells and suppress tumour growth."* **Nature Nanotechnology** `11(11):977–985`. `PMID: 27668796` | `PMCID: PMC5108575` | `DOI: 10.1038/nnano.2016.164` (reversed in vivo by `liproxstatin-1`).

#### B. Additional Verified 2020–2026 Bradbury Lab Engineering, Protein Design, & C'-Dot Therapeutics Papers
- **Pourbaghi M, Elemento O, Bradbury MS (2026).** *"Scaffold-Constrained Generative Design Overcomes Failure Modes in Computational Protein Engineering."* **bioRxiv** (`2026-08-07`). `PMID: 42845323` | `PMCID: PMC13640583` | `DOI: 10.64898/2026.08.07.743551`.
- **Sofias AM, Derks Y, Bradbury MS, Chen X, ..., Lammers T (2026).** *"Cancer theranostics."* **Nature Reviews Cancer** (2026). `PMID: 42918868` | `DOI: 10.1038/s41568-026-00983-3`.
- **Chen F, Bradbury MS (2025).** *"Engineering bispecific HER2xVEGF designed ankyrin repeat proteins for targeted nanobiotechnology applications."* **Journal of Nanobiotechnology** (2025). `PMID: 41046281` | `PMCID: PMC12495870` | `DOI: 10.1186/s12951-025-03631-4`.
- **Zhang L, Aragon-Sanabria V, Aditya A, ..., Bradbury MS (2023).** *"Engineered Ultrasmall Nanoparticle Drug-Immune Conjugates with 'Hit and Run' Tumor Delivery to Eradicate Gastric Cancer."* (Anti-HER2 scFv + topoisomerase inhibitor + **deferoxamine** moieties on sub-8 nm C' dots). **Advanced Therapeutics** (2023). `PMID: 37007587` | `PMCID: PMC10061546` | `DOI: 10.1002/adtp.202370009`.
- **Wu F, Chen PM, ..., Bradbury MS, Wiesner U, Ma K (2022).** *"Ultrasmall Folate Receptor Alpha Targeted Enzymatically Cleavable Silica Nanoparticle Drug Conjugates Augment Penetration and Therapeutic Efficacy in Models of Cancer."* **ACS Nano** (2022). `PMID: 36264003` | `DOI: 10.1021/acsnano.2c05342`.
- **Aragon-Sanabria V, Aditya A, Zhang L, Chen F, ..., Rudin CM, Bradbury MS (2022).** *"Ultrasmall Nanoparticle Delivery of Doxorubicin Improves Therapeutic Index for High-Grade Glioma."* **Clinical Cancer Research** (2022). `PMID: 35499557` | `PMCID: PMC9250642` | `DOI: 10.1158/1078-0432.CCR-21-4053`.
- **Zanoni DK, Stambuk HE, Madajewski B, ..., Bradbury MS, Patel SG (2021).** *"Use of Ultrasmall Core-Shell Fluorescent Silica Nanoparticles for Image-Guided Sentinel Lymph Node Biopsy in Head and Neck Melanoma: A Nonrandomized Clinical Trial."* **JAMA Network Open** (2021). `PMID: 33734415` | `PMCID: PMC7974643` | `DOI: 10.1001/jamanetworkopen.2021.1936`.
- **Madajewski B, Chen F, Yoo B, ..., Bradbury MS, Brennan C (2020).** *"Molecular Engineering of Ultrasmall Silica Nanoparticle-Drug Conjugates as Lung Cancer Therapeutics."* **Clinical Cancer Research** (2020). `PMID: 32723835` | `PMCID: PMC7686858` | `DOI: 10.1158/1078-0432.CCR-20-0851`.
- **Urbanska AM, Khanin R, ..., Bradbury MS, McDevitt MR (2020).** *"A Genomic Profile of Local Immunity in the Melanoma Microenvironment Following Treatment with α Particle-Emitting Ultrasmall Silica Nanoparticles."* **Cancer Biotherapy & Radiopharmaceuticals** (2020). `PMID: 32013538` | `PMCID: PMC7462037` | `DOI: 10.1089/cbr.2019.3150`.
- **Zhang X, Chen F, ..., Wiesner U, Bradbury MS, Quinn TP (2020).** *"Targeted melanoma radiotherapy using ultrasmall 177Lu-labeled α-melanocyte stimulating hormone-functionalized core-shell silica nanoparticles."* **Biomaterials** (2020). `PMID: 32120314` | `PMCID: PMC7171978` | `DOI: 10.1016/j.biomaterials.2020.119858`.
- **Juthani R, Madajewski B, Yoo B, Zhang L, ..., Bradbury MS, Brennan CW (2020).** *"Ultrasmall Core-Shell Silica Nanoparticles for Precision Drug Delivery in a High-Grade Malignant Brain Tumor Model."* **Clinical Cancer Research** (2020). `PMID: 31515460` | `PMCID: PMC6942644` | `DOI: 10.1158/1078-0432.CCR-19-1834`.

---

## 5. Actionable Cross-Check Checklist for `@verifier`

1. **AlphaGenome 2 ToS Ban on Training Downstream ML Models:** Flag any proposal in `02_alphagenome2_and_fms.md` or `03_datasets_and_architectures.md` that uses AlphaGenome 2 API or AlphaGenome Atlas outputs as **training labels, distillation targets, or precomputed training features** — `google-deepmind/alphagenome` `README.md` and `faqs.md` explicitly forbid training other ML models on AlphaGenome API/Atlas outputs! AlphaGenome 2 (`v2.0.0`, released Nov 11, 2026) should be used **strictly at inference/evaluation time** (e.g., scoring patient/cell-line non-coding variants or conditioning on Bradbury Lab scRNA-seq via `cell_conditioning=` with `<=256` cells to inspect enhancer-to-gene wiring of ferroptosis loci).
2. **`Tahoe-100M` Compound Ground Truth:** Flag any claim in `03_datasets_and_architectures.md` that `Tahoe-100M` contains `Erastin`, `RSL3`, `ML210`, `ML162`, `FIN56`, `IKE`, `Sulfasalazine`, `Sorafenib`, `Ferrostatin-1`, `Liproxstatin-1`, or `Deferoxamine` — **none of those are in `Tahoe-100M`'s 379-drug library** (`metadata/drug_metadata.parquet`). Instead, `Tahoe-100M` (`95,624,334` cells, `CC0-1.0`) provides 22 clinical redox/lipid/ferroptosis-pathway drugs (`Artesunate`, `Dihydroartemisinin`, `Auranofin`, `Dimethyl fumarate`, `Pravastatin`, `Pitavastatin`, `Vitamin K4`, `Menadione`, `Ciclopirox`, `Zileuton`, `Malotilate`, `Rosiglitazone`, `Pioglitazone`, `Altretamine`, `Dexamethasone`, `Regorafenib`, `Bortezomib`, `Ixazomib`, `Panobinostat`, `Belinostat`, `Lapatinib`, `Neratinib`, plus 8 `mTOR` inhibitors), and must be paired with **CTRPv2 / PRISM / LINCS L1000** for direct `Erastin`/`RSL3`/`ML210`/`ML162`/`FIN56` pharmacology.
3. **FerrDb V3 Exact Counts & Download Format:** Ensure `03_datasets_and_architectures.md` cites **FerrDb V3** (`PMID: 41171133`, `DOI: 10.1093/nar/gkaf1119`, *NAR* Jan 2026) with exact ferroptosis counts (**734 drivers, 1,021 suppressors, 59 markers, 143 unclassified genes, 563 inducers, 510 inhibitors, 437 disease associations** across 22 RCDs) downloaded as **CSV/TXT** via `POST https://data.zhounan.org/pm2505/api/download/data/` (not JSON, which is V1-only).
4. **Bradbury Lab 2025–2026 Papers & Experimental Alignment:** Ensure `01_literature_and_bradbury_lab.md` and `03_datasets_and_architectures.md` include Dr. Bradbury's three landmark **2025–2026 papers** (`Cancer Research` 2026 PSMA-C' dot TLR-ferroptosis & immunometabolism in *MYC* prostate cancer `PMID: 42290280`; `Developmental Cell` 2026 lysosome rupture & GSH-gated single-cell vs. propagative ferroptosis `PMID: 41709461`; and `Nature Nanotechnology` 2025/2026 self-therapeutic C' dots activating STING/IL-6/PD-L1 & IFN immunity in melanoma `PMID: 41461940`), alongside her current primary title (**Professor of Pharmacology**, Weill Cornell Medicine, 2026–present) and active grant **`NCI R01 CA253658-05`**!
