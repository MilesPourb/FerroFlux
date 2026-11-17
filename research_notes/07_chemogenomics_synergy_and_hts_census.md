# 07 — Live Chemogenomics, High-Throughput BioAssay, Drug-Combination Synergy, and Morphological Perturbation Census

**Verification Date:** November 13, 2026  
**Verification Node:** Purdue Anvil HPC (`a076.anvil.rcac.purdue.edu`)  
**Verification Method:** Direct live HTTPS/REST/FTP/S3 verification (`safe_fetch` with strict zero-CAPTCHA/zero-bypass enforcement) across EBI ChEMBL 37, NCBI PubChem PUG REST & Entrez `pcassay`, Wellcome Sanger FTP & Cell Model Passports, NCI DTP / CellMiner / NCI-ALMANAC, Zenodo (DrugComb `v1.5`/`v1.4`), AWS Cell Painting Gallery (`cpg0016-jump`), and HuggingFace (`recursionpharma/rxrx3-core`).  
**Companion Manifest:** [pubchem_chembl_ferroptosis_assays.tsv](file:///anvil/projects/x-cis261485/Ferroptosis_ML/data_manifests/pubchem_chembl_ferroptosis_assays.tsv)

---

## 1. Executive Summary & Live Portal Status Audit (November 2026)

To ground **FerroFlux-FM** in quantitative small-molecule bioactivities, high-throughput phenotypic ferroptosis counter-screens, pairwise drug-combination synergy matrices, trace-metal/metabolomic cell-line profiles, and high-content morphological perturbation embeddings, we audited every major public chemogenomics and HTS repository live from Purdue Anvil.

### Critical Live Infrastructure Findings (Preventing Broken Ingestion Pipelines)
1. **EBI ChEMBL is now at `ChEMBL_37` (released May 1, 2026):** Live query to `https://www.ebi.ac.uk/chembl/api/data/status.json` (`HTTP 200 OK`) confirms **`24,527,044` bioactivities**, **`2,921,148` distinct compounds**, **`18,552` targets**, and **`101,100` publications** (`chembl_37_sqlite.tar.gz` = `5.76 GB`; `chembl_37_postgresql.tar.gz` = `1.90 GB`). Notably, ChEMBL 37 already indexes **GPX4 PROTAC ternary complex targets** (`CRBN/GPX4` `CHEMBL5465208`, `VHL/GPX4` `CHEMBL5465231`, `BIRC2/GPX4` `CHEMBL6196147`), alongside `935` assays matching `"ferroptosis"`.
2. **Legacy DrugComb & DrugCombDB Web Portals Are Offline — Use Official Zenodo Mirrors:**
   - `https://drugcomb.fimm.fi/` → `Connection refused`
   - `https://drugcomb.org/` → `Connection refused`
   - `http://drugcombdb.denglab.org/` → `TimeoutError`
   - **Live Canonical Replacement (`HTTP 200 OK`):** Official **DrugComb v1.5** (`summary_v_1_5.csv`, `1,421.69 MB`, `739,964` combination blocks across `8,397` drugs and `2,320` cell lines) is live at **`https://zenodo.org/records/15235991`** (`DOI: 10.1093/nar/gkab438`, `PMID: 34060634`), and raw dose-response matrices + drug/cell SMILES/InChIKey tables (`drugcomb_data_v1.4.csv` `2,008.12 MB`, `DrugComb_drug_identifiers.xlsx` `5.86 MB`) are live at **`https://zenodo.org/records/18449193`** (updated `2026-02-01`).
3. **Sanger GDSC & Project SCORE Legacy URLs Return `410`/`404` — Live Endpoints Verified:**
   - `https://www.cancerrxgene.org/downloads/bulk_download` → `HTTP 410 Gone`; live Release 8.4 CSVs (`GDSC1` `333,161` curves; `GDSC2` `242,036` curves) are served directly at **`https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/`** (`HTTP 200 OK`).
   - `https://score.depmap.sanger.ac.uk/` → `HTTP 404 Not Found`; live **Project SCORE v2** archives (`Project_Score2_fitness_scores_Sanger_v2_Broad_21Q2_20250624.zip` `198.87 MB`, `Organoids_fitness_scores_20260723.zip` `21.46 MB`, `Project_score_raw_counts.zip` `2.85 GB`) are served at **`https://cog.sanger.ac.uk/cmp/download/`** (`HTTP 200 OK`).
4. **Massive Ultra-HTS (>300k–575k Compound) Quantitative Screens Exist in PubChem BioAssay:** Excluding ChEMBL single-compound literature imports (`NOT "ChEMBL"[SourceName]`) uncovered **198 primary deposited HTS/qHTS screens**, headlined by `575,118`-compound `ALOX12` qHTS (`AID 1671190`), `407,090`-compound `NFE2L2` activator qHTS (`AID 624171`), `392,548`-compound selenoprotein `TXNRD1` qHTS (`AID 588453`), `364,851`-compound `HT-1080` synthetic-lethality qHTS (`AID 686970`/`686971`), `303,344`-compound Broad MLPCN `BJeLR` RAS-selective lethality screen (`AID 1554` — discovery campaign of `ML162` & `ML210`), `11,046`-compound recombinant selenoprotein `GPX4` qHTS (`AID 1845192`), and Tox21 `ARE-bla` (`NFE2L2`) qHTS across `9,305–9,667` chemicals (`AID 743219`, `1919970`, `2283544`).
5. **Near-Complete Genetic & Chemical Coverage in JUMP-Cell Painting (`cpg0016-jump`):** Exact `InChIKey` connectivity matching against `115,796` JUMP compounds and gene-symbol matching against `7,977` CRISPR knockouts + `15,132` ORFs confirms **36/38 (`94.7%`) canonical ferroptosis genes** (`33/38` CRISPR, `26/38` ORF) and **34/46 (`73.9%`) canonical ferroptosis probes/drugs** (`Erastin`, `IKE`, `ML210`, `FIN56`, `CIL56`, `Ferrostatin-1`, `Liproxstatin-1`, `UAMC-3203`, all 4 iron chelators, `Sorafenib`, `Sulfasalazine`, `Artesunate`, `Brequinar`, `BSO`, `APR-246`, all 6 statins, `Zileuton`) have matched morphological profiles in `s3://cellpainting-gallery/cpg0016-jump/`.

### Master Live Portal & Bulk Archive Status Table

| Resource | Release / Version | Live Verified Endpoint (Nov 2026) | HTTP Status | Size | License / Access |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EBI ChEMBL** | `ChEMBL_37` (`2026-05-01`) | `https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_37_sqlite.tar.gz` | `200 OK` | `5.76 GB` (`1.90 GB` PG; `314 MB` HDF5) | CC-BY-SA 3.0 (Open) |
| **PubChem BioAssay** | Live (`2026-11-12`) | `https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/<AID>/CSV` | `200 OK` | Per-AID CSV (`<1 MB`–`180 MB`) | Public Domain (Open) |
| **NCI-60 DTP GI50 / Raw** | Current (`v20/v21`) | `https://wiki.nci.nih.gov/download/attachments/147193864/GI50.zip?api=v2` | `200 OK` | `39.21 MB` (`DOSERESP.zip` `349.29 MB`) | Public Domain (Open) |
| **NCI CellMiner Omics** | Current (`58` zips) | `https://discover.nci.nih.gov/cellminer/loadDownload.do` | `200 OK` | `58` archives (`ZSCORE`, `SWATH`, `RNA-seq`) | Public Domain (Open) |
| **NCI-ALMANAC Combos** | Nov 2017 (`101` NSCs) | `https://wiki.nci.nih.gov/download/attachments/338237347/ComboDrugGrowth_Nov2017.zip?api=v2` | `200 OK` | `85.97 MB` (`ORCESTRA` RDS `438.5 MB`) | Public Domain (Open) |
| **DrugComb v1.5 Summary** | `v1.5` (`739,964` blocks) | `https://zenodo.org/api/records/15235991/files/summary_v_1_5.csv/content` | `200 OK` | `1,421.69 MB` | CC-BY 4.0 (Open; `drugcomb.org` dead) |
| **DrugComb Raw & IDs** | `v1.4` (`2026-02-01` rev) | `https://zenodo.org/records/18449193` (`drugcomb_data_v1.4.csv` + `.xlsx` IDs) | `200 OK` | `2,008.12 MB` + `5.86 MB` | CC-BY 4.0 (Open) |
| **Sanger GDSC1 & GDSC2** | Rel `8.4` (`24Jul22`) | `https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/` | `200 OK` | `52 MB` + `38 MB` fitted (`2.74 GB` raw) | Open Academic (`cancerrxgene` bulk `410`) |
| **Sanger Project SCORE** | `Score2` (`2025`/`2026`) | `https://cog.sanger.ac.uk/cmp/download/Project_Score2_fitness_scores_Sanger_v2_Broad_21Q2_20250624.zip` | `200 OK` | `198.87 MB` (`Organoids` `21.46 MB`) | CC-BY 4.0 (`score.depmap` `404`) |
| **JUMP-Cell Painting** | `cpg0016-jump` | `s3://cellpainting-gallery/cpg0016-jump/source_4/workspace/profiles/` | `200 OK` | `~13.23 MB/plate` (`~33.4 GB` / `2,525` parquet) | CC0 Public Domain (Open) |
| **Recursion RxRx3-core** | `rxrx3-core` (`2024`) | `https://huggingface.co/datasets/recursionpharma/rxrx3-core` | `200 OK` | `17.39 GB` (`OpenPhenom` parquet `<1 GB`) | CC-BY-NC-SA 4.0 (`gated: False`) |

---

## 2. PubChem BioAssay (`AID`) Quantitative HTS & Counter-Screen Census

Querying NCBI Entrez `db=pcassay` and PubChem PUG REST (`assay/aid/.../summary/JSON`) identified `1,736` total ferroptosis/target-related BioAssays (`872` non-trivial screening/panel assays with `≥10` tested non-ChEMBL substances or `≥25` tested compounds), from which **306 verified rows (`12` columns)** are exported in [pubchem_chembl_ferroptosis_assays.tsv](file:///anvil/projects/x-cis261485/Ferroptosis_ML/data_manifests/pubchem_chembl_ferroptosis_assays.tsv): **`220` PubChem BioAssays** (`115` direct target qHTS, `46` Tox21/NCGC `NFE2L2`/`ARE-bla` qHTS, `32` `BJeLR`/RAS synthetic-lethality HTS, `21` NCI-60 renal/melanoma `GI50` panels, `6` phenotypic ferroptosis/`HT-1080` qHTS), **`50` EBI ChEMBL 37 target/complex bioactivity rollups**, and **`36` representative ChEMBL 37 multi-compound & sparse-target SAR assays**.

### Landmark Ultra-High-Throughput (`10,000–575,000+` Compound) PubChem BioAssays

| PubChem AID | Depositor | Target / Cell System | Readout Type | Tested Compounds | Active Hits | Biological Relevance to FerroFlux-FM |
| :--- | :--- | :--- | :--- | ---: | ---: | :--- |
| **`AID 1671190`** | NCGC (NIH NCATS) | Human `ALOX12` (12S-LOX) | Confirmatory qHTS | **575,118** | **6,069** | Largest public lipoxygenase qHTS (`Round 2`); directly models p53-ALOX12 lipid peroxidation inhibition. |
| **`AID 624171`** | NCGC (NIH NCATS) | Human `NFE2L2` (Nrf2) | Confirmatory qHTS | **407,090** | **1,243** | Ultra-scale qHTS for small-molecule Nrf2 activators driving anti-ferroptotic transcription (`SLC7A11`, `FTH1`, `GCLC`, `NQO1`). |
| **`AID 588453`** | NCGC (NIH NCATS) | Mammalian `TXNRD1` (TrxR1) | Confirmatory qHTS | **392,548** | **4,037** | Direct selenoprotein `TXNRD1` inhibitor qHTS (plus `AID 588456` substrate screen across `391,276` compounds, `56` actives); captures `Auranofin`/`TRi-1` chemotypes and selenocysteine electrophiles. |
| **`AID 504444`** | NCGC (NIH NCATS) | Human `NFE2L2` (Nrf2) | Confirmatory qHTS | **364,195** | **7,472** | Quantitative HTS for **Nrf2 pathway inhibitors** (paired with cytotoxicity counter-screen `AID 504648`, `387,005` tested, `611` cytotoxic) — ideal for discovering Nrf2-addicted cancer ferroptosis sensitizers. |
| **`AID 686970`** / **`686971`** | NCGC (NIH NCATS) | `HT-1080-NT` vs `HT-1080-IDH1(R132H)` | Confirmatory qHTS | **364,851** | **5,979 / 7,452** | Massive multi-concentration viability qHTS in canonical ferroptosis model line **`HT-1080`** ($\pm$ oncogenic 2-HG, a known ferroptosis sensitizer via GPX4/SIRT1 modulation). |
| **`AID 504523`** | Broad Institute | `KEAP1–NFE2L2` PPI | Primary HTS (FP) | **337,116** | **489** | Direct fluorescence polarization screen for non-electrophilic `KEAP1–NRF2` PPI disruptors (confirmatory dose-response in `AID 651798`/`651801`/`651807`, `92` compounds). |
| **`AID 1554`** | Broad Institute (MLPCN) | **`BJeLR`** (`HRAS-G12V`) Viability | Primary MLPCN HTS | **303,344** | **517** | **Foundational Schreiber/Stockwell Broad MLPCN RAS-selective lethality campaign** in engineered `BJeLR` fibroblasts that yielded covalent GPX4 inhibitors **`ML162`** and **`ML210`**! Paired with `BJeLR` confirmatory/counter-screens `AID 2275` (`30,036` tested, `193` active), `AID 1933` (`BJ-TERT`), `AID 1934` (`DRD`), `AID 1935` (`BJ-TERT-LT-ST`), `AID 1936` (`BJeLR` dose confirmation), and project summary `AID 1674`. |
| **`AID 1452`** | NCGC (NIH NCATS) | Human `ALOX12` (12-hLO) | Confirmatory qHTS | **153,607** | **187** | First-generation quantitative concentration-response screen for selective `ALOX12` inhibitors (`AID 2162` confirmation: `95` tested, `25` active). |
| **`AID 887`** | NCGC (NIH NCATS) | Human `ALOX15` (15-hLO-1) | Confirmatory qHTS | **74,290** | **1,054** | Large-scale qHTS for inhibitors of canonical ferroptosis lipoxygenase `ALOX15` (PEBP1–15LOX complex partner). |
| **`AID 1845192`** | NCGC (NIH NCATS) | **Recombinant Selenoprotein `GPX4`** | Confirmatory qHTS | **11,046** | **258** | **Direct biochemical enzymatic qHTS against human selenoprotein GPX4** across `11,046` compounds — critical training anchor separating direct GPX4 catalytic inhibitors from indirect GSH-depleting or iron-redox compounds! |
| **`AID 743219`** (`743202`/`743203`) | Tox21 (NCATS/NTP/EPA) | HepG2 **`ARE-bla`** (`NFE2L2`) | Summary & qHTS | **9,305** | **1,147** | Official **Tox21 10K library quantitative HTS** for `NFE2L2`/antioxidant response element (`ARE-bla`) activation with matched cytotoxicity counter-screen; extended/re-curated in **`AID 1919970`** (`9,667` tested, `1,038` active) and NICEATM ICE **`AID 2283544`** (`9,315` tested, `1,673` ratio actives) + viability **`AID 2283569`** (`802` cytotoxic). |

> **Note on Academic Phenotypic Screens (`Stockwell`, `Conrad`, `Dixon`):** While Broad/MLPCN `BJeLR` (`AID 1554`, `303,344` compounds) and NCGC `GPX4` (`AID 1845192`, `11,046` compounds) are deposited in full in PubChem, smaller academic library screens — such as Conrad's `Gpx4-/-` MEF screen that discovered `Liproxstatin-1` (Friedmann Angeli et al. *Nat Cell Biol* 2014, `PMID: 25402683`, `DOI: 10.1038/ncb3064`) and Conrad/Olzmann's `10,000`-compound `GPX4-KO + FSP1` rescue screen that discovered `iFSP1` (Doll et al. *Nature* 2019, `PMID: 31634900`, `DOI: 10.1038/s41586-019-1705-2`) — were not deposited as full 10k-compound primary PubChem `AID`s, though their hit optimization series ARE captured in **ChEMBL 37** (`935` ferroptosis assays) and recent BindingDB/patent depositions (`AID 2282745` & `AID 2060659` for `EBP`/`DHCR7`, `AID 2061113` for `DHODH`, `AID 2060460` for `12/15-LOX`).

---

## 3. EBI ChEMBL 37 (`2026-05-01`) Quantitative Target Bioactivity & SAR Census

We queried EBI ChEMBL 37 live across 30 canonical ferroptosis driver, suppressor, lipid-remodeling, radical-trapping, and iron-handling proteins (`Homo sapiens` UniProt accessions), counting total assays, total bioactivities, and standardized **`pChEMBL` (`-log10 M` `IC50`/`EC50`/`Ki`/`Kd`/`AC50`)** quantitative records:

| Gene Symbol | UniProt | ChEMBL Target ID | Target Type | Assays | Total Activities | `pChEMBL` Quant. | Key Chemotypes & Modeling Notes |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| **`NFE2L2`** | `Q16236` | `CHEMBL1075094` | SINGLE PROTEIN | 579 | 96,441 | **654** | Includes qHTS imports + quantitative ARE `EC50` series |
| **`KEAP1 / NFE2L2`** | `Q14145`/`Q16236` | `CHEMBL3038498` | PPI COMPLEX | 280 | 4,799 | **1,372** | Rich non-covalent & covalent `KEAP1–NRF2` PPI `IC50`/`Kd` series |
| **`KEAP1`** | `Q14145` | `CHEMBL2069156` | SINGLE PROTEIN | 246 | 1,835 | **581** | Kelch-domain binders, Bardoxolone, DMF/itaconate analogs |
| **`ALOX15`** | `P16050` | `CHEMBL2903` | SINGLE PROTEIN | 242 | 7,353 | **5,926** | Deepest quantitative ferroptosis enzyme SAR (`PD146176`, `ML351`, etc.) |
| **`ALOX5`** | `P09917` | `CHEMBL215` | SINGLE PROTEIN | 837 | 7,092 | **3,563** | `Zileuton` and 3,500+ quantitative `pChEMBL` inhibitors (`+ FLAP` `CHEMBL2111402`: `37`) |
| **`ALOX12`** | `P18054` | `CHEMBL3687` | SINGLE PROTEIN | 106 | 3,506 | **2,847** | `ML355` and selective 12-LOX inhibitors |
| **`DHODH`** | `Q02127` | `CHEMBL1966` | SINGLE PROTEIN | 297 | 6,675 | **2,558** | `Brequinar`, `Teriflunomide`, `BAY 2402234` mitochondrial CoQ2H2 axis |
| **`HMGCR`** | `P04035` | `CHEMBL402` | SINGLE PROTEIN | 140 | 2,495 | **367** | Statins (`mevalonate -> IPP/tRNA-Sec & CoQ10`) + `CRBN`/`VHL` PROTACs (`13` acts) |
| **`GPX4`** | `P36969` | `CHEMBL4295754` | SINGLE PROTEIN | 144 | 502 | **132** | `RSL3`, `ML162`, `ML210`, `JKE-1674`, chloroacetamides, propiolamides |
| **`GPX4` PROTACs** | `P36969` | `CHEMBL5465208` / `5465231` / `6196147` | PPI (`CRBN` / `VHL` / `BIRC2`) | 32 | 93 | **4** | **2024–2026 GPX4 degrader ternary complexes** (`DC50` / ternary `Kd`)! |
| **`GSR`** | `P00390` | `CHEMBL2755` | SINGLE PROTEIN | 114 | 482 | **114** | Glutathione disulfide reductase inhibitors |
| **`TXNRD1`** | `Q16881` | `CHEMBL1927` (`+2096978`) | SINGLE / FAMILY | 131 | 540 | **199** | `Auranofin`, `TRi-1`, `TRi-2` (`96` single-protein + `103` family `pChEMBL`) |
| **`SLC7A11`** | `Q9UPY5` | `CHEMBL1075149` | SINGLE PROTEIN | 72 | 224 | **49** | `Erastin`, `IKE`, `Sulfasalazine`, `Sorafenib`, cystine uptake `IC50` |
| **`EBP`** | `Q15125` | `CHEMBL4931` (`+612409`) | SINGLE / AEBS | 56 | 269 | **100** | 7-DHC accumulation axis (`76` EBP + `24` AEBS `EBP/DHCR7` complex `pChEMBL`) |
| **`DHCR7`** | `Q9UBM7` | `CHEMBL2169735` (`+612409`) | SINGLE / AEBS | 43 | 163 | **39** | **Endogenous radical-trapping sterol `7-DHC` regulator** (`AY9944`, `Tamoxifen`, antipsychotics) |
| **`POR`** | `P16435` | `CHEMBL2169731` | SINGLE PROTEIN | 21 | 114 | **13** | ER cytochrome P450 oxidoreductase (`H2O2`/lipid peroxidation initiator) |
| **`SQLE`** | `Q14534` | `CHEMBL3592` | SINGLE PROTEIN | 23 | 100 | **67** | Squalene epoxidase inhibitors (squalene accumulation suppresses ferroptosis) |
| **`HMOX1`** | `P09601` | `CHEMBL2823` | SINGLE PROTEIN | 23 | 66 | **23** | Heme degradation / labile $\text{Fe}^{2+}$ release axis |
| **`ACSL3` / `ACSL4`** | `O95573` / `O60488` | `CHEMBL4680023` / `4680022` | SINGLE PROTEIN | 4 | 26 | **7** | Sparse direct target annotation (`5` `ACSL3`, `2` `ACSL4` `pChEMBL`; thiazolidinediones) |
| **`GCLC` / `GSS`** | `P48506` / `P48637` | `CHEMBL4055` / `6066521` | SINGLE PROTEIN | 15 | 16 | **6** | `BSO` and GSH synthesis inhibitors (`4` `GCLC`, `2` `GSS` `pChEMBL`) |
| **`SLC3A2` / `TFRC` / `FTH1`** | `P08195` / `P02786` / `P02794` | `CHEMBL6066973` / `3712860` / `6066220` | SINGLE PROTEIN | 4 | 11 | **11** | Emerging chemical proteomics / chemogenomic liganded sites |
| **`AIFM2` (`FSP1`)** | `Q9BRQ8` | `CHEMBL6067200` | SINGLE PROTEIN | 1 | 2 | **2** | **CRITICAL CAVEAT:** Only Kinobead pulldown (`CHEMBL5650866`) is mapped to `CHEMBL6067200`; `iFSP1`/`icFSP1`/`FSEN1` series reside under phenotypic/unassigned ChEMBL assays! |
| **`LPCAT3` / `MBOAT1`** | `Q6P1A2` / `Q6ZNC8` | `CHEMBL6066416` / `6067162` | SINGLE PROTEIN | 2 | 4 | **4** | Very sparse direct target SAR (`2` `pChEMBL` each; `GCH1` `P30793` has `0` human ChEMBL rows). |

### Key Chemoinformatics Insight for `FerroFlux-FM` Data Ingestion
Naive target-ID filtering (`target_chembl_id = CHEMBL6067200` for `FSP1`/`AIFM2`) misses >95% of medicinal chemistry FSP1 and radical-trapping antioxidant (`RTA`) SAR series (`Ferrostatin-1`, `Liproxstatin-1`, `UAMC-3203`, `iFSP1`, `icFSP1`, `FSEN1`), because ChEMBL curates cell-based ferroptosis rescue (`HT-1080 + RSL3/Erastin`) and unmapped biochemical assays as `assay_type = 'F'` (Functional/Phenotypic). Querying `assay_description__icontains=ferroptosis` (`935` ChEMBL assays) and `FSP1` recovers these exact multi-compound `EC50`/`IC50` series (included in Part C of `pubchem_chembl_ferroptosis_assays.tsv`).

---

## 4. NCI-60 DTP Pharmacology + Metabolomics, Lipidomics & Trace-Metal (`ICP-MS`) Audit

### 4.1 Live NCI-60 & CellMiner Pharmacogenomic Files (`HTTP 200 OK`)
- **NCI DTP Bulk Pharmacology (`https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/147193864/NCI-60+Growth+Inhibition+Data`):**
  - `public_nscs.csv` (`2.59 MB`): **`334,952` public NSC compound identifiers** (`>55,000` tested in full 5-concentration 48h SRB dose-response across 60 human tumor lines spanning 9 tissues, including renal `786-0`, `A498`, `ACHN`, `CAKI-1`, `RXF 393`, `SN12C`, `TK-10`, `UO-31` and melanoma `LOX IMVI`, `MALME-3M`, `M14`, `SK-MEL-2`, `SK-MEL-5`, `SK-MEL-28`, `UACC-62`, `UACC-257`).
  - `GI50.zip` (`39.21 MB`), `DOSERESP.zip` (`349.29 MB`), `TGI.zip`, `LC50.zip`, `IC50.zip`, plus updated 384-well `HTS384_GI50.csv` / `HTS384_DOSERESP.csv`.
- **NCI CellMiner Processed Multi-Omics (`https://discover.nci.nih.gov/cellminer/loadDownload.do`, Reinhold et al. *Cancer Res* 2012, `PMID: 22802077`, `DOI: 10.1158/0008-5472.CAN-12-1370`):**
  - `58` directly downloadable `.zip` bundles (`HTTP 200`), including `DTP_NCI60_ZSCORE.zip`, `DTP_HTS384_NCI60_ZSCORE.zip`, `DTP_NCI60_ALMANAC_COMBO_SCORE.zip`, `nci60_RNA__RNA_seq_composite_expression.zip`, `nci60_Protein__SWATH_(Mass_spectrometry)_Protein.zip`, `nci60_DNA__Exome_Seq_Protein_function_affecting.zip`, and `IonTransporter_OSU.zip` (custom oligonucleotide/RT-PCR array quantifying **iron, copper, zinc, and solute carrier transporters** across NCI-60!).

### 4.2 Exact Status of NCI-60 Metabolomics, Lipidomics & ICP-MS Trace Metals
- **Extracellular Consumption & Release (`CORE`) Metabolomics (Public, Verified):** Jain et al., *Science* 2012 (`PMID: 22628656`, `DOI: 10.1126/science.1218595`) quantified LC-MS/MS consumption and release rates of **219 media metabolites** across all 60 NCI-60 cell lines in biological duplicates — directly measuring **cystine/cysteine, glutamate, glycine, glutathione precursors, serine, and polyunsaturated fatty acid uptake**!
- **Intracellular Untargeted Metabolomics & Phospholipid Ions (Public, Verified):** Ortmayr et al., *Nat Commun* 2019 (`PMID: 31015463`, `DOI: 10.1038/s41467-019-09695-9`) profiled intracellular extracts across **54 adherent NCI-60 cell lines** by FIA-TOF-MS (`>2,000` deprotonated/protonated `m/z` ions annotated against HMDB/KEGG, covering GSH/GSSG, CoQ precursors, sterols, and glycerophospholipids; complementary to CCLE LC-MS polar/lipid profiling of **225 metabolites across 928 lines** in Li/Ghandi et al. *Nature* / *Nat Med* 2019, `PMID: 31068700`).
- **ICP-MS Trace Metals (`Fe`, `Cu`, `Zn`, `Mn`, `Se`) Reality Check:** CellMiner does **NOT** host a native pan-NCI-60 total elemental ICP-MS `.zip` table (its metal-related file `IonTransporter_OSU.zip` is mRNA expression of metal/ion transporters `TFRC`, `SLC11A2`/`DMT1`, `SLC40A1`/`FPN1`, `STEAP3`, `ATP7A/B`, `SLC31A1`/`CTR1`). Published NCI-60 SEC-ICP-MS / ICP-MS studies (`PMID: 30237619` and lineage-specific breast/ovarian/renal metallomics panels) only cover subsets of lines or exogenous metallodrugs (`cisplatin`, `oxaliplatin`, `auranofin`). Therefore, **FerroFlux-FM should NOT assume a dense public 60-line × `[Fe2+, Cu+, Zn2+, Se]` ICP-MS matrix exists**; instead, labile iron pool (`LIP`) and selenoprotein flux must be modeled as mechanistic latent state variables constrained by `TFRC`/`FTH1`/`FTL`/`NCOA4`/`SLC40A1`/`HMOX1`/`SEPSECS`/`LRP8` expression + iron-chelator (`DFO`/`Deferasirox`/`Ciclopirox`) and `Erastin`/`RSL3` pharmacological sensitivity, ready for fine-tuning on Bradbury lab's quantitative nanoparticle/radiotracer/metallomics readouts!

---

## 5. Pairwise Drug-Combination Synergy Census (`DrugComb`, `NCI-ALMANAC`, `O'Neil`, `AZ-Sanger`, `Forcina`)

### 5.1 Major Public Combination Synergy Benchmarks
1. **DrugComb `v1.5` (`Zenodo 15235991`, `summary_v_1_5.csv` `1.42 GB`) & `v1.4` (`Zenodo 18449193` raw dose matrices `2.01 GB` + `Zenodo 11102665` summary `193.18 MB`):** Harmonizes **`739,964` drug combination blocks** (`466,033` combo blocks + `709,187` single-agent blocks in `v1.4`) across **`8,396`–`8,397` unique compounds** and **`2,320` cell lines** from `34` studies (`ALMANAC`, `ONEIL`, `ASTRAZENECA`, `FORCINA`, `CLOUD`, `FLOATING`, etc.) with standardized **ZIP, Bliss, Loewe, HSA**, and **CSS (Combination Sensitivity Score)** metrics (`PMID: 34060634`, `DOI: 10.1093/nar/gkab438`; `PMID: 31066443`, `DOI: 10.1093/nar/gkz337`).
2. **NCI-ALMANAC (`ComboDrugGrowth_Nov2017.zip` `85.97 MB`, Holbeck et al. *Cancer Res* 2017, `PMID: 28446549`, `DOI: 10.1158/0008-5472.CAN-17-0489`):** **`101` unique NSC FDA-approved oncology drugs** tested pairwise (`>5,200` pairs × `60` NCI-60 cell lines ≈ `304,549` combination experiments in $3\times 3$ or $5\times 3$ matrices). Direct NSC audit of `ComboCompoundNames_all.txt` confirms inclusion of **`Sorafenib` (`NSC 747971`), `Vorinostat` (`NSC 701852`), `Bortezomib` (`NSC 681239`), `Altretamine` (`NSC 13875`, GPX4 inhibitor), `Arsenic trioxide` (`NSC 92859`), `Tretinoin` (`NSC 122758`), `Sirolimus` (`NSC 226080`), `Everolimus` (`NSC 733504`), `Dasatinib` (`NSC 732517`), `Lapatinib` (`NSC 745750`), `Celecoxib` (`NSC 719627`), `Tamoxifen` (`NSC 180973`), `Raloxifene` (`NSC 747974`, anti-ferroptotic RTA!), `Cisplatin`, `Oxaliplatin`, `Gemcitabine`, `Paclitaxel`, `Doxorubicin`, `Vemurafenib`, `Sunitinib`, `Imatinib`, `Gefitinib`, `Erlotinib`**.
3. **Merck / O'Neil Screen (O'Neil et al. *Mol Cancer Ther* 2016, `PMID: 26983881`, `DOI: 10.1158/1535-7163.MCT-15-0843`):** **`38` oncology drugs** (`583` pairs × `39` cell lines × $4\times 4$ matrix × 4 replicates = **`22,737` experiments**, fully integrated in DrugComb), featuring deep synergy matrices for **`Sorafenib`, `Vorinostat`, `Bortezomib`, `Dexamethasone`, `Lapatinib`, `Dasatinib`, `Rapamycin`/`MK-8669`, `Olaparib`, `Gemcitabine`, `Oxaliplatin`, `Paclitaxel`, `Doxorubicin`**.
4. **AstraZeneca-Sanger DREAM Challenge (Menden et al. *Nat Commun* 2019, `PMID: 31209238`, `DOI: 10.1038/s41467-019-09799-2`):** **`118` targeted drugs** (`910` pairs across `85` GDSC/CMP molecularly profiled cancer cell lines = **`11,576` $6\times 6$ dose-matrix experiments**, integrated in DrugComb `v1.5`), enriching RTK/PI3K/mTOR/MEK/BCL2/DDR combinations.
5. **Forcina / Dixon Kinetic Death Modifier Screen (Forcina et al. *Cell Syst* 2017, `PMID: 28601558`, `DOI: 10.1016/j.cels.2017.05.002`):** Screened **`1,833` bioactive compounds** using time-lapse STACK imaging in `T98G` glioblastoma and `U-2 OS` cells (integrated as study `FORCINA` in DrugComb `v1.5`), providing kinetic cell-death modulatory interactions directly relevant to ferroptosis sensitization and suppression.

### 5.2 Exact Pairwise Combination Counts Verified Directly from Live DrugComb (`Zenodo 18449193` + `11102665`)

Parsing all `8,396` entries in `DrugComb_drug_identifiers.xlsx` by 14-character `InChIKey` connectivity matched **60 ferroptosis probes, redox modulators, and oncology synergy partners**, and scanning all `1,175,220` rows (`466,033` combo blocks) of `summary_table_v1.4.csv` yielded exact empirical combination availability:

| Compound | Role in Ferroptosis / Redox | DrugComb Combo Blocks | Unique Partner Drugs | Cell Lines | Notable Verified Synergy / Antagonism Pairs (`max ZIP` / `max Bliss`) |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **`Sorafenib`** | System $x_c^-$ / multikinase ferroptosis inducer | **11,967** | **386** | **96** | **`+ Bortezomib`** (`203` blocks, `82` lines, `ZIP=32.22`); **`+ Lapatinib`** (`212` blocks, `ZIP=29.78`); **`+ Paclitaxel`** (`206` blocks, `ZIP=27.49`); **`+ Dexamethasone`** (`156` blocks, `ZIP=22.07`); **`+ Dasatinib`** (`214` blocks, `ZIP=21.76`); **`+ Altretamine`** (`60` lines, `ZIP=9.15`); **Antagonized by `Zileuton` (`ZIP=-30.04`) & `Rosiglitazone` (`ZIP=-19.58`)!** |
| **`Dasatinib`** | SRC/RTK inhibitor (EMT/mesenchymal state) | **11,967** | **210** | **96** | `+ Dexamethasone` (`ZIP=44.86`); `+ Lapatinib` (`ZIP=23.99`); `+ Sorafenib` (`ZIP=21.76`) |
| **`Lapatinib`** | EGFR/HER2 inhibitor (autophagy/ferroptosis) | **11,862** | **130** | **94** | `+ Paclitaxel` (`ZIP=34.42`); `+ Sorafenib` (`ZIP=29.78`); `+ Bortezomib` (`ZIP=23.49`) |
| **`Bortezomib`** | Proteasome inhibitor (NRF2/UPR/GSH stress) | **11,436** | **413** | **98** | `+ Sorafenib` (`ZIP=32.22`); `+ Oxaliplatin` (`ZIP=25.74`); **`+ Disulfiram`** (`ZIP=10.01`, `Bliss=14.38`) |
| **`Altretamine`** | Clinical alkylator / **GPX4 inhibitor** (`NSC 13875`) | **6,473** | **363** | **65** | `+ Paclitaxel` (`60` lines, `ZIP=11.06`); `+ Bortezomib` (`61` lines, `ZIP=10.50`); `+ Sorafenib` (`60` lines, `ZIP=9.15`) |
| **`Disulfiram`** | ALDH/NPL4/Cu-redox ferroptosis enhancer | **398** | **348** | **6** | **`+ Artesunate`** (`ZIP=13.10`, `Bliss=11.46`); **`+ Bortezomib`** (`ZIP=10.01`, `Bliss=14.38`) |
| **`Everolimus`** | mTORC1 inhibitor (GPX4 translation axis) | **337** | **323** | **3** | Broad oncology combo panel (`+ Sirolimus` `5` blocks, `+ Temsirolimus` `3` blocks in `v1.4`; expanded in `v1.5` ALMANAC/ONEIL) |
| **`Auranofin`** | **TXNRD1 inhibitor** | **318** | **313** | **5** | `313` partner screen across `5` lines |
| **`Dihydroartemisinin` / `Artesunate`** | Endoperoxide $\text{Fe}^{2+}$-Fenton ferroptosis inducers | **314 / 154** | **312 / 76** | **4 / 5** | **`Artesunate + Disulfiram`** (`ZIP=13.10`) |
| **`Zileuton` / `Rosiglitazone`** | **ALOX5 / ACSL4 anti-ferroptotic inhibitors** | **290 / 288** | **288 / 286** | **6 / 5** | **Mechanistic rescue antagonism** against `Sorafenib` (`ZIP=-30.04` and `-19.58` respectively!) |
| **`Withaferin A`** | Covalent **GPX4 + KEAP1** natural product inducer | **111** | **90** | **8** | `90` partner combinations across `8` lines |
| **`Statins` (`Simva/Fluva/Lova/Pita/Atorva/Prava`)** | HMGCR / CoQ10 / tRNA-Sec axis | **27 total** (`7/6/5/2/2/2`) | **2–7 each** | **2–5** | Sparse in `v1.4` (`27` blocks total); must leverage chemogenomic/morphological transfer! |
| **Tool Probes (`Sulfasalazine`, `Ciclopirox`, `Deferasirox`, `DMF`, `Idebenone`, `Menadione`, `BSO`, `APR-246`, `Erastin`, `RSL3`, `Brequinar`)** | Canonical ferroptosis probes | **2–7 each** (`Erastin=3`, `RSL3=2`) | **2–7 each** | **1–5** | **KEY MODELING REALITY:** Pure tool probes (`Erastin`, `RSL3`, `ML210`, `iFSP1`) have `<10` multi-dose matrix pairs in public oncology combo screens (`v1.4`), whereas `v1.5` adds `FORCINA` fixed-dose modifiers and `Sorafenib`/`Altretamine`/`Auranofin`/`Artesunate`/`Disulfiram` provide `>20,000` dense dose-matrix blocks! |

> **Architectural Implication for `FerroFlux-FM` Synergy Prediction:** Notice that our live DrugComb scan uncovered **mechanistic antagonism (`ZIP = -30.04` for `Sorafenib + Zileuton` and `-19.58` for `Sorafenib + Rosiglitazone`)** alongside **strong synergy (`ZIP = +32.22` for `Sorafenib + Bortezomib` and `+13.10` for `Artesunate + Disulfiram`)**! Because tool probes like `IKE`, `RSL3`, `ML210`, and `iFSP1` have sparse direct pairwise matrices in NCI-ALMANAC/O'Neil, a black-box drug-pair MLP will fail out-of-distribution on novel ferroptosis combinations. Grounding compounds onto **target/pathway flux nodes** (via ChEMBL 37 `pChEMBL` affinities + JUMP-CP morphological embeddings + CTRPv2/PRISM single-agent curves) enables zero-shot mechanistic synergy/rescue prediction!

---

## 6. Sanger `GDSC1` / `GDSC2` & `Project SCORE v2` Census

### 6.1 Sanger GDSC1 & GDSC2 (`Release 8.4`, `https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/`)
- **Scale:** `GDSC1` contains **`333,161` fitted dose-response curves** (`402` drugs × `970` cell lines); `GDSC2` (improved 1,536-well acoustic dispensing + CellTiter-Glo) contains **`242,036` fitted curves** (`295` drugs × `969` cell lines).
- **Verified Ferroptosis, Redox & Synergy Partner Coverage in GDSC:**
  - **Present with dense pan-cancer coverage (`732–966` cell lines):** `Sorafenib` (`GDSC1 ID=30`: `396` lines; `GDSC2 ID=1085`: `958` lines), `Elesclomol` (`GDSC1 ID=1031`: `921` lines; `GDSC2`: `225` lines), `PRIMA-1MET` / **`APR-246`** (`GDSC2 ID=1131`: `894` lines), `Piperlongumine` (ROS/GSTP1 electrophile, `GDSC1 ID=1243`: `911` lines), `BAY ACCi` (`ACACA` MUFA/PUFA lipogenesis inhibitor, `GDSC1 ID=1362`: `835` lines), `DMOG` (`HIF-PH` iron/2-OG dioxygenase inhibitor, `GDSC1 ID=165`: `876` lines), `GSK2606414` (`PERK` ER-stress inhibitor, `GDSC2 ID=1618`: `886` lines), `glutathione` (`GDSC2 ID=2439`: `732` lines), `alpha-lipoic acid` (`GDSC2 ID=2498`: `735` lines), `N-acetyl cysteine` (`GDSC2 ID=2499`: `735` lines), `Vorinostat` (`959` lines), `Bortezomib` (`957` lines), `Rapamycin` (`950` lines), `Temsirolimus` (`922` lines), `Dasatinib` (`962` lines), `Lapatinib` (`966` lines).
  - **Absent from GDSC1/GDSC2 (Must Pull from Broad CTRPv2 & PRISM):** Canonical tool probes `Erastin`, `RSL3`, `ML162`, `ML210`, `FIN56`, `CIL56`, `iFSP1`, `Ferrostatin-1`, and `Liproxstatin-1` are **NOT** in GDSC `screened_compounds_rel_8.4.csv` — confirming that **Broad CTRPv2 (`Erastin`, `1S,3R-RSL3`, `ML162`, `ML210`, `CIL56`) + PRISM Repurposing** are mandatory primary sources for canonical GPX4/System-$x_c^-$ probe sensitivity, while **GDSC2** adds unique `APR-246` (`894` lines), `Elesclomol` (`921` lines), `ACACA` (`835` lines), and thiol-redox perturbation (`GSH`/`NAC`/`α-LA`) curves!

### 6.2 Sanger Project SCORE v2 & Organoid CRISPR Screens (`https://cog.sanger.ac.uk/cmp/download/`)
- **Verified Live Downloads (`HTTP 200 OK`):**
  1. `Project_Score2_fitness_scores_Sanger_v2_Broad_21Q2_20250624.zip` (`198.87 MB`) — harmonized genome-wide CRISPR-Cas9 dropout fitness scores combining Wellcome Sanger (`KY` sgRNA library, Behan et al. *Nature* 2019, `PMID: 30971821`, `DOI: 10.1038/s41586-019-1103-9`; Pacini et al. *Nat Commun* 2019, `PMID: 31862961`; Dwane et al. *NAR* 2021, `PMID: 33068406`) and Broad (`Avana` library) across `>900` cancer cell lines.
  2. `Organoids_fitness_scores_20260723.zip` (`21.46 MB`, **July 23, 2026 release**) — CRISPR knockout fitness scores in **3D patient-derived cancer organoids** (`GPX4`, `AIFM2`, `SLC7A11`, `ACSL4`, `DHCR7`, `MBOAT1/2` 3D spheroid/organoid dependencies — critical bridge between 2D plastic monolayers, where ferroptosis sensitivity is artificially inflated by hyperoxia/low density, and 3D tumor architecture!).
  3. `CRISPR_KO_supplemental_information_20250624.csv` (`1.18 MB`) & `Project_score_raw_counts.zip` (`2,848.00 MB`).

---

## 7. High-Content Morphological Perturbation Atlases (`JUMP-CP cpg0016` & `RxRx3-core`)

### 7.1 JUMP-Cell Painting Consortium (`cpg0016-jump`, Chandrasekaran et al. *Nat Methods* 2024, `PMID: 38594452`, `DOI: 10.1038/s41592-024-02241-6`)
- **Verified Exact Metadata Counts (`https://raw.githubusercontent.com/jump-cellpainting/datasets/main/metadata/`):**
  - **`115,796` unique small molecules** (`compound.csv.gz`)
  - **`7,977` CRISPR-Cas9 gene knockouts** (`crispr.csv.gz`)
  - **`15,132` ORF gene overexpressions** (`orf.csv.gz`)
  - **`2,525` multi-partner 384-well plates** (`plate.csv.gz`, U2OS cells, 8 organelle channels)
- **Genetic Coverage of 38 Core Ferroptosis Genes (`94.7%` Combined!):**
  - **JUMP CRISPR (`33/38` = `86.8%` present):** `ACSL3`, `ACSL4`, `AIFM2`, `ALOX12`, `ALOX15`, `ALOX5`, `DHCR7`, `DHODH`, `EBP`, `EEFSEC`, `FTH1`, `GCH1`, `GCLC`, `GCLM`, `GPX4`, `GSR`, `GSS`, `HMGCR`, `HMOX1`, `KEAP1`, `LRP8`, `NCOA4`, `NFE2L2`, `POR`, `PROM2`, `PSTK`, `SECISBP2`, `SLC3A2`, `SLC7A11`, `SQLE`, `TFRC`, `TXNRD1`, `TXNRD2`.
  - **JUMP ORF (`26/38` = `68.4%` present):** Adds **`FTL`, `MBOAT2`, and `SEPSECS`** on top of `GPX4`, `AIFM2`, `SLC7A11`, `NFE2L2`, `KEAP1`, `DHCR7`, `EBP`, `SQLE`, `HMGCR`, `FTH1`, `NCOA4`, `TFRC`, `PROM2`, `ALOX15`, `ALOX5`, `DHODH`, `GCLC`, `GCLM`, `HMOX1`, `PSTK`, `SECISBP2`, `TXNRD1`, `TXNRD2` (only `LPCAT3` and `MBOAT1` absent from both CRISPR and ORF).
- **Exact `InChIKey` Coverage of 46 Ferroptosis Probes & Clinical Modulators (`34/46` = `73.9%` Present with Verified `JCP2022` IDs!):**
  - **Canonical Inducers Present:** `Erastin` (`JCP2022_006961`), `Imidazole ketone erastin (IKE)` (`JCP2022_070698`), `ML210` (`JCP2022_094079`), `FIN56` (`JCP2022_040500`), `CIL56` (`JCP2022_106962`), `Sulfasalazine` (`JCP2022_058046`), `Sorafenib` (`JCP2022_054847`), `Lenvatinib` (`JCP2022_100166`), `Regorafenib` (`JCP2022_021751`), `Altretamine` (`JCP2022_091723`), `Brequinar` (`JCP2022_068563`), `Buthionine sulfoximine` (`JCP2022_045057`), `Artesunate` (`JCP2022_020823`), `Dihydroartemisinin` (`JCP2022_006696`), `Elesclomol` (`JCP2022_006922`), `Disulfiram` (`JCP2022_004039`), `APR-246` (`JCP2022_006103`), `Menadione` (`JCP2022_054618`). *(Absent: `RSL3`, `ML162`, `FINO2`, `iFSP1`, `icFSP1`, `Auranofin`, `Withaferin A`.)*
  - **Canonical Suppressors, Iron Chelators & Lipid/Mevalonate Modulators Present:** `Ferrostatin-1` (`JCP2022_089600`), `Liproxstatin-1` (`JCP2022_107188`), `UAMC-3203` (`JCP2022_032391`), `Deferoxamine` (`JCP2022_088158`), `Deferasirox` (`JCP2022_007613`), `Deferiprone` (`JCP2022_087849`), `Ciclopirox` (`JCP2022_082294`), `Alpha-Tocopherol` (`JCP2022_028032`), `Zileuton` (`JCP2022_056985`), `Dimethyl fumarate` (`JCP2022_048675`), `Simvastatin` (`JCP2022_081555`), `Lovastatin` (`JCP2022_067776`), `Atorvastatin` (`JCP2022_106098`), `Fluvastatin` (`JCP2022_021041`), `Pitavastatin` (`JCP2022_093888`), `Pravastatin` (`JCP2022_086894`).

### 7.2 Recursion `RxRx3-core` (`https://huggingface.co/datasets/recursionpharma/rxrx3-core`, Open / Ungated)
- **Verified Structure (`metadata_rxrx3_core.csv`, `222,601` wells / `1,335,606` 6-channel images in HUVEC):**
  - `736` CRISPR gene entries (including `ALOX5`, `DHODH`, `HMGCR`, `SQLE`; full `recursionpharma/rxrx3` covers `~17,000` genes under `gated: 'auto'` non-commercial click-through).
  - `1,674` multi-concentration small molecules, including **29 ferroptosis/redox/synergy drugs** — notably **supplementing JUMP-CP with `Rosiglitazone` and `Pioglitazone` (`ACSL4` inhibitors)** alongside `Sorafenib`, `Lenvatinib`, `Regorafenib`, `Sulfasalazine`, `Artesunate`, `Altretamine`, `Disulfiram`, `Dimethyl fumarate`, `Zileuton`, all 4 clinical iron chelators (`Deferoxamine`, `Deferasirox`, `Deferiprone`, `Ciclopirox`), all 7 clinical statins (`Simva`, `Lova`, `Atorva`, `Pitava`, `Rosuva`, `Fluva`, `Prava`), and synergy partners (`Vorinostat`, `Bortezomib`, `Lapatinib`, `Dasatinib`, `Rapamycin`, `Everolimus`, `Temsirolimus`, `Olaparib`).
  - Precomputed vision-foundation embeddings directly downloadable as single `.parquet` files (`OpenPhenom_rxrx3_core_embeddings.parquet`, `MAE-G_8`, `MAE-L_8`, `CellProfiler_features_rxrx3_core.parquet`).

### 7.3 Strict Purdue Anvil HPC Storage & `<1M` Inode Protection Protocol
- **DO NOT** sync `s3://cellpainting-gallery/cpg0016-jump/source_all/images/jump_lite_v1.0/images_compressed/jpegxl_lossy_d20.zarr/` directly onto `/anvil/projects/x-cis261485/` — our live S3 inspection confirmed it contains **millions of `~20 KB` Zarr chunk files (`0.0.0`, `.zarray`, `.zattrs`)** that will immediately exhaust Anvil's project inode limit!
- **RECOMMENDED INGESTION (`<2,600` files total, `<45 GB` disk):**
  1. Sync ONLY `s3://cellpainting-gallery/cpg0016-jump/source_4/workspace/profiles/` (and partner profile `.parquet` tables): **`~13.23 MB/plate` × `2,525` plates = `~33.4 GB` in `2,525` Parquet files** (or consolidate into a single partitioned DuckDB/Parquet dataset of `<10` files!).
  2. Download ONLY `metadata_rxrx3_core.csv` + `OpenPhenom_rxrx3_core_embeddings.parquet` from `recursionpharma/rxrx3-core` (`<1.5 GB`, `2` files, zero raw JP2/TIFF extraction needed!).

---

## 8. Zero-Leakage Audit Against Downstream MSKCC Bradbury / Overholtzer / Wiesner Validation

To guarantee strict benchmark integrity for downstream validation on PI Michelle S. Bradbury's proprietary MSKCC ferroptosis datasets:
1. **PubChem BioAssay (`db=pcassay`):** Live searches for `"Bradbury"`, `"Overholtzer"`, and `"Wiesner"` intersected with `(ferroptosis OR nanoparticle OR melanoma OR renal)` all returned **`0` assays**.
2. **EBI ChEMBL 37 (`document/search.json`):** Live search for `"Overholtzer"` returned **`0` documents**. Search for `"Bradbury"` returned `46` legacy medicinal chemistry papers (`1988–2010`, e.g. `CHEMBL1125716` by R.H. Bradbury on renin/muscarinic inhibitors) with **`0` papers from Michelle S. Bradbury, `0` C' dots / Cornell-dots screens, and `0` ferroptosis assays**.
3. **Conclusion:** Pre-training **FerroFlux-FM** on ChEMBL 37, PubChem qHTS, DrugComb v1.5, NCI-60/ALMANAC, GDSC1/2, Project SCORE v2, JUMP-CP `cpg0016`, and `RxRx3-core` has **zero data leakage** into any MSKCC Bradbury/Overholtzer/Wiesner nanoparticle (`αMSH-PEG-C' dots`, `DFO-C' dots`, `²²⁵Ac`/`¹⁷⁷Lu` radioimmunoconjugates) or patient-derived validation cohorts.

---

## 9. Tiered Ingestion Plan & Purdue Anvil Storage / Inode Budget (`07` Delta)

To integrate cleanly into `@verifier`'s master Round-2 audit (`10_complete_data_audit.md`), all chemogenomics, qHTS, drug-synergy, and morphological perturbation assets are stratified by tier:

| Tier | Dataset / Modality | Processed Format & Exact Live Source | Disk Footprint | File / Inode Count | Compute (`4× H100` Node) |
| :--- | :--- | :--- | ---: | ---: | :--- |
| **Tier 1 (Core Pre-Training)** | **EBI ChEMBL 37** (`24.5M` activities; `30` ferroptosis targets + `935` phenotypic assays) | `chembl_37_sqlite.tar.gz` (`ftp.ebi.ac.uk/.../latest/`) → filtered Parquet | `5.76 GB` (`0.40 GB` filtered) | `2` | `<0.5` GPU-h (RDKit/Morgan/MolFormer featurization) |
| **Tier 1 (Core Pre-Training)** | **PubChem BioAssay qHTS** (`AID 1671190`, `624171`, `588453`, `504444`, `686970/71`, `1554`, `1845192`, Tox21 `743219`/`1919970`) | PUG REST CSVs (`pubchem_chembl_ferroptosis_assays.tsv`) → unified Parquet | `1.85 GB` | `8` shards | `~1.5` GPU-h |
| **Tier 1 (Core Pre-Training)** | **DrugComb `v1.5` + Raw Dose Matrices (`v1.4`)** (`739,964` blocks; `8,397` drugs; `ALMANAC`/`ONEIL`/`AZ`/`FORCINA`) | Zenodo `15235991` (`summary_v_1_5.csv`) + `18449193` (`drugcomb_data_v1.4.csv` + ID `.xlsx`) → Parquet | `3.44 GB` | `4` | `~2.0` GPU-h |
| **Tier 1 (Core Pre-Training)** | **Sanger GDSC1/GDSC2 (`Rel 8.4`) + Project SCORE v2 + 3D Organoids (`2026-07-23`)** | `ftp.sanger.ac.uk` fitted CSVs (`90 MB`) + `cog.sanger.ac.uk/cmp/download/` CRISPR zips (`221.5 MB`) | `0.31 GB` | `6` | `<0.5` GPU-h |
| **Tier 1 (Core Pre-Training)** | **NCI-60 DTP (`GI50` + `ZSCORE` + `ALMANAC` + `CORE` 219-metabolite & `FIA-TOF` ions)** | NCI Wiki (`GI50.zip`, `ComboDrugGrowth_Nov2017.zip`) + CellMiner `.zip`s | `0.65 GB` | `12` | `<0.5` GPU-h |
| **Tier 2 (Morphological Auxiliary Prior)** | **JUMP-Cell Painting (`cpg0016-jump`)** (`115,796` compounds [`34/46` probes], `7,977` CRISPR [`33/38`], `15,132` ORF [`26/38`]) | `s3://cellpainting-gallery/cpg0016-jump/source_4/workspace/profiles/` plate Parquets (consolidated) | `33.40 GB` | `2,525` (`-> 16` merged shards) | `~4.0` GPU-h |
| **Tier 2 (Morphological Auxiliary Prior)** | **Recursion `RxRx3-core` Embeddings** (`222,601` wells; `29` clinical ferroptosis/redox drugs + `736` KOs) | `huggingface.co/datasets/recursionpharma/rxrx3-core` (`OpenPhenom` + `CellProfiler` `.parquet` only) | `1.45 GB` | `4` | `~0.5` GPU-h |
| **Tier 3 (Quarantine Holdout)** | **Bradbury / Overholtzer / Wiesner HTS Audit** | Verified `0` PubChem `AID`s and `0` ChEMBL 37 documents (defer to GEO/PMC/Zenodo quarantine list) | `0.00 GB` | `0` | `0` |
| **Tier 4 (Excluded / Offline / Inode-Unsafe)** | Raw JUMP-CP `jpegxl_lossy_d20.zarr` (`>5M` tiny chunk files); raw `RxRx3` full gated images (`~2.2M` images); dead `drugcomb.org` / `drugcombdb` portals; unverified pan-NCI-60 ICP-MS total metal matrix | Excluded (use pre-extracted `.parquet` embeddings/profiles only) | `0.00 GB` | `0` | `0` |
| **TOTAL (`Tier 1 + Tier 2`)** | **All Chemogenomics, qHTS, Synergy & Cell Painting Profiles** | **Parquet / SQLite / Compressed CSV** | **`46.86 GB`** (`12.01 GB` Tier 1 + `34.85 GB` Tier 2) | **`2,561` raw (`52` merged)** | **`~9.0` H100 GPU-h** |

