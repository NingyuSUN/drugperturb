# DrugPerturb v5: candidate datasets (survey, 2026-10-22)

**Recommendation: Tahoe-100M.** It is the only public single-cell chemical-perturbation dataset this project has never scored that has enough structure-defined compounds for a scaffold split with v4's per-context gates. Every other unscored single-cell candidate has at most 92 compounds. The larger Arc–Biohub–Tahoe dataset would be better still, but it has not been released.

**How this survey was done.** It used dataset cards, repository listings, READMEs, papers and press releases only. No expression matrix, metadata table or cell count was downloaded or opened. Every number below is as reported by its source. The ones v5 depends on are confirmed by the design audit before the protocol is committed ([protocol_v5.yaml](../configs/protocol_v5.yaml), `order_of_operations`).

**What v4 needed:**

- compounds that can be split by scaffold;
- a strict matched vehicle in every batch;
- at least 20 eligible test compounds in at least 10 scaffold groups per context.

**Already scored by this project:** sci-Plex3 at 24 h (A549, MCF7, K562; v1–v4) and OP3 (GSE279945; v4). `git grep` on `main` finds no other single-cell perturbation dataset.

## Candidates

| Dataset | Compounds with SMILES | Cells | Doses, time | Replicates | Matched vehicle | Size | Licence | Prior public use |
|---|---|---|---|---|---|---|---|---|
| **Tahoe-100M** | 379 drugs; canonical SMILES and PubChem CID in `drug_metadata` | 50 cancer cell lines pooled in every well (Mosaic spheroids), SNP-demultiplexed | 0.05, 0.5, 5 µM; 1,135 drug-dose combinations; 24 h | one well per drug-dose; plate 14 repeats plate 6; no donors | DMSO_TF on every plate, reported as two wells (H11, H12), all 50 lines in each | 100.6M cells; 14 plates, 1,344 samples; 3,388 parquet shards (~429 GB) or 14 h5ad (337 GB) | CC0-1.0 | heavy: training data for State (Arc) and Tahoe-x1; in Chem-PerturBridge and other benchmarks; per-condition pseudobulk DE published with the data |
| Arc–Biohub–Tahoe (announced 2026-01-12) | ~1,400 chemical scaffolds, plus 100 cytokines | ~50 cell lines | three doses each | not stated | Mosaic design, as Tahoe-100M | 120M+ cells, 225,000 conditions | to be open-sourced after partner sharing | none; **not released** as of this survey |
| sci-Plex3 A549 72 h (GSE139944) | the sci-Plex3 compounds | A549 | 4 doses; 72 h | 2 wells | same-plate vehicle | part of GSE139944 | GEO | never scored here, but all test compounds' A549 24 h responses are open and the split was built on them |
| OP3 multiome and other OP3 files (GSE279945 series) | the 146 OP3 compounds | PBMCs, the same 3 donors | 1 µM; 24 h | 3 donors | as OP3 | — | GEO | v4 test responses are open; not untouched |
| sci-Plex-GxE ([Cell Genomics 2024](https://doi.org/10.1016/j.xgen.2023.100487)) | 4 kinase-pathway inhibitors; 23 in a validation experiment | A172, T98G, U87MG with CRISPRi of 522 kinases | 1 and 10 µM; 72 h | 2 transductions | DMSO | ~1.05M cells | CC BY-NC-ND 4.0 (article); GEO accession not checked | pertpy loader |
| MIX-seq ([Nat Commun 2020](https://doi.org/10.1038/s41467-020-17440-w); [figshare 10298696](https://figshare.com/articles/MIX-seq_data/10298696)) | about a dozen compounds | pools of 24–99 cancer cell lines | not checked | per pool | DMSO per pool | 2.1 GB | not checked | scPerturb, pertpy, many model papers |
| sci-Plex Müller glia ([eLife](https://elifesciences.org/articles/92091), GSE239731) | 92 (TocriScreen stem cell library) | mouse primary Müller glia with induced Ascl1 | 0.1, 1, 10 µM; day 5 | not stated per compound; 3 control wells per collection day | DMSO in control wells | 27,152 cells in 265 wells (~100 per well) | not stated | the source paper |
| scGeneScope ([NeurIPS 2025 D&B](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ce02df43d66626bb7087ec699e20c7ea-Abstract-Datasets_and_Benchmarks_Track.html)) | 28 | U2-OS | not stated in the abstract | 2 generation rounds | not stated | not stated | not stated | its own benchmark |
| HepaRG TempO-LINC ([bioRxiv 2025.06.10.657506](https://www.biorxiv.org/content/10.1101/2025.06.10.657506v1.full), GSE299113) | 7 stressors | differentiated HepaRG | 3 doses each; 24 h | 1 culture per condition | 0.1% DMSO | 33,742 cells | not stated | the source paper |

**Not single-cell, so outside this survey's scope.** The v4 arms train on cells. These are listed only because they have many compounds:

- DRUG-seq in four primary cell types: 89 compounds × 6 concentrations ([bioRxiv 2025.06.03.657593](https://www.biorxiv.org/content/10.1101/2025.06.03.657593v1.full)).
- DILImap: 301 compounds in one context ([Nat Commun 2025](https://doi.org/10.1038/s41467-025-65690-3)).
- LINCS L1000: audited in v2; only 73 of the 2,000 frozen genes are landmarks.
- Chem-PerturBridge: a harmonized compendium of existing data, not new measurements ([arXiv 2605.31522](https://arxiv.org/abs/2605.31522)).

## Fit to the v4 design

| Dataset | Scaffold-splittable | Strict matched vehicle | Eligible test compounds per context | Verdict |
|---|---|---|---|---|
| **Tahoe-100M** | yes: 379 structures, scaffold count to be fixed by the audit | yes: same plate and cell line. The two vehicle wells per plate are shared by about 90 compound wells. | ~95 expected (25% of 379), less count failures; median 1,287 cells per condition against a gate of 30 | **recommended** |
| Arc–Biohub–Tahoe | yes, by design | as Tahoe-100M | several hundred expected | best option if it is released before the v5 protocol is committed |
| sci-Plex3 A549 72 h | yes | yes | 26 | excluded: compounds and split already exposed |
| OP3 other files | yes | yes | 26–33 | excluded: test responses open |
| sci-Plex-GxE | no: 4 compounds | yes | < 20 | excluded |
| MIX-seq | no: ~12 compounds | yes | < 20 | excluded |
| sci-Plex Müller glia | borderline: 92 compounds give ~23 test compounds | yes | ~23 at most, with ~100 cells per well | not recommended: mouse, a reprogramming context, and at most just over the 20-compound gate |
| scGeneScope | no: 28 compounds | unclear | < 20 | excluded |
| HepaRG TempO-LINC | no: 7 compounds | yes | < 20 | excluded |

## Tahoe-100M in detail

**Source and version.**

- **Copy used:** Hugging Face [`tahoebio/Tahoe-100M`](https://huggingface.co/datasets/tahoebio/Tahoe-100M), revision `349e2a55f5538a3e73b32a06c6bdd8e9e87011e8` (HEAD on 2026-10-22).
- **What changed recently:** the two newest commits (2026-10-14) change only the dataset card. Data files last changed on 2025-07-23, when the pseudobulk DE files were added.
- **Hashes:** every file's sha256 is its LFS oid at that revision.
- **Second copy:** per-plate h5ad files, release `2025-02-25`, in `gs://arc-institute-virtual-cell-atlas/tahoe100M/` (Requester Pays). The README announced the deprecation of the older free bucket `gs://arc-ctc-tahoe100/` for 2026-03-31.
- **Paper:** [bioRxiv 10.1101/2025.02.20.639398](https://www.biorxiv.org/content/10.1101/2025.02.20.639398v1.full), now also published in *Cell* (2026).

**Design, as reported:**

- **Plates:** 14 plates of 96 wells, 1,344 samples.
- **Compounds:** 379 drugs at three doses, giving 1,135 drug-dose combinations. The "1,100+ perturbations" in the dataset card counts drug-doses.
- **Cells:** 50 cell lines in every well. The `cell_line_id` field holds Cellosaurus IDs.
- **Controls:** DMSO_TF marks the vehicle. The dataset card says to match it by plate.
- **Size:** 100,648,790 cells in `obs_metadata` and 95.6M in `expression_data`. The median is 1,287 cells per cell line-drug-dose, with a QC minimum of 50.

**Settled by the design audit on 2026-10-22** ([receipts/v5/design_audit.json](../receipts/v5/design_audit.json); summary in [V5_DESIGN_NOTE.md](V5_DESIGN_NOTE.md)):

- `drug_metadata` has 379 drugs, each at all three doses. One of them, Adagrasib, is an inter-plate reference at 0.05 µM on every plate.
- Every plate has exactly two DMSO_TF samples.
- The 1,344 samples are 14 plates × 96. Plate 14 repeats plate 6, plate 13 mixes doses and repeats, and every other plate holds one dose.

The inconsistencies as listed before the audit:

- Sources give 379, 380 or "1,100+" drugs, and 52,886 or 56,829 conditions.
- 1,344 samples exceed what 1,135 drug-doses, 2 vehicle wells × 14 plates and a 96-well replicate plate account for. Some drug-doses may be on more than one plate. One reanalysis compares the same drug and dose across plates.
- Whether all doses of a drug share a plate is not stated.

**What fits v5:**

- **Compound counts.** About three times OP3's test compounds per context.
- **Contexts.** 50 cell lines × 3 doses give far more than any earlier study.
- **Vehicle units.** About 2,600 cells per plate and line, so vehicle noise is small. That makes Tahoe a real test of v4's anchoring explanation: B1a's gain over B1c should shrink there if vehicle noise drove it.
- **Access.** CC0 licence; SMILES provided; a pinned revision with per-file hashes.

**Risks:**

- **Exposure.** Tahoe-100M is a common pretraining corpus, and per-condition DE results are published beside the data. "Untouched" means untouched by this project, as for OP3. v5 uses no pretrained model, and the DE files are on the never-read list.
- **No biological replicates.** Each drug-dose is normally one well. Between-well variance is estimable only for compounds on plates 6 and 14.
- **Shared vehicle.** About 90 compound wells on a plate share two vehicle wells, so test compounds on one plate are dependent. The protocol adds a plate-anchored baseline (B1p) and a leave-one-plate-out sensitivity analysis.
- **Pooled spheroids.** 50 lines share a well, so cell-to-cell effects between lines are possible.
- **Size.** ~429 GB to stream for the selected lines. A metadata column (`sample_metadata` mean_* fields) summarises expression and must not be read before the protocol commit.

## Sources

- Tahoe-100M dataset card, file listing and commit log: <https://huggingface.co/datasets/tahoebio/Tahoe-100M>, `https://huggingface.co/api/datasets/tahoebio/Tahoe-100M` (read 2026-10-22)
- Tahoe-100M preprint: <https://www.biorxiv.org/content/10.1101/2025.02.20.639398v1.full>; *Cell* version: <https://www.cell.com/cell/fulltext/S0092-8674(26)01008-1> (paywalled, not read)
- Arc Virtual Cell Atlas README: <https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/tahoe-100M/README.md>; plate file sizes: <https://docs.lamin.ai/arc-virtual-cell-atlas>
- Vehicle wells per plate: the preprint (H11, H12) and a reanalysis, <https://gliostate.com/validation/tahoe-glioma-holdout/>
- Arc–Biohub–Tahoe announcement: <https://arcinstitute.org/news/tahoe-arc-biohub>; design figures: <https://www.genengnews.com/topics/artificial-intelligence/tahoe-arc-institute-and-biohub-join-forces-on-massive-virtual-cell-dataset/>
- Chem-PerturBridge dataset table: <https://arxiv.org/html/2605.31522v1>
- sci-Plex-GxE: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10312454>; MIX-seq: <https://figshare.com/articles/MIX-seq_data/10298696>; Müller glia sci-Plex: <https://elifesciences.org/articles/92091>; scGeneScope: NeurIPS 2025 proceedings (link above); HepaRG TempO-LINC: bioRxiv (link above)
