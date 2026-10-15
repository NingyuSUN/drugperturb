# Data card

Public data qualification and the frozen benchmark are complete. Source measurements and derived model targets have separate provenance records.

## Public experimental source

GEO GSE139944 / GSM4150378, sci-Plex3. The analysis is A549, 24 hours, 1,000 nM (1 µM). The design contains 188 treatment labels and 376 drug wells before cell QC and chemical-structure exclusions. These are published nuclear-expression measurements, not newly performed experiments.

The source hash table has 4,992 rows (experimental wells, not nuclei). The compressed metadata is 64,715 bytes, SHA256 `98a65191d343d149c5742cf6e7b734d403e853d165da800fa2daea41b978a2ff`.

## Source-processing boundary

At author revision `079639c50811dd43a206a779ab2f0199a147c98f`, Notebook3 retains human ENSG rows and author hash-QC cells. Its >5% expression gene selection is an argument to PCA preprocessing, not a counts-row filter. The actual RDS contains 58,347 human genes and 143,015 A549 nuclei; nonnegative integer counts and source hash QC were verified. The selected 392 wells contain 36,848 nuclei and 59,281,450 UMI, with exact count conservation. Only raw counts and provenance metadata were reused; author PCA, UMAP, regressed coordinates, size factors and viability scores are not model inputs.

Author hash QC requires assigned well and plate hashes, both hash counts >=5, both top/second ratios >=5, and rt_well !=1. No unsupported mitochondrial threshold will be added.

## Controls and structures

Author Notebook1 identifies culture_plate with top_oligo_P, and joins top_oligo_W + top_oligo_P to well_oligo + plate_oligo (Combo). The main matching keys are cell type, time, replicate, plate and dose pattern. Vehicle dose is zero and must be retained. Eight relevant culture plates contain 16 vehicle wells; strict pattern matching selects eight. Strict matching is this study's prespecified choice, not proof of different solvent concentrations. Both-vehicle pooling is a sensitivity analysis.

Two replicates are normalized and contrasted separately, then averaged equally. Shared controls induce within-plate dependence not fully covered by scaffold bootstrap.

SMILES are supplied by the original authors. Preserve CXSMILES annotations in originals. S4246 includes an undefined atom and S1703 is a multi-active valproate formulation; do not silently guess or replace their structures. S2018 and S1181 share a standardized parent; the free-base S1181 is the prespecified representative. There are 185 structurally qualified parent representatives before cell QC, 173 labels passing cell QC, and 171 passing both. These exclusions overlap, so their counts must not be added blindly.

## Access and redistribution

Public source: https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/
Author code: https://github.com/cole-trapnell-lab/sci-plex/tree/079639c50811dd43a206a779ab2f0199a147c98f/large_screen
GEO usage notice: https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html

Public availability does not supply an explicit dataset-specific CC license. Retain source URLs and retrieval hashes; do not bundle third-party raw data or author code in the public repository. The repository contains its own implementation, source manifests and derived summaries; raw data and trained binary artifacts are excluded from Git.

## Frozen prediction dataset

Each treatment and exact matched control must contain at least 30 author-QC nuclei in both experimental replicates. The primary dataset uses 33,055 treatment nuclei and 886 nuclei in eight unique strict-control wells. Data at thresholds 20/30/50 retain 178/171/150 standardized parent compounds; only 30 defines the primary analysis.

| Split | Compounds | Structure groups | Ring scaffolds | Acyclic parent groups |
|---|---:|---:|---:|---:|
| Train |119|106|104|2|
| Validation |26|24|23|1|
| Final test |26|24|22|2|

Structure identity groups do not imply independent experimental errors: treatments share control wells and plates. Two thousand output genes and 20 target PCs are fitted from training responses only. All evaluation truth uses measured selected-gene changes. The full raw human-gene set supplies normalization denominators.

See [eligibility](../manifests/eligibility.tsv), [split](../manifests/split_manifest.tsv), [source hashes](../manifests/source_manifest.tsv), and [model card](MODEL_CARD.md).
