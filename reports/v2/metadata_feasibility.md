# Cross-condition cohort qualification

This report records metadata capacity and completed count-level qualification for the prespecified v2 study. Model performance is evaluated separately.

## sci-Plex3 cohort capacity

The [original public sci-Plex3 metadata](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378) contain 4,992 well records. The audit covers twelve contexts: A549, MCF7 and K562 at 24 hours, each at 10, 100, 1,000 and 10,000 nM. The 72-hour A549 experiment is outside this study's scope.

Every audited context contains 188 treatment labels and two replicate wells per label. The original structure policy retains 185 parent representatives; three source labels are excluded by the chemical identity/duplicate policy. Fourteen additional representatives are outside the original v1 count-qualified cohort. The frozen 171-compound cohort and its 119/26/26 train/validation/test assignments are retained; excluded labels remain in the manifest.

| Prespecified role | Context | Metadata capacity |
|---|---|---:|
| Source training | A549, 24 h, 1 µM | 119 compounds |
| Source validation | A549, 24 h, 1 µM | 26 compounds |
| Source test locked for new head | A549, 24 h, 1 µM | 26 compounds |
| Primary new-condition test | MCF7, 24 h, 1 µM | 26 compounds / 24 structure groups |
| Secondary tests | K562 1 µM; MCF7 0.01, 0.1 and 10 µM, all 24 h | 26 compounds / 24 groups in each context |

Each context has 376 source treatment wells, eight strict matched vehicle wells and sixteen same-plate vehicle wells. The combined mapping has 4,704 context/well rows but only 4,608 unique physical wells because controls can recur across context inventories. Extraction must deduplicate physical wells; later qualification must keep experimental contexts separate.

The original A549 test responses were previously evaluated in v1. MCF7 represents a new held-out response context for those compounds, not a new never-analyzed chemical test set. The fixed models use chemical structure alone; their predictions are constant across contexts. Transfer performance therefore measures transport of the learned source response, not learning of cell or dose dependence.

## Count-qualified target cohorts

Public source extraction retained 292,010 MCF7 nuclei and 146,752 K562 nuclei, each with all 58,347 supplied human genes. Source and aggregated UMI totals agree exactly. Each cell line has 1,536 selected physical wells; one MCF7 well is empty, while all K562 wells are nonempty. Gene and cell identifiers match the original assay dimension names exactly. These extraction totals include all four doses and all source treatments, not only the final test drugs.

Both treatment replicates and their strictly matched vehicles must each contain at least 30 cells. The original test denominator is 26 compounds in every context:

| Target condition, all 24 h | Eligible / original test compounds | Structure groups | Fixed evaluation gate |
|---|---:|---:|---|
| MCF7, 1 µM — primary | 26 / 26 | 24 | Passed |
| K562, 1 µM | 25 / 26 | 23 | Passed |
| MCF7, 0.01 µM | 25 / 26 | 23 | Passed |
| MCF7, 0.1 µM | 26 / 26 | 24 | Passed |
| MCF7, 10 µM | 18 / 26 | 17 | Descriptive only: fewer than 20 compounds |

The high-dose condition retains its original role and thresholds. Its point estimates are descriptive, with no inferential interval. Low cell counts alone do not establish cytotoxicity. The same drugs recur across conditions; summing the row counts does not create independent chemical test samples.

[Count qualification](context_count_qc.tsv), [compound exclusions](../../manifests/v2/compound_count_qc.tsv), [well-level checks](../../manifests/v2/well_count_qc.tsv), and [extraction provenance](../../receipts/v2_context_extraction.json) provide the exact denominators. Nine of twelve inventory contexts have complete count QC: existing A549 at 1 µM and all four doses in each target cell line. The other three A549 doses were not extracted and are outside the prespecified target evaluation. Count eligibility does not establish model accuracy.

## Independent-study feasibility

The prespecified independent-study candidate is [LINCS Phase I GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742), restricted to chemical treatment, A549, 24 hours and an actual concentration of exactly 1 µM.

The rounded `pert_idose` display label produces 115 signatures and 89 perturbation identifiers, but includes actual doses such as 1.11 and 1.11111 µM. Numeric `pert_dose`, `pert_time` and their units must determine exact matching; the [official glossary](https://clue.io/connectopedia/glossary) distinguishes rounded display fields from numeric metadata.

Only three signatures meet the exact numeric condition: prostaglandin-a1, CYT-997 and calcitriol. All three structures standardized successfully, with no missing structure records. None matches any of the frozen 171 parents, including the 26 test compounds. All 171 project structures reproduced their original parent/scaffold identity during this audit.

| Criterion | Observed | Required |
|---|---:|---:|
| Matched test compounds | 0 of 26 | At least 20 |
| Matched test structure groups | 0 of 24 | At least 10 |
| Current unambiguous direct-landmark matches | 73 of 2,000 output genes | At least 100 |

The 73 genes use exact symbol matching against 978 directly measured landmarks; historical alias mapping was not exhaustively resolved. This is a conservative mapping count, not an upper bound on all possible mappings. The zero-compound intersection independently fails the study gate before any signature-quality filtering, so a quality filter cannot rescue it.

**Decision: the strict independent-study endpoint is not feasible with this release.** No Level 5 expression matrix was acquired and no prediction-versus-LINCS performance was calculated. Dose, cell line and gene criteria were not relaxed. This result is a dataset-compatibility finding, not evidence that the model or biological transfer failed. It does not establish that the test compounds are absent from LINCS under other conditions.

Even a compatible cohort would require separate interpretation: L1000 Level 5 signatures and paired-vehicle sci-Plex deltas differ in platform, reference and scale. A cross-platform rank comparison would not establish calibrated expression prediction, target binding, drug efficacy or clinical validity.

## Reproducible metadata entry

From the repository root, with the declared Python environment and the original metadata downloaded separately:

```bash
PYTHONPATH=src python scripts/audit_v2_metadata.py \
  --metadata data/raw/GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz \
  --split manifests/split_manifest.tsv \
  --chemistry manifests/chemistry_manifest.tsv \
  --output-root .
```

The frozen input contract rejects changed source identities, splits or protocol bytes. [The inventory](design_inventory.tsv), [roles](../../manifests/v2/context_roles.tsv), [well mapping](../../manifests/v2/well_mapping.tsv) and [audit receipt](../../receipts/v2_metadata_audit.json) describe metadata-only outputs. Count extraction, biological-head training, prediction sealing and transfer evaluation are separate analysis stages.


The independent-study gate is also executable from the four metadata files listed in [the pinned LINCS source manifest](../../manifests/v2/lincs_sources.tsv):

```bash
PYTHONPATH=src python scripts/audit_lincs_v2.py \
  --metadata-dir data/raw/lincs_phase1 \
  --output-root .
```

[The external feasibility receipt](../../receipts/v2_external_feasibility.json) records all three exact-condition signatures, their resolved structures, all 26 test-compound matching decisions and the 73 unambiguous direct-landmark mappings. The audit checks source bytes, frozen project identities and code hashes without accessing expression matrices. Its reproducible metadata result is 473,647 source signatures, three exact-condition candidates and zero matched test compounds.
