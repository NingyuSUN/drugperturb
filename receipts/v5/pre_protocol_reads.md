# Tahoe-100M reads before the v5 protocol was committed

Every read below took place on 2026-10-22, before `configs/protocol_v5.yaml` was committed. Files came from Hugging Face `tahoebio/Tahoe-100M` at revision `349e2a55f5538a3e73b32a06c6bdd8e9e87011e8`, and each file's SHA256 matched its LFS oid ([source.tsv](../../manifests/v5/source.tsv)).

**Not read:**

- any expression value;
- any expression summary, such as the `mean_*` columns of `sample_metadata` or the per-cell count, mitochondrial and cell-cycle columns of `obs_metadata`;
- any cell count of a compound or reference-control sample;
- the published `pseudobulk_differential_expression` files.

## Survey: documentation only

| Read | Purpose |
|---|---|
| The dataset card (README, LICENSE), the Hugging Face API file listing, the commit log and the `paths-info` sizes and LFS oids. The dataset page was read through a summarizing fetch tool. That page also renders the Hugging Face viewer's preview of the first rows; the summary returned contained only field descriptions, counts and one example concentration, and no expression value. | identify, pin and hash the source |
| The Arc Virtual Cell Atlas README and plate file listing; the preprint text and a reanalysis page, design sections only | describe the design |

## Exploration while writing the audit

| Read | Purpose |
|---|---|
| Parquet schemas (column names and types) and row counts of `sample_metadata`, `drug_metadata`, `cell_line_metadata` and `obs_metadata` | find the design columns |
| `sample_metadata` columns `sample`, `plate`, `drug` and `drugname_drugconc`: value counts, doses by plate, plates per drug-dose | settle the plate layout and the vehicle and replicate structure |
| `drug_metadata` columns `drug`, `canonical_smiles` and `pubchem_cid`, standardized with the v1 policy | count exclusions, duplicate parents and scaffold groups |
| `obs_metadata` columns `sample`, `cell_line`, `cell_name` and `pass_filter`, for the rows of the 28 DMSO_TF samples only. The row filter was applied inside the parquet read. | the values of `pass_filter` and the line labels in vehicle cells |

## The scripted audit

`scripts/v5_design_audit.py` makes the same kinds of read. Its receipt, [design_audit.json](design_audit.json), records:

- the exact columns and the row filter;
- input hashes, output hashes and code hashes;
- package versions and the command.

`--check` regenerates every table in `manifests/v5/` from the verified files and requires it to equal the committed version. It passed on 2026-10-22.

**What the audit decided from these reads, all before any response is seen:**

- **Compound list:** structures only.
- **Split:** structures only, seed 42.
- **The ten cell lines:** vehicle-unit sizes and a seeded draw.
