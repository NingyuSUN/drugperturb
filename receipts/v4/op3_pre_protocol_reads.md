# OP3 reads before the v4 protocol was committed

All reads were of `GSE279945_sc_counts_processed.h5ad` at its GEO URL through HTTP range requests (no full download), on 2026-10-20, before `configs/protocol_v4.yaml` was committed. No expression value was read and no cell count of any compound well was computed.

| Read | Purpose |
|---|---|
| GEO SOFT family file (series title, summary, design, sample list) | identify the dataset and its design |
| h5ad structure: group names, matrix shapes and dtypes; `obs` column names and category lists; first 20 stored values of `X` and `raw/X` | confirm raw integer counts exist and find the design columns |
| `obs` design columns for all cells: plate, well, row, column, donor, library, compound name and LINCS ID, SMILES, dose, time, control flag, the competition split label | well layout, vehicle matching, compound list and structures |
| Cell-type labels (`cell_type`, `cell_type_orig`) for all cells: a dataset-wide crosstab of the two labelings (all wells pooled), and cell-type counts per DMSO well | choose the context labels and check that vehicle wells can meet a 30-cell gate |
| File size (HTTP Content-Length); a streamed SHA256 computation without saving the file, still running when the protocol was committed | pin the source |

The resulting design tables are regenerated from the verified file in the extraction kernel and must equal the committed manifests (`scripts/v4_op3_design.py --check`). The receipt of the scripted design read is [op3_design_audit.json](op3_design_audit.json).
