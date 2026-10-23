# DrugPerturb v5: extraction, preparation and comparator sealing

These are the CPU stages of [protocol_v5.yaml](../configs/protocol_v5.yaml) that run before any fit. They read Tahoe-100M expression for the first time. None of them computes a test-compound response. Synthetic tests cover every step ([test_v5_pipeline.py](../tests/test_v5_pipeline.py)). **No Kaggle job has been run.** Each one needs the user's approval.

## Stages

| Kaggle kernel | Script | What it does |
|---|---|---|
| v5-E1, `dp-v5-extract` (CPU, internet) | [kaggle/v5_extract.py](../kaggle/v5_extract.py) | Runs the v5 tests. Downloads the five pinned metadata files and checks their SHA256. Replays `v5_design_audit.py --check`. Runs `v5_extract.py` over one range of shards. |
| v5-P1, `dp-v5-prepare` (CPU) | [kaggle/v5_prepare.py](../kaggle/v5_prepare.py) | Runs `v5_prepare.py` on every mounted extraction part, then `v5_baselines.py`. |

**Pushing.** `python kaggle/push.py code --dataset drugperturb-v5-code` stages the code dataset, then `python kaggle/push.py kernel v5-e1` and `kernel v5-p1` push the kernels. Both are private.

**If one session is too short.** `--shards START:STOP` pushes one range of shards as its own kernel, `dp-v5-extract-START-STOP`. In that case, list every part kernel in the P1 kernel's sources.

## Extraction ([scripts/v5_extract.py](../scripts/v5_extract.py), [drugperturb.v5.extract](../src/drugperturb/v5/extract.py))

- **Source.**
  - Hugging Face revision `349e2a55…`: 3,388 expression shards totalling 337.6 GB, and the gene vocabulary of 62,710 genes.
  - Every file's size and SHA256 must match [expression_source.tsv](../manifests/v5/expression_source.tsv), which was built from the LFS oids returned by the Hugging Face API.
  - Shards are downloaded two ahead of processing and deleted after use.
- **Which cells.** Cells are matched to the design through `obs_metadata`, by `BARCODE_SUB_LIB_ID`. A cell is kept when:
  - `pass_filter == full`;
  - it belongs to one of the ten selected lines;
  - it is in a vehicle sample or a sample of an accepted compound.

  The shard's `sample` and `cell_line_id` must agree with `obs_metadata`, and a cell seen twice raises.
- **Decoding.**
  - A cell's leading entry with a negative expression is the CLS marker and is dropped, as the Tahoe tutorial does.
  - Any other negative or non-integer value raises.
  - Tokens outside the vocabulary are dropped, counted and reported.
  - The all-gene library is the sum over the 62,710 vocabulary genes.
- **What one pass accumulates:**
  - all-gene sums and cell numbers per (sample, line) unit;
  - gene detection in vehicle cells;
  - the per-cell all-gene counts of the model cells.
- **Model cells.**
  - At most 32 per (sample, line), and only from training-compound, validation-compound and vehicle samples.
  - They are drawn from `obs_metadata` with `default_rng(42)` before any count is read: units in sorted order, barcodes sorted within a unit.
  - Test-compound units contribute no model cell.
- **Anchors.** They run when the parts are merged, and preparation stops if any fails:
  - every shard is processed exactly once;
  - unit cell numbers equal the `obs_metadata` pass-filter counts;
  - every selected model cell is found exactly once;
  - unit libraries equal unit sums;
  - model cells equal the unit sums where every cell of the unit is a model cell, and never exceed them elsewhere (amendment A1).

## Preparation ([scripts/v5_prepare.py](../scripts/v5_prepare.py), [drugperturb.v5.prepare](../src/drugperturb/v5/prepare.py))

- **Pairs.** One row per (compound, context, replicate sample). Each replicate's matched vehicle is the pooled vehicle unit of its plate and line, both DMSO_TF samples together. Two samples on one plate share that unit.
- **Count QC.** Gates are checked for every context (G3) and for the study as a whole (G4).
  - A (compound, context) is eligible when every replicate sample and every matched vehicle unit has at least 30 cells.
  - A context needs at least 20 eligible test compounds in at least 10 scaffold groups.
  - G4 needs at least 6 passing contexts covering at least 3 lines and 2 doses.
  - The result is written to the receipt. A failed G4 makes the study descriptive; it does not stop the pipeline.
- **Genes.**
  - Candidates are detected in at least 5% of all pass-filter vehicle cells of the ten lines.
  - The 2,000 genes with the highest variance of eligible training-compound deltas are kept (`drugperturb.v4.design.select_genes`).
- **Truth.**
  - For each key, the equal-weight mean over replicates of treated minus vehicle log1p CPM, using all-gene libraries.
  - Computed for training and validation keys only, plus per-replicate training deltas for B1p.
  - No test key enters any delta computation.
- **Outputs, in the v3/v4 layout:**
  - `model_cells.npz`: capped training, validation and vehicle cells on the frozen genes. A vehicle cell's `Combo` is its pooled vehicle unit, so counterfactual inference sees up to 64 vehicle cells per unit.
  - `queries.tsv` for every split, and `vehicle_contexts.tsv`;
  - `vehicle_levels.npz`: all-cell log1p CPM of every vehicle unit;
  - count QC, frozen genes and a receipt with every input and output hash.

## Comparators ([scripts/v5_baselines.py](../scripts/v5_baselines.py), [drugperturb.v5.baselines](../src/drugperturb/v5/baselines.py))

**Per context, with the v3/v4 estimators.** These use the context's eligible training compounds only, and the Ridge alpha is chosen on its validation compounds:

- B0, zero change;
- B1c, the training mean;
- B2c, the nearest training compound;
- M1c, the fingerprint Ridge.

**Anchored versions.** B1a and M1a add the context's mean training vehicle level and subtract the query's own vehicle level. Both levels are computed from **all** vehicle cells, as the protocol requires ("truth and baselines use all cells").

**B1p.** For each replicate, it averages the deltas of the context's training compounds on that replicate's plate, counting each compound once. A plate with fewer than 5 such compounds falls back to B1a for that replicate. B1p is the mean over the query's replicates.

**Sealing.**

- One seal holds the test and validation predictions of every comparator. Contexts without enough training, validation or test compounds are recorded and skipped.
- The synthetic tests check two things:
  - changing test-compound counts changes none of the prepared inputs or sealed comparators;
  - splitting the shards into parts changes no output.

## Not yet done

- **v5 fit script.** v4's `v4_fit.py` assumes a fixed replicate count per key and v3 context names, so v5 needs its own fit script on this prepared layout. The D6 descriptor seal gate must be merged before any fit is sealed.
- **Resources for E1.** These are estimates, and nothing has run.
  - Downloading 337.6 GB.
  - Holding the barcode index of about 20M cells of the ten lines.
  - About 3 GB of int32 unit sums, plus about 4–5 GB of model-cell counts in memory.
  - The run should fit in one 12-hour CPU session with 30 GB of RAM. If it does not, split it with `--shards`.
