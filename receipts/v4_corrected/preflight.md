# Corrected OP3 analysis: checks before any corrected fit

These checks ran on 2026-10-21, after the plan ([v4_corrected_posthoc.yaml](../../configs/v4_corrected_posthoc.yaml)) and the code path were written and before any corrected fit. They were run locally in WSL: `~/venvs/dpcore` (Python 3.12.14) and `~/venvs/cellmodels` (Python 3.11.16, torch 2.5.1+cpu, descriptastorus).

## Local copies of the v4 inputs

The downloaded Kaggle outputs that the local evaluation will read all match their committed hashes or seals:

| Input | Source kernel | Check |
|---|---|---|
| `prepared/` (6 files) | dp-v4-op3-prepare | SHA256 equal to `receipts/v4/op3/prepare_receipt.json` outputs |
| `extraction/unit_counts.npz` | dp-v4-op3-extract | SHA256 equal to the input the preparation recorded |
| context baselines seal | dp-v4-op3-prepare | JSON equal to `receipts/v4/op3/context_baselines_op3_seal.json`; archive hash matches |
| anchored baselines seal | dp-v4-anchored | JSON equal to `receipts/v4/anchored/anchored_baselines_op3_seal.json`; archive hash matches |
| four v4 arm seals and selection | dp-v4-fit-op3 version 2 | JSON equal to `receipts/v4/op3/sealed/`; archive hashes match |

## Local replay of the v4 evaluation

`scripts/v4_evaluate_op3.py`, without the new option and run locally on the inputs above, reproduced `reports/v4/op3` exactly:

- `per_compound.tsv`, `summary.tsv`, `contrasts.tsv` and `test_diagnostics.tsv` are byte-identical;
- `summary.json` is equal as data.

The run took 13 s on CPU. This shows that the local environment reproduces the Kaggle evaluation (dp-v4-evaluate version 1), so the corrected evaluation can run locally. `scripts/v4c_evaluate_local.py` repeats this replay and records it in its receipt before it scores the corrected seals.

## The default path is unchanged

`scripts/v4c_check_default_path.py` ran the synthetic OP3 fixture twice through all 12 fits (2 epochs), the seal and the evaluation. One run used the scripts of the base commit cde0566; the other used this branch's scripts without `--descriptor-tolerance`. Result in [default_path_check.json](default_path_check.json): `unchanged: true`.

- **Identical:** every prediction array, training history (except its wall-clock `minutes` column), seal array, the selection, and all four evaluation tables, byte for byte.
- **JSON records:** these differ only in wall-clock fields and in four input paths (the two runs used different directories). Even the archive hashes are equal.

This is a software check on synthetic data. Committed v4 outputs were not regenerated.

## Tests

- **dpcore:** full suite, 359 passed, 6 skipped. The skips are the descriptastorus-only checks, which now include the corrected pipeline test.
- **cellmodels:** full suite, 372 passed.

New tests:

- `tests/test_v4_corrected.py`: the plan matches the code constant; labels are required and never mixed; the kernel passes the option to every fit and the seal and has its own code dataset; the side-by-side tables; the replay comparison.
- `tests/test_v4_op3_pipeline.py::test_corrected_descriptor_path_is_opt_in_labelled_and_never_mixed`: the option reaches the scaler (fewer descriptor columns, equal to `compound_table(min_std=1e-8)`); fit, seal, selection and summary are labelled; a default seal of corrected fits, a mixed fit set and a default evaluation of corrected seals are all refused.

The five CLI tests that call `git rev-parse` ran with `GIT_DIR`, `GIT_WORK_TREE` and `safe.directory` set as environment variables (see `docs/POST_V4_ENGINEERING.md`).
