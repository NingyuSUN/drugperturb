# Engineering fixes after v4

These are two engineering problems found during v4, and a seal-time [descriptor gate](#seal-time-descriptor-gate) that stops the first kind before any test scoring. None of these changes a committed v1–v4 result:

- The v3 and v4 scripts still call the descriptor scaler with its v3 default.
- The new truth loader returns the same keys and values as the old code.
- The gate adds fields to fit records and seals and changes no existing value. It can refuse a seal; the committed seals were written before it existed.

The fixes themselves refit and rescore nothing. A separate post hoc analysis later refitted the OP3 fits with the descriptor tolerance and reported them next to v4 ([corrected report](../reports/v4_corrected/report.md)). It changes no committed v4 output.

## Descriptor columns that are constant up to rounding

**Problem.** `TrainingDescriptorScaler` (`src/drugperturb/v3/descriptors.py`) keeps every training column whose standard deviation is above 0. RDKit2DNormalized values lie in [0, 1].

When every training compound has the same value in a column, the training mean can be off by a rounding error. `np.std` then returns that error, up to 3×10⁻¹⁵, instead of 0. Under the v3 rule, 25 such columns survive in sci-Plex3 and 28 in OP3. They act in two ways:

- **A compound equal to training in such a column gets exactly ±1 there.** The offset and the standard deviation are the same rounding error. Every sci-Plex3 compound and the DMSO vehicle gets the same value, so these columns gave the v3/v4 sci-Plex3 models no information about compounds.
- **A compound that differs from training gets a z-score of order 10¹⁵.** In OP3 this happens to ixabepilone, alvocidib and dasatinib (test) and CEP-37440 (validation). They get z-scores of 5–9×10¹⁵ ([v4 report](../reports/v4/report.md)).

**Change.** `TrainingDescriptorScaler(min_std=...)` and `compound_table(..., min_std=...)` drop training columns with a standard deviation at or below `min_std`.

- The default, `min_std=0`, is the v3/v4 rule. The v3 and v4 scripts do not pass it, so they behave as before.
- `NOISE_STD_TOLERANCE = 1e-8` is the value proposed for a future protocol.

**Why 1e-8.** The value comes from the training columns only; descriptors carry no response. In both training sets, columns that are constant up to rounding have standard deviations of at most 3×10⁻¹⁵. The least variable informative column (VSA_EState9) has 2×10⁻³. Any tolerance in that gap keeps the same columns: exactly those that are not constant on the training compounds.

The problem itself was found after OP3 test scoring. So the option is not applied to v4; a new protocol has to prespecify it.

**Effect** (`tests/test_descriptor_scaling.py`, cell-model environment):

| | sci-Plex3 | OP3 |
|---|---|---|
| Columns kept, v3 rule | 196 | 196 |
| Columns kept, `min_std=1e-8` | 171 | 168 |
| Largest \|z\|, v3 rule | 10.9 | 9.0×10¹⁵ |
| Largest \|z\|, `min_std=1e-8` | 10.9 | 9.2 |

Values in the columns that both rules keep are identical. With the tolerance, the four affected OP3 compounds have a largest |z| of 2.5–5.2.

**Limits.**

- Descriptor values depend on the descriptastorus and RDKit versions in the cell-model lock. The test pins the 25 and 28 dropped columns for that environment.
- **sci-Plex3 was not refitted with the option.** sci-Plex3 models would receive 171 instead of 196 inputs. The fits would therefore not be bit-identical, even though the dropped columns were constant.
- **OP3 was refitted with the option** (`--descriptor-tolerance`), only in the post hoc [corrected analysis](../reports/v4_corrected/report.md). The validation rule then selects I2 rather than I1. Neither analysis supports E1, E2 or E3.

## Seal-time descriptor gate

**Why.** The descriptors alone show the v4 bug: four OP3 compounds had z-scores of order 10¹⁵. It was still found only after OP3 test scoring. The gate makes this kind of problem stop the pipeline before any prediction is sealed.

**What each fit records.** `scripts/v4_fit.py` and `scripts/v3_fit.py` add a `descriptor_scaling` field to every fit record. It is computed from the scaled `compound_table` matrix the model receives, so it uses descriptors only and no response data.

- `max_abs_z`: the largest |z| over the kept columns, for each split: training, validation and test compounds, and the DMSO vehicle (`vehicle`).
- `above_limit`: for each split, the compounds whose largest |z| exceeds `z_limit`, with that value.
- `z_limit`: `DESCRIPTOR_Z_LIMIT` (`src/drugperturb/v3/descriptors.py`, next to `NOISE_STD_TOLERANCE`).

A |z| beyond the float32 range is stored as `null`. It counts as above any limit.

**The gate.** `scripts/v4_seal.py` first checks every fit record of the setting (smoke flag, test blindness, descriptor labels), then calls `descriptor_gate`. It refuses to seal, before writing any archive, in two cases:

- the largest |z| of any split exceeds `DESCRIPTOR_Z_LIMIT`;
- a fit record has no `descriptor_scaling` (records written before the gate cannot be checked).

The error names each compound with its split and |z|. `--override-descriptor-gate` seals anyway.

Every seal and the OP3 selection store the gate summary as `descriptor_gate`. It holds the limit, the largest |z| per split, the compounds above the limit, the fits without diagnostics, `passed` and `override`.

**Why 1e3.** The limit uses training-only facts.

- Under the tolerance rule, the largest |z| of any compound is 9.2 in OP3 and 10.9 in sci-Plex3; for DMSO it is 4.5. A kept training-constant column gives a differing compound 10¹⁵–10¹⁶. The limit is about two orders of magnitude above the first and at least twelve below the second.
- RDKit2DNormalized values lie in [0, 1], so |z| ≤ 1/std in every column. The least variable informative training column (VSA_EState9) has std 2.4×10⁻³ in sci-Plex3 and 2.1×10⁻³ in OP3. That bounds |z| at 416 and 485, so with the tolerance no compound can reach the limit on these training sets. Under the v3 rule, only a column with training std below 10⁻³ can trip it; on these data those are the training-constant columns.

**What it would have done in v4.** These figures come from the committed structures (`test_gate_names_exactly_the_v4_op3_compounds_and_passes_the_tolerance`, cell-model environment):

| Largest \|z\| (train / validation / test / DMSO) | sci-Plex3 | OP3 |
|---|---|---|
| v3 rule | 10.9 / 10.9 / 10.9 / 4.5 | 9.2 / 5.0×10¹⁵ / 9.0×10¹⁵ / 4.5 |
| `min_std=1e-8` | 10.9 / 10.9 / 10.9 / 4.5 | 9.2 / 9.2 / 9.2 / 4.5 |
| Gate, v3 rule | passes | refuses |
| Gate, `min_std=1e-8` | passes | passes |

Under the v3 rule the OP3 gate names exactly ixabepilone, alvocidib and dasatinib (test) and CEP-37440 (validation). The v4 OP3 seal would therefore have stopped before any evaluation; sci-Plex3 would have sealed as it did.

**Replays.** Re-sealing any committed v4 or corrected fit now needs `--override-descriptor-gate`, because those records have no diagnostics. The override adds `descriptor_gate` to the seals. Nothing was re-sealed:

- the committed v4 and corrected seals stay as written;
- the Kaggle kernels (`kaggle/v4_fit.py`, `kaggle/v4c_fit_op3.py`) were not changed or rerun.

`v3_seal.py` does not gate. v3 fit records carry the diagnostics for reference only. `v4_evaluate_op3.py` does not check the gate either; the seals and the selection carry the override.

**The default path keeps its values.** `scripts/v4c_check_default_path.py` replayed the synthetic OP3 fixture twice, through all 12 fits (2 epochs), the seal and the evaluation. One run used the scripts of 4930620 (`origin/main`), the other this branch's scripts, both on the same `src/`.

Two opt-in options of the check were added for this run:

- `--additive descriptor_scaling --additive descriptor_gate`: the keys only the new scripts write;
- `--new-seal-option=--override-descriptor-gate`: needed because the fixture trips the gate under the v4 rule.

Without these options the check is unchanged.

Result in [default_path_check.json](../receipts/descriptor_gate/default_path_check.json): `unchanged: true`, 62 of 62 checks.

- **Identical:** every prediction array, every training history (except its wall-clock `minutes` column), every seal array, the selection, and all four evaluation tables, byte for byte.
- **JSON records:** these differ only in wall-clock fields, four input paths and the new keys. `descriptor_scaling` appears in fit records and in the fits inside seals. `descriptor_gate` appears in seals, in the selection and in the selection copy inside `summary.json`.

The run used WSL and `~/venvs/cellmodels` (Python 3.11.16) and took 6 min. `base/` was `git archive 4930620 scripts configs`; `new/` was a copy of this branch's `scripts/` and `configs/`. The command was:

```
python scripts/v4c_check_default_path.py --base base --new new --work work \
  --output receipts/descriptor_gate/default_path_check.json \
  --additive descriptor_scaling --additive descriptor_gate --new-seal-option=--override-descriptor-gate
```

This is a software check on synthetic data. No committed output was regenerated.

**Tests.**

- `tests/test_descriptor_scaling.py`, synthetic and run in both environments, covers three cases with the `NOISY` training-constant column:
  - the gate refuses it and names the validation compound;
  - it passes under `min_std=NOISE_STD_TOLERANCE`;
  - the override is recorded, and a record without diagnostics is refused.

  It also checks that a |z| beyond float32 counts as above the limit. The committed-structure check above runs only in the cell-model environment.
- `tests/test_v4_op3_pipeline.py` (cell-model environment):
  - Under the v4 rule, the synthetic fixture has noise columns. Some validation and test compounds and DMSO get |z| of 10¹⁶–10²⁵, so the seal refuses, names them and writes nothing. With the override, every seal and the selection record `override: true`.
  - The corrected path seals without an override and records `passed: true`.

**Limits.**

- The gate looks only at the scaled descriptor matrix. It does not catch other input errors, such as a wrong structure, or a column that is uninformative but finite.
- The limit is an absolute |z|, justified for descriptors in [0, 1] on these two training sets. A new descriptor set needs its own justification.
- Synthetic fixtures check software behaviour only.

## Truth loading in the v3 scripts

**Problem.** Each `z['delta']` on an `NpzFile` decompresses the whole `delta` member. `v3_fit.py`, `v3_context_baselines.py` and `v3_diagnose_chemcpa.py` built their truth dictionaries as `z['delta'][i]` per key. Each row was therefore a view into its own full copy of the matrix, and that view kept the copy alive.

Loading N rows took N decompressions and held N copies. For the sci-Plex3 training truth this was estimated at about 45 GB during v4.

**Change.** The three scripts now use `drugperturb.v4.counterfactual.load_truth`, which v4 already used. It decompresses the matrix once and returns row views of that one array. The scripts' local copies (`load_truth`, `truth_table`) are removed.

The dictionaries are equal element by element to the old ones, so outputs do not change. The committed v3 outputs were produced with the old code and were not rerun.

**Regression tests** (`tests/test_v3_core.py`):

- `test_single_load_truth_reads_delta_once` counts reads of the `delta` member while loading a 50-row archive: 50 with the old pattern, 1 with the helper. It also checks that the two dictionaries are equal.
- `test_v3_scripts_load_truth_through_the_single_load_helper` checks that none of the three scripts calls `np.load`, and that each imports the v4 helper.
