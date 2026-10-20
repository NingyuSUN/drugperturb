# Engineering fixes after v4

These are two engineering problems found during v4. Neither fix changes a committed v1–v4 result:

- The v3 and v4 scripts still call the descriptor scaler with its v3 default.
- The new truth loader returns the same keys and values as the old code.

Nothing was refit or rescored.

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
- No model was fitted with the option. sci-Plex3 models would receive 171 instead of 196 inputs, so fits would not be bit-identical, even though the dropped columns were constant.

## Truth loading in the v3 scripts

**Problem.** Each `z['delta']` on an `NpzFile` decompresses the whole `delta` member. `v3_fit.py`, `v3_context_baselines.py` and `v3_diagnose_chemcpa.py` built their truth dictionaries as `z['delta'][i]` per key. Each row was therefore a view into its own full copy of the matrix, and that view kept the copy alive.

Loading N rows took N decompressions and held N copies. For the sci-Plex3 training truth this was estimated at about 45 GB during v4.

**Change.** The three scripts now use `drugperturb.v4.counterfactual.load_truth`, which v4 already used. It decompresses the matrix once and returns row views of that one array. The scripts' local copies (`load_truth`, `truth_table`) are removed.

The dictionaries are equal element by element to the old ones, so outputs do not change. The committed v3 outputs were produced with the old code and were not rerun.

**Regression tests** (`tests/test_v3_core.py`):

- `test_single_load_truth_reads_delta_once` counts reads of the `delta` member while loading a 50-row archive: 50 with the old pattern, 1 with the helper. It also checks that the two dictionaries are equal.
- `test_v3_scripts_load_truth_through_the_single_load_helper` checks that none of the three scripts calls `np.load`, and that each imports the v4 helper.
