# Corrected OP3 analysis: diagnostics and arm selection (post hoc)

**This note uses only training and validation compounds. It was committed before the corrected predictions were scored on test compounds.** The analysis itself is post hoc: the descriptor bug was found from the v4 OP3 test outcomes, which have been open since the v4 evaluation ([plan](../../configs/v4_corrected_posthoc.yaml)).

The only change from v4 is the descriptor scaler. It drops the 28 descriptor columns that are constant on the training compounds up to rounding, so each fit received 168 descriptor columns instead of 196. All 12 OP3 fits were rerun (kernel dp-v4c-fit-op3; [kaggle_runs.md](../../receipts/v4_corrected/kaggle_runs.md)).

Tables:

- [op3_diagnostics.tsv](diagnostics/op3_diagnostics.tsv) and [per seed](diagnostics/op3_diagnostics_per_seed.tsv), from `scripts/v4_diagnostics_report.py`;
- [op3_validation_by_compound.tsv](diagnostics/op3_validation_by_compound.tsv), from `scripts/v4c_validation_by_compound.py`.

The selection record is [selection_op3.json](../../receipts/v4_corrected/op3/sealed/selection_op3.json). The metrics are those of the [v4 diagnostics](../v4/diagnostics.md).

## Validation compounds: v4 against corrected

These are the 69 validation pairs: 21 compounds over four cell types. The values are seed-averaged predictions over three seeds.

The reference rows were recomputed from training truth. They equal v4's exactly:

| Method | Effect size ratio | r | r vs B1a | Validation RMSE |
|---|---:|---:|---:|---:|
| Zero change (B0) | 0 | — | — | 0.4672 |
| Training mean (B1c) | 1.00 | 0.260 | 0.073 | 0.4523 |
| Fingerprint Ridge (M1c) | 1.03 | 0.266 | 0.082 | 0.4510 |
| Anchored training mean (B1a) | 2.16 | 0.622 | 0 | 0.3633 |
| Anchored Ridge (M1a) | 2.18 | 0.625 | 0.056 | **0.3628** |

The four arms, with v4's values before the arrow and the corrected values after it:

| Arm | Effect size ratio | r | r vs B1a | Validation RMSE | Latent ratio | Non-collapsed |
|---|---|---|---|---|---|---|
| A0, v3 reference | 0.19 → 0.29 | 0.114 → 0.134 | 0.105 → 0.108 | 0.4660 → 0.4649 | 0.037 → 0.044 | yes → yes |
| I1, NB likelihood | 0.35 → 0.33 | 0.196 → 0.160 | 0.119 → 0.106 | 0.4606 → 0.4646 | 0.058 → 0.039 | yes → yes |
| I2, counterfactual pseudobulk | 0.59 → 0.49 | 0.166 → 0.139 | 0.108 → 0.104 | 0.4599 → 0.4629 | 0.059 → 0.070 | yes → yes |
| I3, NB + pseudobulk | 1.53 → 1.51 | 0.203 → 0.197 | 0.090 → 0.069 | 0.4913 → 0.4956 | 0.52 → 0.57 | no → no |

On training pairs, the corrected arms reach:

| Arm | Effect size ratio | r | Training RMSE |
|---|---:|---:|---:|
| A0 | 0.25 | 0.18 | — |
| I1 | 0.33 | 0.25 | — |
| I2 | 0.54 | 0.25 | — |
| I3 | 1.93 | 0.65 | 0.321 (zero change 0.476) |

## Selection: I2 instead of I1, by a margin smaller than the seed spread

The prespecified rule picks, among the non-collapsed interventions, the one with the lowest mean per-seed validation RMSE:

| Arm | v4: mean per-seed RMSE (range) | Corrected: mean per-seed RMSE (range) |
|---|---|---|
| A0 (not a candidate) | 0.4674 (0.4668–0.4680) | 0.4674 (0.4663–0.4693) |
| I1 | **0.4645** (0.4642–0.4650), selected | 0.4680 (0.4676–0.4685) |
| I2 | 0.4655 (0.4647–0.4668) | **0.4676** (0.4642–0.4738), selected |
| I3 | 0.5317, not non-collapsed | 0.5275, not non-collapsed |

**The corrected selection is I2, the pseudobulk loss arm.** It beats I1 by 0.0004. That margin is smaller than I2's own spread across seeds (0.0096), and it is about as small as v4's margin in the other direction (0.0010). Both selections are near-ties between I1 and I2. Every arm remains worse than B1c and M1c on validation (seed-averaged 0.4629–0.4956 against 0.4510–0.4523), and far worse than B1a and M1a (about 0.363).

**What changed I1's validation RMSE was the refit, not CEP-37440.** CEP-37440 is the one validation compound with v4 descriptor z-scores near 10¹⁶. Under v4 its four validation pairs were predicted about as well as the others: the mean pair RMSE was 0.470 for A0, I1 and I2 and 0.580 for I3 (op3_validation_by_compound.tsv). Without CEP-37440, I1's selection metric still rises from 0.4642 (v4) to 0.4675 (corrected); I2's moves only from 0.4652 to 0.4654.

The refit changes the drug encoder's input width (168 against 196 columns), and with it the initialization and the training path. In v4, every compound took the same constant value (±1) in the 28 noise columns, so those columns carried no information. Seed-level variation of this size is therefore expected; it is not evidence that one rule is better.

## Fits

| Arm | Best epochs (seeds 42, 123, 2026) | Stopped | Total minutes |
|---|---|---|---:|
| A0 | 20, 30, 10 | patience | 39 |
| I1 | 10, 20, 20 | patience | 46 |
| I2 | 10, 20, 10 | patience | 66 |
| I3 | 10, 10, 10 | patience | 69 |

The fits ran on Kaggle T4 ×2 and took 1 h 52 min of wall time. No fit had a non-finite validation evaluation. Every fit record carries `analysis: v4_corrected_posthoc`, `descriptor_min_std: 1e-08` and `descriptor_columns: 168`. The in-kernel test run passed 31 tests.

## What this does and does not establish

These are training and validation results only, and validation compounds chose the checkpoints and the arm. They show that removing the noise columns leaves every arm's validation behaviour within seed variation of v4, and that the selection flips between two arms that are practically tied.

The test-compound comparison comes from the corrected evaluation. It is post hoc in every case: OP3 test outcomes were opened by the v4 evaluation, and the descriptor bug was found from them.
