# DrugPerturb v4, corrected descriptor scaler: the OP3 conclusion does not change (post hoc)

**With the descriptor-scaling bug fixed and all 12 OP3 fits rerun, the prespecified validation rule selects the pseudobulk arm (I2) instead of the NB arm (I1). I2 does not beat zero change on OP3 test compounds, and it is reliably worse than the per-context training mean, the Ridge and the vehicle-anchored baselines. That is the v4 conclusion, now without the ixabepilone artifact.**

**This is a post hoc analysis.** The bug was found from the v4 OP3 test outcomes, which have been open since the v4 evaluation. The prespecified [v4 result](../v4/report.md) remains the confirmatory result. This analysis is reported next to it and does not replace it.

- **Plan:** [v4_corrected_posthoc.yaml](../../configs/v4_corrected_posthoc.yaml), committed before any corrected fit.
- **Selection and diagnostics:** [diagnostics.md](diagnostics.md), committed before the corrected predictions were scored.
- **Tables:** [op3/](op3), and side by side in [op3/side_by_side/](op3/side_by_side).
- **Receipts:** [receipts/v4_corrected/](../../receipts/v4_corrected).

## What changed

v4 inherited `TrainingDescriptorScaler` from v3. It keeps every training descriptor column with a nonzero standard deviation. On OP3, 28 RDKit2DNormalized columns are constant on the 85 training compounds up to floating-point rounding (std ≤ 3×10⁻¹⁵). The effect depended on the compound:

- **Compounds that match training in those columns** got exactly ±1 there, the same value for all of them. The columns therefore carried no information.
- **Compounds that differ from training in those columns** got z-scores near 10¹⁶: ixabepilone, alvocidib and dasatinib (test), and CEP-37440 (validation).

The corrected analysis uses `compound_table(..., min_std=1e-8)`, which drops exactly those 28 columns and keeps 168 of 196 ([POST_V4_ENGINEERING.md](../../docs/POST_V4_ENGINEERING.md)). The largest |z| on OP3 falls from 9×10¹⁵ to 9.2.

Nothing else changed:

- **Same as v4:** data, split, genes, arms, hyperparameters, seeds, training schedule, the selection rule, the comparators and the endpoints.
- **Rerun:** all 12 fits, because the bug also reached a validation compound, and so could have affected checkpoint and arm selection.
- **Reused:** the sealed v4 baselines, because none of them uses descriptors.

## Arm selection on validation compounds

The table shows the mean per-seed validation RMSE, the prespecified selection metric.

| Arm | v4 | Corrected |
|---|---:|---:|
| I1, NB likelihood | **0.4645**, selected | 0.4680 |
| I2, counterfactual pseudobulk | 0.4655 | **0.4676**, selected |
| I3, NB + pseudobulk | 0.5317, fails the criteria | 0.5275, fails the criteria |

Both selections are near-ties. The corrected margin, 0.0004, is smaller than I2's spread across seeds, 0.0096.

The flip does not come from CEP-37440. Under v4 its validation pairs were predicted as well as the others. Without CEP-37440, I1 still moves from 0.4642 to 0.4675. The refit changed the training path of every arm, as a new run of the same configuration would. Details are in [diagnostics.md](diagnostics.md).

## Confirmatory endpoints, side by side

Each analysis scores its own selected arm. Endpoints are pooled over the three gate-passing contexts (T cells 33, B cells 28 and myeloid cells 26 test compounds), with paired scaffold-group bootstrap 95% intervals (2,000 draws, seed 2026). Negative values favour the arm.

| Endpoint | Contrast | v4, prespecified: I1 | Corrected, post hoc: I2 |
|---|---|---|---|
| E1, beats zero change | arm − B0 | +0.171 [−0.004, +0.532] | +0.004 [−0.003, +0.012] |
| E2, beats delta baselines | arm − training mean (B1c) | +0.183 [+0.009, +0.544] | +0.016 [+0.009, +0.025] |
| | arm − Ridge (M1c) | +0.184 [+0.009, +0.546] | +0.017 [+0.009, +0.025] |
| E3, beats anchored baselines | arm − anchored mean (B1a) | +0.258 [+0.082, +0.619] | +0.091 [+0.082, +0.102] |
| | arm − anchored Ridge (M1a) | +0.258 [+0.081, +0.619] | +0.091 [+0.081, +0.102] |
| | **Supported** | none | none |

v4's intervals were wide because one compound dominated them. The corrected intervals are narrow, and they are on the same side of zero as before.

## Every arm, pooled ([pooled.tsv](op3/side_by_side/pooled.tsv))

The v4 value comes first, then the corrected value, each with its interval.

| Arm | − zero change | − training mean | − Ridge | − anchored mean |
|---|---|---|---|---|
| A0 | −0.001 [−0.002, +0.001] → −0.000 [−0.003, +0.003] | +0.012 → +0.012 [+0.004, +0.020] | +0.012 → +0.013 [+0.004, +0.022] | +0.087 → +0.087 [+0.077, +0.100] |
| I1 | +0.171 [−0.004, +0.532] → +0.004 [−0.001, +0.009] | +0.183 → +0.016 [+0.008, +0.024] | +0.184 → +0.017 [+0.008, +0.025] | +0.258 → +0.092 [+0.082, +0.103] |
| I2 | +0.001 [−0.007, +0.010] → +0.004 [−0.003, +0.012] | +0.014 → +0.016 [+0.009, +0.025] | +0.014 → +0.017 [+0.009, +0.025] | +0.089 → +0.091 [+0.082, +0.102] |
| I3 | +0.196 [+0.002, +0.588] → +0.023 [+0.004, +0.044] | +0.208 → +0.036 [+0.021, +0.052] | +0.209 → +0.036 [+0.022, +0.052] | +0.283 → +0.111 [+0.095, +0.126] |

What the pooled results show:

- **The fix removes the artifact.** I1 and I3 fall from about +0.17–0.20 against zero change to +0.004 and +0.023.
- **The corrected arms do no better than the v4 arms that the bug did not touch.** I1 now behaves like I2 and A0: within 0.004 of zero change, and about 0.016 worse than the training mean.
- **v4's post hoc exclusion analysis anticipated this.** Without ixabepilone, it found I1 − training mean = +0.015 [+0.007, +0.023].
- **I3 is still reliably worse than zero change.**

## Every method, every context ([per_context.tsv](op3/side_by_side/per_context.tsv))

The table shows mean per-compound RMSE on OP3 test compounds, v4 → corrected. The baselines are the same sealed predictions in both analyses.

| Method | T cells | B cells | Myeloid | NK (below gate) |
|---|---:|---:|---:|---:|
| Zero change (B0) | 0.3225 | 0.5859 | 0.3354 | 0.8914 |
| Training mean (B1c) | 0.3050 | 0.5701 | 0.3312 | 0.8900 |
| Fingerprint Ridge (M1c) | 0.3038 | 0.5705 | 0.3311 | 0.8905 |
| Anchored training mean (B1a) | **0.2537** | 0.4402 | **0.2874** | **0.6605** |
| Anchored Ridge (M1a) | 0.2544 | **0.4400** | 0.2877 | 0.6607 |
| A0, v3 reference | 0.3211 → 0.3204 | 0.5844 → 0.5854 | 0.3367 → 0.3374 | 0.8900 → 0.8903 |
| I1, NB likelihood | 0.4767 → 0.3233 | 0.7570 → 0.5858 | 0.5228 → 0.3468 | 0.8931 → 0.8911 |
| I2, pseudobulk loss | 0.3184 → 0.3212 | 0.5847 → 0.5878 | 0.3435 → 0.3465 | 0.8883 → 0.8907 |
| I3, both | 0.4896 → 0.3321 | 0.7902 → 0.6116 | 0.5517 → 0.3696 | 0.9555 → 0.9428 |

The selected arms are I1 in v4 and I2 in the corrected analysis. In every context, the corrected I2 is above the training mean, the Ridge and the anchored baselines.

## The affected test compounds ([affected_compounds.tsv](op3/side_by_side/affected_compounds.tsv))

The table shows per-compound RMSE, v4 → corrected.

| Compound, context | A0 | I1 | I2 | I3 |
|---|---|---|---|---|
| Ixabepilone, T cells | 0.209 → 0.224 | 5.289 → 0.247 | 0.209 → 0.251 | 5.678 → 0.363 |
| Ixabepilone, B cells | 0.440 → 0.450 | 5.249 → 0.455 | 0.440 → 0.462 | 5.650 → 0.600 |
| Ixabepilone, myeloid | 0.240 → 0.239 | 5.013 → 0.280 | 0.240 → 0.259 | 5.508 → 0.348 |
| Dasatinib, T cells | 0.470 → 0.454 | 0.469 → 0.439 | 0.468 → 0.405 | 0.364 → 0.333 |

- **Ixabepilone.** Under the NB log link, ixabepilone's 10¹⁶ inputs saturated I1's and I3's decoder mean. With the fix, both are back in the range of the other compounds. The Gaussian arms A0 and I2 were not affected.
- **Dasatinib.** It is count-eligible only in T cells.
- **Alvocidib.** It was never count-eligible: its wells kept 0–6 cells of each type against the required 30. So it was never scored in either analysis.

Repeating v4's exclusion analysis on the corrected predictions ([posthoc_sensitivity.tsv](op3/posthoc_sensitivity.tsv)) moves no contrast by more than 0.005. No single compound dominates any more.

## Does the conclusion change?

No. Across all three conclusions, the corrected analysis supports v4 with tighter intervals:

- **No chemCPA arm beats zero change on OP3 test compounds.** The selected arm's E1 interval includes zero in both analyses.
- **No arm beats the per-context training mean or the fingerprint Ridge.** Every corrected arm is reliably worse by 0.012–0.036.
- **Every arm is far worse than the vehicle-anchored baselines,** by about 0.09.

What does change is the reason for the primary arm's failure:

- **In v4,** I1's failure was mostly the descriptor artifact.
- **In the corrected analysis,** the selected arm fails because it predicts about as well as zero change, which a per-context training mean already beats. This is the result v4 reported as post hoc after excluding ixabepilone, now obtained without excluding any compound.

## Limits

- **Post hoc, on open test outcomes.** The bug and its fix were found by inspecting OP3 test results. The tolerance value came from training descriptors only. The analysis plan was committed before the corrected fits, and the selection before the corrected scoring. Still, the decision to run this analysis at all was prompted by test outcomes, so nothing here is confirmatory. A claim that any arm works would need a new protocol and untouched data.
- **Two refits, not a single-variable contrast.**
  - **What differs:** the corrected fits differ from v4 in the descriptor input and also in run-to-run variation. That includes initialization through the input width, and cuDNN nondeterminism, which v4 does not remove.
  - **Size of the variation:** differences between the two analyses smaller than about 0.01 in validation RMSE are within seed variation (I2 ranges 0.4642–0.4738 across seeds).
  - **Consequence for the selection:** the selection of I2 over I1 is a near-tie in both analyses and should not be read as evidence that I2 is the better arm.
- **sci-Plex3 was not refitted.** No sci-Plex3 compound had a z-score above 50, and its 25 noise columns were constant for every compound. The v4 sci-Plex3 results are not affected by the artifact. They would still not be bit-identical under the fix, because the input width changes.
- **Local evaluation.** The corrected predictions were scored locally, not on Kaggle.
  - **Environment check:** before scoring, the same code and environment replayed the v4 evaluation from the v4 seals and reproduced `reports/v4/op3` byte for byte (`receipts/v4_corrected/evaluate_local.json`).
  - **Inputs:** every input matched its committed hash or seal.
- **Scope.** Three PBMC cell types from three donors; 87 test compound–context pairs. Nothing here bears on drug efficacy, toxicity or mechanism.

## Reproducibility

1. **Plan and code.**
   - Commit a9950ee holds the code path (`--descriptor-tolerance` in `v4_fit.py`, `v4_seal.py` and `v4_evaluate_op3.py`), its tests, and a check that the default path is unchanged ([default_path_check.json](../../receipts/v4_corrected/default_path_check.json)).
   - The preflight is in [preflight.md](../../receipts/v4_corrected/preflight.md).
2. **Fits.**
   - Kaggle kernel `dp-v4c-fit-op3` version 1 ran on T4 ×2 for 1 h 52 min, with code archive bb9eb191… from its own code dataset `drugperturb-v4c-code`. Its input was the `dp-v4-op3-prepare` output ([kaggle_runs.md](../../receipts/v4_corrected/kaggle_runs.md)).
   - The fit records, seals and selection were committed in ea74549, before scoring.
3. **Scoring.** One local command, recorded with its versions and all input and output hashes in [evaluate_local.json](../../receipts/v4_corrected/evaluate_local.json):

   ```
   python scripts/v4c_evaluate_local.py --extraction-root <dp-v4-op3-extract>/extraction --prepared <dp-v4-op3-prepare>/prepared \
     --baselines-sealed <dp-v4-op3-prepare>/sealed --anchored-sealed <dp-v4-anchored>/sealed \
     --v4-sealed <dp-v4-fit-op3 v2>/sealed --corrected-sealed <dp-v4c-fit-op3>/sealed --replay-root <scratch> --commit ea74549
   ```

   It replays v4, scores the corrected seals with `v4_evaluate_op3.py --descriptor-tolerance`, repeats `v4_posthoc_op3.py`, and writes the side-by-side tables (`v4c_side_by_side.py`).
4. **Unchanged outputs.** No committed v1–v4 report, seal or receipt was changed.
