# DrugPerturb v6: pre-test diagnostics

**Status: written from training and validation outputs only, before any test scoring.** The protocol is [protocol_v6.yaml](../../configs/protocol_v6.yaml); the selection record is `selection_v6.json`. No test response and no evaluation-half test delta has been read.

## Can the endpoint separate methods at all

| Quantity | Value | What it means |
|---|---:|---|
| Split noise floor | 0.8232 | The direction accuracy the selection half itself reaches against the evaluation half: the ceiling any model could reach given the split. Gate G8 requires 0.60. |
| B1p on its own | 0.8190 | Whether the plate mean already beats chance (0.5) on a compound's own DE genes. |
| Responders | 4946 of 9700 | Keys above the between-well null's 95th percentile (threshold 2.772). |
| Residual share | 0.652 | Variance of the measured residual over B1p as a fraction of the variance of the measured delta: how much is left to predict. |

## Arms on validation

| Arm | Direction accuracy, responders | All keys | Validation RMSE | Residual r | Effect ratio | Non-collapsed |
|---|---:|---:|---:|---:|---:|---|
| M1_pseudobulk_raw | 0.6848 | 0.6278 | 0.3147 | 0.049 | 0.686 | yes |
| M2_pseudobulk_residual | 0.8182 | 0.7483 | 0.2568 | -0.007 | 1.457 | yes |

Chance is 0.5 by construction, because a zero prediction counts as a tie. An arm must reach at least 0.51 and an effect size ratio of 0.10 to be non-collapsed.

## Arm selection

Metric: mean over seeds of the per-seed validation direction accuracy over responder keys, pooled over contexts. Leader: **M2_pseudobulk_residual**. Near tie: no; primary arms M2_pseudobulk_residual at interval level 0.9500.

- M1_pseudobulk_raw: behind the leader by 0.1334 with a Welch standard error of 0.0082, so it is outside 1.96 standard errors.

## Does chemical structure help any method

Each pair is a structure-using method against the method it extends, on validation compounds. These are the cheap tests of S3; they need no GPU fit.

| Contrast | Validation RMSE difference |
|---|---:|
| M1c minus B1c | -0.0004 |
| M1r minus B1p | -0.0003 |
| B2c minus B1c | 0.1110 |

Residual contexts fitted: 30 of 30. Scale factors: 0.82, 1.00, 1.00, 0.30, 1.00, 1.00, 0.44, 1.00, 1.00, 0.90, 1.00, 1.00, 1.00, 1.00, 1.00, 0.00, 0.86, 0.09, 0.45, 1.00, 1.00, 0.16, 1.00, 1.00, 0.80, 1.00, 1.00, 0.55, 1.00, 1.00.

## Context gates

Contexts passing the count gate: 27. Study gate G4: met (27 contexts, 9 lines, 3 doses).

## What this report does not contain

- No test response, and no evaluation-half delta of any test key.
- No endpoint estimate: every number here is from training or validation compounds.
- The arm selection is fixed by this record and is not revisited after test scoring.
