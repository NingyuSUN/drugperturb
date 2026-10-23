# DrugPerturb v5: training/validation diagnostics and arm selection

**This note uses training and validation compounds only. It was committed before the evaluation kernel read any test response.** Test truth has not been computed. The protocol is [protocol_v5.yaml](../../configs/protocol_v5.yaml) (prespecified, with engineering amendment A1).

- **Tables:** [diagnostics.tsv](diagnostics/diagnostics.tsv), [per seed](diagnostics/diagnostics_per_seed.tsv) and [selection_summary.json](diagnostics/selection_summary.json), from `scripts/v5_diagnostics_report.py`.
- **Seals and the selection record:** [receipts/v5/sealed/](../../receipts/v5/sealed), written by kernel dp-v5-seal (code af874d62… for the fits, 0ec7cc0a… for the seal).
- **Runs:** [kaggle_runs.md](../../receipts/v5/kaggle_runs.md).

## Fits

All 33 prespecified fits completed on Kaggle T4 GPUs: I1, I2 and I3 with 10 seeds each, and A0 with 3. None had a non-finite validation evaluation. Every fit used 173 descriptor columns, after dropping the training-constant ones.

**Descriptor gate (D6).** The largest scaled |z| is 14.2 for training, validation and test compounds, and 3.6 for the vehicle. The limit is 1,000, so the gate passed and no override was used.

| Arm | Stopped by | Median best epoch | Fit-minutes, all seeds |
|---|---|---:|---:|
| A0, v3 reference | the 200-epoch maximum for seeds 123 and 2026; patience for seed 42 | 160 | 152 |
| I1, NB likelihood | patience, all seeds | 20 | 305 |
| I2, counterfactual pseudobulk | the 60-minute cap for 9 of 10 seeds; patience for seed 42 | 110 | 615 |
| I3, NB + pseudobulk | patience, all seeds | 10 | 422 |

**I2 was still improving when stopped.** Six of its nine capped fits had their best evaluation within the last 10 epochs. The prespecified 60-minute cap therefore bounded I2's training; a longer budget might have lowered its validation error.

## Validation compounds

Zero change has a validation RMSE of 0.3740. All rows use the same validation pairs: 50 compounds over 30 contexts, 1,462 pairs. The arms' rows use seed-averaged predictions.

**Column definitions:**

- **Effect size ratio:** the size of the predicted change relative to the context training mean.
- **r:** the gene-wise correlation with the measured change.
- **r vs B1a:** the correlation after removing the anchored training mean from both.

| Method | Effect size ratio | r | r vs B1a | Validation RMSE | Mean per-seed RMSE (range) | Non-collapsed |
|---|---:|---:|---:|---:|---|---|
| Training mean (B1c) | 1.00 | 0.407 | 0.354 | 0.3476 | — | — |
| Fingerprint Ridge (M1c) | 1.00 | 0.409 | 0.356 | 0.3471 | — | — |
| Anchored training mean (B1a) | 1.79 | 0.508 | 0 | 0.3400 | — | — |
| Anchored Ridge (M1a) | 1.79 | 0.509 | 0.018 | 0.3401 | — | — |
| Plate-anchored training mean (B1p) | 1.52 | 0.615 | 0.485 | **0.3005** | — | — |
| A0, v3 reference | 0.19 | 0.029 | 0.306 | 0.3753 | 0.3772 (0.3746–0.3785) | no |
| I1, NB likelihood | 0.34 | 0.058 | 0.306 | 0.3777 | 0.3827 (0.3807–0.3880) | no |
| I2, counterfactual pseudobulk | 0.86 | 0.417 | 0.352 | 0.3392 | 0.3526 (0.3427–0.3688) | **yes** |
| I3, NB + pseudobulk | 0.82 | 0.379 | 0.347 | 0.3517 | 0.3592 (0.3522–0.3637) | yes |

The non-collapse criteria are an effect size ratio of at least 0.10, r of at least 0.10, and RMSE below zero change. **I1 fails** on r and on RMSE (worse than zero change). **A0 fails** on r: v3 chemCPA collapses on Tahoe-100M, as it did on sci-Plex3.

## Selection

The prespecified rule is the lowest mean per-seed validation RMSE among non-collapsed candidates, with the near-tie rule ([selection_v5.json](../../receipts/v5/sealed/selection_v5.json)).

- **Pool.** The non-collapsed candidates are I2 and I3.
- **Leader.** I2 has the lowest mean per-seed RMSE, 0.3526.
- **Near-tie check.** I3 is behind by d = 0.0066, with Welch SE = 0.0030 over the 10 + 10 seeds. Since d is above 1.96 × SE = 0.0058, there is no near tie.
- **Result.** **I2 is the single primary arm.** P1 is tested at the 95% level with no Bonferroni adjustment.

| Arm | Seal (predictions SHA256) |
|---|---|
| A0 | 6224a89e… |
| I1 | 012d9a4e… |
| I2 | 93ecfa79… |
| I3 | cb3eb452… |
| Comparators (P1) | b05c4c9c… |

## What the validation results do and do not show

These are validation compounds. They chose the checkpoints, the arm and the Ridge alphas, so they are not evidence for any endpoint.

- **The vehicle-anchoring gain is small here:** B1a is 0.0076 below B1c, against 0.075 on OP3 test compounds. Tahoe's pooled vehicle units hold about 2,800 cells (median), against about 40–340 in OP3, so the vehicle measurement carries little sampling noise to cancel. This was the expectation recorded in the design note.
- **B1p is much better than B1a on validation (0.3005 against 0.3400).** It averages the training compounds measured on the same plate. Plate-level effects beyond the vehicle wells therefore carry a large share of the measured change. Within a context, a test compound's plate-mates come from the training set.
- **The chemCPA results:**
  - The pseudobulk loss makes the drug term substantial and correlated with the truth (I2 r = 0.42, I3 r = 0.38).
  - Seed-averaged I2 lies 0.0008 below B1a and M1a on validation, a difference much smaller than I2's per-seed range (0.0261).

The confirmatory comparison is P1 on the 84 test compounds (2,464 eligible pairs). It has not been computed yet.
