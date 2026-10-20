# v4 diagnostics: does each change make chemCPA use the drug?

**This report uses training and validation compounds only and was committed before any v4 test truth was computed.** It answers the first half of the v4 question: does each intervention give chemCPA a non-trivial drug term? It does not say whether any arm predicts held-out compounds well. That is the job of the sealed test evaluation.

Protocol: [protocol_v4.yaml](../../configs/protocol_v4.yaml) · methods: [V4_METHODS.md](../../docs/V4_METHODS.md) · tables: [sciplex_diagnostics.tsv](diagnostics/sciplex_diagnostics.tsv), [op3_diagnostics.tsv](diagnostics/op3_diagnostics.tsv), and per seed [sciplex](diagnostics/sciplex_diagnostics_per_seed.tsv) and [OP3](diagnostics/op3_diagnostics_per_seed.tsv)

How to read the tables:

- **Effect size ratio** is the size of the predicted change relative to the context's average training effect (B1c). v3 chemCPA's was about 0.007.
- **r** is the gene-wise correlation between predicted and measured changes, averaged over (compound, context) pairs.
- **r vs B1a** is the same correlation after removing the vehicle-anchored training mean (amendment A1).
- **Latent ratio** is the norm of the drug-minus-vehicle term relative to the basal latent of vehicle cells.
- An arm is **non-collapsed** when its effect size ratio is at least 0.10, its r is at least 0.10 and its validation RMSE is below zero change. These thresholds were prespecified.
- Every arm row uses seed-averaged predictions: three seeds for sci-Plex3 Q1 and OP3, one for sci-Plex3 Q2.

## sci-Plex3: the drug term can be switched on, but its gains are vehicle anchoring

### Q1: A549, 1 µM, 26 validation compounds (exploratory)

| Method | Effect size ratio | r | r vs B1a | Validation RMSE | Latent ratio | Non-collapsed |
|---|---:|---:|---:|---:|---:|---|
| Zero change (B0) | 0 | — | — | 1.3029 | — | — |
| Training mean (B1c) | 1.00 | 0.266 | −0.002 | 1.2570 | — | — |
| Fingerprint Ridge (M1c) | 1.13 | 0.294 | 0.002 | 1.2432 | — | — |
| Anchored training mean (B1a) | 2.68 | 0.769 | 0 | **0.8389** | — | — |
| Anchored Ridge (M1a) | 2.80 | 0.761 | 0.018 | 0.8588 | — | — |
| A0, v3 reference | 0.007 | −0.008 | 0.005 | 1.3029 | 0.020 | no |
| I1, NB likelihood | 0.25 | 0.027 | 0.010 | 1.3040 | 0.18 | no |
| I2, counterfactual pseudobulk | 1.62 | 0.635 | 0.015 | 1.0326 | 0.33 | yes |
| I3, NB + pseudobulk | 2.78 | 0.748 | 0.041 | 0.8731 | 1.00 | yes |

On training compounds, the effect size ratio and r of each arm are:

| Arm | Effect size ratio | r |
|---|---:|---:|
| A0 | 0.007 | 0.000 |
| I1 | 0.28 | 0.13 |
| I2 | 1.64 | 0.69 |
| I3 | 2.88 | 0.86 |

### Q2: twelve contexts, 303 validation pairs, seed 42 only (exploratory)

| Method | Effect size ratio | r | r vs B1a | Validation RMSE | Latent ratio | Non-collapsed |
|---|---:|---:|---:|---:|---:|---|
| Zero change (B0) | 0 | — | — | 0.9154 | — | — |
| Training mean (B1c) | 1.00 | 0.323 | 0.002 | 0.8687 | — | — |
| Fingerprint Ridge (M1c) | 1.06 | 0.339 | 0.008 | 0.8621 | — | — |
| Anchored training mean (B1a) | 1.95 | 0.692 | 0 | **0.6574** | — | — |
| Anchored Ridge (M1a) | 2.03 | 0.686 | 0.019 | 0.6665 | — | — |
| A0, v3 reference | 0.009 | 0.001 | 0.024 | 0.9153 | 0.006 | no |
| I1, NB likelihood | 0.27 | 0.036 | 0.028 | 0.9240 | 0.031 | no |
| I2, counterfactual pseudobulk | 0.49 | 0.180 | 0.023 | 0.8935 | 0.036 | yes |
| I3, NB + pseudobulk | 1.14 | 0.153 | 0.023 | 0.9420 | 0.62 | no |

What sci-Plex3 shows:

- **A0 is v3.** It collapses as v3 did, with an effect size ratio of 0.007–0.009 and r ≈ 0. Its best epochs equal v3's exactly (Q1 seeds 40, 30 and 20; Q2 70 of 120), and its first-epoch training loss matches v3's CPU diagnostic to three decimals (−0.927; local CPU engineering smoke).
- **The NB likelihood alone moves the drug term without informing it.** The effect size ratio rises from 0.007 to about 0.25 and the latent ratio from 0.02 to 0.18, and training r reaches 0.13. On validation compounds, though, r is about 0.03 and RMSE is slightly worse than zero change. The likelihood hypothesis explains part of the collapse, the size of the drug term, but not its direction.
- **The counterfactual pseudobulk loss makes the drug term large and correlated with the truth.** I2 meets every criterion in Q1 and Q2; I3 meets them in Q1, but overfits in Q2, where its best epoch was 10 and its validation RMSE is above zero change.
- **The apparent gains are explained by vehicle anchoring, not by compound-specific prediction.** The anchored training mean B1a ignores compound identity and uses only the query's measured vehicle level. It beats every arm in both settings: 0.839 against I3's 0.873 in Q1, and 0.657 against I2's 0.894 in Q2. After B1a is removed, the arms' remaining correlation with the truth is 0.015–0.041. This finding motivated amendment A1, committed before any OP3 fit output was read. B1a and M1a were added as comparators and E3 as an endpoint.

## OP3: no arm beats the delta baselines on validation compounds

69 validation pairs: 21 compounds over four cell types, with only 10 eligible in NK cells; three seeds.

| Method | Effect size ratio | r | r vs B1a | Validation RMSE | Latent ratio | Non-collapsed |
|---|---:|---:|---:|---:|---:|---|
| Zero change (B0) | 0 | — | — | 0.4672 | — | — |
| Training mean (B1c) | 1.00 | 0.260 | 0.073 | 0.4523 | — | — |
| Fingerprint Ridge (M1c) | 1.03 | 0.266 | 0.082 | 0.4510 | — | — |
| Anchored training mean (B1a) | 2.16 | 0.622 | 0 | 0.3633 | — | — |
| Anchored Ridge (M1a) | 2.18 | 0.625 | 0.056 | **0.3628** | — | — |
| A0, v3 reference | 0.19 | 0.114 | 0.105 | 0.4660 | 0.037 | yes, narrowly |
| I1, NB likelihood | 0.35 | 0.196 | 0.119 | 0.4606 | 0.058 | yes |
| I2, counterfactual pseudobulk | 0.59 | 0.166 | 0.108 | 0.4599 | 0.059 | yes |
| I3, NB + pseudobulk | 1.53 | 0.203 | 0.090 | 0.4913 | 0.52 | no |

On training pairs, the effect size ratio and r of each arm are:

| Arm | Effect size ratio | r | Training RMSE |
|---|---:|---:|---:|
| A0 | 0.19 | 0.16 | — |
| I1 | 0.46 | 0.30 | — |
| I2 | 0.57 | 0.26 | — |
| I3 | 1.92 | 0.72 | 0.293 (zero change 0.476) |

Per-seed validation RMSE ranges: A0 0.4668–0.4680, I1 0.4642–0.4650, I2 0.4647–0.4668 and I3 0.5147–0.5572.

What OP3 shows:

- **v3's chemCPA collapses only partly here.** Its drug term has an effect size ratio of 0.19, not 0.007, and it beats zero change on validation by 0.001. The sci-Plex3 collapse therefore depends on the data as well as on the model: OP3 cells have 197 nonzero frozen genes per cell on average, sci-Plex3 cells 59.
- **The NB likelihood and the pseudobulk loss both enlarge the drug term and improve its direction.** r is 0.20 for I1 and 0.17 for I2, against 0.11 for A0. Both lower validation RMSE relative to A0, by 0.005–0.006.
- **I3 memorizes the training deltas.** Its training RMSE of 0.29 is far below zero change (0.48), while its validation error is above zero change.
- **Every arm is still worse than the training mean and the Ridge on validation compounds.** The comparison is 0.4599–0.4660 against 0.4510–0.4523. Every arm is also far worse than the anchored baselines (about 0.363).

**Primary arm, chosen by the prespecified rule on validation only:** I1, the NB likelihood. Its mean per-seed validation RMSE is 0.4645, against 0.4655 for I2. I3 does not meet the criteria. The selection record is [selection_op3.json](../../receipts/v4/op3/sealed/selection_op3.json).

## Fits

| Setting | Arm | Best epochs (per seed) | Stopped | Total minutes |
|---|---|---|---|---:|
| sci-Plex3 Q1 | A0 / I1 / I2 / I3 | 40,30,20 / 40,10,10 / 30,70,40 / 10,10,10 | patience | 6 / 7 / 12 / 11 |
| sci-Plex3 Q2 | A0 / I1 / I2 / I3 | 70 / 10 / 70 / 10 | I2 at the 60-minute cap, others by patience | 55 / 33 / 62 / 53 |
| OP3 | A0 / I1 / I2 / I3 | 30,10,20 / 30,30,100 / 10,20,20 / 10,50,10 | patience | 42 / 72 / 71 / 90 |

Kaggle T4 ×2. The OP3 fits are from kernel version 2, after amendment A2 capped divergent decoder outputs. Version 1 stopped when two fits overflowed; its outputs are archived unread. No OP3 v2 fit had a non-finite validation evaluation. Run records are in [kaggle_runs.md](../../receipts/v4/kaggle_runs.md).

## What this does and does not establish

These are training and validation results. They show that each intervention changes how chemCPA uses the drug term, and how. Validation compounds also chose the checkpoints and the primary arm, so their errors are optimistic for the selected configurations. Test-compound performance, including whether the selected I1 beats zero change, the training mean or the Ridge, comes only from the sealed evaluation.
