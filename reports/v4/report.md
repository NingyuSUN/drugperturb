# DrugPerturb v4: fixing chemCPA's collapse does not make it beat simple baselines

**Changing chemCPA's objective made it use the drug, but no version predicted held-out compounds better than a per-context training mean or Ridge on data this project had never scored.** The study ran on OP3, PBMCs from three donors, with 35 test compounds. It was prespecified, and its predictions were sealed before testing.

The primary arm, chosen on validation compounds, uses a negative binomial (NB) likelihood. It failed the confirmatory test, ending up worse than zero change on average. Almost all of that came from one compound, ixabepilone. Its chemical descriptors were mis-scaled by a bug in the descriptor scaling inherited from v3 (z-scores near 10¹⁶). Excluding it post hoc, the NB arm matches zero change and is still reliably worse than the training mean and the Ridge.

On sci-Plex3, the counterfactual pseudobulk loss produced large apparent gains. A vehicle-anchored training mean, which uses no compound information beyond the context mean, does better still. The gains therefore come from the matched vehicle well's measurement, not from predicting compound effects.

Design: [V4_METHODS.md](../../docs/V4_METHODS.md) · protocol and amendments: [protocol_v4.yaml](../../configs/protocol_v4.yaml) · training/validation diagnostics, committed before test scoring: [diagnostics.md](diagnostics.md) · tables: [op3/](op3), [sciplex/](sciplex)

Error is the mean over compounds of per-compound RMSE across the 2,000 frozen genes (lower is better). Intervals are paired scaffold-group bootstrap 95% intervals (2,000 draws, seed 2026). Pooled estimates average the context-level differences over the OP3 contexts that pass the count gate, with one shared draw per bootstrap replicate. Test truth was first computed by the evaluation kernel, after every prediction had been sealed.

## Diagnose, intervene: does the drug term come alive?

v3 found chemCPA's drug term at 0.6% of the latent norm and its predictions equal to zero change. v4 tested two mechanisms:

- **A count likelihood, I1.** The NB likelihood with library offsets puts the decoder mean on the scale of the response.
- **A counterfactual pseudobulk loss, I2.** It trains the exact path used for prediction, vehicle cells plus the drug term, against measured well-level training deltas.

I3 combines both. A0 is the unchanged v3 model.

The table shows validation compounds only (from [diagnostics.md](diagnostics.md)). The effect size ratio is the size of the predicted change relative to the average training effect; r is the gene-wise correlation with the measured change.

| Arm | sci-Plex3 Q1: ratio / r | sci-Plex3 Q2: ratio / r | OP3: ratio / r | OP3 validation RMSE (zero change 0.467, B1c 0.452) |
|---|---|---|---|---:|
| A0, v3 reference | 0.007 / −0.01 | 0.009 / 0.00 | 0.19 / 0.11 | 0.466 |
| I1, NB likelihood | 0.25 / 0.03 | 0.27 / 0.04 | 0.35 / 0.20 | 0.461 |
| I2, pseudobulk loss | 1.62 / 0.64 | 0.49 / 0.18 | 0.59 / 0.17 | 0.460 |
| I3, both | 2.78 / 0.75 | 1.14 / 0.15 | 1.53 / 0.20 | 0.491 |

Both mechanisms make the drug term non-trivial:

- **The NB likelihood enlarges the drug term everywhere** (ratio about 0.25–0.35 against 0.007), but its direction is informative only on OP3.
- **The pseudobulk loss makes the drug term large and correlated with the truth.**
- **On OP3, even v3's model is only partly collapsed**, with a ratio of 0.19. OP3 cells carry 197 nonzero frozen genes on average, against 59 in sci-Plex3.

By the prespecified criteria, I1 and I2 are non-collapsed on OP3 and I3 overfits; I1 has the lowest validation RMSE and was selected.

## Confirmatory result on OP3

OP3 test compounds: T cells 33, B cells 28, myeloid cells 26. NK cells have only 9 eligible test compounds, below the gate of 20, so they are descriptive only.

| Endpoint | Contrast | Difference [95% interval] | Supported |
|---|---|---|---|
| E1, beats zero change | I1 − zero change | +0.171 [−0.004, +0.532] | no |
| E2, beats delta baselines | I1 − training mean (B1c) | +0.183 [+0.009, +0.544] | no, reliably worse |
| | I1 − Ridge (M1c) | +0.184 [+0.009, +0.546] | |
| E3, beats vehicle-anchored baselines (amendment A1) | I1 − anchored mean (B1a) | +0.258 [+0.082, +0.619] | no, reliably worse |
| | I1 − anchored Ridge (M1a) | +0.258 [+0.081, +0.619] | |

### Every method, every context (OP3 test)

| Method | T cells | B cells | Myeloid | NK (below gate) |
|---|---:|---:|---:|---:|
| Zero change (B0) | 0.3225 | 0.5859 | 0.3354 | 0.8914 |
| Training mean (B1c) | 0.3050 | 0.5701 | 0.3312 | 0.8900 |
| Fingerprint Ridge (M1c) | 0.3038 | 0.5705 | 0.3311 | 0.8905 |
| Nearest training compound (B2c) | 0.4188 | 0.7966 | 0.4762 | 1.3079 |
| Anchored training mean (B1a) | **0.2537** | 0.4402 | **0.2874** | **0.6605** |
| Anchored Ridge (M1a) | 0.2544 | **0.4400** | 0.2877 | 0.6607 |
| A0, v3 reference | 0.3211 | 0.5844 | 0.3367 | 0.8900 |
| I1, NB likelihood (primary) | 0.4767 | 0.7570 | 0.5228 | 0.8931 |
| I2, pseudobulk loss | 0.3184 | 0.5847 | 0.3435 | 0.8883 |
| I3, both | 0.4896 | 0.7902 | 0.5517 | 0.9555 |

All arms, pooled over the three gate-passing contexts (prespecified secondary contrasts, unadjusted):

| Arm | − zero change | − training mean | − Ridge | − anchored mean |
|---|---|---|---|---|
| A0 | −0.001 [−0.002, +0.001] | +0.012 [+0.004, +0.021] | +0.012 [+0.003, +0.022] | +0.087 [+0.076, +0.100] |
| I1 | +0.171 [−0.004, +0.532] | +0.183 [+0.009, +0.544] | +0.184 [+0.009, +0.546] | +0.258 [+0.082, +0.619] |
| I2 | +0.001 [−0.007, +0.010] | +0.014 [+0.006, +0.023] | +0.014 [+0.006, +0.023] | +0.089 [+0.078, +0.100] |
| I3 | +0.196 [+0.002, +0.588] | +0.208 [+0.019, +0.599] | +0.209 [+0.020, +0.599] | +0.283 [+0.090, +0.674] |

The per-context training mean beats zero change on OP3, as it did on sci-Plex3: −0.013 [−0.023, −0.003], a descriptive post hoc contrast. Adding the matched vehicle level (B1a − B1c) lowers error by a further 0.075 [0.064, 0.086].

## Why the primary arm failed: a descriptor-scaling artifact inherited from v3

This diagnosis is post hoc; it was found after test scoring.

I1's error is dominated by one test compound. Ixabepilone has a per-compound RMSE of 5.0–5.3 in every gate-passing context, against 0.3–0.6 for the others. The v3 descriptor scaler keeps any training column with nonzero standard deviation. In both datasets, 25–28 RDKit2DNormalized columns are constant in training up to floating-point noise, with standard deviations down to 10⁻³³.

- **In OP3, four compounds hit those columns:** ixabepilone, dasatinib, alvocidib (all test) and CEP-37440 (validation). They get z-scores of about 5–9×10¹⁵.
- **The arms differ in how they handle the extreme input.** Under the NB log link, the decoder mean saturates for ixabepilone, while the Gaussian arms A0 and I2 do not blow up.
- **No sci-Plex3 compound has a z-score above 50**, so neither the v3 nor the sci-Plex3 v4 results are affected.

Post hoc sensitivity analysis ([posthoc_sensitivity.tsv](op3/posthoc_sensitivity.tsv); not prespecified):

| Contrast | All eligible (prespecified) | Excluding ixabepilone | Excluding the 3 affected test compounds |
|---|---|---|---|
| I1 − zero change | +0.171 [−0.004, +0.532] | +0.001 [−0.005, +0.008] | +0.001 [−0.006, +0.009] |
| I1 − training mean | +0.183 [+0.009, +0.544] | +0.015 [+0.007, +0.023] | +0.014 [+0.006, +0.022] |
| I1 − Ridge | +0.184 [+0.009, +0.546] | +0.015 [+0.007, +0.023] | +0.014 [+0.006, +0.023] |
| I2 − training mean | +0.014 [+0.006, +0.023] | +0.015 [+0.007, +0.024] | +0.014 [+0.006, +0.023] |

Even without the artifact, no arm beats the per-context training mean or Ridge on OP3. The conclusion does not depend on ixabepilone. A tolerance-based column filter is the obvious fix; it is not applied retroactively, because the OP3 test outcomes are now open.

## sci-Plex3 (exploratory: every test outcome was opened in v1–v3)

Fresh truth reproduces the committed v3 per-compound RMSE, to at most 2.2×10⁻¹⁶, for zero change, the sealed B1c and M1c, and the sealed v3 chemCPA, in all 12 contexts ([truth_anchors.tsv](sciplex/truth_anchors.tsv)).

**Q1, A549 1 µM, 26 compounds, three seeds:**

| Method | RMSE |
|---|---:|
| Zero change | 1.2694 |
| Training mean (v1 B1) | 1.2170 |
| Fingerprint Ridge (v1 M1) | 1.2430 |
| Molecular GNN (v1 M3) | 1.2144 |
| Anchored training mean (B1a) | **0.8042** |
| Anchored Ridge (M1a) | 0.8183 |
| v3 chemCPA / A0 | 1.2694 / 1.2694 |
| I1, NB likelihood | 1.2670 |
| I2, pseudobulk loss | 1.0047 |
| I3, both | 0.8372 |

The contrasts for the two pseudobulk arms are:

| Contrast | Difference [95% interval] |
|---|---|
| I3 − v1 training mean | −0.380 [−0.410, −0.353] |
| I3 − anchored mean | +0.033 [+0.004, +0.060] |
| I3 − anchored Ridge | +0.019 [−0.002, +0.042] |
| I2 − v1 training mean | −0.212 [−0.230, −0.193] |
| I2 − anchored mean | +0.200 [+0.186, +0.216] |

**Q2, pooled over the six contexts first scored in v3, seed 42 only:**

| Arm | − zero change | − training mean | − Ridge | − anchored mean |
|---|---|---|---|---|
| A0 | −0.000 [−0.000, −0.000] | +0.069 [+0.060, +0.077] | +0.057 [+0.038, +0.074] | +0.306 [+0.300, +0.313] |
| I1 | −0.005 [−0.015, +0.001] | +0.064 [+0.051, +0.075] | +0.052 [+0.032, +0.071] | +0.302 [+0.290, +0.311] |
| I2 | **−0.026 [−0.029, −0.022]** | +0.043 [+0.038, +0.048] | +0.031 [+0.013, +0.048] | +0.280 [+0.276, +0.285] |
| I3 | +0.043 [+0.016, +0.069] | +0.112 [+0.084, +0.139] | +0.100 [+0.071, +0.128] | +0.349 [+0.320, +0.378] |

The truth T − V contains the sampling noise of the matched vehicle well. The A549 1 µM context has only eight vehicle wells, with about 110 cells each. A model given those vehicle cells can partly cancel that noise; a delta-only baseline cannot. The anchored training mean shows how much the vehicle measurement alone is worth: 0.80 against 1.22 in A549 1 µM.

I3's apparent 30% gain over every v1 method is mostly this effect. It stays above the anchored mean, which knows nothing about compound structure. In Q2, I2 is the only chemCPA version that reliably beats zero change, but it remains worse than the per-context training mean.

## What changed from the prespecified plan

| Change | When | Information used |
|---|---|---|
| A1: vehicle-anchored comparators B1a and M1a, and endpoint E3 | after the sci-Plex3 fits, before any OP3 fit output or test truth | sci-Plex3 validation diagnostics |
| A2: cap per-cell values at log1p(10⁴) before the pseudobulk mapping; rerun all 12 OP3 fits | after OP3 fit kernel v1 stopped (2 of 12 fits overflowed) | the two error messages; v1 outputs archived unread |
| OP3 SHA256 recorded at download rather than before the protocol | a local stream broke at 6.4 of 14.2 GB | none; an independent local download later matched the hash |

E1 and E2 were not changed. All tables above report them as prespecified.

## What this does and does not show

It shows that chemCPA's collapse on sci-Plex3 has two separable parts:

- **Size.** The likelihood can switch the drug term's magnitude on.
- **Direction.** The prediction path has to be trained directly before the drug term points the right way.

It also shows that neither fix, alone or combined, beats a per-context training mean or Ridge on 87 held-out compound–context pairs in three PBMC cell types, within this compute budget (60 minutes per fit on a Kaggle T4) and with published hyperparameters. More broadly, any model conditioned on the matched vehicle cells must be compared with a vehicle-anchored baseline: otherwise vehicle-noise cancellation looks like predictive skill.

It does not show that chemCPA cannot work with LINCS pretraining, other descriptors, a corrected descriptor scaler or tuned hyperparameters. OP3 is untouched only in the sense that this project had never scored it; it is public and others have analysed it. Nothing here bears on drug efficacy, toxicity or mechanism.

## Reproducibility

Kernels, versions and code-archive hashes are listed in [kaggle_runs.md](../../receipts/v4/kaggle_runs.md).

- **OP3 source:** [op3_source.tsv](../../manifests/v4/op3_source.tsv). SHA256 591afcbf…, confirmed by two independent downloads.
- **Pipeline:** extraction, preparation, anchored baselines, fits (three T4 ×2 sessions: sci-Plex3 2.1 h, OP3 v1 2.1 h, which failed, and OP3 v2 2.4 h; about 6.5 hours of Kaggle GPU quota), sealing, evaluation.
- **Receipts:** every receipt chains input and output hashes.
- **Tests:** synthetic tests cover the design, QC, truth, n-donor contrast, diagnostics, anchoring, both objectives, the A0 = v3 identity and an end-to-end synthetic OP3 run. These are software checks, not biological validation.
