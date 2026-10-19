# DrugPerturb v3 results: chemCPA and biolord as baselines

**Neither published single-cell perturbation model beat information-matched simple baselines.** chemCPA collapsed to zero change; biolord predicted weak, non-zero effects that were no better than zero change. When the models were given cell type, dose and target-context training data (Q2), a per-context fingerprint Ridge and even the per-context training mean were better than both models in every context. They were also better than zero change in every context, which reverses the v2 conclusion that zero change was hard to beat.

Design and boundaries: [V3_METHODS.md](../../docs/V3_METHODS.md) · protocol: [protocol_v3.yaml](../../configs/protocol_v3.yaml) · tables: [summary.tsv](summary.tsv), [contrasts.tsv](contrasts.tsv), [per_compound.tsv](per_compound.tsv)

Error is the mean over compounds of per-compound RMSE across the 2,000 frozen genes (lower is better). Intervals are paired scaffold-group bootstrap 95% intervals (2,000 draws). Test truth was first computed after all predictions were sealed. Recomputed truth reproduces the published v1/v2 per-compound RMSE in all six previously opened contexts to 2.2×10⁻¹⁶ ([truth_anchors.tsv](truth_anchors.tsv)).

## Q1: same information as v1 (A549, 24 h, 1 µM; 26 compounds, 24 scaffold groups)

| Method | Mean compound RMSE |
|---|---:|
| Zero change (B0) | 1.2694 |
| Training mean (B1) | 1.2170 |
| Nearest training compound (B2) | 1.7149 |
| Fingerprint Ridge (M1, v1) | 1.2430 |
| Molecular GNN (M3, v1) | **1.2144** |
| chemCPA | 1.2694 |
| biolord | 1.2654 |

| Contrast | Difference [95% interval] |
|---|---|
| chemCPA − Ridge | +0.026 [−0.006, +0.057] |
| chemCPA − training mean | +0.052 [+0.046, +0.059] |
| chemCPA − zero change | −0.0000 [−0.0001, +0.0000] |
| biolord − Ridge | +0.022 [−0.012, +0.055] |
| biolord − training mean | +0.049 [+0.033, +0.061] |
| biolord − zero change | −0.004 [−0.016, +0.003] |

Both models are reliably worse than the training mean and are not distinguishable from zero change.

## Q2: context-aware prediction, confirmatory endpoint on six never-scored contexts

Pooled over A549 and K562 at 10 nM, 100 nM and 10 µM (all six pass the count gate; one scaffold-group draw shared across contexts):

| Contrast | Difference [95% interval] | Superiority |
|---|---|---|
| chemCPA − zero change | −0.0000 [−0.0001, +0.0000] | no |
| chemCPA − context Ridge (M1c) | **+0.057 [+0.038, +0.074]** | no, reliably worse |
| biolord − zero change | −0.005 [−0.016, +0.001] | no |
| biolord − context Ridge (M1c) | **+0.052 [+0.031, +0.072]** | no, reliably worse |

The prespecified rule required the upper bound below zero against both comparators; neither model meets it. Because chemCPA's predictions equal zero change to four decimals, the chemCPA − M1c row also measures how far the context Ridge improves on zero change: about 0.057, reliably.

### Every context

| Context | Exposure | Zero change | Context train mean (B1c) | Context Ridge (M1c) | Transported v1 Ridge | chemCPA | biolord |
|---|---|---:|---:|---:|---:|---:|---:|
| A549 10 nM | new | 1.0910 | **1.0229** | 1.0386 | 1.1677 | 1.0910 | 1.0891 |
| A549 100 nM | new | 1.1079 | **1.0330** | 1.0485 | 1.1860 | 1.1079 | 1.1044 |
| A549 1 µM | opened (v1) | 1.2694 | **1.2170** | 1.2430 | 1.2430 | 1.2694 | 1.2603 |
| A549 10 µM | new | 1.1551 | **1.0817** | 1.0889 | 1.2180 | 1.1549 | 1.1418 |
| K562 10 nM | new | 1.0426 | **0.9738** | 0.9927 | 1.1231 | 1.0426 | 1.0403 |
| K562 100 nM | new | 1.0350 | **0.9704** | 0.9859 | 1.1118 | 1.0350 | 1.0312 |
| K562 1 µM | opened (v2) | 1.0400 | **0.9744** | 0.9866 | 1.1207 | 1.0399 | 1.0318 |
| K562 10 µM | new | 1.0754 | 1.0127 | **1.0115** | 1.1483 | 1.0753 | 1.0720 |
| MCF7 10 nM | opened (v2) | 0.4707 | **0.4357** | 0.4427 | 0.6304 | 0.4707 | 0.4686 |
| MCF7 100 nM | opened (v2) | 0.4560 | **0.4273** | 0.4326 | 0.6187 | 0.4560 | 0.4535 |
| MCF7 1 µM | opened (v2) | 0.5325 | **0.4956** | 0.4993 | 0.6732 | 0.5325 | 0.5254 |
| MCF7 10 µM | opened, below gate | 0.5382 | 0.5057 | **0.4967** | 0.6778 | 0.5381 | 0.5203 |

Among the methods in this table, the per-context training mean is lowest in ten of twelve contexts and the per-context Ridge in the other two. In A549 1 µM the v1 GNN (1.2144, Q1 table) is marginally lower still. v2 found zero change best everywhere because its models had seen only A549. Here the simplest use of target-context data is enough to beat zero change; the two deep models add nothing beyond that. This per-context comparison of B1c with zero change is descriptive: it was not a prespecified contrast.

## Why chemCPA predicts nothing

A training/validation-only diagnostic refitted chemCPA on Q1 ([diagnostics](diagnostics)). Even on the 119 **training** compounds, the predicted counterfactual effect is tiny: mean |Δ| 0.0009 against a measured 1.05, gene-wise Pearson r 0.02. In the latent space, the drug term has norm 0.08 against 14.6 for the basal state. The autoencoder reconstructs each cell from its own basal latent, and the drug pathway is effectively unused. This is the disentanglement failure that CPA-type models are prone to: with a learned-variance Gaussian likelihood and weak per-cell drug effects, the model can explain treated cells without moving the mean. LINCS pretraining, which the chemCPA authors recommend, was not performed here.

Early stopping does not explain the collapse. In a second diagnostic without early stopping, the repository-default configuration ran all 200 epochs; its validation RMSE was 1.3027–1.3029 at every one of the 20 evaluations, against 1.3029 for zero change. The biolord-reproducibility configuration reached 131 epochs within the diagnostic's 60-minute cap and stayed at 1.3029 at all 13 evaluations. The effect sizes and latent norms quoted above come from each run's best validation checkpoint.

biolord did learn non-zero effects (mean |Δ| 0.14 on validation compounds), but they correlate weakly with measured effects (gene-wise r ≈ 0.06), so its error stays at the zero-change level.

## The anchored sensitivity is not a fair comparison

The prespecified anchored sensitivity subtracts the **measured** vehicle well instead of the model's own vehicle prediction. For biolord in A549 and K562 it gives much lower errors (for example 0.96 against 1.02 for the context mean in A549 10 nM), but worse ones in MCF7 (0.70–0.74 against 0.43–0.51). This is expected from the error structure, not a sign of a better model. The measured truth is T − V, with V the noisy vehicle well. An anchored prediction P − V shares the same −V, so its error P − T no longer contains the vehicle noise, while every delta predictor's error does. Anchored scores therefore answer a different question, predicting the treated well's absolute level, and are not used for any claim, as prespecified.

## What this does and does not show

It shows that, on sci-Plex3 at 24 h, with these published configurations, without LINCS pretraining and within a Kaggle T4 compute budget (90 minutes per chemCPA fit and 150 per biolord fit; amendment A2), neither model beats the simplest information-matched baselines for held-out compounds. It does not show that these models cannot work with pretraining, more compute, other output spaces, or other datasets, and it says nothing about drug efficacy or mechanism.
