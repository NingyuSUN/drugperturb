# DrugPerturb v1: on held-out compounds, structure-based models have no reliable advantage over simple baselines

**The molecular graph network had the lowest error of the six methods, and a reliable advantage over the
strong simple baselines was not established.** The validation-selected simple reference was the fingerprint
Ridge:

| Contrast | Difference [95% interval] |
|---|---|
| Molecular GNN (M3) − fingerprint Ridge (M1) | −0.0286 [−0.0680, +0.0087] |
| Molecular GNN (M3) − training mean (B1) | −0.0026 [−0.0475, +0.0419] |

Both intervals include zero. The one contrast that does exclude zero is against predicting nothing: M3 − zero
change is −0.0550 [−0.1008, −0.0090].

The study: sci-Plex3 A549 nuclei at 24 h and 1 µM, 26 held-out compounds in 24 chemical structure groups,
2,000 training-selected genes, predictions sealed before test scoring.

This is the original single-context benchmark. What became of its conclusion over the five later versions is
in the [v1–v6 synthesis](../synthesis.md) ([中文](../synthesis.zh-CN.md)); this report covers v1 alone.

- **Design:** [protocol_v1.yaml](../../configs/protocol_v1.yaml), [research decisions](../../docs/RESEARCH_DECISIONS.md), [model card](../../docs/MODEL_CARD.md), [data card](../../docs/DATA_CARD.md).
- **The complete original report**, with every figure: [report.html](../report.html) and [biologist_summary.html](../biologist_summary.html) — both are HTML and need a local clone to render, which is why this markdown edition exists.
- **Tables:** [final_test/](../final_test), [sensitivity/](../sensitivity).
- **Runs and hashes:** [training runs](../../receipts/training_runs.tsv), [model selection](../../receipts/model_selection.json), [prediction seal](../../receipts/prediction_seal.json), [final evaluation](../../receipts/final_evaluation.json).

**Metrics.** Error is the mean over compounds of per-compound RMSE across the 2,000 frozen genes; lower is
better. Intervals are paired bootstrap 95% intervals that resample the 24 whole structure groups with
replacement: 2,000 draws, seed 2026, compounds weighted equally. A negative difference favours the first
method. They are conditional on one split and on the fitted models, so scaffold independence, shared-control
and plate effects and retraining variability are not captured.

## Data

**Source.** Published sci-Plex3 nuclear expression, [GEO GSE139944 / GSM4150378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378), with the pinned author preprocessing. Every file read was hash-checked ([source manifest](../../manifests/source_manifest.tsv)).

**What was used:**

- **Compounds.** Of 188 source treatment labels, 171 distinct standardized parent compounds pass both the chemical and the paired-cell criteria, split 119 training / 26 validation / 26 test. The test set holds 24 structure groups: 22 Bemis–Murcko ring scaffolds and two acyclic parent groups. Canonical parent and group identities do not cross partitions.
- **Cells.** The primary response uses 33,055 treatment nuclei and 886 nuclei in eight unique matched vehicle wells. The prediction unit is a compound with two experimental replicates, not a nucleus.
- **Context.** One cell line, one time point, one dose: A549, 24 h, 1 µM. The models receive chemical structure only — not cell identity, not dose.
- **Response.** Raw counts summed per well, converted to CPM over all human genes, `log1p` transformed, then contrasted against the matched vehicle well. The two replicate contrasts receive equal weight.
- **Output genes.** 2,000 variable genes selected on the 119 training responses only; standardization and the 20-component PCA are also fitted on training only. Every model is scored against measured changes, never against a PCA reconstruction.
- **Chemical novelty.** The maximum Tanimoto similarity to any training compound has median 0.2819 across the test set (range 0.1587–0.5957).

## Every method

| Method | Mean compound RMSE ↓ | Mean compound MAE ↓ | Mean Spearman ↑ | Mean cosine ↑ | Sign agreement ↑ |
|---|---:|---:|---:|---:|---:|
| B0, zero change | 1.2694 | 1.0115 | NA | NA | 0.0000 |
| B1, training mean | 1.2170 | 0.9849 | 0.2698 | 0.2849 | 0.5964 |
| B2, nearest training compound | 1.7149 | 1.3926 | 0.0725 | 0.0809 | 0.5032 |
| M1, fingerprint Ridge | 1.2430 | 1.0105 | 0.2127 | 0.2283 | 0.5718 |
| M2, fingerprint MLP | 1.2532 | 1.0189 | 0.1920 | 0.2062 | 0.5669 |
| **M3, molecular GNN** | **1.2144** | **0.9848** | **0.2755** | **0.2886** | **0.5985** |

M2 and M3 are the arithmetic mean of three seeds. Individually: M3 gives 1.2195 (seed 42), 1.2086 (123) and
1.2209 (2026); M2 gives 1.2649, 1.2483 and 1.2488. Zero change has no defined Spearman or cosine — it is a
constant vector — and its sign agreement is 0 by construction, since only non-zero predictions count.

## Confirmatory contrasts

All eleven prespecified paired comparisons, from [paired_comparisons.json](../final_test/paired_comparisons.json):

| Contrast | Difference [95% interval] | Reliable? |
|---|---|---|
| M3 − M1 (primary) | −0.0286 [−0.0680, +0.0087] | no |
| M3 − B1 | −0.0026 [−0.0475, +0.0419] | no |
| M3 − B0 | −0.0550 [−0.1008, −0.0090] | yes, M3 better |
| M3 − B2 | −0.5005 [−0.6001, −0.3894] | yes, M3 better |
| M1 − B0 | −0.0264 [−0.0570, +0.0061] | no |
| M1 − B1 | +0.0260 [−0.0041, +0.0583] | no |
| M1 − B2 | −0.4719 [−0.5789, −0.3533] | yes, M1 better |
| M2 − M1 | +0.0102 [−0.0044, +0.0256] | no |
| M2 − B0 | −0.0162 [−0.0500, +0.0211] | no |
| M2 − B1 | +0.0362 [+0.0023, +0.0747] | yes, B1 better |
| M2 − B2 | −0.4617 [−0.5714, −0.3426] | yes, M2 better |

Read together: every learned model buries the nearest-neighbour baseline, the GNN reliably beats predicting
nothing, the MLP is reliably worse than the training mean, and no method is reliably better than the
training mean or the Ridge. The training mean itself has lower error than zero change (1.2170 against
1.2694); no interval was prespecified for baseline-against-baseline, so that gap is a point estimate only.

## Where the differences come from

**The measurement is noisier than the response.** For each test compound, the RMSE between its own two
replicate responses has median 2.1009 (range 1.9650–2.2637), while the RMS of its averaged response has
median 1.2471 (1.1762–1.6477). The two replicates disagree by more than the response itself for **26 of 26**
compounds, and their rank correlation has median 0.1802. On an RMSE endpoint over 2,000 genes, that is the
structural reason a conservative prediction is hard to beat — the fact that v6 eventually answered by
changing the endpoint rather than the model.

**Per compound the ranking is a coin flip.** Counting which method has the lowest error on each test
compound: M3 wins 10, B1 7, M2 5, B2 3, M1 1 and B0 none. M3 is below zero change on 20 of 26 compounds but
below the training mean on only 13 of 26 — the pooled dead heat is not an average of consistent small wins.

**Chemical nearest neighbour is the one clear failure.** B2 copies the response of the most similar training
compound and is worse than every other method by roughly 0.46–0.50 RMSE. At a median similarity of 0.28,
the nearest training compound is not a usable surrogate.

**Direction is about 60% right at best.** Sign agreement reaches 0.5985 for M3 and 0.5964 for B1, against
0.5032 for B2. Nothing here resolves which genes move: the three prespecified fixed-rule cases — highest GNN
error (abexinostat), lowest response (fluorouracil) and lowest training similarity (GSK-LSD1) — are retained
in [fixed_rule_cases.tsv](../final_test/fixed_rule_cases.tsv) rather than selected after seeing the results.

## The control definition changes the ranking

The same frozen predictions, scored against a different vehicle definition. Nothing is retrained, no
prediction changes and the output genes are unchanged ([sensitivity/summary.json](../sensitivity/summary.json)):

| Scenario | n | B0 | B1 | B2 | M1 | M2 | M3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Primary, matched vehicle well | 26 | 1.2694 | 1.2170 | 1.7149 | 1.2430 | 1.2532 | **1.2144** |
| Mean of two separately normalized same-plate vehicles | 26 | **1.0131** | 1.0150 | 1.5970 | 1.0368 | 1.0448 | 1.0395 |
| Both treatment replicates ≥ 50 cells | 24 | 1.2660 | 1.2110 | 1.6988 | 1.2333 | 1.2434 | **1.2051** |

Under the two-vehicle definition zero change has the lowest error and every learned model falls behind it.
The absolute errors also drop, because the truth itself changed. This is the first appearance in the project
of the lesson that v4 later turned into a rule and v5 and v6 built on: **what a model is compared against
depends on what the comparison is allowed to see**, and a reference that uses the matched control's own
measurement is a different, stronger reference.

## Development diagnostics

Exploratory, single-seed, with the gene space refitted per fold, so these are not controlled comparisons:
the GNN has higher error than the Ridge in all three compound cross-validation folds (1.2577 against 1.2412,
1.2751 against 1.2378, 1.2366 against 1.2300). On the learning curve the two are indistinguishable at 25%
and 50% of the training compounds (1.2279 against 1.2270, then 1.2284 against 1.2317), and the GNN is lower
only at the full 119 (1.2130 against 1.2432) — so what little separates them appears at full training size
and does not survive the folds.
[development_diagnostics.tsv](../../receipts/development_diagnostics.tsv).

## Biological coverage

All five prespecified gene sets fail the frozen coverage gate — at least 10 output genes and 25% of the
original members — inside the 2,000-gene output space, so **no pathway-recovery claim is made**:

| Set | Original members | In the output space | Fraction |
|---|---:|---:|---:|
| Author S-phase markers | 43 | 2 | 0.047 |
| Author G2/M markers | 54 | 9 | 0.167 |
| HDACs deacetylate histones (R-HSA-3214815) | 60 | 5 | 0.083 |
| HATs acetylate histones (R-HSA-3214847) | 108 | 6 | 0.056 |
| Cellular responses to stress (R-HSA-2262752) | 791 | 55 | 0.070 |

The full 43-gene S and 54-gene G2/M marker sets were used separately for descriptive single-cell scores with
all-human-UMI denominators; those are descriptions, not phase assignments.

## What this does and does not show

**It shows:**

- A graph network trained on 119 compounds reaches the lowest point error of six methods on held-out chemistry, and the interval against the validation-selected reference includes zero. At this sample size the ranking of the top four methods is not resolved.
- Predicting the training mean response is a strong baseline, and copying the nearest training compound is a bad one.
- The conclusion depends on the control definition: the ranking inverts under a two-vehicle reference.
- The measurement's replicate disagreement exceeds the response magnitude for every test compound.

**It does not show:**

- That structure-based models cannot predict these responses. It shows that this benchmark, at 26 test compounds in 24 structure groups, cannot establish an advantage of the size observed.
- That the drugs have no effect. Zero change winning an error endpoint is a property of the endpoint and the noise, not of the biology.
- Anything about other cell lines, doses or time points — v2 tested exactly that and found no transport.
- Efficacy, toxicity, target engagement, clinical benefit or antibody design. No pretrained foundation model and no GPU training are claimed for this version; the small models ran on CPU.

## Reproducibility

- **Protocol** `configs/protocol_v1.yaml`, frozen before any fit; predictions sealed before test truth was scored ([prediction seal](../../receipts/prediction_seal.json)).
- **Selection.** Four fixed candidates per learned family; M2 and M3 each use seeds 42, 123 and 2026, the candidate is chosen by mean seed validation error, and its three predictions are averaged. Checkpoints are selected on measured gene-space error, not on PCA-coordinate loss ([model selection](../../receipts/model_selection.json)).
- **All 28 learned-model fits** are recorded with their trajectories ([training_runs.tsv](../../receipts/training_runs.tsv)).
- **Two independent core runs** on the same fixed host produced identical 24 validation and test prediction arrays. Cross-host retraining did not meet the strict 1e-6 comparison for the eight GNN arrays; deploying the saved primary GNN weights in Docker passed all eight inference comparisons at a maximum difference of 4.44×10⁻¹⁶. Those failed comparisons are retained rather than removed ([execution evidence](../../docs/EXECUTION_EVIDENCE.md)).
- **Inference.** The [S1090 replay](../inference_example/S1090.json) matches its sealed prediction within 1.8×10⁻⁸; the CLI rejects unsupported contexts.
- **Tests** validate software, not biology: no test here checks a biological claim.
