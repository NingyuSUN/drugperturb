# DrugPerturb v2: limited transport of A549-derived perturbation predictions

**The frozen structure-based predictors did not outperform zero change on mean error in any of the five target conditions.** In the primary MCF7, 24-hour, 1 µM test, the molecular graph network had higher gene-level error than fingerprint Ridge regression and the source-training mean. A separate predictor of five predefined gene-set means had a lower point error than the source mean, but its paired interval crossed zero and its error remained higher than zero change. These results do not establish useful transport of the source response to another cell line or dose.

This is a completed, prespecified **within-study condition-transfer evaluation**, using public sci-Plex3 data. It is not independent-study validation, a drug-efficacy study, or a test of a model that receives cell identity or concentration. [中文报告](report.zh-CN.md) · [Methods](../../docs/V2_METHODS.md) · [Frozen protocol](../../configs/protocol_v2.yaml).

## Question, data, and denominators

The source condition was A549, 24 hours, 1 µM. The original split of 119 training, 26 validation, and 26 test parent compounds was retained. The sole primary target was MCF7 at 24 hours and 1 µM; K562 at 1 µM and MCF7 at 10 nM, 100 nM, and 10 µM were prespecified secondary conditions, all at 24 hours. The same test compounds recur across conditions, so summing condition-specific counts does not create additional independent chemical samples. Their A549 outcomes were previously examined in v1; the new target-condition responses were excluded from development and selection.

The two target source assays contained **438,762 retained nuclei**: 292,010 MCF7 and 146,752 K562, across all four doses and all source treatments and vehicles. Each cell line had 1,536 selected physical wells and all 58,347 supplied human gene rows. One MCF7 well was empty; all K562 wells were nonempty. These are extraction totals, **not the numbers of nuclei or wells belonging only to the evaluated test drugs**. The statistical prediction unit is a compound, with two experimental replicates, rather than an individual nucleus. [Extraction record](../../receipts/v2_context_extraction.json).

Both treatment replicates and their strictly matched vehicles had to contain at least 30 retained cells. Eligibility and the feasibility gate of at least 20 compounds and 10 structure groups were applied separately to each condition:

| Target condition, all 24 h | Role | Eligible / original test compounds | Structure groups | Strict-control evaluation | Two-vehicle sensitivity subset |
|---|---|---:|---:|---|---:|
| MCF7, 1 µM | **Primary** | 26 / 26 | 24 | Gate passed | 26 compounds / 24 groups |
| K562, 1 µM | Secondary | 25 / 26 | 23 | Gate passed | 25 / 23 |
| MCF7, 10 nM | Secondary | 25 / 26 | 23 | Gate passed | 18 / 17; descriptive only |
| MCF7, 100 nM | Secondary | 26 / 26 | 24 | Gate passed | 26 / 24 |
| MCF7, 10 µM | Secondary | 18 / 26 | 17 | **Descriptive only; no CI** | 18 / 17; descriptive only |

The MCF7 10 µM result falls below the compound threshold. Its errors are reported without confidence intervals; the threshold was not relaxed. Low retained cell counts alone do not demonstrate cytotoxicity. At 10 nM, the stricter two-vehicle rule removes seven additional compounds because a vehicle has fewer than 30 cells. [Condition status](transfer/context_status.tsv) · [Count qualification](context_count_qc.tsv) · [Compound exclusions](../../manifests/v2/compound_count_qc.tsv).

## What was predicted and measured

Raw counts were summed within each unique well. Each well was normalized to counts per million using **all human genes** in its library denominator, then transformed with `log1p`. The response was the treatment-minus-strictly-matched-vehicle difference within each replicate, followed by an equal-weight mean of the two replicate differences. Required gene identities had to match exactly; an absent gene was never replaced with zero.

**Analysis A** transported the six sealed v1 predictors across the fixed 2,000 output genes. B0 predicts zero change; B1 uses the source-training mean response; B2 uses the chemically nearest training compound; M1 is fingerprint Ridge regression; M2 is a fingerprint multilayer perceptron; M3 is a molecular graph neural network. Neural predictions use the frozen three-seed arithmetic ensembles. Each compound's root mean squared error (RMSE) is calculated across the 2,000 measured gene deltas, then compound RMSEs receive equal weight. This differs from pooling all gene–compound errors before taking one square root.

**Analysis B** directly predicted five fixed gene-set mean changes using a separate fingerprint Ridge model and matched-output B0, B1, and B2 baselines. Source-training-only filtering retained 43 S-phase, 54 G2/M, 40 histone-deacetylation (HDAC), 87 histone-acetylation (HAT), and 685 stress-response genes: 909 memberships covering 874 distinct genes. All five coverage gates passed. Four alpha candidates were fitted on the 119 source training compounds; alpha 100 was selected on the 26 source validation compounds, without a train-plus-validation refit. Memberships and predictions were sealed before target scoring. The error is mean absolute error (MAE) across five readouts within each compound, then equally across compounds. [Coverage](biology_coverage.tsv) · [Selection](../../receipts/v2_biology_selection.json) · [Prediction seal](../../receipts/v2_biology_prediction_seal.json).

All models use chemical structure alone. A compound therefore has the **same prediction in every target condition**. The evaluation asks whether an A549-derived prediction transports; it does not demonstrate learning of cell-specific or dose-dependent responses. Both error measures use log1p CPM delta units, but gene RMSE and five-set MAE describe different output spaces and should not be compared numerically as a common endpoint.

## A. All six gene-level methods

Values are equal-compound mean RMSE under strict matched controls; lower is better. All values below are rounded to six decimals; full precision is retained in the [summary](transfer/summary.json) and [per-compound table](transfer/per_compound.tsv).

| Condition | Compounds / groups | B0: zero | B1: source mean | B2: nearest | M1: Ridge | M2: MLP | M3: GNN |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MCF7, 1 µM — primary** | 26 / 24 | **0.532506** | 0.651318 | 1.403771 | 0.673216 | 0.676733 | 0.719755 |
| K562, 1 µM | 25 / 23 | **1.039957** | 1.107726 | 1.666624 | 1.120734 | 1.121957 | 1.155189 |
| MCF7, 10 nM | 25 / 23 | **0.470665** | 0.601861 | 1.382447 | 0.630400 | 0.632069 | 0.673526 |
| MCF7, 100 nM | 26 / 24 | **0.456026** | 0.590424 | 1.381053 | 0.618665 | 0.621092 | 0.662230 |
| MCF7, 10 µM — descriptive | 18 / 17 | **0.538237** | 0.658578 | 1.410857 | 0.677771 | 0.683493 | 0.743217 |

Zero change has the lowest point mean in every condition. In the primary test, GNN error is 0.719755 versus Ridge 0.673216 and source mean 0.651318. The chemically nearest training response has the highest point error in all five conditions. Structural similarity alone did not provide an accurate response reference under this transfer design.

## B. All four five-readout methods

Values are equal-compound mean five-set MAE under strict matched controls; lower is better. The five outputs are **gene-set readouts, not five individual genes**.

| Condition | Compounds / groups | B0: zero | B1: source mean | B2: nearest | Ridge |
|---|---:|---:|---:|---:|---:|
| **MCF7, 1 µM — primary** | 26 / 24 | **0.084674** | 0.157443 | 0.169654 | 0.140807 |
| K562, 1 µM | 25 / 23 | **0.117593** | 0.193319 | 0.203601 | 0.179629 |
| MCF7, 10 nM | 25 / 23 | **0.065556** | 0.151228 | 0.172768 | 0.130555 |
| MCF7, 100 nM | 26 / 24 | **0.064506** | 0.174265 | 0.174092 | 0.152868 |
| MCF7, 10 µM — descriptive | 18 / 17 | **0.132601** | 0.177093 | 0.219082 | 0.156463 |

The new Ridge predictor has lower point MAE than the source mean in all five conditions, but zero change remains the best point reference throughout. A smaller error than the source mean alone does not establish successful biological prediction.

One secondary comparison, MCF7 at 100 nM, has a Ridge-minus-source-mean difference of −0.021396 with a pointwise 95% interval of [−0.040373, −0.002597]. This supports a conditional improvement over that reference in that condition, while Ridge remains worse than zero change (0.152868 versus 0.064506). It does not change the primary result or establish superiority across conditions after multiple comparisons.

## Primary paired comparisons and uncertainty

Each difference is candidate error minus reference error **within the same compound**. Negative favors the candidate. Confidence intervals resample structure groups with replacement, retaining all compounds and multiplicity within sampled groups; each estimate remains compound-weighted. There are 2,000 resamples with seed 2026.

| Primary MCF7, 1 µM comparison; 26 compounds / 24 groups | Error difference | Pointwise 95% interval |
|---|---:|---:|
| A: GNN − fingerprint Ridge, RMSE | +0.046539 | [+0.002349, +0.104793] |
| A: GNN − source mean, RMSE | +0.068437 | [+0.024519, +0.127574] |
| B: five-set Ridge − source mean, MAE | −0.016636 | [−0.035685, +0.003450] |

Both prespecified gene-level intervals favor the simpler reference over the GNN. The biological-head interval spans zero: its point improvement over the source mean is uncertain, and this does not establish equivalence. These are pointwise intervals conditional on the frozen models and current experiment, not multiplicity-adjusted joint tests or estimates of retraining variability. The lowest point mean for zero change is a descriptive ranking; no unreported interval against zero is implied. [All contrasts](transfer/contrasts.tsv).

![Fixed prediction error contrasts across the primary and secondary conditions](figures/contrast_intervals.png)

## What the five biological readouts reveal

The primary-condition table separates the mean observed transcriptional change across 26 compounds from the mean absolute prediction error. A small signed average can conceal opposing gene or compound responses.

| Fixed readout | Retained genes | Mean observed delta | Mean Ridge prediction | Zero MAE | Source-mean MAE | Ridge MAE |
|---|---:|---:|---:|---:|---:|---:|
| S-phase | 43 | −0.112725 | −0.358207 | 0.167726 | 0.349969 | 0.312117 |
| G2/M | 54 | −0.084471 | −0.277426 | 0.143728 | 0.285341 | 0.236211 |
| HDAC-associated set | 40 | −0.007348 | +0.021083 | 0.036465 | 0.037875 | 0.047554 |
| HAT-associated set | 87 | −0.025570 | −0.048935 | 0.049723 | 0.057799 | 0.055568 |
| Stress-response set | 685 | +0.001532 | −0.045964 | 0.025729 | 0.056231 | 0.052584 |

Ridge and the source mean predict more negative average cell-cycle readouts than observed in MCF7. Ridge reduces MAE relative to the source mean for S-phase by 0.037852 and G2/M by 0.049130, but both remain worse than zero change. The HDAC-associated set instead has **higher** Ridge error by 0.009679. HAT and stress readouts show smaller error reductions of 0.002230 and 0.003647. Thus the aggregate point improvement over the source mean is concentrated in the cell-cycle readouts, rather than being uniform across the five sets.

The same descriptive pattern appears across all five conditions: Ridge improves over the source mean for S-phase and G2/M but has higher error for the HDAC-associated set. The [per-set table](transfer/per_set.tsv) retains every compound and baseline; the heatmap shows equal-compound mean Ridge-minus-source-mean absolute-error differences. These per-set descriptions are not additional confirmatory tests.

![Per-set Ridge minus source-mean absolute errors; blue favors Ridge and gray denotes unavailable values](figures/biology_set_errors.png)

These labels are **means of transcriptional changes**, not pathway-activity scores. Cell-cycle-set shifts do not establish altered proliferation; HDAC/HAT-set shifts do not measure enzyme inhibition, acetylation, or target engagement; a near-zero stress-set mean does not demonstrate absence of stress. Membership overlap, opposing gene changes, and variation among compounds limit mechanistic interpretation. No efficacy conclusion follows from these summaries.

## Sensitivity to the vehicle reference

The alternative response uses exactly two eligible same-plate vehicles, each normalized and transformed before averaging. Both vehicles must have at least 30 cells. Predictions are unchanged. The values below compare **strict → two-vehicle mean error on the identical eligible subset**; they never compare full-cohort strict error against a smaller alternative cohort. The sensitivity figure retains all methods.

| Condition | Paired compounds / groups | A: zero RMSE | A: GNN RMSE | B: zero MAE | B: Ridge MAE | B: source-mean MAE |
|---|---:|---:|---:|---:|---:|---:|
| **MCF7, 1 µM — primary** | 26 / 24 | 0.53251 → 0.44649 | 0.71975 → 0.65665 | 0.08467 → 0.08188 | 0.14081 → 0.14743 | 0.15744 → 0.16469 |
| K562, 1 µM | 25 / 23 | 1.03996 → 0.90504 | 1.15519 → 1.03452 | 0.11759 → 0.11635 | 0.17963 → 0.18779 | 0.19332 → 0.20100 |
| MCF7, 10 nM — descriptive subset | 18 / 17 | 0.45655 → 0.39437 | 0.68017 → 0.63815 | 0.07686 → 0.06075 | 0.12037 → 0.13425 | 0.14142 → 0.15492 |
| MCF7, 100 nM | 26 / 24 | 0.45603 → 0.40528 | 0.66223 → 0.62845 | 0.06451 → 0.06403 | 0.15287 → 0.14277 | 0.17426 → 0.16416 |
| MCF7, 10 µM — descriptive | 18 / 17 | 0.53824 → 0.49192 | 0.74322 → 0.71208 | 0.13260 → 0.13529 | 0.15646 → 0.15015 | 0.17709 → 0.17055 |

Changing the reference changes measured error even with fixed predictions. In primary analysis B, for example, Ridge MAE increases from 0.14081 to 0.14743 while zero-change MAE decreases slightly. Primary GNN-minus-Ridge RMSE remains positive under the alternative reference: +0.049590 [0.002346, 0.111853]. The biological Ridge-minus-source-mean interval still crosses zero: −0.017253 [−0.036017, +0.002302]. The principal interpretation is therefore retained, while the response definition materially affects error magnitude.

At MCF7 10 nM, the full strict cohort contains 25 compounds / 23 groups, but the paired sensitivity subset contains only 18 / 17. Its strict GNN RMSE is 0.68017, not the full-cohort 0.673526; its strict five-set Ridge MAE is 0.12037, not 0.130555. Both sensitivity comparisons fail the compound gate and receive no intervals. The 10 µM strict and sensitivity results are likewise descriptive. [Vehicle eligibility](transfer/sensitivity_pairs.tsv) · [All sensitivity means](transfer/summary.json).

![All-method control sensitivity, comparing strict and two-vehicle references on the same compounds](figures/control_sensitivity.png)

## Independent-study feasibility: exact-condition LINCS comparison unavailable

LINCS L1000 Phase I was assessed before expression acquisition, requiring exact standardized parent identity, A549, 24 hours, and an **actual** concentration of exactly 1.0 µM. Rounded dose display labels were insufficient. Three signatures met the exact condition, all three structures resolved, and none matched any of the 26 frozen test compounds: **0 / 26 drugs and 0 / 24 structure groups**. The fixed gene space had **73 / 2,000 current unambiguous direct-landmark matches**. Historical aliases were not exhaustively resolved, so 73 is not an upper bound on every possible gene mapping.

The prespecified minimums were 20 compounds, 10 structure groups, and 100 directly measured common genes. The zero-drug intersection alone establishes **no-go for this exact-condition cohort**. No LINCS Level 5 expression matrix was acquired and no expression prediction performance was calculated. This is a dataset-compatibility result, not a failed external accuracy estimate; it does not show absence of these compounds at other LINCS conditions. L1000 and sci-Plex also differ in platform, reference, and scale. [Feasibility report](metadata_feasibility.md) · [Exact matching record](../../receipts/v2_external_feasibility.json).

## Scope and limitations

The supported finding is a negative transport result for these fixed models, outputs, compounds, and conditions. Zero change's lower mean error does not mean that drugs produce no transcriptional responses: averaging, small effects, heterogeneity, and the chosen loss can favor conservative predictions. Nor does it establish that every future model or context-aware approach must fail.

Two biological replicates, shared vehicles, plate structure, overlapping gene sets, and recurring drugs across conditions limit independence. Structure-group resampling addresses chemical grouping but does not remove all experimental dependence. The high-dose exclusions can change the evaluated drug mixture, preventing a simple dose-response interpretation of aggregate means. Supplementary correlation and directional summaries retain undefined values as NA; a constant zero prediction cannot be assigned a correlation of zero merely to fill the table.

The models were frozen before target scoring, and no models were fitted during evaluation. The completed analysis remains within sci-Plex3; independent-study reproducibility, target engagement, drug efficacy, and clinical validity are not established. Full-precision results and their input/output hashes are available in the [evaluation record](../../receipts/v2_transfer_evaluation.json), [summary](transfer/summary.json), and [figure-rendering record](../../receipts/v2_transfer_rendering.json).
