# DrugPerturb v2: methods and evaluation boundaries

The v2 protocol separates transport of the frozen v1 expression predictors from a new, direct predictor of five predefined gene-set expression summaries. The study uses public data and includes no new wet-laboratory experiments. This document describes the fixed evaluation design, completed source-condition training, and independent-study feasibility assessment. Target-condition results and count-qualified cohorts are reported separately.

The machine-readable specification is [protocol_v2.yaml](../configs/protocol_v2.yaml). Model provenance and the interpretation of the original results are documented in the [v1 model card](MODEL_CARD.md).

## Data, experimental contexts, and retained splits

The source is [sci-Plex3, GEO GSM4150378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378), within GSE139944. The source training condition is A549, 24 hours, 1,000 nM. The sole primary target condition is MCF7 at the same time and concentration. Prespecified secondary conditions are K562 at 24 hours and 1,000 nM, and MCF7 at 24 hours and 10, 100, and 10,000 nM. These are conditions within the same study, not independent-study validation cohorts.

The original chemically qualified cohort and split are retained: 119 training, 26 validation, and 26 test parent compounds. Canonical-parent assignments, structural groups, and representative formulations are fixed. Neither a failed target sample nor an excluded formulation is replaced by another compound. Structural groups include Bemis–Murcko ring scaffolds and the original parent-based groups for acyclic compounds.

The primary target cohort consists of the original 26 test compounds that satisfy the fixed target-condition quality criteria. Their A549 outcomes were examined in v1. The new evaluation therefore concerns previously withheld responses in a different condition; it does not establish that these compounds have never been examined by the investigators. Original test responses from A549 are excluded from development of the new biological predictor. Target responses are withheld from feature selection, gene filtering, parameter selection, and prediction sealing.

## Count aggregation, controls, and eligibility

Counts are taken from the original counts assay, retaining all supplied human gene rows and the author's documented cell quality criteria. Each unique experimental well is aggregated once by summing raw unique molecular identifier counts across its retained cells. Shared vehicle wells remain shared measurements; repeated references to a vehicle do not create additional experimental replicates.

For each well, counts are divided by the sum over all human genes, multiplied by one million, and transformed with `log1p`. Normalization precedes output-gene or gene-set selection. A treatment is compared with the unique vehicle matching cell type, time, experimental replicate, plate, and dose-pattern metadata. The response is the treated-minus-vehicle difference within each replicate, followed by an equal-weight mean of the two replicate differences.

Both treatment replicates and their matched vehicles must each contain at least 30 cells passing author quality control. Eligibility is evaluated separately for each condition. Missing quality-control records, observed wells with no cells, and low-cell wells are distinguished in exclusions. Compound-level evaluation requires at least 20 eligible original test compounds and 10 structural groups. These are operational feasibility thresholds, not a power calculation or a guarantee of precise estimates. Below-threshold results are descriptive, and thresholds are not relaxed after observing effects.

Gene identities and prediction order must align exactly. A measured zero count is retained. An absent required gene, duplicate identifier, or incompatible mapping is an alignment failure, not a zero expression value or a reason to silently redefine the output space.

## A. Transport of frozen v1 predictions

All six v1 methods are retained: zero change, the training response mean, the chemically nearest training response, fingerprint Ridge regression, the fingerprint multilayer perceptron, and the molecular graph neural network. The selected 2,000 genes, their order, fitted transformations, weights, and three-seed neural ensembles remain fixed. Previously sealed test predictions are reused without retraining.

These predictors receive molecular structure and do not receive cell identity or dose. Consequently, a compound has the same predicted response in every target condition. The experiment measures transport of an A549-derived prediction; it does not show that the predictor learned cell-specific responses or dose dependence.

For each compound, the root mean squared error (RMSE) is calculated against its measured response across the fixed 2,000 genes. The primary summary is the arithmetic mean of these compound RMSEs. It is not a single RMSE pooled across all compound–gene entries. Prespecified comparisons are graph-network minus Ridge error and graph-network minus training-mean error. All six methods are reported. Measured responses, rather than principal-component reconstructions, define the reference values.

The original v1 five-set coverage failures remain part of the v1 result. Members used in the new five-readout predictor are not added to the frozen v1 gene-level outputs.

## B. Direct prediction of predefined gene-set mean changes

The separate biological predictor uses the five source-defined sets in [biology_v1.json](../configs/biology_v1.json): S-phase, G2/M, histone deacetylation, histone acetylation, and cellular responses to stress. Their source versions, original memberships, and identifier mappings are fixed before target evaluation.

Expression filtering uses only wells belonging to the 119 source training compounds and their unique matched vehicle wells. A gene is retained when its CPM is at least one in at least 20% of training treatment wells or at least 20% of unique training vehicle wells. The required number of wells is rounded upward. The completed run used 238 treatment wells and eight unique vehicles, requiring expression in at least 48 treatment wells or two vehicles. Of 58,347 human genes in the input, 15,983 passed this filter. Validation and test counts do not determine membership.

Each set must retain at least 10 unique genes and at least 50% of its original source members. The denominator includes original members lost during mapping; mapped duplicates count once in the numerator. All five sets must pass for the primary five-set predictor. The fixed failure rule is zero new fits and no replacement or reduced-output primary model; the frozen v1 transport analysis remains available independently.

All five sets passed in the completed source-only run:

| Set | Original source members | Unique mapped genes | Retained genes | Retained / original |
|---|---:|---:|---:|---:|
| S-phase | 43 | 43 | 43 | 100.0% |
| G2/M | 54 | 54 | 54 | 100.0% |
| HDAC | 60 | 59 | 40 | 66.7% |
| HAT | 108 | 107 | 87 | 80.6% |
| Stress | 791 | 783 | 685 | 86.6% |

The fixed panel contains 909 set–gene memberships representing 874 distinct genes because sets overlap. Members are retained unchanged across target conditions. For each compound, each set's label is the equal-weight mean of its fixed genes' measured, two-replicate-averaged expression differences. The label is a mean transcriptional change. Opposite gene changes can cancel; it is neither a pathway-activity estimate nor a measurement of enzyme activity, target engagement, or efficacy.

One multi-output Ridge regression predicts the five labels from unscaled binary Morgan fingerprints (radius two, 2,048 bits). Labels are unscaled and an intercept is fitted. Four candidates, with alpha values 0.1, 1, 10, and 100, were fitted using the 119 training compounds. Selection used the 26 source validation compounds and mean absolute error (MAE), averaged equally across the five sets within each compound and then equally across compounds. Exact ties select the first candidate in the fixed order. Alpha 100 was selected; the four validation MAEs were 0.20439, 0.20246, 0.19344, and 0.18303. These values describe model selection, not independent validation. Training and validation were not combined for a subsequent refit.

Zero change, the source-training mean, and the chemically nearest training response provide matched-output baselines. Nearest-neighbor matching uses binary Tanimoto similarity, with ties resolved by sorted training compound identifier. The primary biological comparison is Ridge minus source-mean MAE. Per-set errors accompany the five-set summary so that aggregate performance cannot conceal failure in a particular set.

Four fitted candidates, the selected model, fixed members, validation selection, and predictions for all 26 original test structures were sealed before target scoring. The [coverage record](../reports/v2/biology_coverage.tsv), [selection record](../receipts/v2_biology_selection.json), and [prediction seal](../receipts/v2_biology_prediction_seal.json) document the completed run. The seal binds source inputs, implementation files, and derived artifacts by cryptographic hash.

## Error uncertainty and control sensitivity

Paired comparisons use compound errors: RMSE for analysis A and five-set MAE for analysis B. For each contrast, candidate-minus-reference error is calculated within each compound. Negative differences favor the candidate. The structural-group bootstrap samples the original number of groups with replacement, preserving every compound in each sampled group and its multiplicity. Each resampled estimate is still compound-weighted. Intervals are the 2.5th and 97.5th percentiles of 2,000 resamples with seed 2026.

These are pointwise intervals conditional on the frozen models, selected compound cohort, and current experiment. They do not fully represent retraining variability, shared-vehicle or plate dependence, or variation across independent studies. Cells and repeated optimization seeds are not independent biological replicates. Two separately reported comparison intervals do not constitute a multiplicity-adjusted joint superiority test. An interval spanning zero does not establish equivalence. Lower error in a fixed comparison does not establish biological or clinical usefulness.

The prespecified sensitivity analysis changes only the reference response: each of exactly two same-cell, same-time, same-replicate, same-plate vehicles is separately normalized and transformed, then their transformed profiles are averaged. Both vehicles must contain at least 30 retained cells; there is no single-vehicle fallback. Predictions remain unchanged. Strict-control and alternative-control results are compared on the identical subset satisfying both-vehicle eligibility, alongside the original full strict-control result. Ranking changes are retained. Because the measured reference changes, this is a response-definition sensitivity analysis, not retraining or independent validation.

## Independent-study feasibility

LINCS L1000 Phase I [GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742) was assessed from metadata for an independent-study comparison. Eligibility requires exact standardized parent identity, A549, 24 hours, and an actual concentration of 1.0 µM. Rounded display labels are not accepted as exact doses: a label of “1 µM” can include actual concentrations such as 1.11 µM.

The completed metadata audit found three exact-condition signatures representing three perturbagen identifiers, with **zero matches among the 26 frozen test compounds**. The fixed output space also had 73 unambiguous directly measured landmark-gene matches. Thus the prespecified minimums of 20 test compounds, 10 structural groups, and 100 directly measured common genes were not met. The result is **NO-GO for this exact-condition external cohort**, not a negative model-performance result. No independent-study prediction accuracy is established. The [feasibility record](../receipts/v2_external_feasibility.json) retains matching decisions and denominators.

L1000 is a population-level expression assay, not bulk RNA sequencing. Its differential-signature scale differs from sci-Plex log1p CPM differences. The gated comparison specifies within-compound Spearman rank agreement on directly measured common genes, not raw cross-platform RMSE; inferred genes cannot be used to satisfy the primary coverage threshold. This metric was not evaluated after the exact-condition feasibility gate failed.
