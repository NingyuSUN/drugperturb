# Model card: DrugPerturb v1

## Intended use
A research benchmark and reproducible inference implementation for compound structure to transcriptional-response prediction. Supported context: **A549, 24 hours, 1,000 nM**. The CLI rejects other contexts. Predictions are changes in log1p counts-per-million on 2,000 training-selected human genes, averaged over two experimental replicates.

This version is not an efficacy, toxicity, target-engagement, clinical, or antibody-design model. It has no independently evaluated external dataset. Low chemical similarity is reported, not converted into a calibrated reliability probability.

## Data and split
Public sci-Plex3, GEO GSE139944 / GSM4150378. Of 188 source treatment labels, 173 met paired cell-count criteria, and 171 remained after chemical qualification/parent deduplication. The primary data comprise 33,055 treatment nuclei and 886 nuclei in eight unique matched vehicle wells. Training 119 compounds, validation 26, final test 26. Test has 22 Bemis–Murcko ring-scaffold groups and two acyclic parent groups. Exact parents/groups do not cross splits; related chemistry can remain similar.

Original SMILES and exclusions are retained. Undefined structures and unresolved active mixtures are excluded. Salt/parent duplicates use the least-fragment-removed formulation, then catalog ID, without reading responses. Molecular features do not encode stereochemistry/isotopes; this limits discrimination between those variants.

## Methods
- B0: literal zero change; B1: average training response; B2: response of the training compound with greatest binary Morgan Tanimoto similarity.
- M1: radius 2 / 2,048-bit Morgan fingerprints, Ridge regression with selected alpha 100.
- M2: same fingerprints, one 128-unit ReLU hidden layer, dropout 0.2, learning rate 0.003.
- M3: atom/bond features, two 64-unit bond-aware message-passing layers, mean neighbor and molecular pooling, dropout 0.1, learning rate 0.003. Pure PyTorch implementation; 27,604 trainable parameters.
- Variable-gene selection, StandardScaler and 20-component PCA fit only on 119 training responses. Neural optimization uses PC targets, but checkpoint selection and all reported errors use actual measured selected-gene responses.
- Four fixed candidates per learned model family. M2/M3 each use seeds 42, 123, 2026; select candidate by mean seed validation error, then average its three predictions. AdamW weight decay 1e-4, maximum 250 epochs, patience 30, batch 32. M1 and all baselines use identical compounds/output genes.

## Held-out performance
Mean of per-compound RMSE across 26 test compounds / 2,000 genes; smaller is better. Spearman ranks changes within each drug, not change amplitude.

| Method | Mean compound RMSE | Mean Spearman |
|---|---:|---:|
| B0 | 1.2694 | NA |
| B1 | 1.2170 | 0.2698 |
| B2 | 1.7149 | 0.0725 |
| M1 | 1.2430 | 0.2127 |
| M2 | 1.2532 | 0.1920 |
| M3 | 1.2144 | 0.2755 |

The validation-selected simple reference was M1. GNN-minus-M1 RMSE difference: **−0.0286**, paired 95% structure-group interval **[−0.0680,0.0087]**. GNN-minus-B1 difference: **−0.0026**, interval **[−0.0475,0.0419]**. Both intervals include zero. A reliable advantage over these strong baselines is not established. Do not select the best individual test seed or retrospectively replace the reference.

Intervals use 2,000 resamples of 24 whole structure groups, seed 2026, and weight compounds equally. They condition on one split and fitted models; shared control/plate effects, retraining uncertainty and external study variation are not fully captured. Seed repeats are not biological replicates.

## Biological coverage
All five prespecified sets fail the minimum 10 output genes and 25% of original members gate: S 2/43, G2/M 9/54, HDAC 5/60, HAT 6/108, stress 55/791. No pathway-recovery result is reported. Separately, all 43 S and 54 G2/M markers were used for single-cell descriptive scores with all-human-UMI denominators; these are not phase assignments.

## Limitations and failure analysis
The task has 171 distinct standardized parent compounds, one cell line/time/dose, two experiment replicates and shared vehicles. Selecting variable genes without a separate training-only low-expression filter can favor noisy responses; variance is not evidence of drug specificity. A new output definition requires a new prespecified protocol and untouched evaluation data, not retuning this opened test set.

Per-drug errors, fixed-rule cases, repeat disagreement, chemical support, development CV/learning curves and fixed-prediction sensitivity checks are retained. Under the same-plate two-vehicle response definition, GNN RMSE is 1.0395 versus 1.0131 for zero change and 1.0150 for the training mean; lower absolute errors reflect changed truth. GNN is weaker than Ridge in all three exploratory development folds. These findings reinforce the inconclusive incremental-value result. Development diagnostics use refitted gene spaces and one seed, so they are exploratory, not directly controlled comparisons of training size or chemical novelty.

## Reproducibility and ownership
Historical training source label: `4847c26e05c0283d704a5fab9a85f272d0a5981a` (a pre-publication execution snapshot, not a public Git commit; see [publication provenance](PUBLICATION_PROVENANCE.md)). Actual CPU execution and all 28 learned-model fits are recorded; six methods were evaluated. Source hashes, fixed settings, transformations, all training trajectories/checkpoints and sealed predictions support audit. The source dataset and author code remain externally attributed; project code licensing does not relicense them. Automated checks do not establish independent biological validation.

AWS execution: **NOT EXECUTED**. GPU training: **NOT EXECUTED**; the small models ran on CPU. See the [v1 report](../reports/v1/report.md), [complete HTML results](../reports/report.html), [protocol](../configs/protocol_v1.yaml), [training runs](../receipts/training_runs.tsv), and [reproducibility](REPRODUCIBILITY.md). To read the HTML reports locally, clone the complete repository or extract its GitHub ZIP while preserving the directory structure, then open `reports/report.html` or `reports/biologist_summary.html`; figures and evidence links require the adjacent files.

The project's original code and accompanying documentation use the [MIT License](../LICENSE), copyright 2026 Ningyu Sun. Third-party data, annotations, author code and dependencies retain their own terms and are not relicensed by this project.


## Numerical reproducibility

Two independent core runs on the same fixed host/container produced identical 24 validation/test prediction arrays. Cross-host retraining from identical inputs did not meet 1e-6 for the GNN; those failed comparisons are retained. Loading the saved primary GNN weights in Docker passed all eight seed/ensemble inference arrays at the same tolerance. Reproducible inference, repeatable training within one environment and cross-platform training are distinct claims; see [execution evidence](EXECUTION_EVIDENCE.md).
