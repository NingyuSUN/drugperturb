# DrugPerturb v4: can chemCPA's collapse to zero change be fixed?

v3 found that chemCPA predicts no change at all: on sci-Plex3 its counterfactual effects were about 0.1% of the measured ones, even for training compounds, and its validation error equalled zero change throughout 200 epochs. v4 asks whether a principled change to the training objective stops the collapse and, if it does, whether the changed model then beats zero change and information-matched simple baselines on data this project has never scored. The machine-readable specification is [protocol_v4.yaml](../configs/protocol_v4.yaml); it was committed before any v4 fit.

## Diagnosis and hypotheses

v3's training/validation-only diagnostic showed a drug term of norm 0.08 against 14.6 for the basal latent of vehicle cells, predicted training-compound effects of mean |Δ| 0.0009 against 1.05 measured (gene-wise r 0.02), and a validation RMSE between 1.3027 and 1.3029 at every evaluation, against 1.3029 for zero change. Two mechanisms can each explain this:

- **Likelihood mismatch (H_likelihood).** The response is the log of a library-weighted pooled mean, `log1p` CPM of summed counts. chemCPA fits a Gaussian with a learned variance to per-cell `log1p` values. Their mean is not the log of the pooled mean, and a drug's shift is a tiny fraction of per-cell variance, which the learned variance can absorb.
- **Autoencoder shortcut (H_shortcut).** Reconstruction lets the decoder explain each treated cell from its own basal encoding. The drug term matters only for the population-mean shift, which per-cell losses barely reward. The path used for prediction, vehicle cells plus the drug term, is never trained directly.

## Arms

All arms use v3's selected configuration (repository default C1) without any hyperparameter search. The upstream `ComPert` module is unchanged; only the objective differs.

| Arm | Likelihood | Extra loss | Tests |
|---|---|---|---|
| A0, v3 reference | Gaussian NLL on `log1p` values (v3) | none | whether the collapse replicates |
| I1, NB likelihood | negative binomial on raw counts, mean = library × exp(η + b) | none | H_likelihood |
| I2, counterfactual pseudobulk | Gaussian (v3) | well-level pseudobulk loss, weight 1 | H_shortcut |
| I3, both | negative binomial | pseudobulk loss, weight 1 | both |

**NB likelihood.** The decoder's first gene block is a log mean fraction η, offset by b, the log of each gene's pooled fraction in training cells, and capped at a fraction of 1. The second block, passed through ComPert's softplus, is the inverse dispersion. The vendored upstream `NBLoss` is applied to raw counts. The decoder's last activation is linear, because a ReLU would force every gene to at least its training-mean fraction. The encoder still sees `log1p(count / library × 10⁴)`.

**Counterfactual pseudobulk loss.** Every autoencoder/doser update adds the mean squared gene error between the measured strict delta of 8 sampled training (compound, context) pairs and the model's counterfactual contrast for them. The contrast is computed as at inference: 64 cells drawn with replacement from each replicate's matched vehicle well, under the compound minus under the vehicle, as a library-weighted pseudobulk, averaged over replicates. Modules run in evaluation mode, using BatchNorm running statistics. The only way to reduce this loss is through the drug term. Pair sampling uses its own random generator, so the cell-batch order is identical to A0.

A unit test shows that A0 repeats the v3 training step exactly, parameter for parameter.

**Not selected: LINCS L1000 pretraining.** The chemCPA authors recommend it, and v3 listed its absence as a limitation. The v2 audit found only 73 of the 2,000 frozen genes measured as L1000 landmarks, so pretraining could transfer the perturbation encoder but not the gene decoder in which the drug term collapses. The collapse also appears on training compounds, where pretraining's benefit, generalization to unseen compounds, does not apply. The published pretraining also needs more GPU time than remained in this week's Kaggle quota.

## Data

**sci-Plex3: exploratory only.** All twelve 24 h contexts had their test outcomes opened in v1–v3. The v3 response-free inputs are reused unchanged in two settings: Q1, A549 1 µM with no covariate and three seeds; and Q2, all twelve contexts with cell type as a covariate and one seed, for compute reasons.

**OP3: confirmatory.** [GSE279945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279945) is a single-cell RNA-seq study of 146 small molecules in peripheral blood mononuclear cells from three donors, at 24 h. Raw UMI counts cover 298,087 cells and 21,265 genes. The file's size is pinned. Its SHA256 is recorded by the extraction kernel at download and compared with an independent local download, and every later kernel chains the hashes of the extraction outputs. Before this protocol, the project read only its design metadata and the cell-type counts of the DMSO wells ([record](../receipts/v4/op3_pre_protocol_reads.md)). The data are public and were analysed by others in the NeurIPS 2023 competition. "Untouched" here means untouched by this project: no OP3 expression value informed any choice.

- **Layout.** There are six plates, one donor per plate and two plates per donor. Each plate row is one library containing exactly one DMSO well. Each of the 141 compounds is dosed at 1 µM, in one well per donor. The positive controls belinostat (0.1 µM) and dabrafenib (48 wells each) are excluded, because their replicate structure differs and, for belinostat, so does the dose.
- **Contexts.** The contexts are the authors' four reannotated cell types: T, NK, B and myeloid cells. A context's measurement unit is the cells of that type in one well. The strict matched vehicle is the same cell type in the DMSO well of the same plate row, which shares the library and the donor.
- **Split.** All 144 structures pass the v1 standardization. Whole scaffold groups were split with seed 42 into 60/15/25 using structures only: 85 training, 21 validation and 35 test compounds ([manifest](../manifests/v4/op3_compounds.tsv)).
- **Count QC.** A (compound, context) pair is eligible when all three donors have at least 30 treated cells and at least 30 cells in the matched vehicle unit. A context is confirmatory when it has at least 20 eligible test compounds in at least 10 scaffold groups. Contexts below this gate are descriptive only. In DMSO wells the median unit sizes are 339 T, 104 myeloid, 91 B and 40.5 NK cells, so NK eligibility is expected to be the tightest.
- **Truth.** For each donor, the truth is the `log1p` CPM of the treated unit minus that of the matched vehicle unit, using the all-gene library. The three donors are averaged with equal weight. Test truth is first computed by the evaluation kernel, after every prediction is sealed.
- **Gene space.** The 2,000 genes are those with the highest variance of training-compound deltas (v1's rule), restricted to genes detected in at least 5% of DMSO cells. Only training compounds and vehicle cells inform this choice.

## Diagnostics before any test scoring

For every fit and every arm's seed-averaged predictions, on training and validation pairs:

- **Effect size ratio:** the mean L2 norm of the predicted delta divided by that of the context's training-mean delta.
- **Gene-wise Pearson r** between predicted and measured deltas, with a constant prediction counted as 0.
- **Compound-specific r:** the correlation after removing the arm's own mean training prediction and the measured training mean.
- **RMSE against zero change.**
- **Latent ratio:** the norm of the drug term minus the vehicle term, relative to the basal latent of vehicle cells.

An arm counts as **non-collapsed** on validation compounds when its effect size ratio is at least 0.10, its mean gene-wise r is at least 0.10, and its validation RMSE is below zero change. These diagnostics are reported, and committed, before the evaluation kernel runs.

## Selection, endpoints and comparators

On OP3 the primary arm is chosen on validation compounds only. It is the non-collapsed intervention (I1–I3) with the lowest mean per-seed validation RMSE. If no intervention qualifies, the lowest-RMSE intervention is still evaluated, and the report states that the collapse was not stopped.

The comparators are information-matched and fitted per context on that context's eligible training compounds, with v3's code: zero change (B0), training mean (B1c), nearest training compound (B2c) and fingerprint Ridge (M1c). The metric is the mean over compounds of per-compound RMSE over the 2,000 genes. Intervals are paired scaffold-group bootstrap 95% intervals (2,000 draws, seed 2026). Pooled estimates average context-level differences over gate-passing contexts, using one shared draw per bootstrap replicate.

- **E1, non-collapse on untouched data:** selected arm − B0, supported if the upper bound is below 0.
- **E2, improvement:** selected arm − B1c and selected arm − M1c, supported only if both upper bounds are below 0.

Every other interval, including all arms against all comparators, per-context results and the sci-Plex3 analyses, is secondary and unadjusted.

## Anchors

- **sci-Plex3.** Fresh truth must reproduce reports/v3/per_compound.tsv to 10⁻⁹ for zero change, the sealed v3 B1c and M1c, and the sealed v3 chemCPA.
- **OP3.** No per-compound result in this response space has been published, so the anchors are internal:
  - the regenerated design equals the committed manifests;
  - well cell-type counts equal the authors' `cell_count_by_well_celltype`;
  - per-cell counts sum to unit pseudobulk counts;
  - an identity model reproduces zero change exactly.

## Compute

Kaggle notebooks run on CPU for extraction, preparation and evaluation, and on NVIDIA T4 ×2 for fits, with a limit of 60 minutes per fit. Environments are pinned in [cellmodels-cuda.txt](../locks/cellmodels-cuda.txt) and [python-cpu.txt](../locks/python-cpu.txt). Seeds are fixed, but cuDNN is not forced to be deterministic.

## Amendments

Every change made after the protocol commit is logged in the protocol, with its date, its reason and the information it used.

| ID | Change | Information used |
|---|---|---|
| A1 | Adds vehicle-anchored comparators B1a and M1a and endpoint E3. E1, E2, the arms and the selection rule are unchanged. | sci-Plex3 validation compounds only, before any OP3 fit output or test truth was read |
| A2 | Caps per-cell model values at log1p(10⁴), the value of a gene holding the whole library, before the pseudobulk mapping. A non-finite validation evaluation scores +∞. All twelve OP3 fits are rerun. | the error messages of two diverged OP3 fits; no validation or test metric |

**Why A1.** The measured response T − V contains the sampling noise of the matched vehicle well. A model that receives those vehicle cells can partly cancel that noise; a delta-only baseline cannot. This is the asymmetry v3 identified for its anchored sensitivity. The information-matched simple version adds the vehicle information to a delta baseline: B1a = B1c + V̄_train − V(query), and M1a likewise from M1c. On sci-Plex3 validation compounds, B1a reached a validation RMSE of 0.84 in Q1 and 0.66 in Q2, better than every chemCPA arm. E3 therefore requires the selected arm to beat B1a and M1a before any claim that chemCPA predicts compound responses better than simple baselines.

## Issue found after test scoring

The descriptor scaler inherited from v3 (`TrainingDescriptorScaler`) keeps any training column with nonzero standard deviation. In both datasets, 25–28 RDKit2DNormalized columns are constant in training up to floating-point noise, with standard deviations down to 10⁻³³.

- In OP3, three test compounds (ixabepilone, dasatinib, alvocidib) and one validation compound (CEP-37440) differ from training in those columns. They receive z-scores near 10¹⁶.
- No sci-Plex3 compound exceeds a z-score of 50.

Results are reported with the scaler as prespecified; the post hoc sensitivity analysis is in the [v4 report](../reports/v4/report.md). A future protocol should drop columns whose training standard deviation falls below a tolerance.
