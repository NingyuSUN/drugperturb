# Model card: the cell-level models, v3–v6

The [v1 model card](MODEL_CARD.md) covers the structure-only models that receive chemical structure and
nothing else. This card covers the published single-cell perturbation models used in v3–v6, which also
receive cell identity and dose: **chemCPA** and **biolord**. It describes them as this project implemented,
trained and scored them, not as their authors distribute them.

Every number here is quoted from the version report it belongs to; those reports hold the denominators,
gates and limitations. Results: [v3](../reports/v3/report.md), [v4](../reports/v4/report.md),
[v4 corrected](../reports/v4_corrected/report.md), [v5](../reports/v5/report.md),
[v6](../reports/v6/report.md), and the [v1–v6 synthesis](../reports/synthesis.md).

## Intended use

A research benchmark. These fits exist to answer one question per version — whether a published
perturbation model predicts the transcriptional response of a **held-out compound** better than a baseline
given the same information — and they are not a deployable predictor. There is no inference CLI for them,
unlike v1's; the sealed prediction archives and fit records are the artefacts.

**Not for:** efficacy, toxicity, target engagement, clinical benefit, antibody design, or any decision about
a real compound. No pretrained foundation model is claimed, and LINCS L1000 pretraining — which chemCPA's
authors recommend — was never performed; [V4_METHODS](V4_METHODS.md) records why it was not selected.

## The two models

**chemCPA.** The upstream `ComPert` module is vendored unchanged and SHA256-checked
([third_party](../src/drugperturb/third_party/README.md)). The training step re-implements the published
Lightning module: Gaussian negative log likelihood reconstruction; drug and cell-type adversaries on the
basal latent with a gradient penalty; autoencoder and doser updates that maximize adversary loss;
gradient-norm clipping at 1; Adam with StepLR (γ 0.9) per epoch, the adversary replacing the autoencoder
update every `adversary_steps`-th batch. v3 compared two published configurations — the repository default
(C1) and the configuration from the biolord reproducibility study (C2) — and C1 was selected. **v4, v5 and
v6 reuse C1 with no hyperparameter search at all**, so that each version's result is attributable to the
change that version makes.

**biolord.** biolord 0.0.3 on scvi-tools 1.1.6, with the published sci-Plex3 module and trainer settings:
`n_latent` 256, decoder 4,096 × 4, batch 512, at most 200 epochs, published early stopping, 150 minutes per
fit (amendment A2 of v3). Validation-compound cells form biolord's internal validation split, and
counterfactuals start from vehicle cells whose latent codes are trained. biolord appears in v3 only.

## Inputs and the response space

Cells enter as `log1p(count / all-gene library × 10⁴)` over that version's 2,000 frozen genes. A prediction
for compound *c* in context *k* starts from the cells of the strict matched vehicle unit, applies *c*, the
dose and the cell identity, and converts per-cell values to a well-level `log1p(CPM)` as the
library-weighted mean of `expm1(value) × 100`, negatives clipped to zero. With measured cell values this
conversion equals the v1/v2 well normalization exactly, so an identity model reproduces zero change
exactly — a unit test, not an argument.

The scored prediction is the **counterfactual contrast**: the same vehicle cells under the compound minus
under the vehicle, averaged over replicates. Molecules are RDKit2DNormalized descriptors (descriptastorus
2.8.0), z-scored on training compounds with constant columns dropped; from v5 on, `descriptors.min_std =
1e-8` guards the near-constant columns that produced v4's 10¹⁶ z-scores. Doses are scaled as nM / 10,000.
The DMSO vehicle carries its own descriptors, at dose 1.0 for chemCPA and 0 for biolord, following each
model's published sci-Plex3 convention.

## Arms, by version

Only the objective or the target changes between arms; the module never does.

| Version | Arm | Likelihood | Extra loss | Target | Seeds |
|---|---|---|---|---|---:|
| v3 | chemCPA C1 / C2 | Gaussian NLL on `log1p` | none | measured delta | 3 |
| v3 | biolord L1 | published | none | measured delta | 3 |
| v4–v5 | A0, v3 reference | Gaussian NLL | none | measured delta | 3 |
| v4–v5 | I1, NB likelihood | negative binomial on raw counts | none | measured delta | 3 / 10 |
| v4–v5 | I2, counterfactual pseudobulk | Gaussian | well-level pseudobulk loss, weight 1 | measured delta | 3 / 10 |
| v4–v5 | I3, both | negative binomial | pseudobulk loss | measured delta | 3 / 10 |
| v6 | M1, pseudobulk raw | Gaussian | pseudobulk loss | measured delta | 10 |
| v6 | M2, pseudobulk residual | Gaussian | pseudobulk loss | `Δ − B1p`, the leave-one-out residual over the plate-anchored mean | 10 |

**The NB likelihood** makes the decoder's first gene block a log mean fraction offset by each gene's pooled
training fraction, with the vendored `NBLoss` applied to raw counts and a linear final activation, because
a ReLU would force every gene to at least its training-mean fraction.

**The counterfactual pseudobulk loss** adds, to every autoencoder/doser update, the mean squared gene error
between the measured delta of 8 sampled training (compound, context) pairs and the model's counterfactual
contrast for them, computed exactly as at inference: 64 cells drawn with replacement per replicate vehicle
unit, library-weighted, modules in evaluation mode. The only way to reduce it is through the drug term. Pair
sampling uses its own generator, so the cell-batch order is identical to A0 — and a unit test shows A0
repeats v3's training step parameter for parameter.

## Selection

Arm and checkpoint selection never reads a test response. Checkpoints are chosen by validation-compound
error in the response space every 10 epochs (at most 200 epochs, patience 5 evaluations). v3 and v4 fit 3
seeds per candidate and take the lowest mean validation RMSE; v5 and v6 fit **10 seeds**, after v4 selected
on margins of 0.0010 and 0.0004 while one arm's seed range was 0.0096. v5 adds an explicit near-tie rule
with Bonferroni-adjusted intervals for co-primary arms. The selected arm's seed predictions are averaged,
and every prediction archive is SHA256-sealed before the evaluation script reads test truth; each fit record
certifies that no test outcome was loaded.

Selected arms: **C1** (v3, both questions), **I1** (v4 prespecified, OP3), **I2** (v4 corrected post hoc,
and v5), **M2** (v6).

## Held-out performance

Each row is that version's own confirmatory comparison, in the sign convention of its own endpoint. v3–v5
are error metrics (lower is better, a negative difference favours the model); v6 is an accuracy (higher is
better, a positive difference favours the model). **No version's primary endpoint was met.**

| Version | Data | Contrast | Difference [95% interval] |
|---|---|---|---|
| v3 | sci-Plex3, 6 never-scored contexts | chemCPA − per-context Ridge (M1c) | +0.057 [+0.038, +0.074] |
| v3 | sci-Plex3 | biolord − per-context Ridge (M1c) | +0.052 [+0.031, +0.072] |
| v3 | sci-Plex3 | chemCPA − zero change | −0.0000 [−0.0001, +0.0000] |
| v3 | sci-Plex3 | biolord − zero change | −0.005 [−0.016, +0.001] |
| v4 | OP3, 35 test compounds | I1 − zero change | +0.171 [−0.004, +0.532] |
| v4 | OP3 | I1 − per-context training mean (B1c) | +0.183 [+0.009, +0.544] |
| v4 | OP3 | I2 − per-context training mean (B1c) | +0.014 [+0.006, +0.023] |
| v4 corrected | OP3 | I2 − zero change | +0.004 [−0.003, +0.012] |
| v4 corrected | OP3 | I2 − per-context training mean (B1c) | +0.016 [+0.009, +0.025] |
| v5 | Tahoe-100M, 30 contexts | I2 − vehicle-anchored training mean (B1a) | −0.0047 [−0.0104, +0.0011] |
| v5 | Tahoe-100M | I2 − anchored Ridge (M1a) | −0.0046 [−0.0104, +0.0011] |
| v5 | Tahoe-100M | I2 − zero change | −0.0359 [−0.0399, −0.0319] |
| v6 | Tahoe-100M, 27 contexts | M2 − plate-anchored training mean (B1p) | +0.0010 [−0.0007, +0.0039] |
| v6 | Tahoe-100M | M2 − anchored Ridge (M1a) | +0.0280 [+0.0214, +0.0361] |
| v6 | Tahoe-100M | M2 − zero change, which scores chance | +0.3135 [+0.2999, +0.3265] |

v6's direction accuracy on responder keys, for scale: M2 0.8226, the plate-anchored mean 0.8217, zero change
0.5000 by construction, and the split noise floor — the accuracy the measurement reaches against itself —
0.8232 on validation.

## The collapse, and what fixed it

This is the part of these models' behaviour that the project actually established.

**v3 found chemCPA predicting nothing at all.** On sci-Plex3 its drug term had norm 0.08 against 14.6 for
the basal latent of vehicle cells — about 0.6% — its predicted training-compound effects had mean |Δ| 0.0009
against 1.05 measured, with gene-wise r 0.02, and its validation RMSE sat between 1.3027 and 1.3029 at every
evaluation against 1.3029 for zero change. The collapse appeared on **training** compounds, so it is not a
generalization failure.

**v4 diagnosed it as two candidate mechanisms and tested both.** Likelihood mismatch: the response is the
log of a library-weighted pooled mean while chemCPA fits a Gaussian with learned variance to per-cell
`log1p` values, which can absorb a drug's shift. Autoencoder shortcut: reconstruction lets the decoder
explain each treated cell from its own basal encoding, and the path used for prediction — vehicle cells plus
the drug term — is never trained directly.

Both interventions worked as mechanisms. On validation compounds the effect-size ratio rose from 0.007 (A0)
to 0.25–0.35 under the NB likelihood, and to 1.62 with the pseudobulk loss, whose gene-wise correlation with
the truth reached 0.64. **The answer to the scientific question did not change.** On OP3 even the
non-collapsed arms did not beat a per-context training mean.

**v6 showed what the revived drug term is worth.** Training on the residual over the plate-anchored mean
rather than on the raw delta is worth +0.0612 [+0.0490, +0.0750] — the largest effect in that study — and
what it buys is the plate mean back: the residual the model learns correlates about −0.007 with the real one
on validation, so "predict the residual" converged on "predict no residual".

## Limitations

- **Two models, one implementation.** A different implementation, a larger pretraining corpus, or LINCS
  pretraining is not tested by any of this.
- **No hyperparameter search after v3.** C1 is a published default, chosen on validation in v3 and then held
  fixed. A tuned chemCPA might do better; this project deliberately did not tune it, so that an endpoint or
  objective change could be read as such.
- **Fit-time caps.** 60 minutes per fit in v5 and v6, 90 in v3, 150 for biolord. v6's design note records
  this as a confound it chose not to remove.
- **Full-gene MSE objectives throughout.** v6 scored direction on responsive genes while still training on
  all 2,000 genes. Re-weighting the loss toward responsive genes is the obvious next study and needs its own
  protocol.
- **The comparators are the finding.** Across v4, v5 and v6 the strongest baseline uses no chemistry at all,
  and each stronger anchoring level reversed an earlier conclusion. Any future arm must be compared against
  the plate-anchored mean, not against zero change.
- **GPU determinism.** Seeds are fixed but GPU kernels are not forced deterministic, so retraining can
  differ in the last digits.

## Compute and reproducibility

Kaggle notebooks, NVIDIA T4, with CPU sessions for extraction, preparation and scoring. Environments are
pinned in [cellmodels-cuda.txt](../locks/cellmodels-cuda.txt) (torch 2.5.1+cu121, scvi-tools 1.1.6, biolord
0.0.3) and [python-cpu.txt](../locks/python-cpu.txt). Every kernel records its GPU, driver, CUDA, package
versions, wall time and input/output hashes: [v3](../receipts/v3), [v4](../receipts/v4),
[v5](../receipts/v5/kaggle_runs.md), [v6](../receipts/v6/kaggle_runs.md). v6's 20 GPU fits took 789
T4-minutes in all.

Protocols, each committed before any fit of its version: [protocol_v3.yaml](../configs/protocol_v3.yaml),
[protocol_v4.yaml](../configs/protocol_v4.yaml), [protocol_v5.yaml](../configs/protocol_v5.yaml),
[protocol_v6.yaml](../configs/protocol_v6.yaml). Methods and design notes:
[V3_METHODS](V3_METHODS.md), [V4_METHODS](V4_METHODS.md), [V5_DESIGN_NOTE](V5_DESIGN_NOTE.md),
[V6_DESIGN_NOTE](V6_DESIGN_NOTE.md). Data: [data card, v3–v6](DATA_CARD_V3_V6.md).

Model weights and trajectories are generated by the runs and excluded from Git, as is raw third-party data.
