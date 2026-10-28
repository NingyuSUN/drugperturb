# DrugPerturb v1–v6: six prespecified tests, one answer

**Across six studies, three datasets, two published single-cell models and two kinds of endpoint, no
method that receives chemical structure met a prespecified primary endpoint against a baseline that
receives no chemical structure at all.** Six primary endpoints, six nulls. The one place a structure-using
model beats the strongest baseline with an interval excluding zero is a v6 secondary endpoint, and it is
about the magnitude of the response rather than its direction.

Each version existed to remove one explanation for that result. v1 removed "nothing good has been tried
yet". v2 removed transport. v3 removed "our models are too small". v4 removed "the published model is
broken, not wrong". v5 removed scale. v6 removed the last one on the list — "the metric rewards predicting
nothing" — and the answer did not change.

This document is the through-line. [中文版](synthesis.zh-CN.md) · Every number below is quoted from the version report it belongs to; the
per-version reports hold the denominators, the gates and the limitations.

| | Question | Data | Answer |
|---|---|---|---|
| [v1](report.html) | Do fingerprint and graph models beat simple baselines on held-out compounds? | sci-Plex3 A549 | No reliable advantage over the training mean or Ridge |
| [v2](v2/report.md) | Do A549-trained structure models transfer to other cell lines and doses? | sci-Plex3 | No. Zero change has the lowest error in all five conditions |
| [v3](v3/report.md) | Do chemCPA and biolord beat information-matched simple baselines? | sci-Plex3 | No. Both are worse than a per-context Ridge, and chemCPA collapses to zero change |
| [v4](v4/report.md) | Can chemCPA's collapse to zero change be fixed? | sci-Plex3, OP3 | Partly. The drug term comes alive, but no variant beats a per-context training mean |
| [v5](v5/report.md) | Does an improved chemCPA beat vehicle-anchored baselines on data this project had never scored? | Tahoe-100M | No. It beats zero change but not the anchored baselines, and a plate-anchored mean beats every method |
| [v6](v6/report.md) | On an endpoint where predicting nothing scores chance by construction, does chemCPA beat a plate-anchored mean? | Tahoe-100M, ten new cell lines | No. It matches that mean while beating chance by +0.31, and structure adds nothing |

## The chain

Each row is an explanation that was live before the version ran, and what the version did to it.

**v1 — "nothing good has been tried yet."** Six methods on one context: zero change, training mean,
chemical nearest neighbour, fingerprint Ridge, fingerprint MLP and a bond-aware molecular GNN, with whole
chemical structure groups held out and predictions sealed. The GNN had the lowest point estimate (1.2144
against the training mean's 1.2170 and zero change's 1.2694), and its intervals included zero: −0.0286
[−0.0680, +0.0087] against Ridge and −0.0026 [−0.0475, +0.0419] against the training mean. *Removed:* the
idea that a point-estimate ranking settles anything at this sample size. It also exposed the first baseline
problem — averaging two separately normalized same-plate vehicle wells flips the ranking in favour of the
zero-change and training-mean references.

**v2 — "the model learned A549; it just has to be used elsewhere."** The v1 predictions, splits and gene
space were frozen, then scored on MCF7 at 24 h and 1 µM (primary) plus four secondary conditions. Zero
change had the lowest mean error in all five: 0.53251 against the source mean's 0.65132, Ridge's 0.67322
and the GNN's 0.71975. Gene-level GNN error exceeded Ridge by +0.046539 [+0.002349, +0.104793]. *Removed:*
transport. A model that receives structure but not cell identity or dose has nothing to transport with.

**v3 — "our models are too small; the published single-cell models are the real test."** chemCPA and
biolord, trained on sci-Plex3 cells across 3 cell lines × 4 doses, given cell type, dose **and**
target-context training data, sealed before scoring, on six contexts never scored before. Both were
reliably worse than a per-context fingerprint Ridge: +0.057 [+0.038, +0.074] and +0.052 [+0.031, +0.072].
chemCPA was not distinguishable from zero change at all — −0.0000 [−0.0001, +0.0000] — and a
training-only diagnostic found why: its drug term was 0.6% of the latent norm after 200 epochs, even for
training compounds. *Removed:* the capacity excuse, and the mystery. Given target-context training data,
even the per-context training mean beats zero change in all 12 contexts, which reverses v2's conclusion.

**v4 — "chemCPA is broken, not wrong."** Two principled interventions: a negative binomial count
likelihood, and a counterfactual pseudobulk loss that trains the exact path used at prediction time. Both
worked as mechanisms. The drug-term size ratio went from 0.007 to 0.25–0.35 (NB) and to 1.62 with a
gene-wise correlation of 0.64 against the truth (pseudobulk). On OP3 — PBMCs from three donors, 35 test
compounds, a dataset this project had never scored — the selected NB arm was still worse than zero change
(+0.171 [−0.004, +0.532]) and than the training mean (+0.183 [+0.009, +0.544]); one compound with
mis-scaled descriptors drove that, and excluding it post hoc the arm matches zero change (+0.001 [−0.005,
+0.008]) and remains worse than the training mean (+0.015 [+0.007, +0.023]). *Removed:* the collapse, as an
explanation. It also removed one of our own results: on sci-Plex3 the pseudobulk arms looked up to 30%
better than every v1 method, and a **vehicle-anchored** training mean, which uses only the matched vehicle
well's measurement, does better still. The apparent gain was the vehicle measurement, not the compound
prediction. That became amendment A1, made before any OP3 outcome was read.

**v5 — "sci-Plex3 and OP3 are too small; 100 million cells will show it."** Tahoe-100M: 84 test compounds
in 77 scaffold groups, 10 cancer cell lines × 3 doses, 2,464 test pairs, with the anchored baselines now in
place. The pseudobulk arm became the first chemCPA arm in this project to reliably beat zero change
(−0.0359 [−0.0399, −0.0319]) — and still did not beat the anchored training mean (−0.0047 [−0.0104,
+0.0011]) or the anchored Ridge (−0.0046 [−0.0104, +0.0011]). Its pooled gain over the delta-space
baselines traced to one cell line whose vehicle samples are small, i.e. to v4's lesson. Meanwhile a
**plate-anchored** training mean, which uses no chemistry whatsoever, had the lowest error in every one of
the 30 contexts, beating the vehicle-anchored mean by −0.0428 [−0.0472, −0.0386]. *Removed:* scale. And it
left a sharper target than chemCPA: the plate.

**v6 — "the metric rewards predicting nothing."** This was the strongest remaining objection, and it was
structural: on an RMSE endpoint, predicting no change wins whenever the measurement is noisy. v6 replaced
the endpoint with **direction accuracy on each (compound, context) key's own top-50 differentially
expressed genes**, where a predicted zero counts as a tie and therefore scores exactly 0.5000 — chance, by
construction. The genes are chosen from one half of a key's cells and scored against the disjoint other
half, so no measurement both picks a key's genes and grades it. Ten Tahoe-100M cell lines v5 never touched;
27 of 30 contexts pass the gate; 1,323 responder keys. The endpoint change worked: zero change sits on
0.5000 and the residual arm beats it by +0.3135 [+0.2999, +0.3265]. The primary endpoint was still not met
— M2 − plate-anchored mean = **+0.0010 [−0.0007, +0.0039]**. *Removed:* the metric.

## The ledger

Every version's primary contrast, in the sign convention of its own endpoint. v1–v5 are error metrics (↓,
so a **negative** difference favours the first method); v6 is an accuracy (↑, so a **positive** difference
favours the first method). The two cannot be compared numerically.

| Version | Endpoint | Primary contrast | Difference [95% interval] | Met? |
|---|---|---|---|---|
| v1 | mean compound RMSE ↓ | GNN − fingerprint Ridge | −0.0286 [−0.0680, +0.0087] | No |
| v1 | mean compound RMSE ↓ | GNN − training mean | −0.0026 [−0.0475, +0.0419] | No |
| v2 | gene-level RMSE ↓ | GNN − Ridge, MCF7 24 h 1 µM | +0.046539 [+0.002349, +0.104793] | No, reliably worse |
| v2 | five-set MAE ↓ | direct Ridge head − source mean | −0.016636 [−0.035685, +0.003450] | No |
| v3 | mean compound RMSE ↓ | chemCPA − per-context Ridge | +0.057 [+0.038, +0.074] | No, reliably worse |
| v3 | mean compound RMSE ↓ | biolord − per-context Ridge | +0.052 [+0.031, +0.072] | No, reliably worse |
| v4 | mean compound RMSE ↓ | NB arm − zero change (OP3) | +0.171 [−0.004, +0.532] | No |
| v4 | mean compound RMSE ↓ | NB arm − per-context training mean | +0.183 [+0.009, +0.544] | No, reliably worse |
| v5 | mean compound RMSE ↓ | pseudobulk arm − anchored training mean | −0.0047 [−0.0104, +0.0011] | No |
| v5 | mean compound RMSE ↓ | pseudobulk arm − anchored Ridge | −0.0046 [−0.0104, +0.0011] | No |
| v6 | direction accuracy ↑ | residual arm − plate-anchored mean | +0.0010 [−0.0007, +0.0039] | No |
| v6 | direction accuracy ↑ | residual arm − anchored Ridge | +0.0280 [+0.0214, +0.0361] | Yes |

Six primary endpoints, none met. The one prespecified comparison a model did win, v6's contrast against the
anchored Ridge, it won because the Ridge is in the wrong space for a directional endpoint, not because
chemistry helped: the three contrasts that isolate chemistry all straddle zero (+0.0013 [−0.0003, +0.0027],
+0.0000 [−0.0008, +0.0008], −0.0004 [−0.0011, +0.0003]).

## What the series established, besides six nulls

**A baseline ladder, and the fact that every rung changed a conclusion.** Zero change → training mean →
per-context mean → vehicle-anchored mean → plate-anchored mean. v2's conclusion ("zero change is hard to
beat") was reversed by v3 the moment target-context training data was allowed. v4's apparent 30% gain
disappeared against a vehicle-anchored mean. v5's pooled gain disappeared against a plate-anchored mean.
None of those reversals came from a better model; each came from a better comparator. The strongest
baseline in this project uses no chemistry, no cell-state encoder and no learning: the mean response of the
training compounds measured on the same plate.

**Where the variance lives.** Plate and context dominate compound. In v5 the plate-anchored mean beat every
method in all 30 contexts; in v6 it is above the residual arm in 10 of 27 contexts and the arm is above it
in 15, with 2 ties, and it beats the raw-target arm in all 27. Whatever a model learns about a compound is
small relative to where and when the compound was measured.

**The training target matters more than the model.** v6's largest effect is not a model change: training
chemCPA on the residual over the plate mean rather than on the raw response is worth +0.0612 [+0.0490,
+0.0750]. What it buys is the plate mean back — the residual the model learns has a validation correlation
of about −0.007 with the real one, so "predict the residual" converged on "predict no residual".

**A ceiling, measured rather than assumed.** v6's split-measurement design makes the measurement's accuracy
against itself computable: 0.8232 on validation. The plate-anchored mean reaches 0.8217 on the test half.
A perfect model therefore has roughly 0.0015 of room above a no-chemistry baseline on this endpoint. The
null is not a statement about the models alone; the question as posed has almost no headroom left.

**The one thing a model does add is magnitude, not direction.** v6's secondary endpoint on restricted RMSE
is the single place in six versions where a chemCPA arm beats the strongest baseline with an interval
excluding zero: M2 − B1p = −0.0020 [−0.0049, −0.0002] (an error, so negative favours M2). It is small, it
is secondary, and it is about how large the response is rather than which way it goes — but it is the one
positive result in the series, and it should not be rounded off to zero.

**The open problem is gene identity, not direction.** Overlap between a method's own top-50 predicted genes
and a key's real top-50 is at most 0.13 (Jaccard) for every method, the plate mean included: B1p 0.1275,
M2 0.1254, M1a 0.0938, M1 0.0802. On a given gene set, direction is largely solved by the anchor. Which
genes move is not solved by anything here.

## What stayed constant, and what tightened

The same discipline ran through all six versions: whole chemical structure groups held out so no test
parent or scaffold identity appears in training; gene selection, standardization and PCA fitted on training
compounds only; predictions sealed and hash-verified before test truth was computed for the first time;
paired scaffold-group bootstrap intervals; prespecified count gates; and input/output hash receipts for
every run.

It tightened in three steps worth naming:

- **v4 — baselines must match the information the model receives.** A model conditioned on vehicle cells
  must be compared against a baseline that also sees the matched vehicle well (amendment A1).
- **v5 — the protocol is committed, and so is the pre-test state.** The seal, the arm selection and the
  validation diagnostics were committed and pushed before the evaluation ran, so the pre-test state carries
  an external timestamp. v6 did the same.
- **v6 — the measurement is split.** Selection and evaluation halves are disjoint, so the half that chooses
  a key's genes and decides whether it responded is never the half that grades the prediction.

Post hoc work is reported next to the prespecified result, never in its place: v4's descriptor-scaling fix
and the rerun it required ([corrected report](v4_corrected/report.md)), and v6's amendment
`A3_2026-10-27_RAISED_AND_WITHDRAWN`, which records a reading of a decision rule that was raised, judged a
post hoc convenience, and withdrawn in favour of the committed wording — after which the full prespecified
plan of 20 fits ran.

## What this does not show

- **Three datasets, not the field.** sci-Plex3, OP3 and Tahoe-100M. "Untouched" means this project had not
  scored them before, not that they are private.
- **Two published models, as implemented here.** chemCPA (vendored) and biolord. No pretrained foundation
  model was trained or evaluated, and a different implementation or a larger pretraining corpus is not
  tested by any of this.
- **Benchmarks, not biology.** Zero change winning an error endpoint does not mean the drugs do nothing;
  v6 exists precisely because that endpoint rewards silence. None of this establishes efficacy, toxicity,
  target engagement or clinical benefit.
- **The anchor exists in these designs.** Every dataset here measures training compounds on the same plate
  as the test compounds, so a plate-anchored mean is always available. In a genuinely prospective setting —
  a new plate, a new batch, no training compounds measured alongside — that comparator does not exist, and
  the question is open there. That is the most interesting thing these six versions did not test.
- **Endpoint coverage.** Direction and magnitude on a 2,000-gene space (v1–v5) and direction on each key's
  own top-50 DE genes (v6). Dose–response shape, time course and single-cell heterogeneity were not
  endpoints.

## Pointers

| Version | Report | Protocol | Methods / design |
|---|---|---|---|
| v6 | [report](v6/report.md), [pre-test diagnostics](v6/diagnostics.md), [CPU-only report](v6/cheap_report.md) | [protocol_v6.yaml](../configs/protocol_v6.yaml) | [V6_DESIGN_NOTE.md](../docs/V6_DESIGN_NOTE.md) |
| v5 | [report](v5/report.md), [pre-test diagnostics](v5/diagnostics.md) | [protocol_v5.yaml](../configs/protocol_v5.yaml) | [V5_DESIGN_NOTE.md](../docs/V5_DESIGN_NOTE.md), [V5_EXTRACTION.md](../docs/V5_EXTRACTION.md) |
| v4 | [report](v4/report.md), [corrected](v4_corrected/report.md), [pre-test diagnostics](v4/diagnostics.md) | [protocol_v4.yaml](../configs/protocol_v4.yaml) | [V4_METHODS.md](../docs/V4_METHODS.md) |
| v3 | [report](v3/report.md) | [protocol_v3.yaml](../configs/protocol_v3.yaml) | [V3_METHODS.md](../docs/V3_METHODS.md) |
| v2 | [report](v2/report.md), [中文](v2/report.zh-CN.md) | [protocol_v2.yaml](../configs/protocol_v2.yaml) | [V2_METHODS.md](../docs/V2_METHODS.md) |
| v1 | [report.html](report.html), [biologist summary](biologist_summary.html) | [protocol_v1.yaml](../configs/protocol_v1.yaml) | [RESEARCH_DECISIONS.md](../docs/RESEARCH_DECISIONS.md) |

Cross-cutting: [DATA_CARD.md](../docs/DATA_CARD.md), [MODEL_CARD.md](../docs/MODEL_CARD.md),
[REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md), [EXECUTION_EVIDENCE.md](../docs/EXECUTION_EVIDENCE.md).
