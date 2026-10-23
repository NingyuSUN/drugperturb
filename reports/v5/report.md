# DrugPerturb v5: on Tahoe-100M, chemCPA does not beat vehicle-anchored baselines

**The prespecified primary endpoint was not met.** The selected chemCPA arm was the counterfactual pseudobulk arm, I2. It did not predict held-out compounds better than the vehicle-anchored training mean or the anchored Ridge:

| Contrast | Difference [95% interval] |
|---|---|
| I2 − anchored training mean (B1a) | −0.0047 [−0.0104, +0.0011] |
| I2 − anchored Ridge (M1a) | −0.0046 [−0.0104, +0.0011] |

The study data: 84 test compounds, 30 contexts (10 cancer cell lines × 3 doses) and 2,464 test pairs, from a dataset this project had never scored.

- **I2 against simpler baselines.** I2 did beat zero change. Pooled, it also beat the training mean and the fingerprint Ridge, but that gain comes from one cell line whose vehicle samples are small. Elsewhere, I2 matches the training mean.
- **The strongest method uses no chemistry.** A plate-anchored training mean (B1p), the mean response of the training compounds measured on the same plate, beats every method by a wide margin.
- **Structure adds nothing.** The anchored Ridge equals the anchored training mean.

**How the study was run.** The protocol was prespecified before any fit. Arm selection used validation compounds only. Every prediction was sealed before test truth was first computed, and the validation diagnostics and the selection were committed and pushed before evaluation.

- **Design:** [protocol_v5.yaml](../../configs/protocol_v5.yaml), [design note](../../docs/V5_DESIGN_NOTE.md), [pipeline](../../docs/V5_EXTRACTION.md), [dataset survey](../../docs/V5_DATASET_SURVEY.md).
- **Before test scoring:** [diagnostics.md](diagnostics.md).
- **Tables:** [evaluation/](evaluation).
- **Runs and hashes:** [kaggle_runs.md](../../receipts/v5/kaggle_runs.md).

**Metrics.** Error is the mean over compounds of per-compound RMSE across the 2,000 frozen genes; lower is better. Intervals are paired scaffold-group bootstrap 95% intervals: 10,000 draws, seed 2026, one draw shared by all 30 gate-passing contexts. Pooled estimates average the context-level differences. Negative differences favour the first method.

## Data

**Source.** [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M), Hugging Face revision `349e2a55…`, CC0. It holds 50 cancer cell lines pooled in every well and 379 drugs at 0.05, 0.5 and 5 µM, for 24 h. Every file read was hash-checked.

**What was used:**

- **Compounds.** 336 standardized compounds in 290 scaffold groups, split 202 / 50 / 84 (test: 77 groups).
- **Cell lines.** Ten were drawn by a prespecified rule on vehicle-sample sizes only: SNU-1, A-172, HEC-1-A, MIA PaCa-2, C32, H4, LOX-IMVI, NCI-H2347, SW 1088 and SW1417.
- **Vehicle.** Each compound sample's strict vehicle is all DMSO cells of the same line on the same plate.
- **Count gates.** All 30 contexts pass the count gate, with 77–84 eligible test compounds each, so the study gate G4 is met.
- **Model cells.** Capped at 32 per sample and line.

## Confirmatory and key secondary endpoints

| Endpoint | Contrast | Difference [95% interval] | Result |
|---|---|---|---|
| **P1, beats vehicle-anchored baselines** (primary) | I2 − B1a | −0.0047 [−0.0104, +0.0011] | **not supported** |
| | I2 − M1a | −0.0046 [−0.0104, +0.0011] | |
| S1, beats zero change | I2 − B0 | −0.0359 [−0.0399, −0.0319] | reliably better |
| S2, beats delta baselines | I2 − training mean (B1c) | −0.0067 [−0.0098, −0.0036] | reliably better, but see below |
| | I2 − Ridge (M1c) | −0.0055 [−0.0087, −0.0023] | |
| S3, structure beyond anchoring | M1a − B1a | −0.00002 [−0.0003, +0.0002] | no difference |
| S4, anchoring gain | B1a − B1c | −0.0021 [−0.0080, +0.0036] | not established |

**Why I2 was the single primary arm.** On validation compounds, I2 led I3 by 0.0066 in mean per-seed RMSE, against a near-tie margin of 1.96 × SE = 0.0058. That is outside the margin, so I2 was tested alone at the 95% level.

**S1–S4** are secondary and unadjusted.

**This is the first time a chemCPA arm in this project beat zero change on untouched data.** The v3 reference and the NB arm did not: they collapsed again.

| Contrast | Difference [95% interval] |
|---|---|
| A0 − B0 | +0.0014 [+0.0006, +0.0024] |
| I1 − B0 | +0.0044 [+0.0030, +0.0060] |

## Every method

The table gives mean compound RMSE, averaged over the 30 contexts ([summary.tsv](evaluation/summary.tsv)).

| Method | RMSE | Uses compound structure | Uses the matched vehicle |
|---|---:|---|---|
| Plate-anchored training mean (B1p) | **0.3215** | no | through same-plate training deltas |
| I2, counterfactual pseudobulk (primary) | 0.3597 | yes | vehicle cells are the model input |
| Anchored training mean (B1a) | 0.3643 | no | yes |
| Anchored Ridge (M1a) | 0.3643 | yes | yes |
| Fingerprint Ridge (M1c) | 0.3652 | yes | no |
| Training mean (B1c) | 0.3664 | no | no |
| I3, NB + pseudobulk | 0.3703 | yes | vehicle cells are the model input |
| Zero change (B0) | 0.3956 | no | no |
| A0, v3 chemCPA | 0.3970 | yes | vehicle cells are the model input |
| I1, NB likelihood | 0.4000 | yes | vehicle cells are the model input |
| Nearest training compound (B2c) | 0.5007 | yes | no |

All arms, pooled over the 30 contexts. These are prespecified secondary contrasts, unadjusted ([secondary.tsv](evaluation/secondary.tsv)).

| Arm | − B0 | − B1c | − M1c | − B1a | − B1p |
|---|---|---|---|---|---|
| A0 | +0.0014 [+0.0006, +0.0024] | +0.0306 [+0.0267, +0.0342] | +0.0318 [+0.0278, +0.0355] | +0.0327 [+0.0263, +0.0391] | +0.0755 [+0.0707, +0.0803] |
| I1 | +0.0044 [+0.0030, +0.0060] | +0.0336 [+0.0297, +0.0373] | +0.0348 [+0.0308, +0.0385] | +0.0356 [+0.0293, +0.0421] | +0.0785 [+0.0737, +0.0834] |
| I2 | −0.0359 [−0.0399, −0.0319] | −0.0067 [−0.0098, −0.0036] | −0.0055 [−0.0087, −0.0023] | −0.0047 [−0.0104, +0.0011] | +0.0382 [+0.0339, +0.0426] |
| I3 | −0.0253 [−0.0287, −0.0218] | +0.0039 [+0.0018, +0.0058] | +0.0051 [+0.0030, +0.0071] | +0.0059 [−0.0005, +0.0125] | +0.0488 [+0.0445, +0.0533] |

## Where I2's gain over the delta baselines comes from

The analysis by cell line was prespecified as secondary. Mean RMSE over the three doses of each line:

| Cell line | B0 | B1c | B1a | B1p | I2 | B1a − B1c | I2 − B1c | I2 − B1a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SNU-1 | 0.436 | 0.409 | 0.396 | 0.362 | 0.405 | −0.013 | −0.004 | +0.009 |
| A-172 | 0.287 | 0.254 | 0.273 | 0.225 | 0.257 | +0.019 | +0.003 | −0.016 |
| HEC-1-A | 0.260 | 0.234 | 0.246 | 0.207 | 0.240 | +0.013 | +0.006 | −0.007 |
| MIA PaCa-2 | 0.323 | 0.294 | 0.318 | 0.260 | 0.296 | +0.023 | +0.002 | −0.022 |
| C32 | 0.337 | 0.308 | 0.312 | 0.273 | 0.310 | +0.004 | +0.002 | −0.002 |
| H4 | 0.403 | 0.376 | 0.384 | 0.334 | 0.374 | +0.008 | −0.002 | −0.010 |
| LOX-IMVI | 0.310 | 0.277 | 0.289 | 0.242 | 0.276 | +0.012 | −0.001 | −0.012 |
| NCI-H2347 | 0.258 | 0.228 | 0.254 | 0.201 | 0.233 | +0.026 | +0.005 | −0.021 |
| **SW 1088** | **1.069** | **1.040** | **0.923** | **0.896** | **0.958** | **−0.117** | **−0.082** | **+0.034** |
| SW1417 | 0.272 | 0.244 | 0.249 | 0.215 | 0.248 | +0.006 | +0.004 | −0.002 |

**SW 1088 is the exception.** Its vehicle units hold 74–626 cells per plate, against a dataset median of about 2,800. Its errors are three times those of the other lines, because the measured response T − V carries the sampling noise of a small vehicle unit. There:

- anchoring to the query's own vehicle helps a great deal (B1a − B1c = −0.117);
- I2 also gains (−0.082), because it is conditioned on the vehicle cells;
- I2 still stays worse than B1a.

**In the other nine lines:**

- I2 is within about ±0.006 of the training mean, and reliably worse than it in four of them ([secondary.tsv](evaluation/secondary.tsv)).
- Anchoring hurts B1a in eight of them (B1a − B1c up to +0.026). When the vehicle measurement is precise, replacing the plate-matched delta with "training mean treated level minus this vehicle" discards plate effects that the treated and vehicle wells share.

**Post hoc cut, not prespecified.** Averaged over the 27 contexts without SW 1088:

| Contrast | Difference |
|---|---|
| I2 − B1c | +0.0016 |
| I2 − B1a | −0.0090 |
| B1a − B1c | +0.0106 |

**What this means.** The pooled S2 result therefore reflects vehicle-noise cancellation in one line, not better prediction of compound effects. This is the same mechanism that v4 diagnosed on sci-Plex3, and the reason v5 made B1a and M1a the primary comparators.

## A plate-anchored mean beats everything

B1p beats B1a by −0.0428 [−0.0472, −0.0386] and I2 by 0.0382 [0.0339, 0.0426]. It has the lowest error of all methods in every one of the 30 contexts.

B1p averages the measured deltas of the training compounds on the query's own plate. Its success means that a large share of the measured change in Tahoe-100M is shared by all compounds on a plate: a plate-level effect beyond what the two vehicle wells capture. In Tahoe-100M, each plate holds one dose, so this effect cannot be separated from dose. A model that learns plate-shared signal from training compounds will look predictive without predicting compound-specific effects. Future Tahoe benchmarks need B1p or an equivalent comparator.

## Compound structure

- **Ridge adds almost nothing.** The fingerprint Ridge chose α = 1,000 or 10,000 in every context. It is slightly below the training mean in every context, by 0.0001–0.0045 (mean 0.0012); no interval was prespecified for this contrast. Anchored, it equals B1a to within 0.0003 (S3).
- **Nearest neighbour is worst.** The nearest training compound by Tanimoto similarity is the worst method (0.501).
- **The pseudobulk arms point the right way.** I2 and I3 produce effects of realistic size (effect size ratio 0.83 and 0.85 on test compounds) with gene-wise r of 0.42 and 0.39 ([test_diagnostics.tsv](evaluation/test_diagnostics.tsv)). B1a has r = 0.49 and B1p r = 0.61.

## Robustness of P1 (prespecified secondary)

- **By dose.**

  | Dose | I2 − B1a [95% interval] |
  |---|---|
  | 50 nM | +0.0027 [−0.0042, +0.0094] |
  | 500 nM | −0.0127 [−0.0197, −0.0055] |
  | 5 µM | −0.0040 [−0.0137, +0.0059] |

- **By context.** The 95% interval of I2 − B1a lies below zero in 14 of the 30 contexts and above zero in 4 ([contrasts_per_context.tsv](evaluation/contrasts_per_context.tsv)).
- **Leaving out one plate.** Leaving out the test compounds measured on any one plate moves I2 − B1a between −0.0063 and −0.0004 ([leave_one_plate_out.tsv](evaluation/leave_one_plate_out.tsv)). The interval excludes zero when plate 2, 3, 7 or 13 is left out, and includes it otherwise.

P1 as prespecified, pooled over all contexts, is not supported. The point estimate favours I2 by about 1%, and the evidence for that direction is not robust to which contexts and plates are included.

## Agreement with the validation results

On validation compounds, seed-averaged I2 was 0.0008 below B1a. On test compounds the pooled difference is −0.0047, and its interval includes zero. B1p led on validation (0.3005 against 0.3400) as on test. The selection, the comparators and the frozen genes were fixed before any of these test numbers existed.

## What this does and does not show

**What it shows:**

- **chemCPA with a counterfactual pseudobulk loss beat zero change on held-out compounds** from a dataset this project had never scored. In v4 on OP3, no arm did.
- **It does not beat a vehicle-anchored training mean or an anchored Ridge,** the prespecified primary comparison.
- **Its apparent advantage over the delta baselines comes from vehicle-noise cancellation in one line.** This confirms, on new data, the methodological lesson of v4.
- **The v3 model collapses again.** So does the NB-only arm.
- **On Tahoe-100M, plate-level effects dominate what simple baselines can capture.** A plate-anchored mean beats every method.

**What it does not show:**

- **Whether a longer budget would change the result.** Nine of ten I2 fits stopped at the prespecified 60-minute cap while still improving. A larger budget or tuning, both outside this protocol, might help.
- **Generalization beyond this design.** The result covers ten cancer cell lines, three doses at 24 h, single-well replicates for most compound-doses, and pooled spheroid culture.
- **Untouched by everyone.** Tahoe-100M is public and widely used. "Untouched" means untouched by this project.
- **Anything about drug efficacy, toxicity or mechanism.**
- **Any arm's anchored sensitivity output.** This is the predicted treated level minus the measured level of the arm's own model vehicle cells. At most 64 capped cells per vehicle unit are too few for that measurement, so the output has an RMSE of 0.98–1.44. It is not an endpoint and is not interpreted.

## Reproducibility

| Stage | Kaggle kernel | Code (commit) | Outcome |
|---|---|---|---|
| Extraction | dp-v5-extract v1 (CPU) | 9b94ed2 | 3,388 hash-checked shards; 15,676,406 cells |
| Preparation and comparators | dp-v5-prepare v1 (CPU) | 9b94ed2 | every anchor passed; comparators sealed (b05c4c9c…) |
| Fits | dp-v5-fit-a, dp-v5-fit-b v1 (T4 ×2 each) | 0917195 | 33 fits, about 13.6 GPU session-hours |
| Seals and selection | dp-v5-seal v1 (CPU) | 4b53816 | D6 gate passed; I2 selected; committed in 3a7dd0b and pushed before evaluation |
| Evaluation | dp-v5-evaluate v1 (CPU) | 4b53816 | every seal verified; test truth first computed here |

- **Receipts:** [receipts/v5/](../../receipts/v5), with commands, versions and input and output hashes. The large outputs (prediction archives, unit sums) stay out of Git, and the receipts hash them.
- **Amendments:** one, A1, an engineering restatement of an extraction anchor made before any data were read. No endpoint, comparator, gate or threshold changed after the protocol was committed.
