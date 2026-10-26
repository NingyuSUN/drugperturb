# DrugPerturb v6: design note

**Status: prespecified on 2026-10-26, before any extraction and any fit.** The protocol is [protocol_v6.yaml](../configs/protocol_v6.yaml), committed with `manifests/v6/` and [receipts/v6/design_audit.json](../receipts/v6/design_audit.json). Nothing has been extracted or fitted, and no expression value, expression summary or compound-sample cell count of any v6 cell line has been read. Changes from here on are amendments.

**Decisions.** You chose D1 — change the endpoint — and then the draft defaults for D2–D10. The primary endpoint is the **direction of change on each compound's own differentially expressed genes**. The residual target of the first draft stays in the study as arm M2 and endpoint S4, so its question still gets answered, at a secondary level.

## Why the endpoint was the problem

v1–v5 all scored mean per-compound RMSE over 2,000 genes. In that metric most genes do not respond to most compounds, so the error is dominated by genes carrying no compound-specific signal — and a method wins by not moving. v5 showed this at its limit:

| v5, 30 contexts | Mean compound RMSE | Uses chemistry |
|---|---:|---|
| **B1p, plate-anchored training mean** | **0.3215** | no |
| I2, the selected chemCPA arm | 0.3597 | yes |
| B1a, vehicle-anchored training mean | 0.3643 | no |
| B0, zero change | 0.3956 | no |

Two different things were hiding the question:

- **A nuisance component.** Plate-shared response, which a plate mean captures without any chemistry. The first draft removed this.
- **Dilution.** Most of the 2,000 genes did not move, so they contribute noise to every compound's score. **This is the larger problem**, and it is what v6 removes.

Restricting the score to the genes a compound actually moved should hurt B1p more than it hurts a model with real compound-specific signal: the plate mean captures the generic response of that plate, not what distinguishes one compound from another. That is why this endpoint has a real chance of separating them where v5's could not.

Scoring **direction** rather than magnitude does two more things. It makes scale irrelevant, so a model with the right biology and the wrong magnitude is no longer penalised — the first draft needed a calibration scalar for exactly this, and v6 does not. And it gives the endpoint a built-in null: zero change predicts 0, which counts as a tie, so **B0 scores exactly 0.5 — chance — by construction.**

## The honest-DE problem, and the fix

Here is the trap. If each compound's DE genes are chosen from the same measurement that defines the truth, the endpoint selects on the outcome. A gene picked because its measured delta happened to be large regresses toward the mean on re-measurement. That penalises methods predicting a mean and flatters models — a bias pointing the wrong way for this project, and it would break the rule in [AGENTS.md](../AGENTS.md) about keeping test responses out of selection.

**The fix is to split the cells.** Within each replicate well, the line's cells are partitioned into two disjoint halves, and the matched vehicle unit the same way:

| Half | Formed from | Used for |
|---|---|---|
| **S, selection** | `T_S` vs `V_S` | choosing D(q), the key's DE gene set; deciding responder status |
| **E, evaluation** | `T_E` vs `V_E` | the scoring truth for every method |

The four cell sets are disjoint, so `delta_S` and `delta_E` share no measurement noise. The partition is drawn once with a fixed seed, recorded per cell at extraction, and identical for every method and arm.

**This is feasible here.** In the v5 lines the median vehicle unit held 2,811 cells (5th percentile 122), and the source reports a median of 1,287 cells per cell line–drug–dose. The count gate doubles v5's, to 60 cells per treated well and 120 per vehicle unit, so each half clears v5's own 30-cell threshold.

**What the split costs is power, and the protocol measures it.** The `split_noise_floor` diagnostic scores `delta_S` against `delta_E` — the accuracy a perfect model of the underlying biology would reach given the split's noise. That is the endpoint's practical ceiling, reported before any arm is scored, and gate G8 stops the study if it falls below 0.60.

## The endpoint

For each (compound, context) key q:

1. **D(q)** = the 50 genes of the 2,000 frozen genes with the largest `|delta_S(q)|`.
2. **Responder gate.** q enters the primary population only if `‖delta_S(q)‖₂` exceeds the 95th percentile of a vehicle-versus-vehicle null, subsampled to q's own cell count. Many compounds at 0.05 µM do not measurably respond, and scoring them measures noise.
3. **Direction accuracy** = over D(q), the fraction of genes whose predicted sign matches the sign of `delta_E(q)`. A predicted zero counts as 0.5.

**P1** asks the selected arm to beat **both B1p** (v5's strongest method) **and M1a** (the strongest method that uses chemical structure), pooled over gate-passing contexts.

Everything is also reported as RMSE over D(q) and as full-gene RMSE on `delta_E`, so v6 connects back to v5 and direction is never confused with magnitude.

## Why the test set is new cell lines

v5 scored the 10 lines it drew, so their responses are open. v6 draws 10 of the untouched ones, giving 30 fresh contexts — the same size as v5.

| Lines | Count |
|---|---:|
| Tahoe-100M total | 50 |
| Meet v5's 200-cell median vehicle rule | 48 |
| Drawn and scored by v5 | 10 |
| **Pool for v6** | **38** |
| Left for a study after v6 | 28 |

Because the contexts are new, **no compound's v6 response has been read, whatever its v5 split role**, so the v5 compound split is reused unchanged (train 202, validation 50, test 84) and no compound moves between roles.

**Stated plainly:** the hypothesis and the choice of B1p as a comparator both come from v5's open results. That is how any follow-up works, and it is conservative — we picked the hardest known baselines to beat. What the rule forbids is a v6 test response informing a v6 choice, and none exists until the evaluation kernel runs.

## Arms, and what deliberately does not change

| Arm | Objective | Target | Seeds |
|---|---|---|---:|
| **M1** | I2 counterfactual pseudobulk | the measured delta — v5's selected arm | 10 |
| **M2** | I2 counterfactual pseudobulk | the leave-one-out residual `Δ − B1p` | 10 |

A0, I1 and I3 are dropped: A0 and I1 collapsed in v3, v4 and v5, and I3 was worse than I2 on Tahoe.

**The training objective and the checkpoint rule stay v5's full-gene MSE.** That is deliberate (D5). For M1, the *only* difference between v5 and v6 is the endpoint, which is what makes the result readable as an endpoint result. Re-weighting the loss toward responsive genes would change model and endpoint at once — it is the obvious next study, and it needs its own protocol and lines.

**The cheap comparators come first.** Every comparator is CPU-only, and two of their validation numbers are decisive on their own: whether B1p beats chance on a compound's own DE genes, and whether any structure-using method (M1c, M1a, M1r, B2c) does. D7 is whether a null result there should cut the GPU budget.

## Decisions, as made on 2026-10-26

| # | Decision | Chosen | What was declined |
|---|---|---|---|
| **D1** | Endpoint | **DE-gene direction primary; residual demoted to M2 and S4** | Keeping the residual target as the primary; testing only the 60-minute cap |
| D2 | Lines and draw | 10 from the 38-line pool, `default_rng(6)` | 20 lines (60 contexts, model-cell cap down to ~16); organ-stratified draw |
| D3 | k, the DE set size | 50, with 20 and 100 as prespecified sensitivities | 20 alone (noisier per compound); 100 alone (reaches into genes the split cannot resolve) |
| D4 | Arms | M1 and M2, 10 seeds each | M1 only (10 fits, drops the residual question); adding I3's residual form |
| D5 | Training objective | Unchanged from v5 (full-gene MSE) | Re-weighting the loss toward responsive genes — plausibly the bigger win, but then a positive result could not be attributed to the endpoint |
| D6 | Fit time cap | 60 minutes, as v5 | 120 minutes (14–27 GPU hours, removes v5's cap confound) |
| D7 | Cheap test as a budget gate | A null structure result on validation cuts the budget to M1 × 3 seeds | Running all 20 fits regardless |
| D8 | Responder gate | 95th percentile of the cell-count-matched vehicle null; primary population is responders | No gate (diluted by non-responders); a stricter percentile |
| D9 | Bootstrap draws | 10,000, seed 2026 | 2,000, as v4 |
| D10 | Gene space | Re-run v5's rule on the new lines | Reusing v5's 2,000 genes, ranked by variance in the v5 lines |

## The draw, and what it came out as

The audit read only `manifests/v5/cell_lines.tsv` and `manifests/v5/vehicle_units.tsv`, each verified against the SHA256 in the v5 audit receipt. The v5 audit had already counted vehicle cells for all 50 lines on all 14 plates, so **the v6 draw read no Tahoe-100M file at all.**

| Draw order | Line | Organ | Median vehicle cells |
|---:|---|---|---:|
| 1 | SW 900 | Lung | 2,815 |
| 2 | KATO III | Esophagus/stomach | 3,898 |
| 3 | LS 180 | Bowel | 3,103 |
| 4 | NCI-H661 | Lung | 224 |
| 5 | hTERT-HPNE | Pancreas | 2,591 |
| 6 | CFPAC-1 | Pancreas | 3,718 |
| 7 | Panc 03.27 | Pancreas | 2,859 |
| 8 | J82 | Bladder/urinary tract | 3,907 |
| 9 | SHP-77 | Lung | 3,258 |
| 10 | PANC-1 | Pancreas | 6,892 |

**Two things about this draw are worth stating plainly.**

**The organ spread is narrower than v5's** — 5 distinct organs against 7, with 7 of the 10 lines pancreas or lung. That is what the prespecified random rule produced. **It was not redrawn**, because redrawing a seeded draw because its spread looks unlucky is design tuning, and the rule was fixed before the draw was run. The cost is a narrower basis for any claim about generalization across tissues, and it is reported with the result. If you would rather have an organ-stratified rule, that is an amendment — legitimate while no expression value has been read, and recorded as a rule changed after seeing a draw.

**NCI-H661 is a small-vehicle line** (median 224 cells, against 2,591–6,892 for the other nine), and one of its 14 vehicle units holds 114 cells, just under the 120 the split gate needs. In v5, the small-vehicle line SW 1088 was exactly where chemCPA's apparent pooled gain came from. Having one here is informative rather than unfortunate — S6 and the per-line analysis test that explanation again on new data — but every pooled estimate is now reported with and without NCI-H661 as a prespecified sensitivity.

**Split feasibility:** 139 of the drawn lines' 140 vehicle units hold at least 120 cells, so the split leaves at least 30 per half; 9 of the 10 lines clear it on every plate.

## Compute estimate

**Extraction.** One pass over ~429 GB of pinned parquet in Kaggle CPU kernels, keeping the 10 new lines and recording the S/E partition per cell. v5's extraction code is reused with a different line list and that one addition.

**Fits.** 20 fits — 2 arms × 10 seeds — at 20–40 minutes each: about **7–13 T4 hours** at a 60-minute cap, 14–27 at 120. v5 used 13.6 session-hours for 33 fits. Kaggle's free quota is about 30 GPU hours a week.

**Cheap first.** Extraction, preparation, D(q), the responder gate, the split noise floor and every comparator are CPU-only, and they settle S3 before any GPU time is spent.

## Next steps

**Done on 2026-10-26:** the decisions, the design audit and its draw, and the commit of the protocol (status `PRESPECIFIED_BEFORE_V6_FITTING`) with `manifests/v6/`.

**Still to do, in order:**

1. **Your approval for the Kaggle jobs.** Nothing below has started.
2. **Extraction** (CPU): v5's extraction code with the v6 line list, plus recording the S/E partition per cell. This is the one code change extraction needs.
3. **Preparation** (CPU): the gene space, D(q) and responder status from the selection half, the residual targets, and every comparator sealed.
4. **The cheap report**: S3, the responder fractions and the split noise floor on validation — the D7 decision point for the GPU budget, and gate G8.
5. **Fits** (GPU), then the diagnostics and selection record, then sealing, then evaluation.

## What this study can and cannot show

**It can show** whether a chemCPA model gets the direction of change right on the genes a held-out compound actually moved, on cell lines this project has never scored, and whether chemical structure gives directional information in *any* method including Ridge.

**It cannot show** that such a model predicts responses. A positive v6 beside a negative v5 would mean the model carries directional information that full-gene RMSE cannot see — genuinely useful, and much weaker than "predicts drug responses". Direction ignores magnitude, so an arm that beats P1 is not usable where effect sizes matter.

**Three further limits.** D(q) comes from a noisy half-measurement, so it estimates a compound's DE genes rather than knowing them. The primary estimand is conditional on a measured response, though S2 reports the all-keys version. And because the loss still optimises full-gene MSE, a negative result says this model trained this way does not predict direction — not that no model can.

**The outcome is genuinely uncertain this time**, which was not true of the first draft. v5's `M1a − B1a` of −0.00002 says structure added nothing *in delta space over all genes*; no v1–v5 endpoint ever asked about direction on responsive genes. S3 answers it for linear models in CPU minutes, whatever the GPU arms do.
