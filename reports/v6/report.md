# DrugPerturb v6: on a directional endpoint, chemCPA matches the plate-anchored mean and does not beat it

**The prespecified primary endpoint was not met.** The selected arm was the residual arm, M2, which learns the deviation from a plate-anchored training mean. It did not predict the direction of held-out compounds better than that mean:

| Contrast | Difference [95% interval] |
|---|---|
| M2 − plate-anchored training mean (B1p) | +0.0010 [−0.0007, +0.0039] |
| M2 − anchored Ridge (M1a) | +0.0280 [+0.0214, +0.0361] |

P1 needs both lower bounds above zero. The second clears it; the first does not.

The study data: 84 test compounds in 77 scaffold groups, 27 gate-passing contexts (9 cancer cell lines × 3 doses), 2,417 eligible test keys of which 1,323 are responders — ten cell lines of Tahoe-100M that v5 never touched.

- **The endpoint change worked.** v5's result was weak because zero change won on an RMSE endpoint. Here zero change (B0) scores exactly 0.5000 **by construction**, and M2 beats it by +0.3135 [+0.2999, +0.3265]. Predicting nothing is no longer a winning strategy, and the result is still negative.
- **Chemical structure adds nothing.** All three prespecified structure contrasts straddle zero: M1c − B1c +0.0013 [−0.0003, +0.0027], M1a − B1a +0.0000 [−0.0008, +0.0008], M1r − B1p −0.0004 [−0.0011, +0.0003].
- **What the training target is worth: a lot.** Training chemCPA on the residual over B1p instead of on the raw delta is worth +0.0612 [+0.0490, +0.0750] (S4). It is the largest effect in the study, and it is about the target, not about chemistry.
- **What it buys: B1p back.** M2's validation residual correlation is about zero (−0.007), so what the residual arm learned is close to "predict no residual", which returns the plate mean.
- **There is almost nothing left to win.** The split noise floor — the accuracy the measurement reaches against itself — is 0.8232 on validation, and B1p already reaches 0.8217 on the test half. A perfect model has roughly 0.0015 of room above the plate mean.

**How the study was run.** The protocol was prespecified before any fit. Arm selection used validation compounds only. Every prediction was sealed and hash-verified before the evaluation half's test truth was computed for the first time, and the validation diagnostics and the selection were committed and pushed before the evaluation kernel ran.

- **Design:** [protocol_v6.yaml](../../configs/protocol_v6.yaml) (amendments A1, A2 and the withdrawn A3 are logged there).
- **Before test scoring:** [cheap_report.md](cheap_report.md) (CPU only, no arm fitted), [diagnostics.md](diagnostics.md), [per_seed_validation.tsv](per_seed_validation.tsv).
- **Tables:** [evaluation/](evaluation).
- **Runs and hashes:** [kaggle_runs.md](../../receipts/v6/kaggle_runs.md).

**Metrics.** The endpoint is direction accuracy on each (compound, context) key's own top-50 differentially expressed genes, chosen from the **selection** half of that key's cells and scored against the disjoint **evaluation** half. A predicted zero counts as 0.5, which is why chance is exactly 0.5 and why B0 sits on it. Higher is better, so a positive difference favours the candidate — the opposite of v1–v5, which were error metrics. Intervals are paired scaffold-group bootstrap 95% intervals: 10,000 draws, seed 2026, one draw shared by all 27 gate-passing contexts. Pooled estimates average the context-level differences.

## Data

**Source.** [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M), Hugging Face revision `349e2a55…`, CC0. Every file read was hash-checked.

**What was used:**

- **Cell lines.** Ten lines v5 did not use, drawn by a prespecified rule on vehicle-sample sizes only, from the committed v5 manifests with no new expression read ([v6_design_check.json](../../receipts/v6/e1/v6_design_check.json)).
- **Compounds.** The v5 split, reused byte-for-byte: 336 standardized compounds in 290 scaffold groups, 202 / 50 / 84. The test set's 77 scaffold groups are the bootstrap unit; 76 of them contain at least one responder key, so the responder analyses resample 76 groups and the all-keys analyses resample 77.
- **Scale.** 3,388 shards, 11,760 units, 17,698,679 pass-filter cells, each found exactly once.
- **The measurement split.** Each unit's cells, and those of its matched vehicle unit, are partitioned into two disjoint halves (seed 606; median 678 selection and 679 evaluation cells per unit). The selection half chooses each key's genes and decides whether it is a responder; the evaluation half is the truth the endpoint scores against. This is what makes the gene choice honest: no measurement both picks a key's genes and grades it.
- **Count gates.** 27 of 30 contexts pass. All three failures are the same line, CVCL_1577, whose halves leave no responder; the study gate G4 is met with 9 lines and 3 doses.

## Confirmatory and key secondary endpoints

| Endpoint | Contrast | Difference [95% interval] | Result |
|---|---|---|---|
| **P1** (confirmatory) | M2 − B1p | **+0.0010 [−0.0007, +0.0039]** | **not met** |
| **P1** (confirmatory) | M2 − M1a | **+0.0280 [+0.0214, +0.0361]** | met |
| S1 | M2 − B0 (chance) | +0.3135 [+0.2999, +0.3265] | far above chance |
| S2 (all keys, not only responders) | M2 − B1p | +0.0004 [−0.0007, +0.0022] | null |
| S2 (all keys) | M2 − M1a | +0.0280 [+0.0231, +0.0334] | met |
| S3 | M1c − B1c | +0.0013 [−0.0003, +0.0027] | null |
| S3 | M1a − B1a | +0.0000 [−0.0008, +0.0008] | null |
| S3 | M1r − B1p | −0.0004 [−0.0011, +0.0003] | null |
| **S4** | M2 − M1 (residual target vs raw) | **+0.0612 [+0.0490, +0.0750]** | the residual target wins |
| S5 (restricted RMSE, lower is better) | M2 − B1p | **−0.0020 [−0.0049, −0.0002]** | M2 wins, narrowly |
| S6 (restricted RMSE) | B1p − B1a | −0.0239 [−0.0293, −0.0186] | plate anchoring wins |
| S6 (restricted RMSE) | B1a − B1c | −0.0492 [−0.0617, −0.0374] | vehicle anchoring wins |

S4 is estimable because both candidate arms were fitted at all ten prespecified seeds. Decision D7, whose committed trigger is a margin over chance, did not fire; amendment A3 records that the alternative reading was raised before any GPU fit and withdrawn.

## Every method

Direction accuracy over each key's own top-50 genes, and restricted RMSE on the same genes. 2,417 keys, of which 1,323 are responders.

| Method | Uses structure | Responders | All keys | Restricted RMSE |
|---|---|---:|---:|---:|
| M2_pseudobulk_residual (selected arm) | yes | **0.8226** | 0.7591 | **0.6274** |
| B1p, plate-anchored training mean | no | 0.8217 | 0.7588 | 0.6283 |
| B1r, Ridge on the residual | no | 0.8212 | 0.7586 | 0.6287 |
| M1r_unscaled, fingerprint Ridge on the residual | yes | 0.8212 | 0.7584 | 0.6287 |
| M1r, the same, scaled | yes | 0.8212 | 0.7584 | 0.6287 |
| B1a, vehicle-anchored training mean | no | 0.7925 | 0.7329 | 0.6529 |
| M1a, anchored fingerprint Ridge | yes | 0.7925 | 0.7330 | 0.6526 |
| M1c, contrast fingerprint Ridge | yes | 0.7827 | 0.7198 | 0.6867 |
| B1c, contrast training mean | no | 0.7813 | 0.7182 | 0.6883 |
| M1_pseudobulk_raw (the other arm) | yes | 0.7612 | 0.6953 | 0.7250 |
| B2r, descriptor Ridge on the residual | yes | 0.7432 | 0.6976 | 0.7469 |
| B2c, descriptor Ridge on the contrast | yes | 0.6663 | 0.6293 | 0.8063 |
| B0, zero change | no | **0.5000** | 0.5000 | 0.8406 |
| *Split noise floor (validation): the ceiling* | — | *0.8232* | — | — |

The ordering is the same at k = 20 (M2 0.8232, B1p 0.8222) and k = 100 (M2 0.8240, B1p 0.8232), so nothing here depends on the choice of 50 genes.

## Where the differences come from

**The plate anchor is the whole story.** The three anchoring levels are separated far more than any method is separated within a level: contrast (B1c 0.7813) → vehicle-anchored (B1a 0.7925) → plate-anchored (B1p 0.8217). Inside each level the fingerprint Ridge and the training mean are indistinguishable, and chemCPA's residual arm sits 0.0010 above the plate mean.

**No single plate carries it.** Dropping each of the 14 plates in turn leaves M2 − B1p with a lower bound below zero every time (0 of 14 above zero, point estimates +0.0000 to +0.0019). The null is not an artifact of one plate, which is the failure mode v5 found for its own pooled gain.

**Per context it is a coin flip.** M2 is above B1p in 15 of the 27 gate-passing contexts, below in 10 and equal to four decimals in 2, with a spread of ±0.003. The raw arm M1, by contrast, is below B1p in all 27.

**No method finds the right genes.** Jaccard overlap between a method's own top-50 predicted genes and the selection-half D(q) is at most 0.13 for everything: B1p 0.1275, M2 0.1254, M1a 0.0938, M1 0.0802. Every method gets the direction largely right on genes it could not itself have identified. That, not the direction, is where the problem is unsolved.

**Dropping the small-vehicle line changes nothing**, because CVCL_1577 already fails the count gate — its halves leave no responder. The prespecified sensitivity is therefore a no-op here, not a passed test.

## What this does and does not show

**It shows:**

- On an endpoint where predicting nothing scores exactly chance by construction, chemCPA trained on the residual over a plate-anchored mean is statistically indistinguishable from that mean on the direction of held-out compounds, in ten cell lines never used before.
- Chemical structure adds nothing at any anchoring level, now on a directional endpoint as well as on the error endpoints of v5.
- The choice of training target is worth far more than the choice of model: the residual target beats the raw delta by +0.0612, and the raw-delta arm loses to the plate mean in every context.
- The plate-anchored mean itself is strong, and on restricted RMSE M2 improves on it by a small but interval-excluding-zero margin (−0.0020 [−0.0049, −0.0002]). The magnitude of the response is where the residual arm earns anything.

**It does not show:**

- That a better model could not win. It shows that on this endpoint there is about 0.0015 of room above B1p, measured against a validation-half ceiling. A test-half ceiling was not computed; the floor in the record is the validation figure the seal carried.
- Anything about absolute dose–response, or cell-type generalization beyond these ten lines, 2,000 frozen genes, log-CPM pseudobulk and a single 24-hour read-out.
- Anything about which genes move. Direction accuracy on a key's own top-50 genes answers "which way", given the gene set; the overlap figures say no method recovers the gene set.
- That the halves are as informative as the full data. The S/E split halves the cells behind both the gene choice and the truth, so both are noisier than an unsplit analysis would be — the price of an honest gene choice.

## Reproducibility

- **Protocol** `configs/protocol_v6.yaml`, SHA256 `a40b56d2…`, frozen as `PRESPECIFIED_BEFORE_V6_FITTING` on 2026-10-26.
- **Sealed predictions**, hash-verified by the evaluator before it read any test response: M1 `b8318176…`, M2 `4186534e…`, comparators `2b9be6ca…`.
- **Every fit record** certifies `test_outcomes_loaded: false` and `evaluation_half_test_delta_loaded: false`; so does the selection record. The seal refuses any fit that does not.
- **Anchors**, all passing: zero change scores chance, the truth fed back as a prediction scores 1.0, B0 is identically zero, the halves partition every unit, every model cell is found once.
- **Kernels, versions, code-archive hashes and runtimes:** [receipts/v6/kaggle_runs.md](../../receipts/v6/kaggle_runs.md). 20 GPU fits, 789 T4-minutes; a superseded 3-seed run of M1 is recorded there too and enters nothing. The evaluation ran twice, the second time only to correct a sign label on the two RMSE endpoints; both runs are kept, every estimate is identical, and the per-key and summary tables are byte-identical.
- **Tests.** 440 CPU tests pass locally and in GitHub CI; 7 `test_v6_cellmodels.py` tests ran inside each GPU kernel and the seal kernel; 20 evaluation and endpoint tests ran inside the evaluation kernel. These validate software, not biology: no test here checks a biological claim.
