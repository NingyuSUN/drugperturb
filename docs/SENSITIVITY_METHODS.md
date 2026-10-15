# Predeclared sensitivities with frozen predictions

This script is a post-primary robustness check. It starts only after both `receipts/prediction_seal.json` and `receipts/final_evaluation.json` exist and agree with the frozen dataset and prediction artifacts. It does not train models, modify preprocessing, choose another configuration, generate predictions for new compounds, or change the primary results.

Run from the project root after primary evaluation:

```bash
.venv/bin/python scripts/sensitivity.py
```

A verified existing sensitivity receipt is reused. Changed inputs or outputs fail explicitly instead of silently replacing a completed result. The script records source hashes, fixed settings, denominators and the prediction-seal hash in `receipts/sensitivity.json`; detailed outputs are in `reports/sensitivity/`.

## Two-vehicle control sensitivity

The primary contrast uses the dose-pattern-matched vehicle. The predeclared alternative uses the two existing vehicle wells matched on cell type, time point, biological replicate and culture plate, which have different dose patterns. This choice is fixed by metadata, not by which reference improves model errors.

Each replicate must have **exactly two** such control wells, each explicitly marked `vehicle=True`, each with drug `dose=0`, and each with **at least 30** qualifying cells. Their dose patterns must differ and one must be the primary strict control. Missing QC, insufficient cells, malformed labels or nonunique control groups exclude that compound from the alternative comparison if either biological replicate fails. Both replicate records and exact exclusion reasons remain in `control_qualification.tsv`. Primary treatment/control mapping and cell counts are cross-checked against frozen metadata.

For every required well, normalize raw UMI counts by its complete human-gene library sum, multiply by one million and apply `log1p`. Only afterward select the already frozen model-output genes. Average the two separately normalized control vectors with equal weight, subtract that average from the normalized treatment vector, and finally average the two biological-replicate deltas with equal weight. **Do not pool counts from the two controls before normalization**, and do not weight controls or biological replicates by cell count or library size. The controls can have different library compositions and sizes, so the order of operations changes the response.

As a consistency check, reconstruct strict-control responses from the same raw counts and require agreement with the sealed primary selected-gene truth at relative and absolute tolerance 1e-10. UMI totals must also agree with well-QC totals. This checks arithmetic and identity alignment; it does not validate the biological equivalence of the two vehicle conditions.

Evaluate the same sealed six main predictions (`B0`, `B1`, `B2`, `M1`, `M2`, `M3`) against the alternate measured response, using the same frozen selected genes. Report primary performance on **the identical alternative-qualified compound subset** beside alternate performance, plus paired per-compound changes in RMSE and the magnitude of the change in measured truth. Primary performance on all test compounds is also retained. No model is re-centered to the new reference: even the frozen zero-change baseline is left unchanged.

This is a **changed-truth sensitivity**. The models were trained against the strict primary contrast; the alternative evaluation answers how their existing predictions behave under a different reference definition. It is not an independent validation, an alternate-control-trained benchmark, or a basis for selecting whichever control yields the better result. Shared controls and overlapping compounds induce correlation; the script does not manufacture additional independent replicates or confidence intervals.

## Cell-count sensitivity

Restrict the original test set to compounds for which **both treatment replicates have at least 50 qualifying cells**. Keep the same sealed primary truth and predictions, including the primary control requirements. Report the actual retained and original test denominators. This restriction can change compound composition, so improvement need not be caused only by reduced sampling noise.

Separately copy the existing 20/30/50-cell eligibility counts from `receipts/dataset_freeze.json`, with the original source-label denominator. Those historical eligibility counts apply the stated threshold to both treatment replicates and their strict controls, followed by chemical qualification for the unique-parent count. They are not counts of newly scored test compounds. In particular, the sensitivity does not fit or predict the extra compounds that would become eligible at 20 cells.

## Outputs and limitations

The output includes metadata-only control qualification, per-compound metrics, scenario summaries, paired control-reference RMSE changes, response-shift context, and source eligibility counts. The four fixed scenarios are: all primary test compounds; primary truth restricted to alternative-qualified compounds; alternate truth on that identical subset; and primary truth restricted to both treatment replicates >=50 cells. Six main methods give up to 24 scenario/method summaries; a zero-sized subset is explicitly NA with its denominator, not dropped.

RMSE is calculated across the frozen selected genes for each compound, then averaged equally across compounds. Additional existing compound metrics retain undefined values and their denominators. No analysis here demonstrates new chemical generalization, mechanism, efficacy or clinical validity. Sensitivity results do not justify a post hoc change to the frozen primary benchmark.
