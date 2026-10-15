# Exploratory development diagnostics

These diagnostics address sensitivity to the number and composition of primary training compounds. They do not replace final scaffold-held-out evaluation, and no final-test features or outcomes are loaded. They are deliberately limited to fingerprint Ridge (M1) and the two-layer molecular GNN (M3), with one training seed (42), no new hyperparameter search and no outcome-driven choice of folds or subset seeds.

## Inputs and execution gate

Run only after the main training procedure creates `receipts/model_selection.json` and `receipts/training_runs.tsv`:

```bash
.venv/bin/python scripts/development_diagnostics.py --source-commit "$(git rev-parse HEAD)"
```

Required response inputs are `data/processed/training_allgenes.npz` and `validation_allgenes.npz`, each with `y`, `fp` and `ids`. `y` must contain the original measured compound-mean response for all genes in `gene_metadata.tsv` order, not the main model's selected genes or PCA reconstructions. Training and validation compound IDs must be unique and disjoint. The corresponding `graphs_train.json` and `graphs_validation.json` use the same compound row order as their NPZ files. The four development artifacts must match `receipts/dataset_freeze.json`. The script verifies only these allowed development artifact hashes, not the final-test files named elsewhere in that receipt.

No diagnostics are launched by implementation or by importing the script. It fails before loading data when model selection is absent. Completed outputs are verified and reused. Interrupted runs can reuse individually completed, hash-verified contexts; changed inputs or fixed settings cause failure rather than silent reuse. An interrupted context without its completion receipt may be rerun.

## Fixed choices inherited from main development

M1 uses the alpha selected by the primary validation set. M3 uses the selected hidden dimension, dropout and learning rate. Its diagnostic epoch count is the rounded median of the three selected-candidate `best_epoch` values in the main training receipt. All three prescribed seeds must be present once. The diagnostic model is trained from scratch for this exact budget, with AdamW weight decay 0.0001, primary batch size, two message-passing layers, CPU execution and at most two Torch threads. Initialization, shuffling and dropout use seed 42 with deterministic Torch algorithms.

No outer-fold or learning-curve validation label is passed to the fitting helper. There is no diagnostic early stopping, checkpoint selection or learning-rate adjustment. Labels are selected and scored only after the fitted models have made predictions. Epoch training losses are optimization records, not evaluation losses. The primary validation set already influenced the hyperparameters and fixed epoch budget; reusing it for learning curves is therefore a development diagnostic, not independent validation.

## Three-fold compound cross-validation

Use `KFold(n_splits=3, shuffle=True, random_state=42)` on the primary training compounds only. The unit is a whole compound; cells and biological replicates are never split into separate model examples. Each fold fits `TrainingTargetTransform` from scratch on its two fitting folds: positive-variance gene eligibility, selection of up to 2,000 genes by training variance, label standardization, and up to 20 principal components (further limited by training sample count). Both models use this same fitted representation. Invert predictions to the selected original delta-expression scale and compare against measured, unreconstructed held-out fold responses.

The split is by compound rather than scaffold. Chemical series can occur on both sides, so these folds assess nearby-chemistry development behavior and must not be presented as unseen-scaffold performance. Main hyperparameters were selected using the main training/validation setup, so this is not nested unbiased model selection. Every compound receives one out-of-fold prediction per model, but fold gene sets can differ; report fold-level results and denominators instead of treating the predictions as one common gene matrix.

## Nested learning-curve subsets

Draw one seed-42 permutation of the primary training compound indices. Use its first `ceil(0.25*N)`, `ceil(0.50*N)` and `N` compounds, with a minimum of two compounds per subset; the script requires at least eight available compounds. These subsets are nested. Evaluate each fitted model on all primary validation compounds. Refit the complete gene-selection, standardization and PCA procedure using only the current subset, and apply the same fixed GNN epoch budget at every fraction.

Because each subset selects its own genes, a lower error at a larger fraction is not exclusively attributable to more training data: the evaluation gene set can also change. Record selected gene IDs, gene count, component count and the caveat alongside each score. Do not relabel this as an apples-to-apples fixed-gene learning curve. No subset is changed after observing its result.

## Outputs and interpretation

`receipts/development_diagnostics.tsv` contains 12 summary rows: two methods for three folds and three subset sizes. `receipts/development_diagnostics.json` adds per-compound errors, settings, source/input/artifact hashes and runtime. Each score is mean per-compound gene-space RMSE, with median RMSE and mean MAE also reported. Compounds receive equal weight. The training count, evaluation count, selected gene count, PCA dimension and fixed epoch budget are explicit.

Ignored `models/diagnostics/` contains per-context measured selected-gene truth, predictions, compound IDs and selected gene IDs in NPZ files, the fitted transform, Ridge model, GNN state and training history. Only development responses are stored there. Individual context receipts permit audit and recovery. These are single-seed exploratory diagnostics; they do not quantify neural seed uncertainty, independent biological replication or cross-study generalization. These outputs must not be used to retune the opened final-test benchmark.
