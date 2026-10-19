# DrugPerturb v3: published single-cell perturbation models as baselines

v3 asks whether two published single-cell perturbation models, chemCPA and biolord, change the v1/v2 conclusion that strong simple baselines are not reliably beaten. The machine-readable specification is [protocol_v3.yaml](../configs/protocol_v3.yaml); its amendments are listed at the end of this document. All data are public and no wet-laboratory experiment is involved.

## Questions

**Q1, information-matched source benchmark.** Both models see only the A549 24 h 1 µM wells of the 119 training compounds and the vehicle wells that serve that context, exactly the information available to the frozen v1 methods. They predict the 26 held-out test compounds, which are compared with the sealed v1 predictions (zero change, training mean, nearest neighbour, fingerprint Ridge, MLP and GNN).

**Q2, context-aware prediction.** Both models receive cell type and dose as inputs and train on training-compound cells from all twelve 24 h contexts (A549, K562, MCF7 × 10, 100, 1,000, 10,000 nM). Test-compound cells are absent from every context. Because the models now see target-context responses of training compounds, they are compared with simple baselines given the same information: per-context training mean (B1c), nearest training compound (B2c) and fingerprint Ridge (M1c), each fitted only on that context's eligible training compounds.

## Exposure of test outcomes and the confirmatory endpoint

Six contexts were scored before v3: A549 1 µM (v1) and the five v2 target contexts. Their test outcomes are reported for comparability only. The six contexts whose test rows are marked `metadata_only` in [context_roles.tsv](../manifests/v2/context_roles.tsv) — A549 and K562 at 10 nM, 100 nM and 10 µM — had never been scored. The Q2 confirmatory estimand is the mean, over those never-scored contexts that pass the count gate, of the context-level difference in mean compound RMSE between a model and a comparator (zero change, M1c). Its interval resamples scaffold groups once per bootstrap draw and applies the same draw to every context. Superiority requires the interval's upper bound to lie below zero against both comparators.

All six never-scored contexts passed the count gate (25–26 eligible test compounds in 23–24 scaffold groups each). MCF7 10 µM fails the gate (18 compounds, 17 groups), as in v2, and is descriptive only.

## Data and leakage boundaries

Sources are the three sci-Plex3 24 h RDS files of GSM4150378 ([source_files.tsv](../manifests/v3/source_files.tsv)), verified by SHA256. Extraction ran on Kaggle with conda-forge R 4.3.1, the v2 extraction runtime:

- `export_context_rds.R` (v2, unchanged) writes all-gene well counts; the new MCF7 and K562 outputs are **byte-identical** to the v2 extraction receipts, including the two gzip count files.
- [v3_export_cells.R](../scripts/v3_export_cells.R) writes per-cell counts for the 2,000 frozen genes plus each cell's all-gene library size. Its per-well sums equal the pseudobulk counts exactly in all three cell types; no cell has a zero library.

[v3_prepare.py](../scripts/v3_prepare.py) recomputes count QC for all twelve contexts with the v2 rule and checks it against the committed records: 1,692 v2 compound–context rows agree on every field, and A549 1 µM eligibility and replicate cell counts equal v1. It then writes the model-cell archive, which contains only training-compound, validation-compound and vehicle cells (454,164 nuclei). The 84,572 test-compound nuclei and 43,041 nuclei of excluded formulations are dropped before any model code runs, and GPU kernels mount only this archive. Validation and training truth are computed here; test truth is not.

Before any v3 prediction was scored, an anchors-only run recomputed test truth in the six already-opened contexts and reproduced the published per-compound RMSE of zero change and the frozen v1 Ridge and GNN to at most 2.2×10⁻¹⁶ ([truth_anchors.tsv](../reports/v3/truth_anchors.tsv)). Never-scored contexts were skipped by that run.

## Model inputs and the response space

Cells enter both models as `log1p(count / all-gene library × 10⁴)` for the frozen genes. A prediction for compound *c* in context *k* starts from the cells of the strict matched vehicle well of each test replicate, applies *c*, the context dose and the cell type, and converts per-cell values to a well-level `log1p(CPM)` as the library-weighted mean of `expm1(value) × 100` (negative values clipped to zero). With the measured cell values this conversion equals the v1/v2 well normalization exactly, so an identity model reproduces zero change exactly (a unit test). The primary prediction is the **counterfactual contrast**: the same vehicle cells passed through the compound condition minus the same cells passed through the vehicle condition, averaged over the two replicates. An anchored sensitivity subtracts the measured vehicle well instead.

Molecules are represented by RDKit2DNormalized descriptors (descriptastorus 2.8.0), z-scored with training-compound statistics; constant training columns are dropped. The DMSO vehicle uses its own descriptors, with dose 1.0 for chemCPA and 0 for biolord, following each model's published sci-Plex3 convention. Doses are scaled as nM / 10,000.

## Models

**chemCPA.** The upstream `ComPert` module is vendored unchanged ([third_party](../src/drugperturb/third_party/README.md), SHA256-checked). The training step re-implements the published Lightning module: Gaussian NLL reconstruction; drug and cell-type adversaries on the basal latent with a gradient penalty; autoencoder and doser updates that maximize adversary loss; gradient-norm clipping at 1; Adam with StepLR (γ 0.9) per epoch. The adversary replaces the autoencoder update on every `adversary_steps`-th batch. Two published configurations are candidates: the repository default and the configuration used by the biolord reproducibility study. Checkpoints are chosen by validation-compound RMSE in the response space above every 10 epochs (200 at most, patience 5 evaluations, 90 minutes per fit). LINCS pretraining is not performed.

**biolord.** biolord 0.0.3 on scvi-tools 1.1.6 with the published sci-Plex3 module and trainer settings (n_latent 256, decoder 4,096 × 4, batch 512, at most 200 epochs, early stopping as published, 150 minutes per fit). Validation-compound cells form biolord's internal validation split; counterfactuals start from vehicle cells, whose latent codes are trained.

Each candidate is fitted with seeds 42, 123 and 2026. The candidate with the lowest mean validation RMSE is selected, and its three seeds' predicted deltas are averaged. Predictions are sealed with SHA256 before the evaluation script reads test responses.

## Evaluation

Metrics are computed by the unchanged v1/v2 evaluation code: per-compound RMSE over the 2,000 frozen genes, averaged with equal weight per compound; paired scaffold-group bootstrap intervals (2,000 draws, seed 2026). Q1 compares each model with the frozen v1 Ridge, training mean and zero change. Q2 reports each context separately, the pooled never-scored-context estimand, and comparisons with the transported v1 Ridge for reference.

## Amendments

| ID | Change | Information used |
|---|---|---|
| A1 | M1c Ridge alpha grid widened from 0.1–100 to 0.1–10⁴, because all twelve contexts selected the upper boundary | validation-compound RMSE only |
| A2 | biolord fits capped at 150 minutes | wall-clock timing of engineering smoke fits |

After A1, eleven contexts select alpha 100 and K562 10 µM selects 1,000, both interior values.

## Compute

Kaggle notebooks, NVIDIA T4 ×2 for model fitting, CPU sessions for extraction, preparation and scoring. Environments are pinned in [cellmodels-cuda.txt](../locks/cellmodels-cuda.txt) (torch 2.5.1+cu121, scvi-tools 1.1.6, biolord 0.0.3) and [python-cpu.txt](../locks/python-cpu.txt). Every kernel records its GPU, driver, CUDA, package versions, wall time and input/output hashes. Seeds are fixed; GPU kernels are not forced to be deterministic, so retraining can differ in the last digits.
