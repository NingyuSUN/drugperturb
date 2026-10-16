# Reproducing DrugPerturb v2

The v2 extension separates **scoring published predictions**, **developing a new biological head**, and **retraining the original models**. They establish different kinds of reproducibility. The original [Nextflow/container instructions](REPRODUCIBILITY.md) and their recorded execution evidence cover v1, not this R/Python extension.

## What is supplied

The repository contains the frozen protocol, compound split, gene identities, count-qualified cohorts, gene-set members, source hashes, all result tables and figures. The two small [prediction archives](../reports/v2/predictions/manifest.json) are byte-identical copies of the original sealed test predictions, including the three individual neural seeds. They contain public compound IDs and numerical predictions, not fitted models or raw counts.

The original scoring command, `scripts/evaluate_transfer_v2.py`, verifies the full retained v1 training archive before evaluating new conditions. Its [isolated replay](../receipts/v2_scoring_replay.json) reproduced all six scientific output files exactly, with no new fits. That command requires the complete original archive; a fresh clone alone does not contain it. The separate public replay below evaluates the published predictions without requiring that archive. It does not verify a new training run or enable inference for new compounds.

## Environment and source extraction

Use Python **3.12.14** with [the pinned CPU requirements](../locks/python-cpu.txt), following the v1 environment setup. Use `PYTHONPATH=src` from the repository root, or install the package with `pip install --no-deps -e .`. The count extractor was executed with **R 4.3.1** and uses base R; the Python container does not include this R runtime.

The two compressed sources total approximately **2.28 GB**. Extraction is sequential: the observed MCF7 process used about **14.6 GiB peak resident memory**, and K562 about **3.8 GiB**. These are observed process peaks, not guaranteed capacity bounds; allow additional memory for the operating system and at least 20 GiB of free working storage. Recorded source and output hashes are in the [acquisition](../receipts/v2_source_acquisition.json) and [extraction](../receipts/v2_context_extraction.json) receipts. No GPU or paid cloud service is required for these steps.

From the repository root, after installing the Python environment and selecting the verified R runtime:

```bash
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python scripts/fetch_v2_sources.py --group context \
  --output-dir data/raw/v2 --receipt logs/v2/source_verification.json

Rscript scripts/export_context_rds.R \
  data/raw/v2/GSM4150378_sciPlex3_MCF7_24hrs.RDS.gz \
  manifests/v2/well_mapping.tsv data/processed/v2/MCF7_24h MCF7 24
Rscript scripts/export_context_rds.R \
  data/raw/v2/GSM4150378_sciPlex3_K562_24hrs.RDS.gz \
  manifests/v2/well_mapping.tsv data/processed/v2/K562_24h K562 24

.venv/bin/python scripts/prepare_transfer_inputs_v2.py \
  --context-root data/processed/v2 \
  --manifest-out logs/v2/target_manifest.json
```

Existing matching downloads are reused. Mismatches, partial downloads and existing extraction outputs are preserved and rejected, rather than silently overwritten. Do not rerun extraction into a completed output directory. For an existing download, add `--offline-verify` to the fetch command instead.

The final command verifies all five derived files per cell line against the recorded extraction hashes, checks code/input identities and the small extraction summaries, and creates a local path manifest. It performs no extraction, model fitting or expression-effect calculation. The manifest contains local absolute paths and belongs under ignored `logs/`. Its hash changes when local paths change. If extraction bytes differ, preserve both outputs and investigate the mismatch; do not edit reference hashes to obtain a passing replay. Fresh extraction across different R/compression environments has not been shown to be byte-identical.

## Replay the published predictions

With verified extraction outputs and the private manifest from the preceding section:

```bash
prediction_hash=$(sha256sum reports/v2/predictions/manifest.json | cut -d' ' -f1)
target_hash=$(sha256sum logs/v2/target_manifest.json | cut -d' ' -f1)
contract_hash=$(sha256sum configs/v2_metadata_inputs.json | cut -d' ' -f1)
qc_hash=$(sha256sum receipts/v2_context_qc.json | cut -d' ' -f1)

.venv/bin/python scripts/replay_public_transfer_v2.py \
  --prediction-manifest reports/v2/predictions/manifest.json \
  --prediction-manifest-sha256 "$prediction_hash" \
  --target-manifest logs/v2/target_manifest.json \
  --target-manifest-sha256 "$target_hash" \
  --input-contract-sha256 "$contract_hash" \
  --qc-receipt-sha256 "$qc_hash" \
  --output-root work/v2-public-replay
```

The [recorded public replay](../receipts/v2_public_replay.json) ran in a source-only snapshot of `c8c3b1b977c3532e35cd7bad01318ff66c5539e4`, with relocated verified MCF7/K562 extraction outputs and no A549 data or fitted model directory. All six scientific files matched. This was a new scoring execution using existing extraction outputs, not a fresh R extraction run.

Use a new output directory. The replay checks the source seals, prediction bytes and identifiers, gene/set order, original split, protocol, count-QC chain and target extraction provenance. It scores the fixed five contexts with the original two-control definitions and bootstrap settings, then requires all six scientific output files to match the published evaluation. A failed comparison is a failure, not permission to retune or replace the reference. Its scope is fixed-prediction scoring from public inputs; the internal full-training-archive checks remain a separate operation.

The SHA256 arguments pin the files selected for the invocation. They detect accidental drift after that selection; they are not a signed security attestation. To compare with the published reference, start from a verified repository revision and preserve its committed files.

To rerender the published figures independently:

```bash
.venv/bin/python scripts/plot_transfer_v2.py \
  --result-root . --output-root work/v2-render
```

Rendering checks the evaluation receipt and all six tables before producing images. It does not refit models or select endpoints. Figure bytes can depend on the plotting runtime even if the source numbers are unchanged.

## Reconstruct count qualification or develop the biological head

These are optional development/reconstruction operations, not prerequisites for replaying the supplied predictions. Use an independent writable checkout/output directory and preserve the published receipts.

Count qualification consumes only cell-count QC tables. Include the original A549 QC to reconstruct the complete recorded inventory; without it, A549 rows are correctly marked unavailable:

```bash
.venv/bin/python scripts/qualify_contexts_v2.py \
  --well-qc /ABS/V1/data/processed/well_qc.tsv \
  --well-qc data/processed/v2/MCF7_24h/well_qc.tsv \
  --well-qc data/processed/v2/K562_24h/well_qc.tsv \
  --output-root work/v2-count-qc
```

The new five-set Ridge head uses A549 counts and source provenance produced by the existing [v1 acquisition and core workflow](REPRODUCIBILITY.md). It takes only source training/validation responses for development, retains the original split, and fits four candidates if all coverage gates pass. Set `v1_root` to a coherent completed v1 snapshot; it is a placeholder, not a supplied model archive:

```bash
v1_root=/ABS/V1
contract_hash=$(sha256sum configs/v2_metadata_inputs.json | cut -d' ' -f1)
source_hash=$(sha256sum "$v1_root/receipts/report_artifacts.json" | cut -d' ' -f1)
.venv/bin/python scripts/train_biology_v2.py \
  --counts "$v1_root/data/processed/pseudobulk_counts.tsv.gz" \
  --mapping "$v1_root/manifests/qualified_well_mapping.tsv" \
  --gene-metadata "$v1_root/data/processed/gene_metadata.tsv" \
  --split manifests/split_manifest.tsv --biology configs/biology_v1.json \
  --protocol configs/protocol_v2.yaml --input-contract configs/v2_metadata_inputs.json \
  --input-contract-sha256 "$contract_hash" \
  --source-receipt "$v1_root/receipts/report_artifacts.json" \
  --source-receipt-sha256 "$source_hash" \
  --model-dir work/v2-new-head/models --output-root work/v2-new-head
```

This creates its own member/coverage/selection records and prediction seal. An existing intact seal is verified with zero new fits. Do not substitute new model outputs into the published-prediction replay or mix old seals with newly fitted weights. The full archive evaluator additionally compares original v1 prediction hashes with the public reference; a newly trained v1 snapshot is not a drop-in substitute. There is no claimed one-command fresh retraining path that guarantees the published v2 predictions. V1 cross-runtime GNN retraining already failed its strict comparison, as retained in the original execution evidence.

## Reproduce the external-data feasibility gate

Only four small LINCS metadata files are needed. The exact-condition overlap is zero, so the expression matrix is unnecessary:

```bash
.venv/bin/python scripts/fetch_v2_sources.py --group lincs \
  --output-dir data/raw/lincs_phase1 --receipt logs/v2/lincs_verification.json
.venv/bin/python scripts/audit_lincs_v2.py \
  --metadata-dir data/raw/lincs_phase1 --output-root work/v2-lincs
```

This checks exact numeric dose, time, cell identity and standardized parent structure. It does not relax conditions to increase overlap. Compare the scientific counts and source hashes with the [reference receipt](../receipts/v2_external_feasibility.json); source-revision labels may differ in a later checkout.

## Validation boundaries

The [local validation record](../receipts/v2_validation.json) reports 330 passing Python tests. Python tests use synthetic fixtures for leakage, identity alignment, missing values, controls, provenance and overwrite protection. R tests cover sparse count aggregation, count conservation, gene IDs and compressed-output integrity. The actual scoring replay addresses computational repeatability. None establishes independent-study transport, mechanism, drug efficacy, clinical performance, AWS execution or GPU training. The scientific conclusions are in the [English](../reports/v2/report.md) and [Chinese](../reports/v2/report.zh-CN.md) reports.
