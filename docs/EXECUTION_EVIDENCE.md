# Execution evidence

## Actual public-data analysis

- Primary training source: `4847c26e05c0283d704a5fab9a85f272d0a5981a`.
- Hardware: Primary CPU host; Python 3.12.14, PyTorch 2.5.1+cpu. No GPU model training.
- Public-count extraction: 171.87 seconds, peak resident memory 4,629,040 KiB. Raw UMI conservation and closed-stream gzip integrity checks passed.
- Main training: 28 learned-model fits (4 Ridge settings, 12 MLP runs, 12 GNN runs); six evaluated methods including three references. Process elapsed time 226.39 seconds, peak resident memory 469,804 KiB.
- Development diagnostics: three compound folds and three nested training subsets, M1/M3 only, one exploratory seed; 28.33 seconds, peak resident memory 670,852 KiB.
- Single-cell marker summaries: 87.26 seconds, peak resident memory 4,609,644 KiB; all 36,848 cell denominators matched the raw-count QC output.
- Final evaluation: 26 compounds, 24 structure groups, 2,000 genes; selected predictions were sealed before scoring. See `final_evaluation.json`.
- Actual inference replay: S1090 maximum absolute difference 1.77e-8 from sealed M3 predictions; supported context and all 2,000 gene IDs matched.

Resource measurements are from `/usr/bin/time -v`; they describe these processes, not a general capacity guarantee or an independent validation dataset.

## Separate scopes

Primary analysis, synthetic workflow tests, clean workflow replay and container execution have separate receipts. A successful synthetic example is never substituted for the real experiment. Cloud instructions are not cloud execution.

AWS: **NOT EXECUTED**. External-study validation: **NOT EXECUTED**. GPU training: **NOT EXECUTED**. The local/server CPU workflow is the evidence-bearing implementation for this version.


## Workflow and deployment checks

| Check | Observed result |
|---|---|
| Meaningful local Python tests |108 passed; 97.48 seconds|
| Docker synthetic Nextflow run and resume |Complete; resume explicitly reused a cached process|
| Modified cached synthetic report |Correctly rejected by hash check; original bytes restored|
| Real core workflow, two fresh snapshots |All 7 steps succeeded twice; 245.09/245.98 seconds inside snapshot execution, 239 output hashes verified per run|
| Same host and image, newly trained predictions |24/24 aligned arrays exact; 28 fit records identical except timing|
| Primary CPU host versus Container host retraining |16/24 arrays meet 1e-6; 8 GNN arrays fail; test-ensemble maximum absolute difference 0.01175557|
| Fixed primary GNN weights, Docker inference |8/8 arrays pass 1e-6; maximum absolute difference 4.44e-16|
| Hosted GitHub Actions |108 tests passed; synthetic Nextflow completed with standard Linux resource trace and verified output hashes; [historical receipt](../receipts/hosted_ci.json)|
| AWS / GPU training |Not executed|

Core-replay source: `ccbb7594cf0b49f98036628d16036a15ed208cd9`. Image: `sha256:1ea9b4f7f2880f8c42f35559d15f9d17b950438e09c97c51c1625a1f343716ea`. Exact commands, source/runtime/input hashes, per-step logs and preserved failed attempts are indexed in [workflow_reproduction.json](../receipts/workflow_reproduction.json).

Each prediction array contains 26 compounds × 2,000 genes. Validation and test were aligned separately for each method and seed; they do not supply independent new biological validation. Primary GNN test RMSE remains 1.2144064385; container retraining gives 1.2144486204. Similar average errors do not override a failed per-value numerical criterion.

The Primary CPU host is Intel Core i9-7980XE; the Container host is AMD Ryzen 7 1700. Byte-identical frozen inputs and unchanged scientific source rule out data/code drift. Training trajectories diverge from tiny first-epoch differences; the exact CPU/backend operator responsible is unproven. Same-environment retraining and fixed-weight inference passed, while cross-environment GNN retraining remains explicitly failed.

The recorded Container host did not expose the `/proc/PID/io` counters required by standard Nextflow metrics. Those runs used portable operational status/duration/cache reports and the official DAG; CPU, peak-memory and I/O metrics are unavailable for these container runs. Primary CPU host process measurements above remain separate measurements. The verified container runs requested 6 GB task memory.

Source acquisition and base-R conversion were independently executed upstream. The core Python container starts from verified extracted counts and does not claim an R extraction inside Docker. The workflow is one isolated snapshot process, not a distributed or parallel scientific DAG.

Historical source IDs and source hashes identify original execution snapshots, not commits in the public repository. See [publication provenance](PUBLICATION_PROVENANCE.md) for the original-to-public file hash mapping.

The [historical hosted receipt](../receipts/hosted_ci.json) records a run before publication-history cleanup. Original JUnit outcomes, Nextflow trace, runtime versions and source/output hashes remain available; runner and former run/commit locators are omitted in the labeled public copies. It does not establish that a later source revision passed. The README badge links the current hosted run. These synthetic checks are separate from the real-data model evaluation.
