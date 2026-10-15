# Reproducibility and workflow boundaries

This DSL2 workflow contains **one snapshot process**. The scientific scripts currently derive the project root from their own paths; the wrapper copies source, configuration and declared input bytes into an isolated task-local project before executing them. It never executes scientific scripts in the live checkout. Each snapshot runs the scientific chain sequentially; the workflow does not represent each scientific step as a separate Nextflow process.

## Runtime and evidence

Use Nextflow **25.10.2**, Java 21, Python **3.12.14**, and the exact CPU packages in `locks/python-cpu.txt`. Runtime package versions (including Matplotlib for rendering), raw-mode Rscript version/executable SHA256, and Python executable SHA256 enter the host cache key and final receipt. Container runs require an immutable image ID or digest. Used package versions are checked against the lock before computation. The lock pins versions, not wheel distribution hashes; record the resulting image ID and preserve the image for stronger distribution-byte reproducibility.

Use the default `--resource_metrics true` on compatible Linux for standard Nextflow resource reports. On a kernel without `/proc/PID/io`, select `--resource_metrics false`: standard resource observers are disabled and labeled operational trace/HTML/timeline/JSON provide status, duration and cache counts instead. CPU, memory and I/O are unavailable in that mode. The recorded Container host used this portable mode; its resource evidence must not be inferred from the Primary CPU host's separate measurements.

The runtime setup below is local to the checkout. Primary CPU host and Container host are descriptive execution labels. EC2 configuration is an existing-host execution template; AWS execution has not been performed.

The synthetic smoke test checks software execution using artificial controls, structures and counts. It does not validate drug efficacy, biological generalization or external reproducibility. The real core replay reuses the same public dataset and frozen scientific protocol; agreement with the primary run is computational reproducibility, not independent biological validation.

## Fresh-checkout setup and public acquisition

Start with an installed Python **3.12.14** executable (verify `python3.12 --version`) and Java 21. The Python setup is explicit:

```bash
cd /ABS/PROJECT
python3.12 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r locks/python-cpu.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
```

Install the pinned self-contained Nextflow launcher into the project, verifying official release bytes before execution. With Java 21 already selected in `JAVA_HOME`/`PATH`:

```bash
mkdir -p .runtime/workflow/bin
curl -fL https://github.com/nextflow-io/nextflow/releases/download/v25.10.2/nextflow-25.10.2-dist -o .runtime/workflow/bin/nextflow
printf '%s  %s\n' 59c6f48fce6139157b2e8a28fdca8166bc502a22d9ef1a0a70065ecc9c3ae4a3 .runtime/workflow/bin/nextflow | sha256sum -c -
chmod u+x .runtime/workflow/bin/nextflow
export PATH="$PWD/.runtime/workflow/bin:$PATH" NXF_VER=25.10.2
nextflow -version
```

Select and record a Java 21 runtime. The CI configuration provides checksum-verified Temurin and Nextflow downloads; the execution receipts preserve the versions and runtime hashes used in the recorded runs.

Fetch both public source files declared in the committed source manifest, then verify already present bytes without network access:

```bash
.venv/bin/python scripts/fetch_public.py
.venv/bin/python scripts/fetch_public.py --offline-verify
```

For core mode, either supply the externally verified pseudobulk/QC files with their complete upstream provenance, or explicitly derive them once using a separately installed R runtime and these commands from the project directory:

```bash
.venv/bin/python scripts/prepare_design.py
/ABS/VERIFIED/Rscript scripts/export_rds.R data/raw/GSM4150378_sciPlex3_A549_24hrs.RDS.gz manifests/selected_wells.tsv data/processed
```

`export_rds.R` reads the source count matrix and supplied selected wells; its helper uses base R and inspects serialized slots without requiring the original analysis package. The inspected extraction runtime was separate from this Python container. Raw workflow mode performs the same conversion inside its snapshot when that runtime is explicitly supplied. Source acquisition, conversion and scientific training are distinct steps; synthetic test mode requires none of these public files.

## Synthetic end-to-end run and cache replay

Run from a writable launch directory and preserve it: Nextflow stores its task cache there. Use distinct report directories for separate launches; keep the same work directory and launch directory for resume.

```bash
export PATH=/ABS/PROJECT/.runtime/workflow/bin:$PATH
# Keep the verified Java21 JAVA_HOME/PATH from setup above.
nextflow -C /ABS/PROJECT/nextflow.config run /ABS/PROJECT/main.nf \
  -profile test --project_root /ABS/PROJECT \
  --python /ABS/PROJECT/.venv/bin/python \
  --resource_metrics true --outdir /ABS/PROJECT/work/smoke-first \
  -work-dir /ABS/PROJECT/work/nf-smoke
nextflow -C /ABS/PROJECT/nextflow.config run /ABS/PROJECT/main.nf \
  -profile test --project_root /ABS/PROJECT \
  --python /ABS/PROJECT/.venv/bin/python \
  --resource_metrics true --outdir /ABS/PROJECT/work/smoke-resume \
  -work-dir /ABS/PROJECT/work/nf-smoke -resume
```

Test mode generates 24 synthetic alkane structures, two synthetic replicate plates with matched vehicle controls, and 12 artificial genes. It performs control matching, response construction, structure grouping and split, training-only gene selection/PCA, three baselines, ridge regression, a fingerprint MLP and a two-layer bond-aware GNN. Neural smoke fitting is limited to 12 epochs. Validation selects checkpoints; predictions are sealed before test metrics are formed. A paired scaffold bootstrap and prominently labeled synthetic HTML report complete the software path. No public-data files are consumed in test mode.

Sources are staged with `cache 'deep'`; source bytes, lock, input manifest, input bytes, settings and runtime identity bind the cache. A resumed task must show `CACHED`. The final receipt's output hashes are verified after both fresh and cached tasks, so changing a cached prediction/report fails the wrapper instead of silently accepting file existence. An incomplete snapshot fails and is preserved for inspection. Source/config changes during execution also fail. This is integrity checking against an accidental-change threat model, not a signed security attestation.

Each launch emits `trace.tsv`, `execution_report.html`, `timeline.html` and `dag.dot` in its output directory. Published `snapshot` is a symlink to retained Nextflow work: do not delete the work directory before archiving required outputs. `snapshot/receipts/workflow_run.json` records source/output hashes, runtime, settings and completed commands; per-step logs and failure receipts retain diagnostic evidence.

## Freeze inputs and run the real core chain

Public download and R extraction are explicit upstream operations. The existing project fetcher verifies public-source bytes; core mode starts from already verified pseudobulk matrices. It does not silently fetch or convert an RDS. Preserve the upstream source manifest, including the original RDS URL/hash/size, even though core execution does not need to stage the large RDS.

Keep the committed `manifests/workflow_core_inputs.json` as the original execution exemplar. For newly derived files, create a separate local manifest; the helper refuses to overwrite an existing freeze:

```bash
/ABS/PROJECT/.venv/bin/python /ABS/PROJECT/scripts/smoke_pipeline.py \
  --source-root /ABS/PROJECT --freeze-inputs core \
  --manifest-out /ABS/PROJECT/manifests/workflow_core_inputs.local.json
```

Manifest schema:

```json
{
  "schema_version": 1,
  "mode": "core",
  "artifact_hashes": {
    "data/raw/GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz": "64 lowercase SHA256 hex characters",
    "data/processed/pseudobulk_counts.tsv.gz": "64 lowercase SHA256 hex characters"
  },
  "upstream_sources": [{"filename":"original RDS filename","url":"official source URL","sha256":"original source SHA256","bytes":123}]
}
```

The displayed two artifact entries are illustrative, not a complete manifest. Core requires metadata, `pseudobulk_counts.tsv.gz`, `well_qc.tsv`, `gene_metadata.tsv`, and `extraction_summary.tsv`. The freeze helper also includes available `selected_cell_qc.tsv.gz`, `cell_state_well_summary.tsv`, and `manifests/source_manifest.tsv`. All keys are project-relative paths and must have unique basenames for staging. SHA256 is verified after copying into the snapshot. Prior models, fit/evaluation receipts and sealed response arrays are forbidden inputs. Full upstream provenance is preserved in the workflow receipt before `prepare_design.py` rebuilds its snapshot source manifest.

After committing all intended source changes, run:

```bash
nextflow -C /ABS/PROJECT/nextflow.config run /ABS/PROJECT/main.nf \
  -profile local --project_root /ABS/PROJECT --mode core \
  --input_manifest /ABS/PROJECT/manifests/workflow_core_inputs.local.json \
  --source_commit ACTUAL_COMMIT --allow_real_execution \
  --python /ABS/PROJECT/.venv/bin/python \
  --resource_metrics true --outdir /ABS/PROJECT/work/core-first \
  -work-dir /ABS/PROJECT/work/nf-core
```

The explicit real-execution gate prevents an accidental scientific rerun. Core executes `prepare_design.py → prepare_dataset.py → train.py → development_diagnostics.py → score.py → sensitivity.py → report.py`. Training and diagnostics receive the source-commit label; source hashes independently record exact copied bytes, including any uncommitted changes. The diagnostics and sensitivity steps are mandatory before rendering their claims. Development-only checkpoint selection and the prediction seal remain governed by the existing scientific scripts. The wrapper adds isolation and provenance rather than changing their protocol.

Raw mode uses a `mode: raw` manifest with metadata and the verified A549 RDS. Set `--mode raw --rscript /ABS/VERIFIED/Rscript`; `export_rds.R` then runs after design selection. The selected R runtime and its required packages must already be installed. Raw conversion is not included in the CPU Python container and is not claimed verified by the synthetic/core smoke checks.

Compare actual replay predictions to the primary predictions by aligned method, compound and gene identifiers, with maximum absolute difference at most `1e-6`. Byte equality of timestamped receipts, training timings, model pickle files or compressed archives is not the numerical reproducibility criterion. Do not select a new seed or model configuration after seeing test results.

## CPU container and existing EC2 host

Build from a **small temporary context** containing only `containers/Dockerfile` and `locks/python-cpu.txt`; do not send `.venv`, raw data or model artifacts as Docker build context. For example:

```bash
mkdir -p /TMP/CONTEXT/containers /TMP/CONTEXT/locks
cp /ABS/PROJECT/containers/Dockerfile /TMP/CONTEXT/containers/
cp /ABS/PROJECT/locks/python-cpu.txt /TMP/CONTEXT/locks/
docker build -f /TMP/CONTEXT/containers/Dockerfile -t drugperturb:cpu /TMP/CONTEXT
docker image inspect drugperturb:cpu --format '{{.Id}}'
```

The Dockerfile pins Python and package versions and uses the public CPU Torch index. A controlled build may override `PYTHON_BASE` with an immutable digest. Pass the actual built image ID (`sha256:...`) or repository digest (`name@sha256:...`) with `-profile test,docker --container_image ID`; Docker must access the same absolute task/source mount paths. Nextflow runs outside the image and the Python snapshot process runs inside it. No GPU or PyG binary extension is required.

`-profile ec2` selects local execution on an already provisioned EC2 machine. Configure its absolute project/Python/work paths and verify its runtime before use. On ordinary compatible Linux keep the default `--resource_metrics true` for standard Nextflow resource reports; combine `ec2,docker` only with an accessible matching Docker engine. This profile neither creates AWS infrastructure nor establishes that cloud execution has occurred. EC2, raw R extraction, and external biological validation require their own actual receipts.


## Verified runs and numerical scope

The recorded replay source label is `ccbb7594cf0b49f98036628d16036a15ed208cd9`. Historical source IDs refer to pre-publication snapshots rather than public Git commits; [publication provenance](PUBLICATION_PROVENANCE.md) maps original file hashes to this public copy. The Docker image is `sha256:1ea9b4f7f2880f8c42f35559d15f9d17b950438e09c97c51c1625a1f343716ea`. Both fresh core runs used `--memory '6 GB' --resource_metrics false` on the Container host. Select task resources according to measured host capacity; use standard resource metrics on compatible Linux.

Both clean runs completed all seven steps and verified 239 output hashes each. All 24 aligned validation/test method-and-seed arrays were numerically identical; selected configurations and all 28 fit records also matched apart from timing. This establishes repeatability in this fixed host/container environment.

The separate comparison against Primary CPU host training failed the predeclared 1e-6 criterion for eight GNN arrays; 16 other arrays passed. Every frozen dataset artifact and scientific model/training source was identical. The two CPU models differ, and small first-epoch numerical differences amplified during GNN optimization; the exact operator/backend cause was not established. The threshold was not relaxed and primary results were not replaced. [PyTorch 2.5.1's reproducibility documentation](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/randomness.rst) scopes reproducibility to the execution environment rather than guaranteeing equality across platforms.

Saved-model deployment is a separate test: the primary GNN weights, transform and graph features were loaded in the fixed Docker image without training or test-truth access. All eight validation/test seed-and-ensemble arrays passed 1e-6, with maximum difference 4.44e-16. Read the [workflow receipt](../receipts/workflow_reproduction.json), [same-environment comparison](../receipts/same_runtime_prediction_comparison.tsv), [cross-environment retraining comparison](../receipts/replay_prediction_comparison.tsv), and [fixed-weight inference receipt](../receipts/cross_runtime_inference.json) as distinct evidence.
