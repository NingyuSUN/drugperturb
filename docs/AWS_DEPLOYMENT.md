# AWS CPU deployment — NOT EXECUTED

AWS execution has not been performed. This is a deployment recipe for an existing cloud host, not an execution receipt. Select a budget, region and resource-retention policy before provisioning; no pricing or EC2 performance claim is made.

The Primary CPU host receipts document an on-premises server, **not AWS**. Its raw extraction peaked near 4.6 GiB RAM; primary training used about 470 MiB and took about 3m46s. These measurements exclude cloud provisioning, image/input downloads, and the complete diagnostics/replay chain. A general-purpose **x86-64 CPU host with 4 vCPU, 16 GiB RAM and 20–40 GiB available disk** is a capacity starting point, not an EC2-tested specification. Budget additional space for the OS, image layers, 573 MB RDS if retained, staged inputs, independent snapshots, logs, and archives; measure free space before starting. No GPU is needed.

## Execution model and prerequisites

`-profile ec2,docker` explicitly combines the existing-host configuration in `conf/ec2.config` with Docker in `conf/docker.config`; `-profile ec2` alone does not enable Docker. It runs one isolated snapshot task using Nextflow's **local executor on an already provisioned EC2 host**. It does not provision infrastructure or submit AWS Batch jobs. The [`local` executor runs on the launch machine](https://www.nextflow.io/docs/latest/executor.html).

Use the selected source commit, Nextflow **25.10.2** (as pinned in `nextflow.config`), a supported Java runtime, Docker, and AWS CLI on that host. Resolve the CPU container to an immutable image digest; record its base-image digest and build log. `containers/Dockerfile` installs `locks/python-cpu.txt`; a version tag alone is not an immutable image pin. [Nextflow requires the chosen container runtime and task tools to be available](https://www.nextflow.io/docs/latest/container.html).

The shipped CPU image has **no R**. This recipe uses `--mode core`: the same hash-verified well-count extraction and fixed cell-marker summaries already derived from the public RDS. It refits and evaluates in a new snapshot; it does not import previously fitted targets or predictions. `--mode raw` requires a separately verified R-capable image/runtime and is outside this CPU-image recipe.

Choose and record the region, instance type, AMI, runtime/time limit, disk capacity, budget and retention before provisioning. Attach an [EC2 instance profile](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-ec2_instance-profiles.html) rather than copying access keys into code, images, shell history, or configuration. Scope S3 permissions to:

- Input prefix: `s3:GetObject` for that prefix; `s3:ListBucket` constrained by `s3:prefix`.
- Output prefix: `s3:PutObject`, plus scoped read/list permissions for verification. Grant multipart/KMS permissions only if the selected storage configuration requires them.

Use distinct immutable input and per-run output prefixes. Do not grant whole-account or unrestricted bucket access for convenience. AWS provides [prefix-scoped S3 policy examples](https://docs.aws.amazon.com/AmazonS3/latest/userguide/example-policies-s3.html).

## Prepare the selected host

All placeholders below must be replaced with selected values. They are deliberately not usable account/resource identifiers. Run from a fresh checkout of the selected commit; keep primary frozen results in a separate read-only reference directory.

```bash
TASK_ROOT=/srv/drugperturb
TASK_RUN=selected-UTC-run-id
TASK_INPUT_URI=s3://SELECTED-BUCKET/immutable-input-prefix
TASK_OUTPUT_URI=s3://SELECTED-BUCKET/selected-output-prefix/$TASK_RUN
TASK_IMAGE='SELECTED-REGISTRY/drugperturb@sha256:REPLACE_WITH_64_HEX_DIGEST'
TASK_OUT=/srv/drugperturb-runs/$TASK_RUN
TASK_WORK=/srv/drugperturb-work/$TASK_RUN
TASK_REFERENCE=/srv/drugperturb-primary-reference
cd "$TASK_ROOT"
TASK_COMMIT=$(git rev-parse HEAD)
mkdir -p "$TASK_OUT" "$TASK_WORK"
date -u +%FT%TZ > "$TASK_OUT/started_utc.txt"
git status --porcelain > "$TASK_OUT/source_status.txt"
nextflow -version > "$TASK_OUT/nextflow_version.txt"
docker version > "$TASK_OUT/docker_version.txt"
uname -a > "$TASK_OUT/host.txt"
free -h > "$TASK_OUT/memory.txt"
df -h "$TASK_ROOT" "$TASK_WORK" > "$TASK_OUT/disk.txt"
```

Confirm the source status is clean. Record the selected region, instance ID/type, AMI ID, architecture, vCPU/RAM, EBS sizes, role name, source commit, image digest, and run purpose and budget in `$TASK_OUT/cloud_run_spec.json`; these are deployment evidence, not inferred from the Primary CPU host.

Copy the separately generated `manifests/workflow_core_inputs.local.json` from the input-preparation host; keep the committed `workflow_core_inputs.json` exemplar unchanged. Transfer only that local manifest's declared inputs into the fresh checkout. The input prefix should contain the repository-relative `data/...` tree and original source manifest. This command avoids downloading prior models, labels, selections, or executable source from S3:

```bash
python3 - "$TASK_INPUT_URI" <<'PYINPUT'
import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path.cwd()
manifest = json.loads((root/'manifests/workflow_core_inputs.local.json').read_text())
assert manifest['mode'] == 'core'
for relative, expected in manifest['artifact_hashes'].items():
    path = (root/relative).resolve()
    assert path.is_relative_to(root.resolve()), relative
    path.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['aws','s3','cp',sys.argv[1].rstrip('/')+'/'+relative,
                    str(path),'--no-progress'],check=True)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, relative
print('All declared core inputs downloaded and SHA256 values match')
PYINPUT
docker pull "$TASK_IMAGE"
docker image inspect "$TASK_IMAGE" > "$TASK_OUT/container_inspect.json"
```

`aws s3 cp` downloads only manifest-listed objects; it is not the integrity check. The explicit SHA256 check and workflow manifest bind the bytes. See the [official copy options](https://docs.aws.amazon.com/cli/latest/reference/s3/cp.html). Do not regenerate the frozen input manifest on EC2 to make a mismatch disappear.

## Synthetic check, then the core replay

The commands match the current `main.nf` parameters. Use fresh output/work directories; keep failed partial runs for diagnosis. The real-data gate is explicit. Do not use `-profile test,ec2`: `test` also selects the local configuration that disables Docker.

```bash
nextflow run main.nf -profile ec2,docker \
  --project_root "$TASK_ROOT" --mode test \
  --container_image "$TASK_IMAGE" --source_commit "$TASK_COMMIT" \
  --outdir "$TASK_OUT/synthetic" -work-dir "$TASK_WORK/synthetic"

nextflow run main.nf -profile ec2,docker \
  --project_root "$TASK_ROOT" --mode core --allow_real_execution true \
  --input_manifest "$TASK_ROOT/manifests/workflow_core_inputs.local.json" \
  --container_image "$TASK_IMAGE" --source_commit "$TASK_COMMIT" \
  --cpus 2 --memory '8 GB' --time_limit '1h' \
  --outdir "$TASK_OUT/core" -work-dir "$TASK_WORK/core"
```

Stop on a failed synthetic check, input mismatch, incomplete receipt, or failed process. The resource/time values are starting limits; they are not a cost guarantee or a substitute for monitoring the instance. The workflow writes trace, execution report, timeline, DAG, per-step logs, source hashes and `snapshot/receipts/workflow_run.json`. Cached `-resume` runs also recheck output hashes; preserve the same work directory and image/input identities if resuming a resumed run.

## Verify and archive before releasing compute

Use the immutable container to verify the completed snapshot's output hashes:

```bash
docker run --rm -v "$TASK_OUT/core/snapshot:/snapshot:ro" \
  --entrypoint python "$TASK_IMAGE" \
  /snapshot/scripts/smoke_pipeline.py --verify-output /snapshot
```

Compare the replay's compound IDs, selected gene IDs/order and primary predictions against the separately retained primary frozen run. The predeclared prediction tolerance is **absolute error ≤1e-6, relative tolerance 0**; do not loosen it after seeing a failure. Run this check in the pinned container, mounting the completed snapshot and original reference root read-only:

```bash
docker run --rm -i \
  -v "$TASK_OUT/core/snapshot:/replay:ro" \
  -v "$TASK_REFERENCE:/reference:ro" \
  --entrypoint python "$TASK_IMAGE" - <<'PYCOMPARE'
import pathlib, numpy as np, pandas as pd
replay, reference = pathlib.Path("/replay"), pathlib.Path("/reference")
np.testing.assert_array_equal(
    pd.read_csv(replay/'manifests/selected_genes.tsv',sep='\t').gene_id,
    pd.read_csv(reference/'manifests/selected_genes.tsv',sep='\t').gene_id)
with np.load(replay/'models/run_v1/test_predictions.npz',allow_pickle=False) as a, \
     np.load(reference/'models/run_v1/test_predictions.npz',allow_pickle=False) as b:
    np.testing.assert_array_equal(a['ids'],b['ids'])
    for method in ('B0','B1','B2','M1','M2','M3'):
        np.testing.assert_allclose(a[method],b[method],atol=1e-6,rtol=0)
PYCOMPARE
```

Cross-runtime/CPU numerical equivalence has **not** been established on AWS. A mismatch is a failed reproduction check to investigate, not evidence that the scientific result improved. Record UTC completion, exit status, actual peak resources, source/image/input hashes, maximum numerical differences, trace/report hashes and the upload inventory in the cloud-run receipt.

`publishDir` currently uses symlinks. Archive the complete snapshot, not dangling links or just a report. Preserve work until the upload and downloaded inventory/hash verification succeed:

```bash
aws s3 sync "$TASK_OUT/" "$TASK_OUTPUT_URI/" --follow-symlinks --no-progress
```

The [AWS CLI sync reference](https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html) documents symlink handling. After successful archival—or a preserved, diagnosed failed run—stop/terminate only the explicitly selected instance according to its retention policy, then verify its state. Stopping is not a complete cleanup: [EBS volumes may persist after termination depending on `DeleteOnTermination`](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-termination-works.html). Inspect retained EBS volumes/snapshots, S3 objects/versions and their retention/lifecycle, and any networking resources created for this run; record what remains chargeable. Verify the selected region and resource IDs rather than deleting resources by a broad name filter. No cloud resources were created or terminated for the reported benchmark.

A successful cloud replay would establish computational portability for these same public inputs. It would not provide an independent biological dataset, therapeutic efficacy evidence or clinical validation.
