# Publication provenance

This release presents the completed DrugPerturb benchmark. Original execution records and trained artifacts are preserved separately. Historical source identifiers label the recorded analysis snapshots; they are not commits in this public repository.

## Scientific records and execution labels

Historical source labels identify the actual runs: primary training `4847c26e05c0283d704a5fab9a85f272d0a5981a`, development diagnostics `a9d4bbe83da858b542682733ff851f3a40b65038`, and container replay `ccbb7594cf0b49f98036628d16036a15ed208cd9`. The completed source snapshot was `0821fd31affabb38b3a08e29d0f9781788518b9a`. These labels are retained as recorded facts, not rewritten to imply training occurred at the later publication commit.

Data manifests, model settings, numerical result tables and figures are preserved from the completed snapshot. Subsequent source changes strengthen checks before reusing saved training and evaluation results; they do not change model fitting, selection or scoring calculations. Documentation and execution labels have been updated for public use. No real-data training or model selection was repeated. The exact per-file comparison is recorded in [publication_manifest.json](../manifests/publication_manifest.json).

## Sanitized receipt copies

Execution receipts with an explicit `_publication` field are presentation copies. Machine identifiers, private paths and unrelated operational context have been removed or replaced with descriptive execution labels. Each records the SHA256 of its untouched original. All reported numerical results and original artifact-hash fields remain unchanged. Runtime files with generalized names are marked as renamed in the publication manifest; their original SHA256 identifies the preserved original without publishing its machine-specific filename.

An embedded hash of an original execution receipt refers to that original capture, not necessarily to its sanitized public copy. Use the original-to-public SHA256 mapping in the publication manifest to distinguish the two. Do not interpret a presentation edit as a new execution, or an intentionally changed public receipt hash as scientific-data corruption. The final [publication integrity receipt](../receipts/publication_integrity.json) binds this public release's files.

The unmodified `report_artifacts.json` describes the original generated scientific report. Its code/data hash references remain historical. The public HTML differs only in its execution-host label; scientific figures, result tables and conclusions remain unchanged. The protocol's `compute` section describes execution using generic terms; all scientific settings are unchanged. Public documentation, provenance-checking code and sanitized metadata are identified separately in the publication manifest.

## Reproduction and availability

A new run creates fresh receipts from its actual source and inputs. Follow [reproducibility instructions](REPRODUCIBILITY.md); raw public inputs and binary model artifacts are obtained or generated outside Git. This release contains no private study data, credentials, raw expression matrices or trained binary checkpoints. Third-party data and annotations retain their source terms.

Original prediction seals also bind the training source bytes. Combining historical fitted artifacts with changed source can therefore correctly fail verification. Historical hashes are retained as recorded; they are not rewritten to make an old run appear to have used the updated code. The published result tables remain the record of that original analysis.

Same-environment training repeatability, saved-model inference portability and cross-environment retraining remain separately reported. The GNN cross-environment retraining comparison failed the original numeric tolerance and is not changed by publication.

## Hosted software validation

The [historical hosted CI receipt](../receipts/hosted_ci.json) and workflow receipt under `receipts/ci/` are labeled presentation copies. Former run, commit and archive locators and the JUnit runner hostname are omitted. Original test outcomes, timings, dependency versions, source/output hashes and archive digest are preserved; untouched originals are archived separately. Original evidence hashes and public-copy hashes are listed separately. These historical checks are not attributed to a rewritten or later source revision; the README badge links current GitHub Actions results. The original dated `delivery_verification.json` retains its then-current `NOT_EXECUTED` hosted-CI field. Synthetic verification does not add a new real-data model result.
