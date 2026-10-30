# DrugPerturb

[![CI](https://github.com/NingyuSUN/drugperturb/actions/workflows/ci.yml/badge.svg)](https://github.com/NingyuSUN/drugperturb/actions/workflows/ci.yml)

**Prespecified benchmarks for predicting transcriptional responses to held-out compounds, from structure-based models to chemCPA and biolord.**

sci-Plex3 · OP3 (human PBMCs) · Tahoe-100M · compound-, scaffold- and cell-line-level holdouts · split-measurement endpoints · predictions sealed before testing

Reports: [v1-v6 synthesis](reports/synthesis.md) ([中文](reports/synthesis.zh-CN.md)) · [v6, held-out cell lines](reports/v6/report.md) · [v5, Tahoe-100M](reports/v5/report.md) · [v4, OP3](reports/v4/report.md) · [v4 corrected, post hoc](reports/v4_corrected/report.md) · [v3, sci-Plex3](reports/v3/report.md) · [v2, cross-condition](reports/v2/report.md) ([中文](reports/v2/report.zh-CN.md)) · [v1, A549](reports/v1/report.md)

Each version answers one question under a protocol committed before any fit:

| Version | Question | Data | Answer |
|---|---|---|---|
| v6 | On an endpoint where predicting nothing scores chance by construction, does chemCPA beat a plate-anchored mean? | Tahoe-100M, ten new cell lines | No. It matches that mean (+0.0010 [−0.0007, +0.0039]) while beating chance by +0.31, and chemical structure adds nothing |
| v5 | Does an improved chemCPA beat vehicle-anchored baselines on data this project had never scored? | Tahoe-100M | No. It beats zero change but not the anchored baselines, and a plate-anchored mean beats every method |
| v4 | Can chemCPA's collapse to zero change be fixed? | sci-Plex3, OP3 | Partly. The drug term comes alive, but no variant beats a per-context training mean on OP3 |
| v3 | Do chemCPA and biolord beat information-matched simple baselines? | sci-Plex3 | No. Both are worse than a per-context Ridge, and chemCPA collapses to zero change |
| v2 | Do A549-trained structure models transfer to other cell lines and doses? | sci-Plex3 | No. Zero change has the lowest error in all five conditions |
| v1 | Do fingerprint and graph models beat simple baselines on held-out compounds? | sci-Plex3 A549 | No reliable advantage over the training mean or Ridge |

The [cross-version synthesis](reports/synthesis.md) strings these six answers into one argument:
which explanation each version removed, what the series established besides six nulls, and what it
never tested.

## v6: an endpoint that zero change cannot win

**The prespecified primary endpoint was not met**, on an endpoint built so that the v5 criticism cannot apply.

v5's result was weak for a structural reason: on an RMSE endpoint, predicting no change wins whenever the
measurement is noisy. v6 replaces the endpoint with **direction accuracy on each compound's own top-50
differentially expressed genes**, where a predicted zero counts as a tie and therefore scores exactly 0.5000 —
chance, by construction. The genes are chosen from one half of a compound's cells and scored against the
other, disjoint half, so no measurement both picks a compound's genes and grades it. The test set is ten
Tahoe-100M cell lines v5 never used.

| Pooled over 27 contexts, responder keys | Difference [95% interval] |
|---|---|
| **P1:** residual arm (M2) − plate-anchored training mean (B1p) | **+0.0010 [−0.0007, +0.0039]** |
| **P1:** M2 − anchored Ridge (M1a) | +0.0280 [+0.0214, +0.0361] |
| M2 − zero change (chance) | +0.3135 [+0.2999, +0.3265] |
| **S4:** M2 − the same model trained on the raw delta (M1) | +0.0612 [+0.0490, +0.0750] |
| **S3:** does structure help? M1c − B1c / M1a − B1a / M1r − B1p | +0.0013 / +0.0000 / −0.0004, all straddling zero |

**What the secondary analyses show:**

- **Predicting nothing no longer wins, and the answer is still no.** Zero change sits exactly on chance and
  every method beats it by a wide margin, yet chemCPA still cannot separate itself from a mean that uses no
  chemistry.
- **The training target matters far more than the model.** Training chemCPA on the residual over the plate
  mean rather than on the raw response is worth +0.0612 — the largest effect in the study. What it buys is
  the plate mean back: the residual the model learns has about zero correlation with the real one.
- **There is almost nothing left to win.** The measurement's accuracy against itself is 0.8232 and the plate
  mean already reaches 0.8217, so a perfect model has roughly 0.0015 of room.
- **No single plate or context carries the null.** Dropping each of the 14 plates in turn leaves the interval
  straddling zero every time, and per context M2 is above the plate mean in 15 of 27 and below in 10.
- **No method finds the right genes.** Overlap between a method's own top-50 predicted genes and the
  compound's real ones is at most 0.13 for everything, the plate mean included. Direction is largely solved
  on a given gene set; gene identity is not.

See the [v6 report](reports/v6/report.md), the [pre-test diagnostics](reports/v6/diagnostics.md), the
[CPU-only report written before any fit](reports/v6/cheap_report.md) and the
[protocol](configs/protocol_v6.yaml).

## v5: a confirmatory test on Tahoe-100M

**The prespecified primary endpoint was not met.** The chemCPA arm selected on validation compounds was the counterfactual pseudobulk arm. It did not beat the vehicle-anchored training mean or anchored Ridge on held-out compounds.

The dataset was [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M), which this project had never scored: 84 test compounds in 77 scaffold groups, 10 cancer cell lines × 3 doses, and 2,464 test pairs. The protocol was committed before any fit. The seals, validation diagnostics and arm selection were committed and pushed before test truth was first computed.

| Pooled over 30 contexts | ΔRMSE [95% interval] |
|---|---|
| **P1:** pseudobulk arm (I2) − anchored training mean (B1a) | −0.0047 [−0.0104, +0.0011] |
| **P1:** I2 − anchored Ridge (M1a) | −0.0046 [−0.0104, +0.0011] |
| I2 − zero change | −0.0359 [−0.0399, −0.0319] |
| I2 − training mean (B1c) | −0.0067 [−0.0098, −0.0036] |
| Plate-anchored training mean (B1p) − B1a | −0.0428 [−0.0472, −0.0386] |

**What the secondary analyses show:**

- **First chemCPA win over zero change.** This is the first chemCPA arm in this project to reliably beat zero change on untouched data.
- **Its gain over the delta baselines comes from one line.** It is in SW 1088, whose vehicle samples are small, so the gain reflects vehicle-noise cancellation (v4's lesson). In the other nine lines, the arm matches the training mean.
- **Structure adds nothing.** The anchored Ridge equals the anchored mean.
- **Plate effects dominate.** A plate-anchored training mean, which uses no chemistry, has the lowest error in every one of the 30 contexts. Plate-level effects therefore dominate Tahoe-100M responses.

See the [v5 report](reports/v5/report.md), the [pre-test diagnostics](reports/v5/diagnostics.md), the [protocol](configs/protocol_v5.yaml) and the [design note](docs/V5_DESIGN_NOTE.md).

## v4: diagnose, intervene, improve? Fixing chemCPA's collapse

**Two principled changes made chemCPA use the drug, but neither beat a per-context training mean or Ridge on held-out compounds in a dataset this project had never scored.**

- **Data.** The confirmatory test set is OP3 (GEO GSE279945): peripheral blood mononuclear cells from three donors, with 35 test compounds in T, B and myeloid cells. The protocol was committed before any fit, and predictions were sealed before testing.
- **Interventions.** v3 diagnosed chemCPA's collapse; v4 tested two fixes:
  - a negative binomial count likelihood, which enlarges the drug term;
  - a counterfactual pseudobulk loss, which trains the prediction path directly and makes the drug term large and correlated with the truth.
- **The primary arm failed.** The NB arm, selected on validation compounds, was worse than zero change on OP3: +0.171 [−0.004, +0.532]. One compound drove that result; its descriptors hit a scaling bug inherited from v3. Excluding it post hoc, the arm matches zero change (+0.001 [−0.005, +0.008]) and is still worse than the training mean (+0.015 [+0.007, +0.023]).
- **Methodological finding.** On sci-Plex3, the pseudobulk arms appeared to beat every v1 method by up to 30%. A vehicle-anchored training mean, which uses only the matched vehicle well's measurement, does better. Models conditioned on vehicle cells need that baseline (amendment A1, made before any OP3 result was read).

| OP3, pooled over T, B and myeloid cells | ΔRMSE [95% interval] |
|---|---|
| NB arm (primary) − zero change | +0.171 [−0.004, +0.532] |
| NB arm − per-context training mean | +0.183 [+0.009, +0.544] |
| Pseudobulk arm − zero change | +0.001 [−0.007, +0.010] |
| Pseudobulk arm − per-context training mean | +0.014 [+0.006, +0.023] |
| v3 chemCPA, refitted − zero change | −0.001 [−0.002, +0.001] |

See the [v4 report](reports/v4/report.md), the [pre-test diagnostics](reports/v4/diagnostics.md) and the [v4 methods](docs/V4_METHODS.md).

**Post hoc corrected analysis.** The scaling bug was fixed, and all 12 OP3 fits were rerun with the training-constant descriptor columns removed. The validation rule then selects the pseudobulk arm. On test compounds it matches zero change (+0.004 [−0.003, +0.012]) and is still worse than the training mean (+0.016 [+0.009, +0.025]); the conclusion does not change. This analysis was prompted by the open OP3 test outcomes, so it is reported next to the prespecified result, not in its place. See the [corrected report](reports/v4_corrected/report.md).

## v3: published single-cell perturbation models as baselines

**chemCPA and biolord did not beat information-matched simple baselines.** Both models were trained on sci-Plex3 cells from 3 cell lines × 4 doses, ran on Kaggle T4 GPUs, and had their predictions sealed before test scoring. The confirmatory endpoint pooled six contexts never scored before. There, both were reliably worse than a per-context fingerprint Ridge, and neither was distinguishable from zero change:

| Pooled over 6 never-scored contexts | ΔRMSE [95% interval] |
|---|---|
| chemCPA − per-context Ridge | +0.057 [+0.038, +0.074] |
| biolord − per-context Ridge | +0.052 [+0.031, +0.072] |
| chemCPA − zero change | −0.0000 [−0.0001, +0.0000] |
| biolord − zero change | −0.005 [−0.016, +0.001] |

Given target-context training data, even the per-context training mean beats zero change in all 12 contexts, which reverses the v2 result below. A training/validation-only diagnostic shows chemCPA collapsing to zero change: its drug term is about 0.6% of the latent norm, even for training compounds and after 200 epochs. The fresh extraction reproduced v2's MCF7/K562 outputs byte-for-byte. Recomputed test truth matches published per-compound RMSE to 2×10⁻¹⁶. See the [v3 report](reports/v3/report.md) and [v3 methods](docs/V3_METHODS.md).

## v2: cross-condition results

**Models trained on A549 did not outperform zero-change predictions in any of the five evaluated target conditions.** The primary test is MCF7 at 24 hours and 1 µM, with **26 held-out compounds in 24 structure groups**. Original splits, gene outputs and predictions were frozen before scoring the new conditions.

| Primary MCF7 endpoint ↓ | Zero change | Source mean | Fingerprint Ridge | Molecular GNN |
|---|---:|---:|---:|---:|
| 2,000-gene mean compound RMSE | **0.53251** | 0.65132 | 0.67322 | 0.71975 |
| Five-set mean compound MAE | **0.08467** | 0.15744 | 0.14081 | Not fitted for this endpoint |

The five-set row uses a **separately trained Ridge head**, selected from four fixed candidates using only the source validation set. All five predefined sets passed training-only coverage checks. Its primary error difference from the source mean was −0.01664, with a 95% paired structure-group interval of [−0.03569, +0.00345]; a reliable gain was not established. These two rows have different output spaces and error definitions and should not be compared numerically.

For gene-level transport, GNN error exceeded Ridge by +0.04654 [95% interval +0.00235, +0.10479]. These are pointwise intervals conditional on the frozen models. Zero-change had the lowest mean error for both endpoints across all five conditions; this does **not** imply that the drugs have no biological effects.

![Cross-condition prediction contrasts](reports/v2/figures/contrast_intervals.png)

The evaluation adds K562 at 1 µM and MCF7 at 10 nM, 100 nM and 10 µM. Eligible cohorts contain 18–26 compounds; the 10 µM cohort fails the prespecified sample-count gate and is reported descriptively without an interval. Matched-control sensitivity uses identical compound subsets. LINCS Phase I had **0/26 test compounds** under the exact independent-study condition, so no external expression-performance claim is made.

Read the [English](reports/v2/report.md) or [Chinese](reports/v2/report.zh-CN.md) report for every baseline, biological readout, denominator and limitation. Both the [original-archive scoring replay](receipts/v2_scoring_replay.json) and a [public-files-only replay](receipts/v2_public_replay.json) reproduced all six scientific result files exactly. The latter uses the published prediction archives and verified extracted counts, with no private fitted models. **330 Python tests passed** in the [local validation](receipts/v2_validation.json). These checks do not establish fresh extraction or retraining equivalence.

## Original A549 results (v1)

The molecular graph model achieved the lowest primary point estimate, but **a reliable advantage over strong simple baselines was not established**.

| Method | Held-out mean compound RMSE ↓ |
|---|---:|
| Zero change |1.2694|
| Training mean |1.2170|
| Chemical nearest neighbor |1.7149|
| Fingerprint Ridge |1.2430|
| Fingerprint MLP |1.2532|
| Molecular GNN, three-seed mean |**1.2144**|

All methods predict the same 2,000 genes for the same **26 held-out compounds**. The GNN–Ridge error difference is −0.0286, with a paired 95% structure-group interval of [−0.0680, +0.0087]. Its difference from the training mean is −0.0026 [−0.0475, +0.0419]. Both intervals include zero.

The control definition matters: using the average of two separately normalized same-plate vehicle wells changes the point-estimate ranking in favor of zero-change and training-mean references. GNN also has higher error than Ridge in all three exploratory development folds. These findings are retained in the [sensitivity results](reports/sensitivity/metrics_summary.tsv) and [diagnostics](receipts/development_diagnostics.tsv).

![Held-out model comparison](reports/figures/03_model_comparison.png)

The [v1 report](reports/v1/report.md) covers these results in markdown, with every paired interval, the control-definition sensitivity and the coverage gates. The original HTML reports carry the figures: to view them, clone this repository or choose **Code → Download ZIP** on GitHub and extract the complete archive. Keep the repository directory structure, then open `reports/report.html` or `reports/biologist_summary.html` in a browser. The full report loads figures and linked evidence from adjacent repository directories; downloading an HTML file alone will leave those resources unavailable. Viewing the reports does not require installing the analysis environment.

## What is implemented

- **Traceable public data:** hash-verified source files, raw-count conservation, explicit cell/well mapping, two-replicate QC and matched vehicles.
- **Chemical machine learning:** RDKit parent standardization, Morgan fingerprints, a small MLP and a two-layer bond-aware molecular GNN implemented in PyTorch.
- **Controlled evaluation:** whole chemical structure groups held out; training-only gene selection and target transforms; four fixed candidates, three neural seeds, sealed predictions and compound-level uncertainty.
- **Cross-condition evaluation:** fixed cell/dose tests, a direct five-set Ridge predictor, paired control sensitivity, count-based inference gates and an exact-condition external-data feasibility audit.
- **Split-measurement endpoints:** each unit's cells partitioned into disjoint selection and evaluation halves, so the half that picks a key's differentially expressed genes and decides whether it responded never grades the prediction; direction accuracy on those genes, where a predicted zero scores exactly chance.
- **Scientific diagnostics:** replicate consistency, chemical support, fixed-rule error cases, development learning curves, control sensitivity and prespecified biological coverage checks.
- **Auditable delivery:** model weights and trajectories generated by the run, version-locked environments, Nextflow entrypoints, an inference CLI and input/output hash receipts.

## Original data and evaluation

The workflow starts with published sci-Plex3 nuclear-expression measurements. Of 188 source treatment labels, 171 distinct standardized parent compounds pass both chemical and paired-cell criteria. The primary response uses 33,055 treatment nuclei and 886 nuclei in eight unique matched vehicle wells. Cells estimate responses; they do not multiply the number of drug prediction units.

| Partition | Compounds | Structure groups |
|---|---:|---:|
| Training |119|106|
| Validation |26|24|
| Final test |26|24|

The test groups comprise 22 ring scaffolds and two acyclic parent groups. Canonical parent/group identities do not cross partitions; residual chemical similarity and shared-control dependence remain documented limitations.

Raw counts are summed per well, converted to CPM and log1p transformed, then contrasted against the matched vehicle. Two replicate contrasts receive equal weight. Only training compounds select the 2,000 output genes and fit standardization/PCA. **Every model is scored against measured changes, not PCA-reconstructed truth.**

In v1, all five biological gene sets failed the frozen coverage threshold within the selected 2,000-gene output space; no pathway-recovery claim is made. The v2 direct head separately filters all measured genes using source training wells, yielding five eligible transcriptional readouts without changing the v1 gene space or its failed gates. Full 43-gene S and 54-gene G2/M marker summaries remain available as separate single-cell descriptions.

## Run and inspect

The v2 extension has a [separate CLI recipe and execution scope](docs/V2_REPRODUCIBILITY.md). Existing Nextflow/container receipts below cover v1; they do not certify v2 extraction or fitting.

See [reproducibility instructions](docs/REPRODUCIBILITY.md) for environment setup, the synthetic smoke workflow, a fresh public-data snapshot and execution receipts. The synthetic example is an engineering check; the results above come from the real public experiment.

```bash
# Once the pinned Python environment is installed
PYTHONPATH=src .venv/bin/python -m pytest -q

# Verify existing source downloads without downloading again
.venv/bin/python scripts/fetch_public.py --offline-verify
```

The prediction CLI accepts a valid SMILES string and the supported A549/24h/1000nM context; it returns gene-level deltas plus nearest-training chemical similarity. [S1090 inference replay](reports/inference_example/S1090.json) matches its sealed prediction within 1.8×10⁻⁸. The CLI rejects unsupported biological contexts.

## Verified v1 execution

The real seven-step core workflow ran twice from fresh snapshots in the fixed Docker environment. All **24 validation/test prediction arrays matched exactly**, and all 28 fit records matched apart from timing. Synthetic execution, cache recovery and rejection of modified cached output also passed.

Reproducibility has an explicit boundary: retraining from the same inputs on the Primary CPU host versus the Container host passed the strict `1e-6` comparison for 16/24 arrays; the eight GNN arrays did not. Deploying the **saved primary GNN weights** in Docker passed all eight inference comparisons (maximum difference 4.44×10⁻¹⁶). The original scientific results above remain unchanged. See [execution evidence](docs/EXECUTION_EVIDENCE.md) and the [machine-readable replay receipt](receipts/workflow_reproduction.json).

The [historical hosted CI evidence](receipts/hosted_ci.json) records **108 Python tests** and a successful synthetic Nextflow workflow on GitHub-hosted Ubuntu. Original output hashes were independently verified. Public receipt copies omit former run locators and are explicitly labeled; they do not certify a later source revision. The badge links current hosted tests and workflow results. AWS execution has not been performed.

## Evidence to review

| Question | Artifact |
|---|---|
| Where did the data come from? | [Data card](docs/DATA_CARD.md), [source hashes](manifests/source_manifest.tsv) |
| Which compounds were excluded or held out? | [Eligibility](manifests/eligibility.tsv), [split manifest](manifests/split_manifest.tsv) |
| What was actually trained? | [All 28 learned-model fits](receipts/training_runs.tsv), [model selection](receipts/model_selection.json) |
| Can individual errors be checked? | [Every compound/seed metric](reports/final_test/metrics_by_compound.tsv), [paired intervals](reports/final_test/paired_comparisons.json) |
| Are predictions linked to fixed artifacts? | [Prediction seal](receipts/prediction_seal.json), [final evaluation](receipts/final_evaluation.json) |
| What can the model be used for? | [Model card](docs/MODEL_CARD.md) |

## Scope

The project comprises an original single-context benchmark (v1), a fixed-prediction transport evaluation across five additional conditions within the same study (v2), context-aware published single-cell baselines (v3), interventions on chemCPA's collapse tested on OP3 (v4), a confirmatory test on Tahoe-100M (v5) and a directional, split-measurement endpoint on ten further Tahoe-100M cell lines (v6). The v1/v2 models receive chemical structure, not cell identity or dose; the v3 biolord and v3–v6 chemCPA models receive both. It does not establish drug efficacy, toxicity, target engagement, clinical benefit or antibody design. No pretrained foundation model or paid AWS execution is claimed. OP3 and Tahoe-100M are public datasets, untouched only in the sense that this project had not scored them before. v1 training used the Primary CPU host; v3–v6 cell-level models were trained on Kaggle NVIDIA T4 GPUs (Kaggle notebooks, not AWS).

## Sources

- [Srivatsan et al., *Science* (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7289078/)
- [GEO GSE139944 / GSM4150378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378)
- [Pinned sci-Plex author preprocessing](https://github.com/cole-trapnell-lab/sci-plex/tree/079639c50811dd43a206a779ab2f0199a147c98f)
- [Reactome 97](https://download.reactome.org/97/) and [HGNC](https://www.genenames.org/) for frozen biological annotation.
- [GEO GSE279945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279945) (OP3, v4) and [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) at revision `349e2a55…` (v5, v6).
- [chemCPA](https://github.com/theislab/chemCPA) (vendored model file in `src/drugperturb/third_party/`) and [biolord](https://github.com/nitzanlab/biolord).

Raw third-party data and generated binary models are excluded from Git. Recreate fitted artifacts through the isolated workflow before using the inference CLI. Source terms and project implementation are documented separately.

Historical source IDs in the execution evidence label pre-publication source snapshots; they are not commit IDs in this public repository. [Publication provenance](docs/PUBLICATION_PROVENANCE.md) records original-file SHA256 to public-copy SHA256 mappings and distinguishes historical execution records from the public source history.

## License

The project's original code and accompanying documentation are available under the [MIT License](LICENSE), copyright 2026 Ningyu Sun. Third-party data, annotations, author code and dependencies retain their own terms; this license does not relicense them. See the [data card](docs/DATA_CARD.md) for source attribution and redistribution boundaries.
