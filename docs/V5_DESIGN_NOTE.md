# DrugPerturb v5: design note

**Status: prespecified on 2026-10-22, before any fit.** v4 and its corrected analysis both concluded that a positive claim needs a new protocol on untouched data. This note explains that protocol, [protocol_v5.yaml](../configs/protocol_v5.yaml).

- **Decisions:** the user chose Tahoe-100M and the draft defaults for D2–D11.
- **Design audit:** it was run from metadata only and wrote `manifests/v5/` and [receipts/v5/design_audit.json](../receipts/v5/design_audit.json).
- **Salt policy:** the user reviewed the audit and kept the v1 counter-ion list.
- **Commit:** the protocol was committed with those manifests, and changes from here on are amendments.

When it was committed, nothing had been fitted, and no expression value or compound-sample cell count had been read ([pre-protocol reads](../receipts/v5/pre_protocol_reads.md)). The dataset comparison is in [V5_DATASET_SURVEY.md](V5_DATASET_SURVEY.md).

## Design audit results

| Item | Result |
|---|---|
| Samples | 1,344: 1,298 compound samples of 378 drugs, 28 vehicle samples (exactly two DMSO_TF per plate), and 18 Adagrasib reference-control samples |
| Plate layout | Each plate holds one dose (0.05 µM on plates 1/4/7/10, 0.5 µM on 2/5/8/11, 5 µM on 3/6/9/12/14), except plate 13, which mixes all three. Plate 14 repeats plate 6. A compound's three doses are on three different plates, so **dose and plate are confounded**. |
| Excluded | **Adagrasib (reference control).** It is at 0.05 µM on every plate; it is excluded, as v4 excluded the OP3 positive controls. **22 drugs on structure:** 2 have no SMILES, and 20 are unresolved mixtures, mostly salts whose counter-ions the v1 list lacks. **20 duplicate-parent formulations**, by the v1 rule. |
| Compounds | 336 unique parents in 290 scaffold groups |
| Split (seed 42) | train 202 (168 groups), validation 50 (45), test **84 (77)**; gate G2 met |
| Replicates | 871 compound-doses have one sample, 134 two and 3 three. The multi-sample ones are mostly plate 6/14 and plate 13 repeats. |
| Overlap with scored data | 17 sci-Plex3 parents (4 in test) and 11 OP3 parents (1 in test) |
| Vehicle units | 50 lines; median 2,811 pass-filter cells per (plate, line); 48 lines meet the 200-cell rule |
| Lines drawn | SNU-1, A-172, HEC-1-A, MIA PaCa-2, C32, H4, LOX-IMVI, NCI-H2347, SW 1088 and SW1417: 30 contexts. SW 1088 has the smallest vehicle units, 74–626 cells per plate, so its count gate is the tightest. |

**Limits of the design:**

- **Salts excluded under the v1 policy (kept on review).** The v1 counter-ion list was built for sci-Plex3. As a result, 14 Tahoe salts and solvates are rejected as mixtures; examples are the maleates of neratinib and fluvoxamine, cabozantinib malate and the acetate peptides. Of the other 6, 3 hold two parent copies, 1 has a stray fragment in its source SMILES, and 2 are metal complexes. Extending the list would recover up to 14; dapagliflozin is already tested in free form. It is a structure-only choice, but it is also a policy change, and it would need its own constant, because the v1–v4 manifests depend on the current list.
- **Dose confounded with plate.** Each dose's compounds and vehicles come from its own plates. Within a dose, B1a and M1a anchor each compound to its own plate's vehicle; across doses, plate and dose effects cannot be separated. Every endpoint compares methods within a context, so this limits interpretation of dose effects, not the endpoints themselves.

## Question

On data this project has never scored, does a chemCPA model predict held-out compounds better than **vehicle-anchored** simple baselines? The model is trained with the v4 interventions (NB likelihood, counterfactual pseudobulk loss, or both) and the corrected descriptor scaler. Compounds are held out as whole scaffold groups.

**What we expect.** v3, v4 and the corrected analysis make a negative result the most likely outcome. The study is still informative, for three reasons:

- **A new data regime.** Cancer lines, three doses, about three times as many test compounds per context, and large vehicle units.
- **Whether structure carries any signal.** S3 (M1a − B1a) asks whether structure adds anything beyond the anchored context mean. v1–v4 never established that it does.
- **Whether v4's anchoring explanation holds.** S4 (B1a − B1c) tests it. Tahoe's vehicle units have about 2,600 cells, so vehicle noise is small, and the anchoring gain should shrink if vehicle noise caused it.

## What carries over from v4 unchanged

- **Arms.** A0, I1, I2 and I3, with v3's C1 hyperparameters, no search, and the A2 cap.
- **Training.** The schedule, early stopping, checkpoint rule, 60-minute cap and seed averaging.
- **Response.** Library-weighted pseudobulk, all-gene `log1p` CPM, strict matched-vehicle deltas, and 2,000 genes chosen from training-compound deltas only.
- **Comparators.** B0, B1c, B2c and M1c, sealed before any arm is scored.
- **Diagnostics.** v4's non-collapse criteria, reported before test scoring.
- **Evaluation.** Compound units, per-compound RMSE, a scaffold-group bootstrap shared across contexts, and pooling over gate-passing contexts.
- **Process.** Sealing and the order of operations.

## What v4 taught, prespecified from the start

| Lesson | v4 | v5 |
|---|---|---|
| Descriptor columns constant up to rounding gave ixabepilone z ≈ 10¹⁶ | found after test scoring; fixed post hoc | `descriptors.min_std = 1e-8` (`NOISE_STD_TOLERANCE`), plus a label-free report of dropped columns and the largest \|z\| per split before any fit. A seal gate at \|z\| = 10³ is proposed (D6). |
| Vehicle-anchored baselines beat every arm | B1a, M1a and E3 added by amendment A1 | **B1a and M1a are the primary comparators.** The primary endpoint P1 requires the arm to beat both. The old E1 and E2 become key secondary endpoints. |
| Arm selection was a near-tie twice (margins 0.0010 and 0.0004; I2's seed range 0.0096) | 3 seeds, lowest mean wins | 10 seeds per candidate, which resolves differences of about 0.005. Closer candidates are declared tied by a fixed rule and carried forward as co-primary arms, with Bonferroni-adjusted intervals. |
| Gates were partly implicit | count gate per context | Explicit gates G0–G7 with stated consequences. These include a split gate (≥ 60 test compounds in ≥ 40 scaffold groups) and a study gate (≥ 6 gate-passing contexts covering ≥ 3 lines and ≥ 2 doses, or nothing is confirmatory). |

## Tahoe-specific choices

- **Contexts.** A context is a (cell line, dose). The default is 10 cell lines at all three doses, 30 contexts in all. The lines are drawn at random (seed 42) from those with a median vehicle unit of at least 200 cells, using vehicle samples only.
- **Model cells.** At most 32 cells per well and line are used to train models. Truth and the baselines use all cells.
- **Replicates.** The replicate unit is the well, usually one per (compound, dose). Compounds on both plate 6 and plate 14 are averaged over the two plates, each against its own plate's vehicle. There are no donors, so the study cannot separate biological from technical variance, except for the plate 6/14 compounds.
- **Vehicle.**
  - The strict vehicle is all DMSO_TF cells of the same line on the same plate.
  - Because about 90 compound wells share one vehicle unit, the protocol adds two checks:
    - a plate-anchored training mean, **B1p** (secondary);
    - a leave-one-plate-out sensitivity analysis of P1.
- **Reads before the protocol is committed.**
  - **Allowed:** design columns only.
  - **Never read:**
    - the published pseudobulk DE files;
    - the expression-summary columns of `sample_metadata`;
    - any compound-sample cell count.

## Decisions (made 2026-10-22: D1 Tahoe-100M, D2–D11 as the defaults below)

| # | Decision | Default in the draft | Alternatives and trade-offs |
|---|---|---|---|
| **D1** | **Which dataset** | **Tahoe-100M**, pinned to Hugging Face revision `349e2a55…` | **Wait for the Arc–Biohub–Tahoe release** (~1,400 scaffolds; nobody outside the partners has analysed it), with no release date. **sci-Plex Müller glia** (92 compounds, mouse): at most ~23 test compounds, barely over the gate. |
| D2 | Arms, or the question itself | Candidates I1, I2, I3; A0 as a 3-seed reference | **Drop I3**, which failed non-collapse on OP3 twice; this saves 10 fits. **Reframe the primary question** as "does any structure model beat B1a" (S3 promoted); this ends the chemCPA line. |
| D3 | Contexts and size | 10 lines × 3 doses; 32 model cells per well and line | **All 50 lines:** 150 contexts; the cell cap must fall to about 6 to keep the same number of model cells. **One dose only:** cheaper, less coverage. **A different line rule,** such as one line per organ. |
| D4 | Seeds per candidate | 10 | **5:** ties below ~0.006 instead of ~0.0045, half the GPU time. **More than 10:** little gain; the tie rule handles the rest. |
| D5 | Data route | Hugging Face parquet, streamed in Kaggle CPU kernels (free, ~429 GB) | **Arc GCS per-plate h5ad** (337 GB): Requester Pays, needs a billed GCP project. Every Kaggle job needs your approval. |
| D6 | Descriptor seal gate | Adopt `DESCRIPTOR_Z_LIMIT = 1e3` from branch `v5-zscore-seal-gate` (in progress in another session, not merged) | Without it, `min_std` alone covers the v4 failure, but not a new extreme. |
| D7 | Compounds also in sci-Plex3 or OP3 | Keep them and report the overlap | Exclude them from test. This is cleaner for "new compounds", but they were never scored in Tahoe. |
| D8 | Plate-anchored baseline B1p | Include it as secondary | Omit it. |
| D9 | Bootstrap draws | 10,000, for the Bonferroni tails | 2,000, as in v4. |
| D10 | Near-tie handling | Co-primary arms, Bonferroni-adjusted | **Seed-ensemble of the tied arms** as one primary arm. **Fixed listed order,** as v4 did for exact ties. |
| D11 | Plate 6/14 duplicates | Average both plates | Use plate 6 only, so every compound has one well. |

## Compute estimate

**Extraction.** One pass over ~429 GB of parquet in CPU kernels, keeping the selected lines. The run may need several kernels because of Kaggle's 12-hour limit.

**Fits.** About 33 fits: 3 candidates × 10 seeds, plus A0 × 3. At an estimated 20–40 minutes each, that is 11–22 T4 GPU hours, over two or more sessions. For comparison, v4's three sessions on OP3 and sci-Plex3 used ~6.5 GPU hours in all.

## Next steps

**Done on 2026-10-22:** the decisions, the design audit, the user's review, and the commit of the protocol (status `PRESPECIFIED_BEFORE_V5_FITTING`) with `manifests/v5/`.

**Still to do:**

1. Merge the D6 descriptor seal gate (branch `v5-zscore-seal-gate`) before any fit is sealed.
2. Write the extraction and preparation kernels, and get approval for the Kaggle jobs.
