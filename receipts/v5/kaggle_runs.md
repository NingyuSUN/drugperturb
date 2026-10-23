# DrugPerturb v5: Kaggle runs

The kernels and the code dataset are private, under the account `ningyusunny`. Each job was launched only after the user approved it.

| Kernel | Version | Hardware | Code archive SHA256 (dataset version) | Repository commit | Inputs | Outcome |
|---|---|---|---|---|---|---|
| dp-v5-extract (E1) | 1 | CPU (4 cores, 31 GB), internet | 11d17891… (drugperturb-v5-code v1) | 9b94ed2 | Tahoe-100M at Hugging Face revision 349e2a55… | complete; 3,388 shards in one part; 1 h 45 min extraction |
| dp-v5-prepare (P1) | 1 | CPU | 11d17891… (drugperturb-v5-code v1) | 9b94ed2 | dp-v5-extract v1 | complete; every anchor passed; 30 of 30 contexts pass the count gate; comparators sealed |

## Commands

The commands ran in WSL from the `dp-v5p` worktree, with Kaggle CLI 2.2.4. WSL git needs `GIT_DIR` and `GIT_WORK_TREE` for this worktree. The archive holds the 496 files that git tracks at 9b94ed2; no data or outputs are included.

```
KAGGLE_BUILD_ROOT=$HOME/kb python3 kaggle/push.py code --dataset drugperturb-v5-code
KAGGLE_BUILD_ROOT=$HOME/kb python3 kaggle/push.py kernel v5-e1
```

## E1 (dp-v5-extract, version 1)

- **Timing.** The kernel was pushed at 00:22 UTC on 2026-10-23 and wrote its receipt at 02:09 UTC. The extraction itself took 6,277 s.
- **Code and tests.**
  - The receipt ([e1_extract.json](e1/e1_extract.json)) records the code-archive SHA256 11d17891…, which equals the archive that was pushed.
  - The in-kernel tests passed: 23 tests ([e1_tests.log](e1/e1_tests.log)).
- **Inputs.**
  - The five metadata files matched [source.tsv](../../manifests/v5/source.tsv) and [expression_source.tsv](../../manifests/v5/expression_source.tsv).
  - `v5_design_audit.py --check` regenerated every design manifest exactly ([v5_design_check.json](e1/v5_design_check.json)).
  - Every expression shard matched its size and LFS SHA256.
  - The part receipt records the protocol SHA256 c50b1162…, the protocol at 9b94ed2.
- **Extraction** ([part_receipt.json](e1/part_receipt.json); per shard in [shards.tsv](e1/shards.tsv)):
  - **Rows.** 95,624,334 rows were read, equal to the dataset's published cell count.
  - **Cells.** 15,676,406 were matched to the design: pass-filter cells of the ten lines in vehicle samples and accepted-compound samples, all of them found.
  - **Decoding.** Every matched cell began with the CLS marker. There were no tokens outside the vocabulary and no duplicate tokens.
  - **Units and cells.** 11,760 units; 418,028 vehicle cells; 281,856 model cells.
- **Checks made locally on the downloaded outputs**, before preparation:
  - every output file matches its hash in the part receipt;
  - unit cell numbers equal the `obs_metadata` pass-filter counts for all 11,760 units;
  - the model cells are exactly the seeded selection, each found once.

  The remaining anchors run in P1: unit libraries against unit sums, and model cells within the unit sums (amendment A1). WSL has only 7 GB of RAM, too little to hold the matrices.
- **Not committed.** The outputs stay out of Git, under the repository's rule on large outputs: `unit_counts.npz` (724 MB), `model_cells.npz` (850 MB) and the selection and design tables. The local copy downloaded with `kaggle kernels output ningyusunny/dp-v5-extract` is at `~/kout/v5e1` in WSL, and the part receipt hashes every file.
- **No response computed.** No delta, truth or response of any compound was computed. The receipt status is `EXTRACTED_PART_TEST_RESPONSES_NOT_COMPUTED`.

## P1 (dp-v5-prepare, version 1)

- **Timing.** The kernel was pushed at 02:39 UTC on 2026-10-23 and was complete by 02:45 UTC.
- **Code and inputs.**
  - The code archive is the same as E1's: 11d17891…, commit 9b94ed2 ([p1_prepare.json](p1/p1_prepare.json)).
  - The input was the dp-v5-extract v1 output.
  - Every prepared output matches its hash in [prepare_receipt.json](p1/prepare_receipt.json).
- **Anchors.** All passed:
  - 3,388 shards, each processed once;
  - unit cells equal the obs pass-filter counts;
  - unit libraries equal unit sums;
  - every selected model cell was found once;
  - model cells stay within the unit sums, and equal them where every cell of a unit was kept (amendment A1);
  - model libraries equal model counts;
  - no test key in the truth;
  - no test cell among the model cells;
  - at most 32 model cells per unit.
- **Count QC and gates** ([context_summary.tsv](p1/context_summary.tsv)):
  - All 30 contexts pass the count gate G3, with 77–84 eligible test compounds in 70–77 scaffold groups each. SW 1088 (CVCL_1715) is the smallest.
  - Gate G4 is met: 30 contexts covering 10 lines and 3 doses.
  - 2,464 eligible test pairs.
- **Inputs to the fits.**
  - 7,747 candidate genes (vehicle detection at least 5%), and 2,000 frozen genes chosen by training-compound delta variance ([frozen_genes.tsv](p1/frozen_genes.tsv)).
  - 5,936 training and 1,462 validation truth pairs. Test truth was not computed.
  - 281,856 model cells: 220,877 training, 52,027 validation and 8,952 vehicle.
- **Comparators.** B0, B1c, B2c, M1c, B1a, M1a and B1p were sealed for 2,520 test keys and 1,462 validation keys, with no context skipped ([baselines_v5_seal.json](p1/baselines_v5_seal.json); predictions SHA256 b05c4c9c…).
  - The Ridge chose α = 1,000 or 10,000 in every context.
  - B1p fell back to B1a nowhere: every context and plate has at least 5 training compounds.
- **Local copy.** The full outputs are at `~/kout/v5p1` in WSL (584 MB) and stay out of Git.
