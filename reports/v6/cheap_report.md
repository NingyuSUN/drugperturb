# DrugPerturb v6: the CPU-only report, before any fit

**Status: written from preparation and the sealed comparators only.** No arm has been fitted, no test response and no evaluation-half test delta has been read. Every number below is on VALIDATION compounds.

## Can the endpoint work at all

| Quantity | Value | Threshold | Verdict |
|---|---:|---|---|
| Split noise floor, responders | **0.8232** | gate G8 needs 0.60 | met |
| Split noise floor, all keys | 0.7325 | - | - |
| B1p on its own | 0.8190 | chance 0.50 | above chance |
| Zero change (B0) | 0.5000 | exactly 0.50 by construction | as expected |
| Residual share over B1p | 0.652 | - | how much is left to predict |
| **Headroom above B1p** | **0.0042** | - | the floor minus B1p: all any model can win |

## Are the DE sets compound-specific, or one plate signature

If a plate carried one dominant shared response, every compound on it would pick nearly the same top genes, and the endpoint would be asking about plate-shared biology instead of compound-specific biology. Mean Jaccard overlap of two responders' DE sets, within a context:

| Pair | Mean overlap | Pairs |
|---|---:|---:|
| Same plate | 0.110 | 31056 |
| Different plate | 0.065 | 74960 |

- CVCL_0480_500nM: 311 responders, their DE sets cover 1357 genes, and only 6 genes appear in more than half of them.
- CVCL_0480_5000nM: 298 responders, their DE sets cover 1469 genes, and only 4 genes appear in more than half of them.
- CVCL_0480_50nM: 294 responders, their DE sets cover 1251 genes, and only 6 genes appear in more than half of them.

## Every comparator on its own DE genes

Direction accuracy over each validation key's top-50 selection-half genes. Chance is 0.50; B0 reaches it exactly because a zero prediction counts as a tie.

| Method | Responders | All keys | Uses structure |
|---|---:|---:|---|
| M1r_unscaled | 0.8202 | 0.7492 | yes |
| M1r | 0.8198 | 0.7490 | yes |
| B1r | 0.8197 | 0.7489 | no |
| B1p | 0.8190 | 0.7488 | no |
| M1a | 0.7980 | 0.7299 | yes |
| B1a | 0.7978 | 0.7294 | no |
| M1c | 0.7728 | 0.7019 | yes |
| B1c | 0.7711 | 0.7006 | no |
| B2r | 0.7596 | 0.6964 | yes |
| B0 | 0.5000 | 0.5000 | no |

## Does chemical structure help (a preview of S3)

| Contrast | Difference in direction accuracy |
|---|---:|
| M1c minus B1c | 0.0017 |
| M1a minus B1a | 0.0002 |
| M1r minus B1p | 0.0008 |
| B2r minus B1p | -0.0594 |
| B1r minus B1p | 0.0007 |

**No structure-using method beats its reference by more than 0.01.** Under decision D7 that cuts the GPU budget to M1 at 3 seeds.

## The responder gate

Threshold 2.772, the 95th percentile of 140 between-well vehicle statistics (amendment A1). 4946 of 9700 keys are responders.

| Split | Responders |
|---|---:|
| test | 1323 |
| train | 2945 |
| validation | 678 |

## Context gates

Contexts passing the count gate: 27 of 30. Study gate G4: met (27 contexts, 9 lines, 3 doses).

| Context | Eligible test compounds | Groups | Responders | Gate |
|---|---:|---:|---:|---|
| CVCL_0359_5000nM | 84 | 77 | 55 | met |
| CVCL_0359_500nM | 83 | 76 | 67 | met |
| CVCL_0359_50nM | 81 | 74 | 47 | met |
| CVCL_0371_5000nM | 83 | 76 | 53 | met |
| CVCL_0371_500nM | 83 | 76 | 58 | met |
| CVCL_0371_50nM | 81 | 74 | 55 | met |
| CVCL_0397_5000nM | 83 | 76 | 54 | met |
| CVCL_0397_500nM | 83 | 76 | 58 | met |
| CVCL_0397_50nM | 81 | 74 | 53 | met |
| CVCL_0480_5000nM | 84 | 77 | 70 | met |
| CVCL_0480_500nM | 83 | 76 | 79 | met |
| CVCL_0480_50nM | 81 | 74 | 75 | met |
| CVCL_1119_5000nM | 84 | 77 | 64 | met |
| CVCL_1119_500nM | 83 | 76 | 67 | met |
| CVCL_1119_50nM | 81 | 74 | 56 | met |
| CVCL_1577_5000nM | 54 | 50 | 0 | no |
| CVCL_1577_500nM | 70 | 64 | 0 | no |
| CVCL_1577_50nM | 64 | 58 | 0 | no |
| CVCL_1635_5000nM | 83 | 76 | 40 | met |
| CVCL_1635_500nM | 83 | 76 | 47 | met |
| CVCL_1635_50nM | 81 | 74 | 28 | met |
| CVCL_1693_5000nM | 84 | 77 | 32 | met |
| CVCL_1693_500nM | 83 | 76 | 36 | met |
| CVCL_1693_50nM | 81 | 74 | 24 | met |
| CVCL_1731_5000nM | 84 | 77 | 39 | met |
| CVCL_1731_500nM | 83 | 76 | 46 | met |
| CVCL_1731_50nM | 81 | 74 | 32 | met |
| CVCL_C466_5000nM | 84 | 77 | 32 | met |
| CVCL_C466_500nM | 83 | 76 | 38 | met |
| CVCL_C466_50nM | 81 | 74 | 18 | met |

## What happens next

- Gate G8: **met**.
- Decision D7: the GPU budget is cut to M1 at 3 seeds.
- No arm has been fitted. The GPU fits are the next step and need their own approval.
