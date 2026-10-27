# DrugPerturb v6: Kaggle runs

The kernels and the code dataset are private, under the account `ningyusunny`. Each job was launched only after the user approved it.

| Kernel | Version | Hardware | Code archive SHA256 (dataset version) | Repository commit | Inputs | Outcome |
|---|---|---|---|---|---|---|
| dp-v6-extract (E1) | 1 | CPU (4 cores), internet | 997c3885… (drugperturb-v6-code v1) | 0ff1452 | Tahoe-100M at Hugging Face revision 349e2a55… | complete; 3,388 shards in one part; 17,698,679 cells; 3 h 37 min |
| dp-v6-prepare (P1) | 1 | CPU | 997c3885… (drugperturb-v6-code v1) | 0ff1452 | dp-v6-extract v1 | complete; every anchor passed; 27 of 30 contexts pass the count gate; comparators sealed; CPU-only report written |
| dp-v6-fit-m1 | 3 | GPU (Tesla T4 x2) | b52e1cfe… (drugperturb-v6-code, pushed from 76516f0) | 76516f0 | dp-v6-prepare v1 | complete; M1_pseudobulk_raw at all ten seeds; 441.8 min |
| dp-v6-fit-m2 | 1 | GPU (Tesla T4 x2) | b52e1cfe… (drugperturb-v6-code, pushed from 76516f0) | 76516f0 | dp-v6-prepare v1 | complete; M2_pseudobulk_residual at all ten seeds; 347.6 min |
| dp-v6-seal | 1 | CPU | b52e1cfe… (drugperturb-v6-code, pushed from 76516f0) | 76516f0 | dp-v6-fit-m1 v3, dp-v6-fit-m2 v1, dp-v6-prepare v1 | complete; both arms sealed; M2 selected on validation; pre-test diagnostics written |

**Superseded GPU run.** `dp-v6-fit-m1` version 2 ran M1 at three seeds under the reduced plan of amendment A3, which was then withdrawn (see A3_2026-10-27_RAISED_AND_WITHDRAWN). Version 3 refits all ten seeds from scratch and is the one sealed; version 2's three fits are wasted GPU time and no number from them enters the study.

## Commands

```bash
python kaggle/push.py code --dataset drugperturb-v6-code
python kaggle/push.py kernel v6-e1          # all 3,388 shards in one part
python kaggle/push.py kernel v6-p1          # prepare, comparators, CPU-only report
```

## E1: extraction

- **Design check.** `scripts/v6_design_audit.py --check` regenerated `manifests/v6` exactly, with `expression_read: false` ([v6_design_check.json](e1/v6_design_check.json)).
- **Scale.** 11,760 units, 23,520 half-units, 17,698,679 pass-filter cells of the ten lines, all found exactly once.
- **The measurement split.** Seed 606. Median half sizes 678 (selection) and 679 (evaluation), the extra cell going to the evaluation half as the protocol states. Some half-units hold no cell, because a line can be absent from a well; the count gate removes those keys.
- **Model cells.** 282,496 selected, capped at 32 per unit.
- **Anchors.** Every extraction anchor passed ([part_receipt.json](e1/part_receipt.json)).

## P1: preparation, comparators and the CPU-only report

- **Genes.** 2,000 frozen, chosen from training-compound deltas on whole units.
- **Count gate.** 27 of 30 contexts pass; study gate G4 met ([context_summary.tsv](p1/context_summary.tsv)).
- **Responders.** Threshold 2.772, the 95th percentile of 140 between-well vehicle statistics whose median is 1.866 (amendment A1). 4,946 of 9,700 keys are responders: 2,945 training, 678 validation, 1,323 test.
- **Comparators.** Sealed before any fit ([baselines_v6_seal.json](p1/baselines_v6_seal.json)).
- **No test outcome.** `evaluation_half_test_delta_computed: false` in the prepare receipt.

## The CPU-only report

Full text: [reports/v6/cheap_report.md](../../reports/v6/cheap_report.md).

| Quantity | Value |
|---|---:|
| Split noise floor, responders (gate G8 needs 0.60) | **0.8232** |
| B1p direction accuracy | **0.8190** |
| Headroom above B1p | **0.0042** |
| Zero change (B0) | 0.5000 |
| Residual share over B1p | 0.652 |

Gate G8 is met. No structure-using method beats the method it extends by more than 0.01 (M1c minus B1c +0.0017, M1a minus B1a +0.0002, M1r minus B1p +0.0008), but that is not what decision D7 says. D7's committed trigger is a margin over CHANCE, and every structure-using comparator clears chance by 0.26 or more because B1p, which uses no chemistry, already does. **D7 therefore does not fire and the full prespecified plan runs:** both candidate arms at all ten seeds. Amendment A3_2026-10-27_RAISED_AND_WITHDRAWN records that the discrepancy was raised before any GPU fit and that the user chose the committed wording.

The report was regenerated locally from the fetched P1 outputs twice: first after `de_set_sharing` was added to the script, and again after the script's D7 sentence was corrected to the committed trigger. Both times every other number the Kaggle run produced came back unchanged, which is itself a reproducibility check on the CPU stage.

## The fits

```bash
python kaggle/push.py code --dataset drugperturb-v6-code
python kaggle/push.py kernel v6-fit-m1     # M1_pseudobulk_raw, ten seeds
python kaggle/push.py kernel v6-fit-m2     # M2_pseudobulk_residual, ten seeds
python kaggle/push.py kernel v6-seal       # seal, select, pre-test diagnostics
```

Kaggle allows two batch GPU sessions at a time, so the two arms ran in parallel; `dp-v6-fit-m2` waited about two and a quarter hours for the superseded version 2 session to release its slot.

- **Smoke gate.** Each kernel ran a one-epoch fit per seed before any real fit, and both printed `smoke fits passed`; the seven `tests/test_v6_cellmodels.py` tests passed in both GPU environments and in the seal's CPU environment.
- **Every fit stopped on patience**, none on the epoch ceiling or the 60-minute budget. Per-seed records and training histories are in [fits/](fits/), the per-seed logs in [fits/logs/](fits/logs/).
- **Test blindness.** Every one of the 20 fit records carries `test_outcomes_loaded: false` and `evaluation_half_test_delta_loaded: false`; the seal refuses a fit that does not.
- **Descriptor gate.** Passed without an override; the largest scaled descriptor is 14.2 standard deviations against a limit of 1000.

| Arm | Target | Validation direction accuracy, responders | All keys | Validation RMSE | Residual r | Effect ratio |
|---|---|---:|---:|---:|---:|---:|
| M1_pseudobulk_raw | the delta itself | 0.6848 ± 0.0247 | 0.6278 | 0.3147 | +0.049 | 0.686 |
| M2_pseudobulk_residual | delta minus B1p | 0.8182 ± 0.0008 | 0.7483 | 0.2568 | −0.007 | 1.457 |
| B1p (plate-anchored training mean) | — | 0.8190 | — | — | — | — |
| Split noise floor (the ceiling) | — | 0.8232 | — | — | — | — |

Both arms are non-collapsed by the protocol's criteria. The floor leaves 0.3232 of headroom over chance. M1 recovers 0.1848 of it (57%), B1p recovers 0.3190 (99%), and M2 lands 0.0008 BELOW the mean it was built on, with a residual correlation of about zero.

## The seal and the selection

Full text: [reports/v6/diagnostics.md](../../reports/v6/diagnostics.md); per-seed values in [per_seed_validation.tsv](../../reports/v6/per_seed_validation.tsv); the record itself in [sealed/selection_v6.json](sealed/selection_v6.json).

- **Selected on validation: M2_pseudobulk_residual**, the single primary arm. M1 is 0.1334 behind with a Welch standard error of 0.0082, far outside the near-tie band, so `bonferroni_k` is 1.
- `planned_candidates` equals `fitted_candidates`, `arms_not_fitted` is empty and `reduced_plan` is null: the plan that ran is the plan that was prespecified, so endpoint S4 (M2 minus M1) is estimable.
- The selection record and the pre-test diagnostics are committed in this commit, before the evaluation kernel runs. Neither reads a test response or an evaluation-half delta.
