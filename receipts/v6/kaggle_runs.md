# DrugPerturb v6: Kaggle runs

The kernels and the code dataset are private, under the account `ningyusunny`. Each job was launched only after the user approved it.

| Kernel | Version | Hardware | Code archive SHA256 (dataset version) | Repository commit | Inputs | Outcome |
|---|---|---|---|---|---|---|
| dp-v6-extract (E1) | 1 | CPU (4 cores), internet | 997c3885… (drugperturb-v6-code v1) | 0ff1452 | Tahoe-100M at Hugging Face revision 349e2a55… | complete; 3,388 shards in one part; 17,698,679 cells; 3 h 37 min |
| dp-v6-prepare (P1) | 1 | CPU | 997c3885… (drugperturb-v6-code v1) | 0ff1452 | dp-v6-extract v1 | complete; every anchor passed; 27 of 30 contexts pass the count gate; comparators sealed; CPU-only report written |

**No GPU kernel has run.** The fits need their own approval and the CPU report's decision D7 first, so they are not registered in `kaggle/push.py`.

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

Gate G8 is met. No structure-using method beats its reference by more than 0.01, so decision D7 cuts the GPU budget to M1 at 3 seeds.

The report was regenerated locally from the fetched P1 outputs after `de_set_sharing` was added to the script; every number the Kaggle run produced is unchanged, and the local run adds the DE-set sharing diagnostic and the headroom figure.
