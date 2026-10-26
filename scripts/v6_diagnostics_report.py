"""Write the v6 pre-test diagnostics report from training and validation outputs only.

Committed before the evaluation kernel runs, as the protocol requires. Every
number comes from the sealed arms' validation predictions, the sealed
comparators, the preparation receipt and the selection record: no test response
and no evaluation-half test delta is read.

The report leads with the three numbers that decide whether the primary endpoint
can succeed at all - the split noise floor, which is its ceiling; whether B1p
already beats chance on a compound's own DE genes; and whether any method that
uses chemical structure beats its own reference.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_STATUS = 'PRESPECIFIED_BEFORE_V6_FITTING'
# Anchoring adds the same term to both sides, so it cancels in these differences; M1a against B1a is an
# endpoint (S3) and is estimated in the evaluation, not here.
STRUCTURE_PAIRS = (('M1c', 'B1c'), ('M1r', 'B1p'), ('B2c', 'B1c'))


def table(rows, columns):
    return pd.DataFrame(rows, columns=columns)


def fmt(value, digits=4):
    return 'n/a' if value is None or not np.isfinite(value) else f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sealed', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v6.yaml')
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != PROTOCOL_STATUS:
        raise ValueError('v6 protocol is not prespecified')

    def locate(name):
        found = [r/name for r in args.sealed if (r/name).exists()]
        if len(found) != 1:
            raise ValueError(f'{name}: expected exactly one copy, found {len(found)}')
        return found[0]

    receipt = json.loads((args.prepared/'prepare_receipt.json').read_text())
    selection = json.loads(locate('selection_v6.json').read_text())
    arms = {a: json.loads(locate(f'{a}_v6_seal.json').read_text()) for a in selection['arms']}
    comparators = json.loads(locate('baselines_v6_seal.json').read_text())
    if any(s.get('test_outcomes_loaded') is not False for s in [selection, comparators, *arms.values()]):
        raise ValueError('a sealed input does not certify that no test outcome was loaded')

    first = next(iter(arms.values()))
    floor = first['endpoint']['split_noise_floor']
    b1p_accuracy = first['endpoint']['b1p_direction_accuracy']
    gate = protocol['gates']['G8_split_power']
    lines = ['# DrugPerturb v6: pre-test diagnostics', '',
             f"**Status: written from training and validation outputs only, before any test scoring.** "
             f"The protocol is [protocol_v6.yaml](../../configs/protocol_v6.yaml); the selection record is "
             f"`selection_v6.json`. No test response and no evaluation-half test delta has been read.", '',
             '## Can the endpoint separate methods at all', '',
             '| Quantity | Value | What it means |', '|---|---:|---|',
             f'| Split noise floor | {fmt(floor)} | The direction accuracy the selection half itself reaches '
             f'against the evaluation half: the ceiling any model could reach given the split. Gate G8 requires '
             f'0.60. |',
             f'| B1p on its own | {fmt(b1p_accuracy)} | Whether the plate mean already beats chance (0.5) on a '
             f"compound's own DE genes. |",
             f"| Responders | {receipt['responder']['responders']} of {receipt['responder']['keys']} | "
             f"Keys above the between-well null's "
             f"{receipt['responder']['percentile']:.0f}th percentile (threshold "
             f"{fmt(receipt['responder']['threshold'], 3)}). |",
             f"| Residual share | {fmt(first['endpoint']['residual_share'], 3)} | Variance of the measured "
             f'residual over B1p as a fraction of the variance of the measured delta: how much is left to '
             f'predict. |', '']
    if np.isfinite(floor) and floor < 0.60:
        lines += [f'**Gate G8 is not met.** The split noise floor is {fmt(floor)}, below 0.60, so the endpoint '
                  f'cannot resolve differences between methods. {gate}', '']

    lines += ['## Arms on validation', '',
              '| Arm | Direction accuracy, responders | All keys | Validation RMSE | Residual r | Effect ratio | '
              'Non-collapsed |', '|---|---:|---:|---:|---:|---:|---|']
    for name, seal in arms.items():
        e, v = seal['endpoint'], seal['non_collapse']
        lines.append(f"| {name} | {fmt(e['validation_direction_accuracy'])} | "
                     f"{fmt(e['validation_direction_accuracy_all_keys'])} | "
                     f"{fmt(seal['mean_seed_validation_rmse'])} | {fmt(e['residual_r'], 3)} | "
                     f"{fmt(v['effect_size_ratio'], 3)} | {'yes' if v['non_collapsed'] else 'no'} |")
    lines += ['', f"Chance is {0.5:.1f} by construction, because a zero prediction counts as a tie. An arm must "
                  f"reach at least {0.5+0.01:.2f} and an effect size ratio of 0.10 to be non-collapsed.", '']

    per_seed = [dict(arm=name, seed=seed, direction_accuracy=value, validation_rmse=rmse)
                for name, seal in arms.items()
                for seed, value, rmse in zip(seal['seeds'], seal['seed_direction_accuracy'],
                                             seal['seed_validation_rmse'])]
    lines += ['## Arm selection', '',
              f"Metric: {selection['metric']}. Leader: **{selection['selected']}**. "
              f"Near tie: {'yes' if selection['near_tie'] else 'no'}; "
              f"primary arms {', '.join(selection['primary_arms'])} at interval level "
              f"{selection['interval_level']:.4f}.", '']
    if not selection['any_passing']:
        lines += ['**No candidate met the non-collapse criteria.** The highest-metric candidate is still '
                  'evaluated, and the study reports that no arm met them.', '']
    for arm, c in selection['comparisons_to_leader'].items():
        lines.append(f"- {arm}: behind the leader by {fmt(c['d'])} with a Welch standard error of "
                     f"{fmt(c['se'])}, so it is {'within' if c['within'] else 'outside'} 1.96 standard errors.")
    lines += ['', '## Does chemical structure help any method', '',
              'Each pair is a structure-using method against the method it extends, on validation compounds. '
              'These are the cheap tests of S3; they need no GPU fit.', '',
              '| Contrast | Validation RMSE difference |', '|---|---:|']
    simple = {c: info.get('validation_rmse_simple', {}) for c, info in comparators['selection'].items()}
    # fit_context records one validation RMSE per alpha and keeps the lowest, so that is the chosen model's.
    ridge = {c: (min(info['validation_rmse']) if info.get('validation_rmse') else None)
             for c, info in comparators['selection'].items()}
    for candidate, baseline in STRUCTURE_PAIRS:
        values = []
        for context, info in comparators['selection'].items():
            got, want = None, None
            if candidate == 'M1c':
                got, want = ridge.get(context), simple.get(context, {}).get('B1')
            elif candidate == 'B2c':
                got, want = simple.get(context, {}).get('B2'), simple.get(context, {}).get('B1')
            elif candidate == 'M1r':
                residual = comparators['residual']['per_context'].get(context, {})
                if residual.get('fitted'):
                    got, want = residual.get('validation_residual_rmse'), residual.get('zero_residual_rmse')
            if got is not None and want is not None and np.isfinite(got) and np.isfinite(want):
                values.append(got-want)
        if values:
            lines.append(f'| {candidate} minus {baseline} | {fmt(float(np.mean(values)))} |')
    lines += ['', f"Residual contexts fitted: "
                  f"{sum(1 for v in comparators['residual']['per_context'].values() if v['fitted'])} of "
                  f"{len(comparators['residual']['per_context'])}. Scale factors: "
                  f"{', '.join(fmt(v['scale'], 2) for v in comparators['residual']['per_context'].values() if v['fitted'])}.",
              '', '## Context gates', '',
              f"Contexts passing the count gate: {receipt['contexts_passing_count_gate']}. "
              f"Study gate G4: {'met' if receipt['study_gate_G4']['met'] else 'NOT met'} "
              f"({receipt['study_gate_G4']['n_contexts']} contexts, {receipt['study_gate_G4']['n_lines']} lines, "
              f"{receipt['study_gate_G4']['n_doses']} doses).", '',
              '## What this report does not contain', '',
              '- No test response, and no evaluation-half delta of any test key.',
              '- No endpoint estimate: every number here is from training or validation compounds.',
              '- The arm selection is fixed by this record and is not revisited after test scoring.', '']

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines))
    pd.DataFrame(per_seed).to_csv(args.output.parent/'per_seed_validation.tsv', sep='\t', index=False)
    print(json.dumps(dict(report=str(args.output), arms=list(arms), selected=selection['selected'],
                          split_noise_floor=floor, b1p_direction_accuracy=b1p_accuracy,
                          responders=receipt['responder']['responders']), default=float))


if __name__ == '__main__':
    main()
