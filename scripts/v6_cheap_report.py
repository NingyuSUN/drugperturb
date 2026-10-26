"""The v6 CPU-only report: the endpoint's ceiling, the responder gate and whether structure helps at all.

Written from preparation and the sealed comparators, before any GPU fit. It is
the D7 decision point for the GPU budget and it evaluates gate G8, and it needs
no arm, so it costs CPU minutes.

Three numbers decide whether the primary endpoint can succeed:

- the split noise floor, the accuracy ``delta_S`` itself reaches against
  ``delta_E`` - the ceiling any model could reach (gate G8 requires 0.60);
- whether B1p already beats chance on a compound's own DE genes;
- whether any method that uses chemical structure beats the method it extends.

Everything is computed on VALIDATION keys. No test response and no
evaluation-half test delta is read.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import yaml

from drugperturb.v6 import scoring

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_STATUS = 'PRESPECIFIED_BEFORE_V6_FITTING'
G8_FLOOR = 0.60
CHANCE = 0.5
STRUCTURE_PAIRS = (('M1c', 'B1c'), ('M1a', 'B1a'), ('M1r', 'B1p'), ('B2c', 'B1c'), ('B2r', 'B1p'),
                   ('B1r', 'B1p'))
METHODS = ('B0', 'B1c', 'B2c', 'M1c', 'B1a', 'M1a', 'B1p', 'B1r', 'B2r', 'M1r', 'M1r_unscaled')


def by_key(keys, values):
    return {tuple(k.split('|')): values[i] for i, k in enumerate(np.asarray(keys, str))}


def fmt(value, digits=4):
    return 'n/a' if value is None or not np.isfinite(value) else f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sealed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v6.yaml')
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != PROTOCOL_STATUS:
        raise ValueError('v6 protocol is not prespecified')
    receipt = json.loads((args.prepared/'prepare_receipt.json').read_text())
    seal = json.loads((args.sealed/'baselines_v6_seal.json').read_text())
    if seal['test_outcomes_loaded'] is not False:
        raise ValueError('the sealed comparators do not certify a test-blind seal')

    with np.load(args.prepared/'validation_truth.npz', allow_pickle=False) as z:
        evaluation = by_key(z['keys'], z['delta_evaluation'])
        selection_half = by_key(z['keys'], z['delta_selection'])
        measured = by_key(z['keys'], z['delta'])
    with np.load(args.prepared/'endpoint/de_sets.npz', allow_pickle=False) as z:
        de_sets = {k: z[k] for k in ['keys', 'genes']}
        responder = by_key(z['keys'], z['responder'])
        de_k = int(z['k'])
    with np.load(args.sealed/'baselines_v6_predictions.npz', allow_pickle=False) as z:
        comparators = {}
        for name in METHODS:
            key_field = f'validation_{name}_keys' if f'validation_{name}_keys' in z.files else 'validation_keys'
            if f'validation_{name}' in z.files:
                comparators[name] = by_key(z[key_field], z[f'validation_{name}'])

    keys = sorted(set(evaluation) & {tuple(s.split('|')) for s in np.asarray(de_sets['keys'], str)})
    # B0 is the zero prediction, so it is built rather than read; it is the endpoint's chance anchor and the
    # comparators are sealed with a validation copy only for the methods the diagnostics fit.
    comparators['B0'] = {k: np.zeros_like(evaluation[k]) for k in keys}
    responders = [k for k in keys if bool(responder.get(k, False))]
    floor = scoring.split_noise_floor(selection_half, evaluation, de_sets, responders)
    floor_all = scoring.split_noise_floor(selection_half, evaluation, de_sets, keys)
    accuracy = {name: scoring.mean_accuracy(values, evaluation, de_sets,
                                            [k for k in responders if k in values])
                for name, values in comparators.items()}
    accuracy_all = {name: scoring.mean_accuracy(values, evaluation, de_sets, [k for k in keys if k in values])
                    for name, values in comparators.items()}
    share = scoring.residual_share(measured, comparators['B1p'], [k for k in keys if k in comparators['B1p']])
    g8 = bool(np.isfinite(floor) and floor >= G8_FLOOR)
    structure = {f'{a}_minus_{b}': (accuracy[a]-accuracy[b]) for a, b in STRUCTURE_PAIRS
                 if a in accuracy and b in accuracy}
    any_structure = any(np.isfinite(v) and v > 0.01 for v in structure.values())

    order = sorted(accuracy, key=lambda n: -(accuracy[n] if np.isfinite(accuracy[n]) else -1))
    lines = ['# DrugPerturb v6: the CPU-only report, before any fit', '',
             '**Status: written from preparation and the sealed comparators only.** No arm has been fitted, no '
             'test response and no evaluation-half test delta has been read. Every number below is on '
             'VALIDATION compounds.', '',
             '## Can the endpoint work at all', '',
             '| Quantity | Value | Threshold | Verdict |', '|---|---:|---|---|',
             f'| Split noise floor, responders | **{fmt(floor)}** | gate G8 needs {G8_FLOOR:.2f} | '
             f"{'met' if g8 else 'NOT MET'} |",
             f'| Split noise floor, all keys | {fmt(floor_all)} | - | - |',
             f"| B1p on its own | {fmt(accuracy.get('B1p'))} | chance {CHANCE:.2f} | "
             f"{'above chance' if np.isfinite(accuracy.get('B1p', np.nan)) and accuracy['B1p'] > CHANCE else 'at or below chance'} |",
             f"| Zero change (B0) | {fmt(accuracy.get('B0'))} | exactly {CHANCE:.2f} by construction | "
             f"{'as expected' if np.isclose(accuracy.get('B0', np.nan), CHANCE) else 'UNEXPECTED'} |",
             f'| Residual share over B1p | {fmt(share, 3)} | - | how much is left to predict |', '']
    if not g8:
        lines += [f'**Gate G8 is not met.** The split leaves too little signal for the endpoint to separate '
                  f'methods, so the protocol reports that as the study result rather than comparing against '
                  f'it. The GPU fits would be measuring noise.', '']

    lines += ['## Every comparator on its own DE genes', '',
              f'Direction accuracy over each validation key\'s top-{de_k} selection-half genes. Chance is '
              f'{CHANCE:.2f}; B0 reaches it exactly because a zero prediction counts as a tie.', '',
              '| Method | Responders | All keys | Uses structure |', '|---|---:|---:|---|']
    uses_structure = {'B2c', 'M1c', 'M1a', 'M1r', 'M1r_unscaled', 'B2r'}
    for name in order:
        lines.append(f'| {name} | {fmt(accuracy[name])} | {fmt(accuracy_all[name])} | '
                     f"{'yes' if name in uses_structure else 'no'} |")
    lines += ['', '## Does chemical structure help (a preview of S3)', '',
              '| Contrast | Difference in direction accuracy |', '|---|---:|']
    for name, value in structure.items():
        lines.append(f'| {name.replace("_minus_", " minus ")} | {fmt(value)} |')
    lines += ['', f"**{'At least one structure-using method beats its reference by more than 0.01.' if any_structure else 'No structure-using method beats its reference by more than 0.01.'}** "
                  f"Under decision D7 that {'leaves' if any_structure else 'cuts'} the GPU budget "
                  f"{'as planned (20 fits)' if any_structure else 'to M1 at 3 seeds'}.", '',
              '## The responder gate', '',
              f"Threshold {fmt(receipt['responder']['threshold'], 3)}, the "
              f"{receipt['responder']['percentile']:.0f}th percentile of {receipt['responder']['nulls']} "
              f"between-well vehicle statistics (amendment A1). "
              f"{receipt['responder']['responders']} of {receipt['responder']['keys']} keys are responders.", '',
              '| Split | Responders |', '|---|---:|']
    for split, count in sorted(receipt['responder']['by_split'].items()):
        lines.append(f'| {split} | {count} |')
    summary = pd.read_csv(args.prepared/'count_qc/context_summary.tsv', sep='\t')
    lines += ['', '## Context gates', '',
              f"Contexts passing the count gate: {int(summary.test_count_gate_met.sum())} of {len(summary)}. "
              f"Study gate G4: {'met' if receipt['study_gate_G4']['met'] else 'NOT met'} "
              f"({receipt['study_gate_G4']['n_contexts']} contexts, {receipt['study_gate_G4']['n_lines']} lines, "
              f"{receipt['study_gate_G4']['n_doses']} doses).", '',
              '| Context | Eligible test compounds | Groups | Responders | Gate |', '|---|---:|---:|---:|---|']
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.context_id} | {row.eligible_test_compounds} | {row.eligible_test_groups} | '
                     f"{row.responder_test_compounds} | {'met' if row.test_count_gate_met else 'no'} |")
    lines += ['',
              '## What happens next', '',
              f"- Gate G8: **{'met' if g8 else 'not met'}**.",
              f"- Decision D7: the GPU budget is {'unchanged' if any_structure else 'cut to M1 at 3 seeds'}.",
              '- No arm has been fitted. The GPU fits are the next step and need their own approval.', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines))
    record = dict(status='CPU_ONLY_NO_ARM_FITTED', test_outcomes_computed=False,
                  evaluation_half_test_delta_computed=False, de_k=de_k,
                  split_noise_floor=floor, split_noise_floor_all_keys=floor_all, gate_G8_met=g8,
                  g8_threshold=G8_FLOOR, residual_share=share, validation_keys=len(keys),
                  validation_responders=len(responders), accuracy=accuracy, accuracy_all_keys=accuracy_all,
                  structure_contrasts=structure, any_structure_helps=any_structure,
                  responder=receipt['responder'], study_gate_G4=receipt['study_gate_G4'],
                  contexts_passing_count_gate=int(summary.test_count_gate_met.sum()))
    (args.output.parent/'cheap_report.json').write_text(json.dumps(record, indent=2, default=float)+'\n')
    print(json.dumps({k: record[k] for k in ['split_noise_floor', 'gate_G8_met', 'residual_share',
                                             'validation_responders', 'any_structure_helps', 'accuracy']},
                     indent=2, default=float))


if __name__ == '__main__':
    main()
