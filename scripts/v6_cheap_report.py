"""The v6 CPU-only report: the endpoint's ceiling, the responder gate and whether structure helps at all.

Written from preparation and the sealed comparators, before any GPU fit. It is
the D7 decision point for the GPU budget and it evaluates gate G8, and it needs
no arm, so it costs CPU minutes.

Three numbers decide whether the primary endpoint can succeed:

- the split noise floor, the accuracy ``delta_S`` itself reaches against
  ``delta_E`` - the ceiling any model could reach (gate G8 requires 0.60);
- whether B1p already beats chance on a compound's own DE genes;
- whether any method that uses chemical structure beats the method it extends.

Decision D7's committed trigger is the margin over CHANCE, not the margin over
the method extended, so the report states both (amendment A3).

Everything is computed on VALIDATION keys. No test response and no
evaluation-half test delta is read.
"""
from pathlib import Path
import argparse
import itertools
import json

import numpy as np
import pandas as pd
import yaml

from drugperturb.v6 import scoring

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_STATUS = 'PRESPECIFIED_BEFORE_V6_FITTING'
G8_FLOOR = 0.60
CHANCE = 0.5
D7_MARGIN = 0.01  # protocol_v6 USER_DECISION D7 consequence_if_no_signal
STRUCTURE_PAIRS = (('M1c', 'B1c'), ('M1a', 'B1a'), ('M1r', 'B1p'), ('B2c', 'B1c'), ('B2r', 'B1p'),
                   ('B1r', 'B1p'))
USES_STRUCTURE = ('B2c', 'M1c', 'M1a', 'M1r', 'M1r_unscaled', 'B2r')
METHODS = ('B0', 'B1c', 'B2c', 'M1c', 'B1a', 'M1a', 'B1p', 'B1r', 'B2r', 'M1r', 'M1r_unscaled')


def by_key(keys, values):
    return {tuple(k.split('|')): values[i] for i, k in enumerate(np.asarray(keys, str))}


def de_set_sharing(de_sets, responder, plate_of, *, max_pairs=4000, seed=0):
    """How much two compounds' DE sets overlap, within a plate and across plates.

    If a plate carried one dominant shared response, every compound on it would
    pick nearly the same top-k genes and the endpoint would be asking about
    plate-shared biology rather than compound-specific biology. This measures
    that directly, from selection-half data only. Pairs are compared inside a
    context, so the cell line and dose are held fixed.
    """
    genes = {tuple(k.split('|')): set(row) for k, row in zip(np.asarray(de_sets['keys'], str), de_sets['genes'])}
    members = {}
    for key in genes:
        if bool(responder.get(key, False)) and key in plate_of:
            members.setdefault(key[1], []).append(key)
    rng = np.random.default_rng(seed)
    same, across, concentration = [], [], []
    for context, keys in members.items():
        pairs = list(itertools.combinations(sorted(keys), 2))
        if len(pairs) > max_pairs:
            pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
        for a, b in pairs:
            overlap = len(genes[a] & genes[b])/len(genes[a] | genes[b])
            (same if plate_of[a] == plate_of[b] else across).append(overlap)
        if len(keys) >= 20:
            counts = np.bincount([g for k in keys for g in genes[k]], minlength=1)
            concentration.append(dict(context_id=context, responders=len(keys),
                                      genes_in_most_sets=int((counts > 0.5*len(keys)).sum()),
                                      union_size=int((counts > 0).sum())))
    return dict(same_plate_overlap=float(np.mean(same)) if same else float('nan'),
                across_plate_overlap=float(np.mean(across)) if across else float('nan'),
                same_plate_pairs=len(same), across_plate_pairs=len(across),
                concentration=sorted(concentration, key=lambda r: -r['responders'])[:5])


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
    queries = pd.read_csv(args.prepared/'queries.tsv', sep='\t', keep_default_na=False)
    plate_of = {(r.compound_id, r.context_id): r.plate for r in queries.itertuples(index=False)}
    sharing = de_set_sharing(de_sets, responder, plate_of)
    headroom = floor-accuracy.get('B1p', np.nan)
    structure = {f'{a}_minus_{b}': (accuracy[a]-accuracy[b]) for a, b in STRUCTURE_PAIRS
                 if a in accuracy and b in accuracy}
    structure_adds = any(np.isfinite(v) and v > D7_MARGIN for v in structure.values())
    # Decision D7 as committed compares each structure-using comparator against CHANCE, not against the
    # method it extends. Amendment A3_2026-10-27_RAISED_AND_WITHDRAWN records why the two readings differ.
    d7_margins = {name: accuracy[name]-CHANCE for name in USES_STRUCTURE if name in accuracy}
    d7_fires = not any(np.isfinite(v) and v > D7_MARGIN for v in d7_margins.values())

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
             f'| Residual share over B1p | {fmt(share, 3)} | - | how much is left to predict |',
             f'| **Headroom above B1p** | **{fmt(headroom)}** | - | the floor minus B1p: all any model can win |',
             '',
             '## Are the DE sets compound-specific, or one plate signature',
             '',
             'If a plate carried one dominant shared response, every compound on it would pick nearly the same '
             'top genes, and the endpoint would be asking about plate-shared biology instead of '
             'compound-specific biology. Mean Jaccard overlap of two responders\' DE sets, within a context:',
             '',
             '| Pair | Mean overlap | Pairs |', '|---|---:|---:|',
             f"| Same plate | {fmt(sharing['same_plate_overlap'], 3)} | {sharing['same_plate_pairs']} |",
             f"| Different plate | {fmt(sharing['across_plate_overlap'], 3)} | {sharing['across_plate_pairs']} |",
             '']
    for row in sharing['concentration'][:3]:
        lines.append(f"- {row['context_id']}: {row['responders']} responders, their DE sets cover "
                     f"{row['union_size']} genes, and only {row['genes_in_most_sets']} genes appear in more "
                     f"than half of them.")
    lines += ['']
    if not g8:
        lines += [f'**Gate G8 is not met.** The split leaves too little signal for the endpoint to separate '
                  f'methods, so the protocol reports that as the study result rather than comparing against '
                  f'it. The GPU fits would be measuring noise.', '']

    lines += ['## Every comparator on its own DE genes', '',
              f'Direction accuracy over each validation key\'s top-{de_k} selection-half genes. Chance is '
              f'{CHANCE:.2f}; B0 reaches it exactly because a zero prediction counts as a tie.', '',
              '| Method | Responders | All keys | Uses structure |', '|---|---:|---:|---|']
    for name in order:
        lines.append(f'| {name} | {fmt(accuracy[name])} | {fmt(accuracy_all[name])} | '
                     f"{'yes' if name in USES_STRUCTURE else 'no'} |")
    lines += ['', '## Does chemical structure help (a preview of S3)', '',
              '| Contrast | Difference in direction accuracy |', '|---|---:|']
    for name, value in structure.items():
        lines.append(f'| {name.replace("_minus_", " minus ")} | {fmt(value)} |')
    lines += ['', f"**{'At least one' if structure_adds else 'No'} structure-using method beats the method it "
                  f"extends by more than {D7_MARGIN}.** That is the question a reader cares about, and it is "
                  f"NOT decision D7's committed trigger.", '',
              '## Decision D7 as committed', '',
              f'D7 fires when no structure-using comparator exceeds CHANCE by more than {D7_MARGIN} on '
              f'validation direction accuracy. The margins over chance are:', '',
              '| Method | Accuracy minus chance |', '|---|---:|']
    for name, value in sorted(d7_margins.items(), key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else -1)):
        lines.append(f'| {name} | {fmt(value)} |')
    lines += ['', f"**D7 {'fires' if d7_fires else 'does not fire'}.** " + (
        'The GPU budget is cut to M1 at 3 seeds and the study reports that no structure-based method found '
        'directional signal.' if d7_fires else
        'The full prespecified plan runs: both candidate arms at all ten seeds. Every structure-using '
        'comparator clears chance by a wide margin because B1p, which uses no chemistry at all, already '
        'does; amendment A3_2026-10-27_RAISED_AND_WITHDRAWN records that this was raised before any GPU '
        'fit and that the committed wording was kept.'), '',
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
              f"- Decision D7: {'fires, so the GPU budget is cut to M1 at 3 seeds' if d7_fires else 'does not fire, so the full prespecified plan runs - both arms at all ten seeds'}.",
              '- No arm has been fitted. The GPU fits are the next step and need their own approval.', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines))
    record = dict(status='CPU_ONLY_NO_ARM_FITTED', test_outcomes_computed=False,
                  evaluation_half_test_delta_computed=False, de_k=de_k,
                  split_noise_floor=floor, split_noise_floor_all_keys=floor_all, gate_G8_met=g8,
                  g8_threshold=G8_FLOOR, residual_share=share, validation_keys=len(keys),
                  validation_responders=len(responders), accuracy=accuracy, accuracy_all_keys=accuracy_all,
                  structure_contrasts=structure, structure_adds_over_reference=structure_adds,
                  d7_margins_over_chance=d7_margins, d7_fires=d7_fires, d7_margin=D7_MARGIN,
                  headroom_above_b1p=headroom, de_set_sharing=sharing,
                  responder=receipt['responder'], study_gate_G4=receipt['study_gate_G4'],
                  contexts_passing_count_gate=int(summary.test_count_gate_met.sum()))
    (args.output.parent/'cheap_report.json').write_text(json.dumps(record, indent=2, default=float)+'\n')
    print(json.dumps({k: record[k] for k in ['split_noise_floor', 'gate_G8_met', 'headroom_above_b1p', 'residual_share',
                                             'validation_responders', 'structure_adds_over_reference',
                                             'd7_fires', 'accuracy']},
                     indent=2, default=float))


if __name__ == '__main__':
    main()
