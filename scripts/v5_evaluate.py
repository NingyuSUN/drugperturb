"""Score sealed v5 predictions against measured test deltas (protocol_v5 evaluation); test truth is first computed here.

Runs only after every arm, every comparator and the primary-arm selection are
sealed. Verifies each seal and that the selection names the sealed arms, then
computes test truth from the extraction unit sums for the count-eligible test
pairs. Endpoints follow protocol_v5 evaluation: P1 (each primary arm minus B1a
and M1a, at the Bonferroni level of the selection), key secondary S1-S4, and the
other secondary analyses, pooled over gate-passing contexts with one
scaffold-group draw shared by every context (10,000 draws, seed 2026).
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_evaluate import by_key, load_sealed  # noqa: E402
from drugperturb.evaluation import compound_metrics, summarize_compound_metrics  # noqa: E402
from drugperturb.v3.runner import sha256  # noqa: E402
from drugperturb.v4 import diagnostics  # noqa: E402
from drugperturb.v5 import prepare  # noqa: E402
from drugperturb.v5.selection import ARMS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMPARATORS = ['B0', 'B1c', 'B2c', 'M1c', 'B1a', 'M1a', 'B1p']
DRAWS, SEED = 10000, 2026


def read(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'replicate': str})


def pooled_bootstrap(differences, *, level=0.95, draws=DRAWS, seed=SEED):
    """Mean over contexts of context-mean compound differences; one scaffold-group draw shared by all contexts."""
    groups = sorted({g for table in differences.values() for g in table.scaffold_group})
    position = {g: i for i, g in enumerate(groups)}
    sampled = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    weights = np.stack([np.bincount(row, minlength=len(groups)) for row in sampled])
    estimates, point = np.zeros(draws), []
    for table in differences.values():
        index = table.scaffold_group.map(position).to_numpy()
        values = table.difference.to_numpy()
        w = weights[:, index]
        n = w.sum(axis=1)
        estimates += np.divide(w @ values, n, out=np.full(draws, np.nan), where=n > 0)
        point.append(values.mean())
    estimates /= len(differences)
    valid = np.isfinite(estimates)
    tail = (1-level)/2
    lower, upper = np.quantile(estimates[valid], [tail, 1-tail])
    return dict(point_estimate=float(np.mean(point)), ci_lower=float(lower), ci_upper=float(upper), level=level,
                contexts=len(differences), n_groups=len(groups), draws=draws, valid_draws=int(valid.sum()), seed=seed,
                negative_favors='candidate')


def locate(roots, name):
    found = [r for r in roots if (r/name).exists()]
    if len(found) != 1:
        raise ValueError(f'{name}: expected exactly one sealed copy, found {len(found)}')
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction-root', type=Path, required=True, help='directory holding the part_* outputs')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sealed-root', type=Path, nargs='+', required=True,
                        help='directories holding the arm seals, the selection and the comparator seal')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, default=ROOT/'manifests/v5')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v5.yaml')
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != 'PRESPECIFIED_BEFORE_V5_FITTING':
        raise ValueError('v5 protocol is not prespecified')
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    arms = {}
    for arm in ARMS:
        arrays, seal = load_sealed(locate(args.sealed_root, f'{arm}_v5_seal.json'), arm, 'v5')
        arms[arm] = dict(contrast=by_key(arrays['test_keys'], arrays['test_contrast']),
                         anchored=by_key(arrays['test_keys'], arrays['test_anchored']), seal=seal)
    base, base_seal = load_sealed(locate(args.sealed_root, 'baselines_v5_seal.json'), 'baselines', 'v5')
    comparators = {m: by_key(base['keys'], base[m]) for m in COMPARATORS}
    selection = json.loads((locate(args.sealed_root, 'selection_v5.json')/'selection_v5.json').read_text())
    if selection['status'] != 'SELECTED_ON_VALIDATION_BEFORE_TEST_SCORING' or any(
            selection['arms'][a]['sha256'] != arms[a]['seal']['predictions']['sha256']
            for a in protocol['primary_arm_selection']['candidates']):
        raise ValueError('selection record does not match the sealed arm predictions')
    primary, level = selection['primary_arms'], float(selection['interval_level'])

    # Test truth, first computed here, for count-eligible test pairs only.
    receipt = json.loads((args.prepared/'prepare_receipt.json').read_text())
    parts = sorted(p for p in args.extraction_root.rglob('part_*') if (p/'part_receipt.json').exists())
    if len(parts) != 1:
        raise ValueError('expected one extraction part')
    if sha256(parts[0]/'unit_counts.npz') != receipt['inputs'][f'{parts[0].name}/unit_counts.npz']:
        raise ValueError('extraction unit counts differ from those the preparation read')
    with np.load(parts[0]/'unit_counts.npz', allow_pickle=False) as z:
        counts = sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
        library, n_cells = z['library'], z['n_cells']
    units = read(parts[0]/'units.tsv')
    compounds, contexts = read(args.manifests/'compounds.tsv'), read(args.manifests/'contexts.tsv')
    pairs = prepare.response_pairs(units, compounds, contexts)
    ids, matrix, unit_library, _ = prepare.with_pooled_vehicles(units, counts, library, n_cells)
    del counts
    row_of = {u: i for i, u in enumerate(ids)}
    frozen = read(args.prepared/'frozen_genes.tsv').source_index.to_numpy()
    compound_qc = read(args.prepared/'count_qc/compound_qc.tsv')
    summary_qc = read(args.prepared/'count_qc/context_summary.tsv').set_index('context_id')
    eligible = compound_qc[compound_qc.split.eq('test') & compound_qc.eligible.astype(str).eq('True')]
    keys = sorted(map(tuple, eligible[['compound_id', 'context_id']].to_numpy()))
    table, delta = prepare.replicate_deltas(pairs, keys, row_of, matrix, unit_library, frozen)
    truth = dict(zip(keys, prepare.key_means(table, delta, keys)))
    plates = table.groupby(['compound_id', 'context_id']).plate.agg(lambda s: sorted(set(s))).to_dict()
    del matrix

    gate_contexts = [c for c in contexts.context_id if str(summary_qc.loc[c, 'test_count_gate_met']) == 'True']
    methods = {**{m: comparators[m] for m in COMPARATORS}, **{a: arms[a]['contrast'] for a in ARMS},
               **{f'{a}_anchored': arms[a]['anchored'] for a in ARMS}}
    rows, per_compound, test_diagnostics, compound_rmse = [], [], [], {}
    for context in contexts.context_id:
        cohort = eligible[eligible.context_id.eq(context)].sort_values('compound_id')
        if cohort.empty:
            continue
        cids, groups = cohort.compound_id.tolist(), cohort.scaffold_group.tolist()
        ckeys = [(c, context) for c in cids]
        t = np.vstack([truth[k] for k in ckeys])
        for name, predicted in methods.items():
            frame = compound_metrics(t, np.vstack([predicted[k] for k in ckeys]), cids, groups)
            frame['method'], frame['context_id'] = name, context
            per_compound.append(frame)
            compound_rmse[(name, context)] = frame[['compound_id', 'scaffold_group', 'rmse']]
            rows.append(dict(context_id=context, gate_met=context in gate_contexts, method=name,
                             **summarize_compound_metrics(frame)))
        truth_map = dict(zip(ckeys, t))
        reference = {context: comparators['B1c'][ckeys[0]]}
        b1a = comparators['B1a']
        for name in [*ARMS, 'M1c', 'B1a', 'M1a', 'B1p']:
            predicted = {k: methods[name][k] for k in ckeys}
            s = diagnostics.summarize(predicted, truth_map, reference)
            s['r_vs_B1a'] = float(np.mean([diagnostics.pearson(predicted[k]-b1a[k], truth_map[k]-b1a[k]) for k in ckeys]))
            test_diagnostics.append(dict(context_id=context, method=name, **s))

    def differences(candidate, baseline, chosen_contexts, drop=None):
        result = {}
        for context in chosen_contexts:
            a, b = compound_rmse[(candidate, context)], compound_rmse[(baseline, context)]
            frame = pd.DataFrame(dict(compound_id=a.compound_id, scaffold_group=a.scaffold_group,
                                      difference=a.rmse.to_numpy()-b.rmse.to_numpy()))
            if drop is not None:
                frame = frame[[drop not in plates[(c, context)] for c in frame.compound_id]]
            if len(frame):
                result[context] = frame
        return result

    def pooled(candidate, baseline, chosen_contexts=gate_contexts, level=0.95, drop=None):
        d = differences(candidate, baseline, chosen_contexts, drop)
        return pooled_bootstrap(d, level=level) if d else None

    endpoints = dict(primary_arms=primary, interval_level=level, near_tie=selection['near_tie'],
                     selection_any_passing=selection['any_passing'], gate_passing_contexts=gate_contexts,
                     study_gate_G4=receipt['study_gate_G4'])
    confirmatory = receipt['study_gate_G4']['met']
    p1 = {}
    for arm in primary:
        estimates = {b: pooled(arm, b, level=level) for b in ['B1a', 'M1a']}
        p1[arm] = dict(estimates=estimates, supported=bool(all(v and v['ci_upper'] < 0 for v in estimates.values())))
    endpoints['P1_beyond_vehicle_anchoring'] = dict(per_arm=p1, supported=bool(confirmatory and any(
        v['supported'] for v in p1.values())), confirmatory=confirmatory)
    endpoints['S1_non_collapse'] = {a: pooled(a, 'B0') for a in primary}
    endpoints['S2_delta_baselines'] = {a: {b: pooled(a, b) for b in ['B1c', 'M1c']} for a in primary}
    endpoints['S3_structure_beyond_anchoring'] = pooled('M1a', 'B1a')
    endpoints['S4_anchoring_gain'] = pooled('B1a', 'B1c')
    secondary = []
    doses = sorted(contexts.dose_nM.unique())
    lines = sorted(contexts.cell_line_id.unique())
    subsets = {'pooled': gate_contexts,
               **{f'dose_{d}nM': [c for c in gate_contexts if c.endswith(f'_{d}nM')] for d in doses},
               **{f'line_{l}': [c for c in gate_contexts if c.rsplit('_', 1)[0] == l] for l in lines}}
    for candidate in [*ARMS, 'B1p']:
        for baseline in COMPARATORS:
            if candidate == baseline:
                continue
            for subset, chosen in subsets.items():
                if subset != 'pooled' and candidate == 'B1p':
                    continue
                estimate = pooled(candidate, baseline, chosen)
                if estimate:
                    secondary.append(dict(candidate=candidate, baseline=baseline, subset=subset, **estimate))
    for arm in primary:
        estimate = pooled(arm, 'B1p')
        secondary.append(dict(candidate=arm, baseline='B1p', subset='pooled_primary', **estimate))
    contrasts = []
    for context in gate_contexts:
        for candidate in ARMS:
            for baseline in COMPARATORS:
                estimate = pooled(candidate, baseline, [context])
                contrasts.append(dict(context_id=context, candidate=candidate, baseline=baseline,
                                      primary_arm=candidate in primary, **estimate))
    all_plates = sorted({p for v in plates.values() for p in v}, key=lambda p: int(p[5:]))
    leave_one_plate_out = [dict(arm=arm, baseline=b, dropped_plate=p, **(pooled(arm, b, level=level, drop=p) or {}))
                           for arm in primary for b in ['B1a', 'M1a'] for p in all_plates]
    pd.concat(per_compound, ignore_index=True).to_csv(out/'per_compound.tsv', sep='\t', index=False)
    pd.DataFrame(rows).to_csv(out/'summary.tsv', sep='\t', index=False)
    pd.DataFrame(secondary).to_csv(out/'secondary.tsv', sep='\t', index=False)
    pd.DataFrame(contrasts).to_csv(out/'contrasts_per_context.tsv', sep='\t', index=False)
    pd.DataFrame(leave_one_plate_out).to_csv(out/'leave_one_plate_out.tsv', sep='\t', index=False)
    pd.DataFrame(test_diagnostics).to_csv(out/'test_diagnostics.tsv', sep='\t', index=False)
    result = dict(endpoints=endpoints, a0_minus_b0=pooled('A0_v3_reference', 'B0'), selection=selection,
                  seals={a: arms[a]['seal']['predictions']['sha256'] for a in ARMS},
                  comparator_seal=base_seal['predictions']['sha256'], test_truth_first_computed_here=True,
                  eligible_test_pairs=len(keys), bootstrap=dict(draws=DRAWS, seed=SEED, unit='scaffold_group'))
    (out/'summary.json').write_text(json.dumps(diagnostics.finite(result), indent=2, allow_nan=False)+'\n')
    print(json.dumps(diagnostics.finite(endpoints), indent=2))


if __name__ == '__main__':
    main()
