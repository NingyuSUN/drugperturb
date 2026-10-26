"""Score the sealed v6 predictions on the evaluation half; the only place test truth is computed.

Verifies every seal, recomputes the extraction anchors, computes the
evaluation-half test delta for the first time, and scores the primary endpoint
(direction accuracy over each key's own DE genes, from the selection half) and
every prespecified secondary.

Higher is better for direction accuracy, so a POSITIVE difference favours the
candidate and P1 is supported when the lower bound is above zero - the opposite
sign convention to v1-v5, whose metric was an error.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drugperturb.v4 import diagnostics  # noqa: E402
from drugperturb.v6 import endpoint, extract, prepare, scoring  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_STATUS = 'PRESPECIFIED_BEFORE_V6_FITTING'
DELTA_COMPARATORS = ['B0', 'B1c', 'B2c', 'M1c', 'B1a', 'M1a', 'B1p']
RESIDUAL_COMPARATORS = ['B1r', 'B2r', 'M1r']
DRAWS, SEED = 10000, 2026
CHANCE = 0.5


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'replicate': str})


def by_key(keys, values):
    return {tuple(k.split('|')): values[i] for i, k in enumerate(np.asarray(keys, str))}


def pooled_bootstrap(differences, *, level=0.95, draws=DRAWS, seed=SEED):
    """Mean over contexts of context-mean compound differences; one scaffold-group draw shared by all contexts."""
    groups = sorted({g for table in differences.values() for g in table.scaffold_group})
    position = {g: i for i, g in enumerate(groups)}
    sampled = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    weights = np.stack([np.bincount(row, minlength=len(groups)) for row in sampled])
    estimates, point = np.zeros(draws), []
    for tbl in differences.values():
        index = tbl.scaffold_group.map(position).to_numpy()
        values = tbl.difference.to_numpy()
        w = weights[:, index]
        n = w.sum(axis=1)
        estimates += np.divide(w@values, n, out=np.full(draws, np.nan), where=n > 0)
        point.append(values.mean())
    estimates /= len(differences)
    valid = np.isfinite(estimates)
    tail = (1-level)/2
    lower, upper = np.quantile(estimates[valid], [tail, 1-tail])
    return dict(point_estimate=float(np.mean(point)), ci_lower=float(lower), ci_upper=float(upper), level=level,
                contexts=len(differences), n_groups=len(groups), draws=draws, valid_draws=int(valid.sum()),
                seed=seed, positive_favors='candidate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction-root', type=Path, nargs='+', required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sealed', type=Path, nargs='+', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, default=ROOT/'manifests/v6')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v6.yaml')
    args = parser.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != PROTOCOL_STATUS:
        raise ValueError('v6 protocol is not prespecified')
    de_k = int(protocol['response_endpoint']['de_gene_set']['k'])

    def locate(name):
        found = [r/name for r in args.sealed if (r/name).exists()]
        if len(found) != 1:
            raise ValueError(f'{name}: expected exactly one sealed copy, found {len(found)}')
        return found[0]

    receipt = json.loads((args.prepared/'prepare_receipt.json').read_text())
    if receipt['evaluation_half_test_delta_computed'] is not False:
        raise ValueError('preparation already computed an evaluation-half test delta')
    selection = json.loads(locate('selection_v6.json').read_text())
    primary_arms = selection['primary_arms']
    level = float(selection['interval_level'])

    # Verify every seal before any test delta is computed.
    seals, methods = {}, {}
    for name in [*selection['arms'], 'baselines']:
        seal = json.loads(locate(f'{name}_v6_seal.json').read_text())
        archive = locate(seal['predictions']['path'])
        if sha256(archive) != seal['predictions']['sha256']:
            raise ValueError(f'{name}: sealed predictions do not match their seal')
        if seal['test_outcomes_loaded'] is not False:
            raise ValueError(f'{name} does not certify a test-blind seal')
        seals[name] = seal
        with np.load(archive, allow_pickle=False) as z:
            if name == 'baselines':
                for method in DELTA_COMPARATORS:
                    methods[method] = by_key(z['keys'], z[method])
                for method in RESIDUAL_COMPARATORS:
                    methods[method] = by_key(z[f'{method}_keys'], z[method])
                methods['M1r_unscaled'] = by_key(z[f'M1r_keys'], z['M1r_unscaled'])
            else:
                methods[name] = by_key(z['test_keys'], z['test_contrast'])

    # Rebuild the units and compute the evaluation-half test truth, for the first time.
    parts = sorted({p for root in args.extraction_root for p in root.rglob('part_*')
                    if (p/'part_receipt.json').exists()})
    units = read_tsv(parts[0]/'units.tsv')
    halves = read_tsv(parts[0]/'half_units.tsv')
    expected = read_tsv(parts[0]/'expected_half_cells.tsv')
    sel = read_tsv(parts[0]/'model_selection.tsv')
    shards = read_tsv(args.manifests/'expression_source.tsv')
    cap = int(protocol['model_cells']['cap_per_well_and_line'])
    merged = extract.merge_parts(parts, halves, expected_cells=expected.expected_cells.to_numpy(np.int64),
                                 model_barcodes=sel.barcode.to_numpy(str), cap=cap,
                                 shard_manifest=shards[shards.role.eq('expression_shard')])
    frozen = read_tsv(args.prepared/'frozen_genes.tsv')
    columns = frozen.source_index.to_numpy()
    half = prepare.half_tables(merged['half_ids'], merged['half_counts'], merged['half_library'],
                               merged['half_cells'], units)
    queries = read_tsv(args.prepared/'queries.tsv')
    pairs = queries.rename(columns=dict(treatment_Combo='treatment_unit', vehicle_Combo='vehicle_unit'))
    tests = pairs[pairs.split.eq('test') & pairs.count_eligible]
    keys = sorted(set(map(tuple, tests[['compound_id', 'context_id']].to_numpy())))
    truth = by_key([f'{c}|{x}' for c, x in keys], prepare.half_delta(pairs, keys, half['E'], columns))

    with np.load(args.prepared/'endpoint/de_sets.npz', allow_pickle=False) as z:
        de_sets = {k: z[k] for k in ['keys', 'genes']}
        responder = by_key(z['keys'], z['responder'])
        selection_delta = by_key(z['keys'], z['selection_delta'])
        if int(z['k']) != de_k:
            raise ValueError('the prepared DE sets use a different k from the protocol')

    # Anchors, including the two the endpoint needs.
    zero = {k: np.zeros(len(columns)) for k in keys}
    anchors = dict(merged['checks'])
    anchors['zero_change_scores_chance'] = bool(np.isclose(
        scoring.mean_accuracy(zero, truth, de_sets, keys), CHANCE))
    anchors['truth_fed_as_prediction_scores_one'] = bool(np.isclose(
        scoring.mean_accuracy(truth, truth, de_sets, keys), 1.0))
    anchors['b0_equals_zero'] = bool(all(np.abs(methods['B0'][k]).max() == 0 for k in keys if k in methods['B0']))
    failed = [k for k, v in anchors.items() if v is False]
    if failed:
        raise ValueError(f'evaluation anchors failed: {failed}')

    summary = read_tsv(args.prepared/'count_qc/context_summary.tsv')
    gate_contexts = sorted(summary.loc[summary.test_count_gate_met, 'context_id'])
    groups = dict(zip(queries.compound_id, queries.scaffold_group))
    plates = {(c, x): sorted(set(g.plate)) for (c, x), g in pairs.groupby(['compound_id', 'context_id'])}
    scored = {}
    for name, prediction in methods.items():
        present = [k for k in keys if k in prediction]
        accuracy, dropped = scoring.accuracy_by_key(prediction, truth, de_sets, present)
        rmse = scoring.restricted_rmse_by_key(prediction, truth, de_sets, present)
        scored[name] = dict(accuracy=accuracy, rmse=rmse, dropped_genes=dropped)

    rows = []
    for name, values in scored.items():
        for key, accuracy in values['accuracy'].items():
            rows.append(dict(method=name, compound_id=key[0], context_id=key[1], scaffold_group=groups[key[0]],
                             responder=bool(responder.get(key, False)), direction_accuracy=accuracy,
                             de_rmse=values['rmse'][key]))
    per_compound = pd.DataFrame(rows)
    per_compound.to_csv(out/'per_compound.tsv', sep='\t', index=False)

    def differences(candidate, baseline, contexts, *, responders_only=True, metric='direction_accuracy',
                    drop_plate=None, drop_line=None):
        result = {}
        frame = per_compound[per_compound.responder] if responders_only else per_compound
        for context in contexts:
            if drop_line and context.startswith(drop_line):
                continue
            a = frame[frame.method.eq(candidate) & frame.context_id.eq(context)].set_index('compound_id')
            b = frame[frame.method.eq(baseline) & frame.context_id.eq(context)].set_index('compound_id')
            shared = sorted(set(a.index) & set(b.index))
            if drop_plate is not None:
                shared = [c for c in shared if drop_plate not in plates[(c, context)]]
            if not shared:
                continue
            result[context] = pd.DataFrame(dict(
                compound_id=shared, scaffold_group=[groups[c] for c in shared],
                difference=a.loc[shared, metric].to_numpy()-b.loc[shared, metric].to_numpy()))
        return result

    def pooled(candidate, baseline, contexts=None, level=0.95, **kwargs):
        d = differences(candidate, baseline, contexts or gate_contexts, **kwargs)
        return pooled_bootstrap(d, level=level) if d else None

    confirmatory = bool(receipt['study_gate_G4']['met'])
    floor = seals[primary_arms[0]]['endpoint']['split_noise_floor']
    g8_met = bool(np.isfinite(floor) and floor >= 0.60)
    endpoints = dict(primary_arms=primary_arms, interval_level=level, near_tie=selection['near_tie'],
                     selection_any_passing=selection['any_passing'], gate_passing_contexts=gate_contexts,
                     study_gate_G4=receipt['study_gate_G4'], split_noise_floor=floor, gate_G8_met=g8_met)
    p1 = {}
    for arm in primary_arms:
        estimates = {b: pooled(arm, b, level=level) for b in ['B1p', 'M1a']}
        p1[arm] = dict(estimates=estimates,
                       supported=bool(all(v and v['ci_lower'] > 0 for v in estimates.values())))
    endpoints['P1_direction_beyond_simple_baselines'] = dict(
        per_arm=p1, supported=bool(confirmatory and g8_met and any(v['supported'] for v in p1.values())),
        confirmatory=confirmatory, note='supported needs every lower bound above zero, for one primary arm')
    arm = primary_arms[0]
    endpoints['S1_above_chance'] = dict(vs_B0=pooled(arm, 'B0'),
                                        chance=CHANCE,
                                        mean_accuracy=float(np.mean(list(scored[arm]['accuracy'].values()))))
    endpoints['S2_all_keys'] = {b: pooled(arm, b, responders_only=False) for b in ['B1p', 'M1a']}
    endpoints['S3_structure_baselines'] = {f'{a}_minus_{b}': pooled(a, b)
                                           for a, b in [('M1c', 'B1c'), ('M1a', 'B1a'), ('M1r', 'B1p')]}
    endpoints['S4_residual_arm'] = pooled('M2_pseudobulk_residual', 'M1_pseudobulk_raw') \
        if {'M1_pseudobulk_raw', 'M2_pseudobulk_residual'} <= set(methods) else None
    endpoints['S5_rmse_on_de_genes'] = {f'{arm}_minus_B1p': pooled(arm, 'B1p', metric='de_rmse'),
                                        'note': 'an error, so a NEGATIVE difference favours the candidate'}
    endpoints['S6_replicates_v5'] = {f'{a}_minus_{b}': pooled(a, b, metric='de_rmse')
                                     for a, b in [('B1p', 'B1a'), ('B1a', 'B1c')]}
    secondary = dict(
        every_pair={f'{a}_minus_{b}': pooled(a, b) for a in [*selection['arms'], 'M1r']
                    for b in ['B1p', 'B1c', 'B0'] if a in methods},
        unscaled_M1r=pooled('M1r_unscaled', 'B1p') if 'M1r_unscaled' in methods else None,
        without_small_vehicle_line={b: pooled(arm, b, drop_line='CVCL_1577') for b in ['B1p', 'M1a']},
        leave_one_plate_out={p: {b: pooled(arm, b, drop_plate=p) for b in ['B1p']}
                             for p in sorted({p for v in plates.values() for p in v})},
        de_overlap={name: float(np.mean(endpoint.de_overlap(
            scoring.stack(methods[name], [k for k in keys if k in methods[name]]),
            scoring.gene_sets(de_sets, [k for k in keys if k in methods[name]]))))
            for name in [*selection['arms'], 'B1p', 'M1a'] if name in methods},
        per_context=per_compound.groupby(['method', 'context_id'], as_index=False)
        .agg(direction_accuracy=('direction_accuracy', 'mean'), de_rmse=('de_rmse', 'mean'),
             compounds=('compound_id', 'nunique'), responders=('responder', 'sum')).to_dict('records'))
    for label, sensitivity in {'k20': 20, 'k100': 100}.items():
        k = min(sensitivity, len(frozen))
        if k == de_k:
            continue  # the primary k; a sensitivity at the same k would repeat the primary estimate
        alt = endpoint.de_gene_sets(scoring.stack(selection_delta, keys), k=k)
        alt_sets = dict(keys=np.asarray([f'{c}|{x}' for c, x in keys]), genes=alt)
        secondary[f'accuracy_{label}'] = dict(k=k, accuracy={
            name: scoring.mean_accuracy(methods[name], truth, alt_sets,
                                        [key for key in keys if key in methods[name]
                                         and bool(responder.get(key, False))])
            for name in [*selection['arms'], 'B1p', 'M1a', 'B0'] if name in methods})

    overall = pd.DataFrame([dict(method=name,
                                 direction_accuracy=float(np.mean(list(v['accuracy'].values()))) if v['accuracy'] else np.nan,
                                 responder_direction_accuracy=float(np.mean(
                                     [a for k, a in v['accuracy'].items() if responder.get(k, False)])
                                 ) if any(responder.get(k, False) for k in v['accuracy']) else np.nan,
                                 de_rmse=float(np.mean(list(v['rmse'].values()))) if v['rmse'] else np.nan,
                                 keys=len(v['accuracy']))
                            for name, v in scored.items()]).sort_values('responder_direction_accuracy',
                                                                        ascending=False)
    overall.to_csv(out/'summary.tsv', sep='\t', index=False)
    record = dict(status='TEST_SCORED', test_outcomes_computed=True, protocol=sha256(args.protocol),
                  de_k=de_k, anchors=anchors, endpoints=endpoints, secondary=secondary,
                  eligible_test_keys=len(keys), responder_test_keys=int(sum(bool(responder.get(k, False))
                                                                           for k in keys)),
                  bootstrap=dict(draws=DRAWS, seed=SEED, unit='scaffold_group'),
                  seals={k: v['predictions']['sha256'] for k, v in seals.items()},
                  outputs={p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'evaluation_v6.json').write_text(json.dumps(diagnostics.finite(record), indent=2)+'\n')
    print(json.dumps(diagnostics.finite(dict(
        primary_arms=primary_arms, P1=endpoints['P1_direction_beyond_simple_baselines']['supported'],
        split_noise_floor=floor, responder_test_keys=record['responder_test_keys'],
        best=overall.iloc[0].to_dict())), indent=2))


if __name__ == '__main__':
    main()
