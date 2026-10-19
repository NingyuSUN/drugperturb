"""Score sealed v3 predictions against measured test deltas (Q1 and Q2).

Runs only after every prediction archive is sealed. Before scoring new methods,
the recomputed test truth is anchored to published v1/v2 per-compound results:
zero-change RMSE (a function of truth alone) and the frozen v1 predictions must
reproduce the committed tables.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import yaml

from drugperturb.evaluation import compound_metrics, paired_scaffold_bootstrap, summarize_compound_metrics
from drugperturb.transfer_v2 import alternate_controls, measured_responses, required_wells
from drugperturb.v3.runner import PROTOCOL_CONTEXTS, sha256

ROOT = Path(__file__).resolve().parents[1]
V1_METHODS = ['B0', 'B1', 'B2', 'M1', 'M2', 'M3']


def read(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'dose_pattern': str, 'replicate': str})


def load_sealed(root, method, protocol):
    seal = json.loads((root/f'{method}_{protocol}_seal.json').read_text())
    archive = root/seal['predictions']['path']
    if (seal['status'] != 'SEALED_BEFORE_TEST_SCORING' or seal['test_outcomes_loaded'] is not False
            or sha256(archive) != seal['predictions']['sha256']):
        raise ValueError(f'{method} {protocol}: seal verification failed')
    with np.load(archive, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, seal


def by_key(keys, values):
    return {tuple(k.split('|')): values[i] for i, k in enumerate(keys)}


def frozen_v1():
    with np.load(ROOT/'reports/v2/predictions/gene_predictions.npz', allow_pickle=False) as z:
        ids = z['ids'].tolist()
        return {m: {cid: z[m][i] for i, cid in enumerate(ids)} for m in V1_METHODS}


def pooled_bootstrap(differences, draws=2000, seed=2026):
    """Mean over contexts of context-mean differences; one group draw shared by all contexts."""
    groups = sorted({g for table in differences.values() for g in table.scaffold_group})
    sampled = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    weights = np.stack([np.bincount(row, minlength=len(groups)) for row in sampled])
    estimates = np.zeros(draws)
    point = []
    for table in differences.values():
        position = {g: i for i, g in enumerate(groups)}
        index = table.scaffold_group.map(position).to_numpy()
        values = table.difference.to_numpy()
        w = weights[:, index]
        n = w.sum(axis=1)
        estimates += np.divide(w @ values, n, out=np.full(draws, np.nan), where=n > 0)
        point.append(values.mean())
    estimates /= len(differences)
    valid = np.isfinite(estimates)
    lower, upper = np.quantile(estimates[valid], [0.025, 0.975])
    return dict(point_estimate=float(np.mean(point)), ci_lower=float(lower), ci_upper=float(upper),
                contexts=list(differences), n_groups=len(groups), draws=draws, valid_draws=int(valid.sum()),
                seed=seed, negative_favors='candidate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sealed-root', type=Path, action='append', default=[])
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--anchors-only', action='store_true',
                        help='Check recomputed truth against published v1/v2 results in already-opened contexts only; '
                             'reads no new predictions and computes no never-scored-context truth')
    args = parser.parse_args()
    anchors_only = args.anchors_only
    protocol = yaml.safe_load((ROOT/'configs/protocol_v3.yaml').read_text())
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    locate = lambda name: next(r for r in args.sealed_root if (r/name).exists())
    sealed, new_methods = {}, [] if anchors_only else ['chemcpa', 'biolord']
    for method in new_methods:
        for q in ['Q1', 'Q2']:
            arrays, seal = load_sealed(locate(f'{method}_{q}_seal.json'), method, q)
            sealed[(method, q)] = dict(contrast=by_key(arrays['test_keys'], arrays['test_contrast']),
                                       anchored=by_key(arrays['test_keys'], arrays['test_anchored']), seal=seal)
    if not anchors_only:
        baselines, _ = load_sealed(locate('context_baselines_Q2_seal.json'), 'context_baselines', 'Q2')
        context_simple = {m: by_key(baselines['keys'], baselines[m]) for m in ['B0', 'B1c', 'B2c', 'M1c']}
    v1 = frozen_v1()

    wells = read(ROOT/'manifests/v2/well_mapping.tsv')
    compounds = read(args.prepared/'count_qc/compound_qualification.tsv')
    qualified = read(args.prepared/'count_qc/well_qualification.tsv')
    summary = read(args.prepared/'count_qc/context_summary.tsv').set_index('context_id')
    gene_ids = read(ROOT/'manifests/selected_genes.tsv').gene_id.tolist()
    fresh = protocol['known_test_exposure']['never_scored_before_v3']
    rows, contrasts, anchors, per_compound = [], [], [], []
    differences = {('chemcpa', 'B0'): {}, ('chemcpa', 'M1c'): {}, ('biolord', 'B0'): {}, ('biolord', 'M1c'): {}}
    for context in PROTOCOL_CONTEXTS['Q2']:
        cell = context.split('_')[0]
        cohort = compounds[compounds.context_id.eq(context) & compounds.split.eq('test')
                           & compounds.eligible.astype(str).eq('True')].sort_values('compound_id')
        ids, groups = cohort.compound_id.tolist(), cohort.scaffold_group.tolist()
        gate = str(summary.loc[context, 'frozen_test_count_gate_met']) == 'True'
        if not ids or (anchors_only and context in fresh):
            continue
        ctx_wells = wells[wells.context_id.eq(context)]
        ctx_q = qualified[qualified.context_id.eq(context)]
        qc = read(args.extraction_root/f'pseudobulk/{cell}_24h/well_qc.tsv')
        pairs = alternate_controls(ctx_wells, ctx_q, qc, ids)
        keys = required_wells(ctx_q, pairs, ids)
        counts = pd.read_csv(args.extraction_root/f'pseudobulk/{cell}_24h/pseudobulk_counts.tsv.gz', sep='\t',
                             usecols=['gene_id', *keys], index_col='gene_id')
        truth, _ = measured_responses(counts, ctx_q, pairs, ids, gene_ids)
        methods = {'B0': np.zeros_like(truth)}
        if context == 'A549_24h_1000nM':
            for m in V1_METHODS[1:]:
                methods[m] = np.vstack([v1[m][c] for c in ids])
            for m in new_methods:
                methods[f'{m}_Q1'] = np.vstack([sealed[(m, 'Q1')]['contrast'][(c, context)] for c in ids])
                methods[f'{m}_Q1_anchored'] = np.vstack([sealed[(m, 'Q1')]['anchored'][(c, context)] for c in ids])
        for m in [] if anchors_only else ['B1c', 'B2c', 'M1c']:
            methods[m] = np.vstack([context_simple[m][(c, context)] for c in ids])
        for m in ['M1', 'M3']:
            methods[f'v1_transport_{m}'] = np.vstack([v1[m][c] for c in ids])
        for m in new_methods:
            methods[f'{m}_Q2'] = np.vstack([sealed[(m, 'Q2')]['contrast'][(c, context)] for c in ids])
            methods[f'{m}_Q2_anchored'] = np.vstack([sealed[(m, 'Q2')]['anchored'][(c, context)] for c in ids])
        status = 'fresh' if context in fresh else 'opened'
        tables = {}
        for name, prediction in methods.items():
            table = compound_metrics(truth, prediction, ids, groups)
            table['method'], table['context_id'], table['exposure'] = name, context, status
            tables[name] = table
            per_compound.append(table)
            s = summarize_compound_metrics(table)
            rows.append(dict(context_id=context, exposure=status, gate_met=gate, method=name, **s))
        pairs_to_test = [('chemcpa_Q2', 'B0'), ('chemcpa_Q2', 'M1c'), ('biolord_Q2', 'B0'), ('biolord_Q2', 'M1c'),
                         ('chemcpa_Q2', 'v1_transport_M1'), ('biolord_Q2', 'v1_transport_M1')]
        if context == 'A549_24h_1000nM':
            pairs_to_test += [(f'{m}_Q1', c) for m in ['chemcpa', 'biolord'] for c in ['M1', 'B1', 'B0']]
        for candidate, baseline in pairs_to_test:
            if not gate or anchors_only:
                continue
            b = paired_scaffold_bootstrap(truth, methods[candidate], methods[baseline], ids, groups)
            contrasts.append(dict(context_id=context, exposure=status, candidate=candidate, baseline=baseline,
                                  **{k: b[k] for k in ['point_estimate', 'ci_lower', 'ci_upper', 'n_compounds', 'n_groups']}))
        if status == 'fresh' and gate and not anchors_only:
            for (m, base) in differences:
                frame = pd.DataFrame(dict(scaffold_group=groups, difference=tables[f'{m}_Q2'].rmse.to_numpy()
                                          - tables[base].rmse.to_numpy()))
                differences[(m, base)][context] = frame
        published = None
        if context == 'A549_24h_1000nM':
            published = read(ROOT/'reports/final_test/metrics_by_compound.tsv')
            published = published[published.method.isin(['B0', 'M1', 'M3'])]
        elif status == 'opened':
            published = read(ROOT/'reports/v2/transfer/per_compound.tsv')
            published = published[published.context_id.eq(context) & published.analysis.eq('A')
                                  & published.comparison.eq('strict_full') & published.method.isin(['B0', 'M1', 'M3'])]
        if published is not None and len(published):
            for method, group in published.groupby('method'):
                mine = tables['B0' if method == 'B0' else (method if context == 'A549_24h_1000nM' else f'v1_transport_{method}')]
                merged = group.merge(mine[['compound_id', 'rmse']], on='compound_id', suffixes=('_published', ''))
                anchors.append(dict(context_id=context, method=method, compounds=len(merged),
                                    max_abs_rmse_difference=float(np.max(np.abs(merged.rmse_published.astype(float)-merged.rmse)))))
    anchor_table = pd.DataFrame(anchors)
    if anchor_table.empty or (anchor_table.max_abs_rmse_difference > 1e-9).any():
        raise ValueError(f'recomputed truth does not reproduce published results:\n{anchor_table}')
    if anchors_only:
        anchor_table.to_csv(out/'truth_anchors.tsv', sep='\t', index=False)
        print(anchor_table.to_string(index=False))
        return
    pooled = {f'{m}_Q2_minus_{b}': pooled_bootstrap(d) for (m, b), d in differences.items() if d}
    pd.concat(per_compound, ignore_index=True).to_csv(out/'per_compound.tsv', sep='\t', index=False)
    pd.DataFrame(rows).to_csv(out/'summary.tsv', sep='\t', index=False)
    pd.DataFrame(contrasts).to_csv(out/'contrasts.tsv', sep='\t', index=False)
    anchor_table.to_csv(out/'truth_anchors.tsv', sep='\t', index=False)
    selections = {f'{m}_{q}': dict(selected=v['seal']['selected_candidate'],
                                   validation=v['seal']['candidate_mean_validation_rmse'])
                  for (m, q), v in sealed.items()}
    result = dict(pooled_fresh_contexts=pooled, selections=selections,
                  superiority_rule=protocol['evaluation']['superiority_rule'],
                  anchors_max_abs_difference=float(anchor_table.max_abs_rmse_difference.max()))
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
