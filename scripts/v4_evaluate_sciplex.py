"""Score sealed v4 sci-Plex3 predictions (exploratory: every test outcome was opened in v1-v3).

Truth is recomputed with the v3 code. Anchors: per-compound zero-change RMSE and
the v3 sealed chemCPA predictions' RMSE must reproduce reports/v3/per_compound.tsv
in every context before any v4 arm is scored.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_evaluate import by_key, frozen_v1, load_sealed, pooled_bootstrap, read  # noqa: E402
from v4_fit import ARMS  # noqa: E402
from drugperturb.evaluation import compound_metrics, paired_scaffold_bootstrap, summarize_compound_metrics  # noqa: E402
from drugperturb.transfer_v2 import alternate_controls, measured_responses, required_wells  # noqa: E402
from drugperturb.v3.runner import PROTOCOL_CONTEXTS  # noqa: E402
from drugperturb.v4 import diagnostics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FRESH_IN_V3 = ['A549_24h_10nM', 'A549_24h_100nM', 'A549_24h_10000nM', 'K562_24h_10nM', 'K562_24h_100nM', 'K562_24h_10000nM']
Q1_CONTEXT = 'A549_24h_1000nM'


def locate(roots, name):
    found = [r for r in roots if (r/name).exists()]
    if len(found) != 1:
        raise ValueError(f'{name}: expected exactly one sealed copy, found {len(found)}')
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--sealed-root', type=Path, action='append', default=[])
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    arms = {}
    for setting in ['sciplex_Q1', 'sciplex_Q2']:
        for arm in ARMS:
            arrays, _ = load_sealed(locate(args.sealed_root, f'{arm}_{setting}_seal.json'), arm, setting)
            arms[(arm, setting)] = by_key(arrays['test_keys'], arrays['test_contrast'])
    v3_models = {}
    for q in ['Q1', 'Q2']:
        arrays, _ = load_sealed(locate(args.sealed_root, f'chemcpa_{q}_seal.json'), 'chemcpa', q)
        v3_models[q] = by_key(arrays['test_keys'], arrays['test_contrast'])
    baselines, _ = load_sealed(locate(args.sealed_root, 'context_baselines_Q2_seal.json'), 'context_baselines', 'Q2')
    simple = {m: by_key(baselines['keys'], baselines[m]) for m in ['B0', 'B1c', 'B2c', 'M1c']}
    v1 = frozen_v1()
    published = read(ROOT/'reports/v3/per_compound.tsv')

    wells = read(ROOT/'manifests/v2/well_mapping.tsv')
    compounds = read(args.prepared/'count_qc/compound_qualification.tsv')
    qualified = read(args.prepared/'count_qc/well_qualification.tsv')
    gate_table = read(args.prepared/'count_qc/context_summary.tsv').set_index('context_id')
    gene_ids = read(ROOT/'manifests/selected_genes.tsv').gene_id.tolist()
    rows, contrasts, per_compound, anchors, test_diagnostics = [], [], [], [], []
    differences = {}
    for context in PROTOCOL_CONTEXTS['Q2']:
        cell = context.split('_')[0]
        cohort = compounds[compounds.context_id.eq(context) & compounds.split.eq('test')
                           & compounds.eligible.astype(str).eq('True')].sort_values('compound_id')
        ids, groups = cohort.compound_id.tolist(), cohort.scaffold_group.tolist()
        gate = str(gate_table.loc[context, 'frozen_test_count_gate_met']) == 'True'
        ctx_wells, ctx_q = wells[wells.context_id.eq(context)], qualified[qualified.context_id.eq(context)]
        qc = read(args.extraction_root/f'pseudobulk/{cell}_24h/well_qc.tsv')
        pairs = alternate_controls(ctx_wells, ctx_q, qc, ids)
        keys = required_wells(ctx_q, pairs, ids)
        counts = pd.read_csv(args.extraction_root/f'pseudobulk/{cell}_24h/pseudobulk_counts.tsv.gz', sep='\t',
                             usecols=['gene_id', *keys], index_col='gene_id')
        truth, _ = measured_responses(counts, ctx_q, pairs, ids, gene_ids)
        methods = {'B0': np.zeros_like(truth), 'v3_chemcpa_Q2': np.vstack([v3_models['Q2'][(c, context)] for c in ids])}
        for m in ['B1c', 'B2c', 'M1c']:
            methods[m] = np.vstack([simple[m][(c, context)] for c in ids])
        for arm in ARMS:
            methods[f'{arm}_Q2'] = np.vstack([arms[(arm, 'sciplex_Q2')][(c, context)] for c in ids])
        if context == Q1_CONTEXT:
            methods['v3_chemcpa_Q1'] = np.vstack([v3_models['Q1'][(c, context)] for c in ids])
            for m in ['B1', 'M1', 'M3']:
                methods[f'v1_{m}'] = np.vstack([v1[m][c] for c in ids])
            for arm in ARMS:
                methods[f'{arm}_Q1'] = np.vstack([arms[(arm, 'sciplex_Q1')][(c, context)] for c in ids])
        tables = {}
        for name, prediction in methods.items():
            table = compound_metrics(truth, prediction, ids, groups)
            table['method'], table['context_id'] = name, context
            tables[name] = table
            per_compound.append(table)
            rows.append(dict(context_id=context, gate_met=gate, method=name, **summarize_compound_metrics(table)))
        for mine, theirs in [('B0', 'B0'), ('B1c', 'B1c'), ('M1c', 'M1c'), ('v3_chemcpa_Q2', 'chemcpa_Q2'),
                             ('v3_chemcpa_Q1', 'chemcpa_Q1')]:
            if mine not in tables:
                continue
            ref = published[published.context_id.eq(context) & published.method.eq(theirs)]
            merged = ref.merge(tables[mine][['compound_id', 'rmse']], on='compound_id', suffixes=('_published', ''))
            anchors.append(dict(context_id=context, method=mine, compounds=len(merged), expected=len(ids),
                                max_abs_rmse_difference=float(np.max(np.abs(merged.rmse_published.astype(float)-merged.rmse)))))
        truth_map = {(c, context): t for c, t in zip(ids, truth)}
        reference = {context: methods['B1c'][0]}
        for name in [f'{a}_Q2' for a in ARMS]+['v3_chemcpa_Q2', 'M1c']+([f'{a}_Q1' for a in ARMS] if context == Q1_CONTEXT else []):
            s = diagnostics.summarize({(c, context): p for c, p in zip(ids, methods[name])}, truth_map, reference)
            test_diagnostics.append(dict(context_id=context, method=name, **s))
        if not gate:
            continue
        tests = [(f'{a}_Q2', b) for a in ARMS for b in ['B0', 'B1c', 'M1c']]
        if context == Q1_CONTEXT:
            tests += [(f'{a}_Q1', b) for a in ARMS for b in ['B0', 'v1_B1', 'v1_M1']]
        for candidate, baseline in tests:
            b = paired_scaffold_bootstrap(truth, methods[candidate], methods[baseline], ids, groups)
            contrasts.append(dict(context_id=context, candidate=candidate, baseline=baseline, label='exploratory',
                                  **{k: b[k] for k in ['point_estimate', 'ci_lower', 'ci_upper', 'n_compounds', 'n_groups']}))
            if context in FRESH_IN_V3 and candidate.endswith('_Q2'):
                frame = pd.DataFrame(dict(scaffold_group=groups, difference=tables[candidate].rmse.to_numpy()
                                          - tables[baseline].rmse.to_numpy()))
                differences.setdefault((candidate, baseline), {})[context] = frame
    anchor_table = pd.DataFrame(anchors)
    if (anchor_table.max_abs_rmse_difference > 1e-9).any() or (anchor_table.compounds != anchor_table.expected).any():
        raise ValueError(f'recomputed truth does not reproduce the committed v3 results:\n{anchor_table}')
    pooled = {f'{c}_minus_{b}': pooled_bootstrap(d) for (c, b), d in differences.items()}
    pd.concat(per_compound, ignore_index=True).to_csv(out/'per_compound.tsv', sep='\t', index=False)
    pd.DataFrame(rows).to_csv(out/'summary.tsv', sep='\t', index=False)
    pd.DataFrame(contrasts).to_csv(out/'contrasts.tsv', sep='\t', index=False)
    pd.DataFrame(test_diagnostics).to_csv(out/'test_diagnostics.tsv', sep='\t', index=False)
    anchor_table.to_csv(out/'truth_anchors.tsv', sep='\t', index=False)
    result = dict(label='exploratory', pooled_v3_fresh_contexts=pooled,
                  anchors_max_abs_difference=float(anchor_table.max_abs_rmse_difference.max()))
    (out/'summary.json').write_text(json.dumps(diagnostics.finite(result), indent=2, allow_nan=False)+'\n')
    print(json.dumps(diagnostics.finite({k: {kk: vv for kk, vv in v.items() if kk in ('point_estimate', 'ci_lower', 'ci_upper')}
                                         for k, v in pooled.items()}), indent=2))


if __name__ == '__main__':
    main()
