"""Score sealed OP3 predictions against measured test deltas (v4 confirmatory analysis).

Runs only after every arm, the context baselines and the primary-arm selection
are sealed. Test truth is first computed here. Endpoints follow the protocol:
E1 (selected arm minus zero change) and E2 (selected arm minus B1c and M1c),
pooled over gate-passing contexts with one scaffold-group draw shared by all
contexts; every other contrast is secondary.
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
from v3_evaluate import by_key, load_sealed, pooled_bootstrap  # noqa: E402
from v4_fit import ARMS  # noqa: E402
from drugperturb.evaluation import compound_metrics, paired_scaffold_bootstrap, summarize_compound_metrics  # noqa: E402
from drugperturb.v3.runner import sha256  # noqa: E402
from drugperturb.v4 import design, diagnostics  # noqa: E402
from drugperturb.v4.counterfactual import SETTINGS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ['B0', 'B1c', 'B2c', 'M1c']


def read(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'replicate': str})


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
    protocol = yaml.safe_load((ROOT/'configs/protocol_v4.yaml').read_text())
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    sealed = {}
    for arm in ARMS:
        arrays, seal = load_sealed(locate(args.sealed_root, f'{arm}_op3_seal.json'), arm, 'op3')
        sealed[arm] = dict(contrast=by_key(arrays['test_keys'], arrays['test_contrast']),
                           anchored=by_key(arrays['test_keys'], arrays['test_anchored']), seal=seal)
    baselines, _ = load_sealed(locate(args.sealed_root, 'context_baselines_op3_seal.json'), 'context_baselines', 'op3')
    simple = {m: by_key(baselines['keys'], baselines[m]) for m in BASELINES}
    selection = json.loads((locate(args.sealed_root, 'selection_op3.json')/'selection_op3.json').read_text())
    if selection['status'] != 'SELECTED_ON_VALIDATION_BEFORE_TEST_SCORING' or any(
            selection['arms'][a]['sha256'] != sealed[a]['seal']['predictions']['sha256'] for a in ARMS):
        raise ValueError('selection record does not match the sealed arm predictions')
    chosen = selection['selected']

    compound_qc = read(args.prepared/'count_qc/compound_qc.tsv')
    summary_qc = read(args.prepared/'count_qc/context_summary.tsv').set_index('context_id')
    frozen = read(args.prepared/'frozen_genes.tsv')
    with np.load(args.extraction_root/'unit_counts.npz', allow_pickle=False) as z:
        if sha256(args.extraction_root/'unit_counts.npz') != json.loads(
                (args.prepared/'prepare_receipt.json').read_text())['inputs']['unit_counts.npz']:
            raise ValueError('extraction unit counts differ from those the preparation read')
        unit_counts = sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
        unit_library, unit_ids = z['library'], z['unit_ids'].astype(str)
    pairs = design.response_pairs(read(ROOT/'manifests/v4/op3_wells.tsv'), read(ROOT/'manifests/v4/op3_compounds.tsv'))
    position = {u: i for i, u in enumerate(unit_ids)}
    columns = frozen.source_index.to_numpy()
    methods_all = [*BASELINES, *ARMS]
    rows, contrasts, per_compound, test_diagnostics = [], [], [], []
    differences = {}
    for context in SETTINGS['op3']['contexts']:
        cohort = compound_qc[compound_qc.context_id.eq(context) & compound_qc.split.eq('test')
                             & compound_qc.eligible.astype(str).eq('True')].sort_values('compound_id')
        ids, groups = cohort.compound_id.tolist(), cohort.scaffold_group.tolist()
        gate = str(summary_qc.loc[context, 'test_count_gate_met']) == 'True'
        if not ids:
            continue
        keys = [(c, context) for c in ids]
        needed = sorted(set(pairs.loc[pairs.split.eq('test') & pairs.context_id.eq(context), 'treatment_unit'])
                        | set(pairs.loc[pairs.context_id.eq(context), 'vehicle_unit']))
        index = [position[u] for u in needed]
        truth = design.strict_deltas(unit_counts[index].toarray(), unit_library[index], needed, pairs, keys, columns)
        methods = {m: np.vstack([simple[m][k] for k in keys]) for m in BASELINES}
        for arm in ARMS:
            methods[arm] = np.vstack([sealed[arm]['contrast'][k] for k in keys])
            methods[f'{arm}_anchored'] = np.vstack([sealed[arm]['anchored'][k] for k in keys])
        tables = {}
        for name, prediction in methods.items():
            table = compound_metrics(truth, prediction, ids, groups)
            table['method'], table['context_id'] = name, context
            tables[name] = table
            per_compound.append(table)
            rows.append(dict(context_id=context, gate_met=gate, method=name, **summarize_compound_metrics(table)))
        truth_map = dict(zip(keys, truth))
        reference = {context: methods['B1c'][0]}
        for arm in [*ARMS, 'M1c']:
            s = diagnostics.summarize(dict(zip(keys, methods[arm])), truth_map, reference)
            test_diagnostics.append(dict(context_id=context, method=arm, **s))
        if not gate:
            continue
        for candidate in ARMS:
            for baseline in ['B0', 'B1c', 'M1c']:
                b = paired_scaffold_bootstrap(truth, methods[candidate], methods[baseline], ids, groups)
                contrasts.append(dict(context_id=context, candidate=candidate, baseline=baseline,
                                      primary_arm=candidate == chosen,
                                      **{k: b[k] for k in ['point_estimate', 'ci_lower', 'ci_upper', 'n_compounds', 'n_groups']}))
                frame = pd.DataFrame(dict(scaffold_group=groups, difference=tables[candidate].rmse.to_numpy()
                                          - tables[baseline].rmse.to_numpy()))
                differences.setdefault((candidate, baseline), {})[context] = frame
    pooled = {f'{c}_minus_{b}': pooled_bootstrap(d) for (c, b), d in differences.items()}
    e1 = pooled.get(f'{chosen}_minus_B0')
    e2 = {b: pooled.get(f'{chosen}_minus_{b}') for b in ['B1c', 'M1c']}
    endpoints = dict(
        selected_arm=chosen, selection_any_passing=selection['any_passing'],
        E1_non_collapse=dict(estimate=e1, supported=bool(e1 and e1['ci_upper'] < 0)),
        E2_improvement=dict(estimates=e2, supported=bool(all(v and v['ci_upper'] < 0 for v in e2.values()))),
        gate_passing_contexts=[c for c in SETTINGS['op3']['contexts']
                               if str(summary_qc.loc[c, 'test_count_gate_met']) == 'True'])
    pd.concat(per_compound, ignore_index=True).to_csv(out/'per_compound.tsv', sep='\t', index=False)
    pd.DataFrame(rows).to_csv(out/'summary.tsv', sep='\t', index=False)
    pd.DataFrame(contrasts).to_csv(out/'contrasts.tsv', sep='\t', index=False)
    pd.DataFrame(test_diagnostics).to_csv(out/'test_diagnostics.tsv', sep='\t', index=False)
    result = dict(endpoints=endpoints, pooled=pooled, selection=selection,
                  seals={a: sealed[a]['seal']['predictions']['sha256'] for a in ARMS},
                  test_truth_first_computed_here=True)
    (out/'summary.json').write_text(json.dumps(diagnostics.finite(result), indent=2, allow_nan=False)+'\n')
    print(json.dumps(diagnostics.finite(endpoints), indent=2))


if __name__ == '__main__':
    main()
