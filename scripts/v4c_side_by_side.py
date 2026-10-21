"""Tabulate the prespecified v4 OP3 result next to the post hoc corrected one (configs/v4_corrected_posthoc.yaml).

Reads only committed outputs of the two evaluations and their selection records;
computes no new estimate. Writes selection.tsv (validation selection per arm),
endpoints.tsv (E1-E3 for each analysis's selected arm), pooled.tsv (every arm
against every comparator), per_context.tsv (mean compound RMSE per method and
context) and affected_compounds.tsv (the three test compounds whose v4 descriptor
z-scores were near 1e16).
"""
from pathlib import Path
import argparse
import json

import pandas as pd

ARMS = ['A0_v3_reference', 'I1_nb_likelihood', 'I2_counterfactual_pseudobulk', 'I3_nb_plus_pseudobulk']
COMPARATORS = ['B0', 'B1c', 'M1c', 'B1a', 'M1a']
BASELINES = ['B0', 'B1c', 'B2c', 'M1c', 'B1a', 'M1a']
ENDPOINTS = [('E1_non_collapse', 'B0'), ('E2_improvement', 'B1c'), ('E2_improvement', 'M1c'),
             ('E3_beyond_vehicle_anchoring', 'B1a'), ('E3_beyond_vehicle_anchoring', 'M1a')]
# ixabepilone, alvocidib, dasatinib (tests/test_descriptor_scaling.py; CEP-37440 is a validation compound)
AFFECTED = {'LSM-43293': 'Ixabepilone', 'LSM-1011': 'Alvocidib', 'LSM-1020': 'Dasatinib'}
ANALYSES = ['v4', 'corrected']


def interval(estimate):
    return {k: estimate[k] for k in ['point_estimate', 'ci_lower', 'ci_upper']}


def tables(summaries, selections, per_compound, per_context):
    selection = pd.DataFrame([dict(arm=arm, analysis=a, validation_rmse=selections[a]['arms'][arm]['validation_rmse'],
                                   non_collapsed=selections[a]['arms'][arm]['non_collapse']['non_collapsed'],
                                   selected=selections[a]['selected'] == arm)
                              for arm in ARMS for a in ANALYSES])
    selection = selection.pivot(index='arm', columns='analysis').swaplevel(axis=1).sort_index(axis=1)
    selection.columns = [f'{a}_{c}' for a, c in selection.columns]
    endpoints = []
    for name, comparator in ENDPOINTS:
        row = dict(endpoint=name, comparator=comparator)
        for a in ANALYSES:
            e = summaries[a]['endpoints']
            estimate = e[name]['estimate'] if name == 'E1_non_collapse' else e[name]['estimates'][comparator]
            row.update({f'{a}_arm': e['selected_arm'], **{f'{a}_{k}': v for k, v in interval(estimate).items()},
                        f'{a}_supported': e[name]['supported']})
        endpoints.append(row)
    pooled = [dict(candidate=arm, comparator=b, **{f'{a}_{k}': v for a in ANALYSES
                                                   for k, v in interval(summaries[a]['pooled'][f'{arm}_minus_{b}']).items()})
              for arm in ARMS for b in COMPARATORS]
    context = pd.concat([per_context[a].assign(analysis=a) for a in ANALYSES])
    context = context[context.method.isin([*BASELINES, *ARMS])]  # baselines are shared: equal in both analyses
    context = context.pivot_table(index=['method', 'context_id', 'gate_met'], columns='analysis',
                                  values='mean_compound_rmse', sort=False).reset_index()
    context.columns.name = None
    compounds = pd.concat([per_compound[a].assign(analysis=a) for a in ANALYSES])
    compounds = compounds[compounds.compound_id.isin(AFFECTED) & compounds.method.isin(ARMS)]
    compounds = compounds.pivot_table(index=['compound_id', 'context_id', 'method'], columns='analysis',
                                      values='rmse').reset_index()
    compounds.insert(1, 'name', compounds.compound_id.map(AFFECTED))
    compounds.columns.name = None
    return selection.reset_index(), pd.DataFrame(endpoints), pd.DataFrame(pooled), context, compounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v4-report', type=Path, required=True, help='reports/v4/op3')
    parser.add_argument('--corrected-report', type=Path, required=True, help='reports/v4_corrected/op3')
    parser.add_argument('--v4-selection', type=Path, required=True)
    parser.add_argument('--corrected-selection', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    roots = dict(v4=args.v4_report, corrected=args.corrected_report)
    summaries = {a: json.loads((r/'summary.json').read_text()) for a, r in roots.items()}
    selections = dict(v4=json.loads(args.v4_selection.read_text()),
                      corrected=json.loads(args.corrected_selection.read_text()))
    if summaries['v4'].get('analysis') or summaries['corrected'].get('analysis') != 'v4_corrected_posthoc':
        raise ValueError('expected the prespecified v4 summary and the post hoc corrected summary, in that order')
    for a in ANALYSES:
        if summaries[a]['selection'] != selections[a]:
            raise ValueError(f'{a}: selection record differs from the one the evaluation used')
    read = lambda path: pd.read_csv(path, sep='\t', keep_default_na=False, na_values=[''])
    out = tables(summaries, selections, {a: read(r/'per_compound.tsv') for a, r in roots.items()},
                 {a: read(r/'summary.tsv') for a, r in roots.items()})
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, table in zip(['selection', 'endpoints', 'pooled', 'per_context', 'affected_compounds'], out):
        table.to_csv(args.output_root/f'{name}.tsv', sep='\t', index=False)
        print(f'== {name}\n{table.to_string(index=False, float_format=lambda x: f"{x:.4f}")}')


if __name__ == '__main__':
    main()
