"""Seal vehicle-anchored baselines B1a and M1a (amendment A1); never reads test outcomes.

B1a/M1a add, to the sealed per-context B1c/M1c delta predictions, the context's
mean training-pair vehicle level minus the query's measured vehicle level
(drugperturb.v4.anchoring). Inputs are the response-free prepared outputs
(vehicle cells, queries, training truth) and the sealed context baselines.
Validation predictions are kept for the diagnostics.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_diagnostics_report as report  # noqa: E402
from v3_evaluate import by_key, load_sealed  # noqa: E402
from v4_fit import load_setting  # noqa: E402
from drugperturb.v3 import runner  # noqa: E402
from drugperturb.v4 import anchoring, counterfactual  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SEAL = {'op3': 'op3', 'sciplex_Q2': 'Q2'}


def vehicle_context(setting, prepared):
    """Queries, measured vehicle levels and training vehicle means of a setting (no test response)."""
    spec = counterfactual.SETTINGS[setting]
    contexts, R = spec['contexts'], spec['replicates']
    obs, counts, library, query_table, train_rows, vehicle_contexts, truth, _ = load_setting(setting, prepared)
    truth = {s: {k: v for k, v in t.items() if k[1] in contexts} for s, t in truth.items()}
    queries = {}
    for split, rows in [('train', train_rows), ('validation', query_table), ('test', query_table)]:
        built = counterfactual.build_queries(rows, contexts, [split], R)
        queries[split] = built if split == 'test' else {k: v for k, v in built.items() if k in truth[split]}
    train_mask, validation_mask, _ = counterfactual.setting_cells(obs, contexts, vehicle_contexts)
    keep = np.flatnonzero(train_mask | validation_mask)
    sub = obs.iloc[keep].reset_index(drop=True)
    wells = sorted({r['vehicle_well'] for q in queries.values() for v in q.values() for r in v})
    vehicles = runner.vehicle_wells(sub, counts[keep], library[keep], wells)
    levels = anchoring.vehicle_levels(vehicles)
    vbar = anchoring.training_vehicle_means(queries['train'], levels, sorted(queries['train']))
    return queries, levels, vbar, truth, len(vehicles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=list(BASELINE_SEAL), required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--baselines-root', type=Path, required=True, help='directory holding the context-baseline seal')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    contexts = counterfactual.SETTINGS[args.setting]['contexts']
    queries, levels, vbar, truth, n_vehicles = vehicle_context(args.setting, args.prepared)

    name = f'context_baselines_{BASELINE_SEAL[args.setting]}_seal.json'
    sealed, seal = load_sealed(args.baselines_root, 'context_baselines', BASELINE_SEAL[args.setting])
    final, keys = {}, None
    for method in ['B1c', 'M1c']:
        delta = {k: v for k, v in by_key(sealed['keys'], sealed[method]).items() if k in queries['test']}
        if not delta:
            raise ValueError(f'{method}: no sealed test key has a complete counterfactual query')
        anchored = anchoring.anchor(delta, queries['test'], levels, vbar)
        keys = sorted(anchored) if keys is None else keys
        final[f'{method[:2]}a'] = np.vstack([anchored[k] for k in keys])
    final['keys'] = np.asarray([f'{c}|{x}' for c, x in keys])
    references, _, _ = report.reference_predictions(args.setting, args.prepared)
    val_keys = sorted(queries['validation'])
    for method in ['B1c', 'M1c']:
        anchored = anchoring.anchor({k: references[method][k] for k in val_keys}, queries['validation'], levels, vbar)
        final[f'validation_{method[:2]}a'] = np.vstack([anchored[k] for k in val_keys])
    final['validation_keys'] = np.asarray([f'{c}|{x}' for c, x in val_keys])
    inputs = {'protocol': ROOT/'configs/protocol_v4.yaml', 'model_cells': args.prepared/'model_cells.npz',
              'queries': args.prepared/'queries.tsv', 'train_truth': args.prepared/'train_truth.npz',
              'context_baselines': args.baselines_root/seal['predictions']['path'],
              'context_baselines_seal': args.baselines_root/name}
    written = runner.write_seal(args.output_root, args.setting, 'anchored_baselines', final,
                                dict(amendment='A1', definition='delta baseline + Vbar_train(context) - mean_r V_r(query)',
                                     n_training_pairs=len(queries['train']), n_vehicle_wells=n_vehicles), inputs)
    print(json.dumps(dict(setting=args.setting, test_keys=len(keys), validation_keys=len(val_keys),
                          sha256=written['predictions']['sha256'])))


if __name__ == '__main__':
    main()
