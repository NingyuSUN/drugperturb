"""Seal the v5 comparators (B0, B1c, B2c, M1c, B1a, M1a, B1p) from prepared inputs; never reads test outcomes.

Per context, the v3/v4 estimators use only that context's eligible training
compounds (Ridge alpha by the context's validation compounds). B1a and M1a add
the context's mean training vehicle level minus the query's all-cell vehicle
level; B1p is the plate-anchored training mean (drugperturb.v5.baselines).
Validation predictions are kept for the diagnostics. Contexts without at least
two training compounds, one validation compound and one test compound are
recorded and skipped.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_context_baselines as v3  # noqa: E402
from drugperturb.chemistry import morgan_fingerprint  # noqa: E402
from drugperturb.response import baseline_predictions  # noqa: E402
from drugperturb.v3 import runner  # noqa: E402
from drugperturb.v4.counterfactual import load_truth  # noqa: E402
from drugperturb.v5 import baselines  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, default=ROOT/'manifests/v5')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v5.yaml')
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != 'PRESPECIFIED_BEFORE_V5_FITTING':
        raise ValueError('v5 protocol is not prespecified')
    if [float(a) for a in protocol['comparators']['ridge_alphas']] != v3.ALPHAS:
        raise ValueError('v5 Ridge alpha grid must equal the v3 grid used by the shared estimator')
    compounds = pd.read_csv(args.manifests/'compounds.tsv', sep='\t', keep_default_na=False)
    compounds = compounds[compounds.status.eq('accepted')]
    fps = {c: morgan_fingerprint(s).astype(float) for c, s in zip(compounds.compound_id, compounds.canonical_smiles)}
    contexts = pd.read_csv(args.manifests/'contexts.tsv', sep='\t').context_id.tolist()
    p = args.prepared
    train, val = load_truth(p/'train_truth.npz'), load_truth(p/'validation_truth.npz')
    queries = pd.read_csv(p/'queries.tsv', sep='\t', keep_default_na=False, dtype={'replicate': str})
    with np.load(p/'vehicle_levels.npz', allow_pickle=False) as z:
        levels = dict(zip(z['vehicle_unit'].astype(str), z['level']))
    with np.load(p/'train_replicate_deltas.npz', allow_pickle=False) as z:
        replicate_table = pd.DataFrame({c: z[c].astype(str) for c in ['compound_id', 'context_id', 'replicate', 'plate']})
        replicate_delta = z['delta']

    tests = queries[queries.split.eq('test')]
    test_pred = {m: {} for m in ['B0', 'B1c', 'B2c', 'M1c']}
    val_pred = {m: {} for m in ['B1c', 'M1c']}
    selection, skipped = {}, {}
    for context in contexts:
        t_ids = sorted(c for c, x in train if x == context)
        v_ids = sorted(c for c, x in val if x == context)
        q_ids = sorted(set(tests.loc[tests.context_id.eq(context), 'compound_id']))
        if len(t_ids) < 2 or not v_ids or not q_ids:
            skipped[context] = dict(n_train=len(t_ids), n_validation=len(v_ids), n_test_queries=len(q_ids))
            continue
        ty = np.vstack([train[(c, context)] for c in t_ids])
        vy = np.vstack([val[(c, context)] for c in v_ids])
        tf, vf, qf = (np.vstack([fps[c] for c in ids]) for ids in (t_ids, v_ids, q_ids))
        simple = baseline_predictions(ty, tf, qf)
        ridge, info = v3.fit_context(ty, tf, vy, vf, qf)
        ridge_val, _ = v3.fit_context(ty, tf, vy, vf, vf)
        val_simple = baseline_predictions(ty, tf, vf)
        info['validation_rmse_simple'] = {k: v3.rmse(vy, v) for k, v in val_simple.items()}
        info.update(n_validation=len(v_ids), n_test_queries=len(q_ids))
        selection[context] = info
        for i, c in enumerate(q_ids):
            for method, source in [('B0', simple['B0']), ('B1c', simple['B1']), ('B2c', simple['B2']), ('M1c', ridge)]:
                test_pred[method][(c, context)] = source[i]
        for i, c in enumerate(v_ids):
            val_pred['B1c'][(c, context)], val_pred['M1c'][(c, context)] = val_simple['B1'][i], ridge_val[i]

    train_recs = {k: v for k, v in baselines.records(queries, 'train').items() if k in train}
    vbar = baselines.training_vehicle_means(train_recs, levels)
    means, plate_counts = baselines.plate_means(replicate_table, replicate_delta)
    final, fallbacks = {}, {}
    for split, pred in [('test', test_pred), ('validation', val_pred)]:
        recs = baselines.records(queries, split)
        keys = sorted(pred['B1c'])
        recs = {k: recs[k] for k in keys}
        anchored = {f'{m[:2]}a': baselines.anchor(pred[m], recs, levels, vbar) for m in ['B1c', 'M1c']}
        anchored['B1p'], fallbacks[split] = baselines.plate_anchored(recs, pred['B1c'], levels, vbar, means)
        prefix = '' if split == 'test' else 'validation_'
        final[f'{prefix}keys'] = baselines.key_strings(keys)
        for method, values in {**pred, **anchored}.items():
            final[f'{prefix}{method}'] = baselines.stack(values, keys)
    inputs = {'protocol': args.protocol, 'compounds': args.manifests/'compounds.tsv',
              **{name: p/name for name in ['train_truth.npz', 'validation_truth.npz', 'train_replicate_deltas.npz',
                                            'queries.tsv', 'vehicle_levels.npz']}}
    metadata = dict(selection=selection, skipped_contexts=skipped, alphas=v3.ALPHAS,
                    anchoring='delta baseline + Vbar_train(context) - mean_r V_r(query); V from all vehicle cells',
                    b1p=dict(min_compounds=baselines.B1P_MIN_COMPOUNDS, plates_with_enough_compounds=len(means),
                             context_plates=len(plate_counts), replicates_falling_back_to_B1a=fallbacks),
                    n_training_pairs=len(train_recs))
    seal = runner.write_seal(args.output_root, 'v5', 'baselines', final, metadata, inputs)
    print(json.dumps(dict(test_keys=len(final['keys']), validation_keys=len(final['validation_keys']),
                          skipped=sorted(skipped), sha256=seal['predictions']['sha256'])))


if __name__ == '__main__':
    main()
