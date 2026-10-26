"""Seal the v6 comparators from prepared inputs; never reads a test outcome.

The delta-space comparators are v5's, unchanged and computed on whole units:
B0, B1c, B2c, M1c, B1a, M1a and B1p. v6 adds the residual-space ones, which
predict the plate residual and add B1p back:

- B1r, the context mean of the training residuals (a sanity check: near B1p);
- B2r, the nearest training compound's residual;
- M1r, a fingerprint Ridge on the training residuals - the cheap test of
  whether chemical structure predicts the residual at all (endpoint S3).

A training compound's residual uses the leave-one-out plate mean, so its own
delta never enters its own reference. Queries use the plain plate mean.

The residual comparators are fitted in a second pass, because their Ridge alpha
and their scale factor are chosen against the VALIDATION residual, which is the
validation delta minus B1p, and B1p is only known once the delta-space
comparators and the vehicle levels are in hand. The scale factor is least
squares, clipped to [0, 1]; M1r is stored with and without it, and the primary
direction endpoint is unaffected either way because a positive factor preserves
every sign.
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
from drugperturb.v6 import prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_STATUS = 'PRESPECIFIED_BEFORE_V6_FITTING'
RESIDUAL_METHODS = ('B1r', 'B2r', 'M1r')


def scale_factor(predicted, measured):
    """Least squares of measured on predicted, clipped to [0, 1]; 0 where the prediction carries nothing."""
    predicted, measured = np.ravel(predicted), np.ravel(measured)
    denominator = float(predicted@predicted)
    if denominator <= 0 or not np.isfinite(denominator):
        return 0.0
    return float(np.clip((predicted@measured)/denominator, 0.0, 1.0))


def anchored_predictions(queries, split, keys, pred, levels, vbar, means):
    """B1a, M1a and B1p for one split's keys, plus the number of replicates that fell back."""
    recs = {k: baselines.records(queries, split)[k] for k in keys}
    anchored = {f'{m[:2]}a': baselines.anchor(pred[m], recs, levels, vbar) for m in ['B1c', 'M1c']}
    anchored['B1p'], fallback = baselines.plate_anchored(recs, pred['B1c'], levels, vbar, means)
    return anchored, fallback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, default=ROOT/'manifests/v6')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v6.yaml')
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != PROTOCOL_STATUS:
        raise ValueError('v6 protocol is not prespecified')
    if [float(a) for a in protocol['comparators']['ridge_alphas']] != v3.ALPHAS:
        raise ValueError('v6 Ridge alpha grid must equal the v3 grid used by the shared estimator')
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
    with np.load(p/'train_residual_targets.npz', allow_pickle=False) as z:
        residual_of = {tuple(k.split('|')): r for k, r in zip(z['keys'].astype(str), z['residual'])}
        residual_fallbacks = [tuple(k.split('|')) for k in z['fallback_keys'].astype(str)]

    # Pass 1: the delta-space comparators, exactly as v5 computes them.
    tests = queries[queries.split.eq('test')]
    test_pred = {m: {} for m in ['B0', 'B1c', 'B2c', 'M1c']}
    val_pred = {m: {} for m in ['B1c', 'M1c']}
    members, selection, skipped = {}, {}, {}
    for context in contexts:
        t_ids = sorted(c for c, x in train if x == context)
        v_ids = sorted(c for c, x in val if x == context)
        q_ids = sorted(set(tests.loc[tests.context_id.eq(context), 'compound_id']))
        if len(t_ids) < 2 or not v_ids or not q_ids:
            skipped[context] = dict(n_train=len(t_ids), n_validation=len(v_ids), n_test_queries=len(q_ids))
            continue
        members[context] = dict(train=t_ids, validation=v_ids, test=q_ids)
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

    # B1a, M1a and B1p, which the residual pass needs as its reference.
    train_recs = {k: v for k, v in baselines.records(queries, 'train').items() if k in train}
    vbar = baselines.training_vehicle_means(train_recs, levels)
    means, plate_counts = baselines.plate_means(replicate_table, replicate_delta)
    loo_means, loo, _ = prepare.leave_one_out_plate_means(replicate_table, replicate_delta)
    keys = {split: sorted(pred['B1c']) for split, pred in [('test', test_pred), ('validation', val_pred)]}
    anchored, fallbacks = {}, {}
    for split, pred in [('test', test_pred), ('validation', val_pred)]:
        anchored[split], fallbacks[split] = anchored_predictions(queries, split, keys[split], pred, levels, vbar, means)

    # Pass 2: the residual-space comparators, fitted against the validation residual over B1p.
    test_res = {m: {} for m in RESIDUAL_METHODS}
    val_res = {m: {} for m in RESIDUAL_METHODS}
    raw_m1r = {'test': {}, 'validation': {}}
    residual_info = {}
    for context, ids in members.items():
        res_ids = [c for c in ids['train'] if (c, context) in residual_of]
        v_ids, q_ids = ids['validation'], ids['test']
        if len(res_ids) < 2:
            residual_info[context] = dict(training_residuals=len(res_ids), fitted=False)
            continue
        ry = np.vstack([residual_of[(c, context)] for c in res_ids])
        rf = np.vstack([fps[c] for c in res_ids])
        vf, qf = (np.vstack([fps[c] for c in group]) for group in (v_ids, q_ids))
        validation_residual = np.vstack([val[(c, context)]-anchored['validation']['B1p'][(c, context)]
                                         for c in v_ids])
        ridge_val, info = v3.fit_context(ry, rf, validation_residual, vf, vf)
        ridge_test, _ = v3.fit_context(ry, rf, validation_residual, vf, qf)
        lam = scale_factor(ridge_val, validation_residual)
        residual_info[context] = dict(training_residuals=len(res_ids), fitted=True,
                                      alpha=info.get('alpha'), scale=lam,
                                      validation_residual_rmse=v3.rmse(validation_residual, lam*ridge_val),
                                      zero_residual_rmse=v3.rmse(validation_residual, np.zeros_like(ridge_val)))
        res_simple_test = baseline_predictions(ry, rf, qf)
        res_simple_val = baseline_predictions(ry, rf, vf)
        for i, c in enumerate(q_ids):
            test_res['B1r'][(c, context)] = res_simple_test['B1'][i]
            test_res['B2r'][(c, context)] = res_simple_test['B2'][i]
            test_res['M1r'][(c, context)] = lam*ridge_test[i]
            raw_m1r['test'][(c, context)] = ridge_test[i]
        for i, c in enumerate(v_ids):
            val_res['B1r'][(c, context)] = res_simple_val['B1'][i]
            val_res['B2r'][(c, context)] = res_simple_val['B2'][i]
            val_res['M1r'][(c, context)] = lam*ridge_val[i]
            raw_m1r['validation'][(c, context)] = ridge_val[i]

    final = {}
    for split, pred, res in [('test', test_pred, test_res), ('validation', val_pred, val_res)]:
        prefix = '' if split == 'test' else 'validation_'
        final[f'{prefix}keys'] = baselines.key_strings(keys[split])
        for method, values in {**pred, **anchored[split]}.items():
            final[f'{prefix}{method}'] = baselines.stack(values, keys[split])
        for method, values in res.items():
            present = [k for k in keys[split] if k in values]
            final[f'{prefix}{method}'] = baselines.stack(
                {k: anchored[split]['B1p'][k]+values[k] for k in present}, present)
            final[f'{prefix}{method}_keys'] = baselines.key_strings(present)
        unscaled = raw_m1r[split]
        present = [k for k in keys[split] if k in unscaled]
        final[f'{prefix}M1r_unscaled'] = baselines.stack(
            {k: anchored[split]['B1p'][k]+unscaled[k] for k in present}, present)
    inputs = {'protocol': args.protocol, 'compounds': args.manifests/'compounds.tsv',
              **{name: p/name for name in ['train_truth.npz', 'validation_truth.npz', 'train_replicate_deltas.npz',
                                            'train_residual_targets.npz', 'queries.tsv', 'vehicle_levels.npz']}}
    metadata = dict(selection=selection, skipped_contexts=skipped, alphas=v3.ALPHAS,
                    anchoring='delta baseline + Vbar_train(context) - mean_r V_r(query); V from all vehicle cells',
                    b1p=dict(min_compounds=baselines.B1P_MIN_COMPOUNDS, plates_with_enough_compounds=len(means),
                             context_plates=len(plate_counts), replicates_falling_back_to_B1a=fallbacks),
                    residual=dict(reference='B1p, leave-one-out for training compounds',
                                  scale='least squares on validation residuals, clipped to [0, 1]',
                                  per_context=residual_info, leave_one_out_plates=len(loo_means),
                                  leave_one_out_keys=len(loo),
                                  training_keys_without_a_residual=len(residual_fallbacks)),
                    n_training_pairs=len(train_recs))
    seal = runner.write_seal(args.output_root, 'v6', 'baselines', final, metadata, inputs)
    print(json.dumps(dict(test_keys=len(final['keys']), validation_keys=len(final['validation_keys']),
                          residual_contexts=sum(1 for v in residual_info.values() if v['fitted']),
                          skipped=sorted(skipped), sha256=seal['predictions']['sha256'])))


if __name__ == '__main__':
    main()
