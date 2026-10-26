"""Fit one v6 arm and seed on Tahoe-100M prepared inputs; never reads a test outcome.

Both arms use v4's I2 objective and v3's C1 hyperparameters, unchanged: the only
difference between them is what the counterfactual pseudobulk loss is trained
against (protocol_v6 arms).

- M1_pseudobulk_raw trains against the measured whole-unit delta, so for this arm
  the ONLY difference from v5 is the endpoint it is later scored on.
- M2_pseudobulk_residual trains against the leave-one-out residual
  ``delta - B1p`` from preparation, and its prediction of a delta is B1p plus
  what it predicts. B1p comes from the comparators sealed before any fit.

Validation is reported in delta space for both arms, so the checkpoint rule and
arm selection compare like with like. The fit also records the validation
direction accuracy, which is the arm-selection metric, and the residual
correlation, which is the quantity the primary endpoint depends on.
"""
from pathlib import Path
import argparse
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_fit import arm_spec  # noqa: E402
from drugperturb.v3 import runner as v3_runner  # noqa: E402
from drugperturb.v3.cells import load_model_archive  # noqa: E402
from drugperturb.v3.descriptors import NOISE_STD_TOLERANCE, VEHICLE_ID, compound_table, scaling_diagnostics  # noqa: E402
from drugperturb.v3.pseudobulk import normalize_counts  # noqa: E402
from drugperturb.v4 import chemcpa_v4, diagnostics  # noqa: E402
from drugperturb.v4.counterfactual import load_truth  # noqa: E402
from drugperturb.v5 import chemcpa as v5_chemcpa, counterfactual  # noqa: E402
from drugperturb.v6 import scoring  # noqa: E402
from drugperturb.v6.selection import ARMS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_STATUS = 'PRESPECIFIED_BEFORE_V6_FITTING'
OBJECTIVE = 'I2_counterfactual_pseudobulk'  # both v6 arms; only the target differs
RESIDUAL_ARM = 'M2_pseudobulk_residual'


def read_tsv(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'replicate': str})


def by_key(keys, values):
    return {tuple(k.split('|')): values[i] for i, k in enumerate(np.asarray(keys, str))}


def load_baseline(path, name, split):
    """One sealed comparator's predictions for a split, keyed by (compound, context)."""
    prefix = '' if split == 'test' else f'{split}_'
    with np.load(path, allow_pickle=False) as z:
        return by_key(z[f'{prefix}keys'], z[f'{prefix}{name}'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--baselines', type=Path, required=True, help='sealed baselines_v6_predictions.npz')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-epochs', type=int, help='Engineering smoke runs only; recorded in the fit record')
    parser.add_argument('--manifests', type=Path, default=ROOT/'manifests/v6')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v6.yaml')
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != PROTOCOL_STATUS:
        raise ValueError('v6 protocol is not prespecified')
    if args.seed not in protocol['training']['seeds']['candidates']:
        raise ValueError('seed is not prespecified for this arm')
    v4 = yaml.safe_load((ROOT/'configs/protocol_v4.yaml').read_text())
    training, common, arm = protocol['training'], v4['arms']['common'], arm_spec(v4, OBJECTIVE)
    residual_arm = args.arm == RESIDUAL_ARM
    name = f'{args.arm}_v6_s{args.seed}'
    out = args.output_root/'fits'
    out.mkdir(parents=True, exist_ok=True)
    if (out/f'{name}.json').exists():
        raise ValueError(f'fit {name} already exists')

    contexts = read_tsv(args.manifests/'contexts.tsv').context_id.tolist()
    lines = counterfactual.cell_lines(contexts)
    obs, counts, library = load_model_archive(args.prepared/'model_cells.npz')
    query_table = v3_runner.read_query_table(args.prepared/'queries.tsv')
    vehicle_contexts = pd.read_csv(args.prepared/'vehicle_contexts.tsv', sep='\t')
    truth = {s: load_truth(args.prepared/f'{s}_truth.npz') for s in ['train', 'validation']}
    with np.load(args.prepared/'validation_truth.npz', allow_pickle=False) as z:
        evaluation_truth = by_key(z['keys'], z['delta_evaluation'])
        selection_truth = by_key(z['keys'], z['delta_selection'])
    with np.load(args.prepared/'endpoint/de_sets.npz', allow_pickle=False) as z:
        de_sets = {k: z[k] for k in ['keys', 'genes']}
        responder = by_key(z['keys'], z['responder'])
    with np.load(args.prepared/'train_residual_targets.npz', allow_pickle=False) as z:
        residual_target = by_key(z['keys'], z['residual'])
        residual_missing = [tuple(k.split('|')) for k in np.asarray(z['fallback_keys'], str)]
    b1p = {s: load_baseline(args.baselines, 'B1p', s) for s in ['validation', 'test']}

    train_mask, validation_mask = counterfactual.setting_cells(obs, contexts, vehicle_contexts)
    keep = np.flatnonzero(train_mask | validation_mask)
    sub = obs.iloc[keep].reset_index(drop=True)
    sub_counts, sub_library = counts[keep], library[keep]
    queries = {}
    for split in ['train', 'validation', 'test']:
        built = counterfactual.build_queries(query_table, contexts, [split])
        queries[split] = built if split == 'test' else {k: v for k, v in built.items() if k in truth[split]}
    for split in ['train', 'validation']:
        if set(queries[split]) != set(truth[split]):
            raise ValueError(f'{split} queries do not cover the {split} truth')
    wells = sorted({r['vehicle_well'] for q in queries.values() for v in q.values() for r in v})
    vehicles = v3_runner.vehicle_wells(sub, sub_counts, sub_library, wells)
    if not set(sub.loc[sub.Combo.isin(wells), 'role']) <= {'vehicle'}:
        raise ValueError('counterfactual source wells must be vehicle units')
    width = sub_counts.shape[1]
    reference = diagnostics.context_means(truth['train'])

    compounds = read_tsv(args.manifests/'compounds.tsv')
    compounds = compounds[compounds.status.eq('accepted')].reset_index(drop=True)
    train_ids = compounds.loc[compounds.split.eq('train'), 'compound_id'].tolist()
    descriptors, kept = compound_table(compounds.compound_id.tolist(), compounds.canonical_smiles.tolist(), train_ids,
                                       min_std=NOISE_STD_TOLERANCE)
    scaling = scaling_diagnostics(descriptors, dict(zip(compounds.compound_id, compounds.split)))  # gated by v6_seal.py
    vocabulary = [VEHICLE_ID, *compounds.compound_id.tolist()]
    X = normalize_counts(sub_counts, sub_library)
    train_index = np.flatnonzero(train_mask[keep])

    # The one difference between the arms: what the pseudobulk loss is trained against.
    target_keys = sorted(k for k in queries['train'] if not residual_arm or k in residual_target)
    if not target_keys:
        raise ValueError('no training key has a target for this arm')
    target_values = np.vstack([(residual_target if residual_arm else truth['train'])[k] for k in target_keys])
    target = v5_chemcpa.WeightedPseudobulkTarget(
        pairs=counterfactual.pseudobulk_pairs({k: queries['train'][k] for k in target_keys}, target_keys),
        truth=target_values, vehicle_cells=vehicles,
        pairs_per_update=arm['pseudobulk']['pairs_per_update'],
        cells_per_well=arm['pseudobulk']['cells_per_vehicle_well'], seed=args.seed+1, device=args.device)
    model = v5_chemcpa.ChemCPAv5(
        likelihood=arm['likelihood'], gene_offset=None, pseudobulk_weight=arm['pseudobulk_weight'],
        pseudobulk_target=target, hparams=common['hparams'],
        additional=dict(doser_type=common['doser_type'], decoder_activation=arm['decoder_activation']),
        n_genes=width, drug_vocabulary=vocabulary,
        drug_matrix=np.vstack([descriptors[c] for c in vocabulary]), cell_types=lines, use_cell_type=True,
        device=args.device, seed=args.seed)
    predict = lambda well, condition: model.predict_mean(well['values'], condition)
    deltas = lambda q: counterfactual.counterfactual_deltas(predict, q, vehicles, gene_count=width)

    def in_delta_space(predicted, split):
        """An arm's prediction of the measured delta; the residual arm adds B1p back."""
        if not residual_arm:
            return predicted
        offset = b1p[split]
        return {k: v+offset[k] for k, v in predicted.items() if k in offset}

    nonfinite = []

    def validate(m):
        """Delta-space validation RMSE; a non-finite evaluation scores +inf and is never selected (v4 A2)."""
        try:
            predicted = in_delta_space(deltas(queries['validation'])[0], 'validation')
            return v3_runner.mean_query_rmse(predicted, {k: truth['validation'][k] for k in predicted})
        except ValueError as error:
            if 'finite' not in str(error):
                raise
            nonfinite.append(str(error))
            return float('inf')

    t = sub.iloc[train_index]
    result = chemcpa_v4.fit(model, X[train_index], t.compound_id.tolist(), t.dose_nM.fillna(0).tolist(),
                            t.cell_type.tolist(), batch_size=common['hparams']['batch_size'],
                            max_epochs=args.max_epochs or training['max_epochs'],
                            evaluate_every=training['evaluate_every_epochs'],
                            patience=training['early_stopping_patience_evaluations'],
                            max_minutes=training['max_minutes_per_fit'], seed=args.seed, validate=validate,
                            counts=None, library=sub_library[train_index])
    history = pd.DataFrame(result.pop('history'))
    raw = {s: deltas(q) for s, q in queries.items()}
    predictions = {s: (in_delta_space(contrast, s) if s != 'train' else contrast, anchored)
                   for s, (contrast, anchored) in raw.items()}
    train_means = diagnostics.context_means(raw['train'][0])
    summary = {}
    for split in ['train', 'validation']:
        against = residual_target if (residual_arm and split == 'train') else truth[split]
        predicted = {k: v for k, v in predictions[split][0].items() if k in against}
        summary[split] = diagnostics.summarize(predicted, {k: against[k] for k in predicted}, reference, train_means)

    # The endpoint quantities, on validation responder keys: the arm-selection metric and residual_r.
    validation_keys = sorted(k for k in predictions['validation'][0] if k in evaluation_truth)
    responders = [k for k in validation_keys if bool(responder.get(k, False))]
    endpoint_summary = dict(
        validation_direction_accuracy=scoring.mean_accuracy(predictions['validation'][0], evaluation_truth,
                                                            de_sets, responders),
        validation_direction_accuracy_all_keys=scoring.mean_accuracy(predictions['validation'][0], evaluation_truth,
                                                                     de_sets, validation_keys),
        b1p_direction_accuracy=scoring.mean_accuracy(b1p['validation'], evaluation_truth, de_sets, responders),
        split_noise_floor=scoring.split_noise_floor(selection_truth, evaluation_truth, de_sets, responders),
        residual_r=scoring.residual_correlation(predictions['validation'][0], truth['validation'],
                                                b1p['validation'], validation_keys),
        residual_share=scoring.residual_share(truth['validation'], b1p['validation'], validation_keys),
        n_validation_responders=len(responders), n_validation_keys=len(validation_keys))

    latent = {}
    rng = np.random.default_rng(2026)
    for split in ['train', 'validation']:
        per_line = {}
        for line in lines:
            keys = [k for k in queries[split] if queries[split][k][0]['treated']['cell_type'] == line]
            if not keys:
                continue
            source = sorted({r['vehicle_well'] for k in keys for r in queries[split][k]})
            values = np.vstack([vehicles[w]['values'] for w in source])
            values = values[rng.choice(len(values), size=min(2000, len(values)), replace=False)]
            conditions = [(k[0], queries[split][k][0]['treated']['dose_nM']) for k in keys]
            per_line[line] = model.latent_summary(values, conditions, line)
        latent[split] = dict(per_cell_line=per_line, **{
            k: float(np.mean([v[k] for v in per_line.values()])) for k in next(iter(per_line.values()))})

    arrays = {}
    for split, (contrast, anchored) in predictions.items():
        for k, v in v3_runner.prediction_arrays(contrast, anchored, sorted(contrast)).items():
            arrays[f'{split}_{k}'] = v
    for split, (contrast, _) in raw.items():
        arrays[f'{split}_raw'] = scoring.stack(contrast, sorted(contrast))
    np.savez_compressed(out/f'{name}.npz', **arrays)
    history.to_csv(out/f'{name}_history.tsv', sep='\t', index=False)
    record = dict(name=name, analysis='v6', arm=args.arm, objective=OBJECTIVE, residual_target=residual_arm,
                  arm_spec={k: v for k, v in arm.items() if k != 'pseudobulk'}, max_epochs_override=args.max_epochs,
                  engineering_smoke=args.max_epochs is not None, n_training_cells=int(train_index.size),
                  n_validation_cells=int(validation_mask.sum()), n_target_keys=len(target_keys),
                  n_training_keys_without_a_target=len(residual_missing) if residual_arm else 0,
                  n_train_queries=len(queries['train']), n_validation_queries=len(queries['validation']),
                  n_test_queries=len(queries['test']), n_vehicle_units=len(vehicles), cell_lines=lines,
                  descriptor_columns=len(kept), descriptor_min_std=NOISE_STD_TOLERANCE, descriptor_scaling=scaling,
                  seconds=time.perf_counter()-started, device=args.device,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                  test_outcomes_loaded=False, evaluation_half_test_delta_loaded=False,
                  diagnostics=summary, endpoint=endpoint_summary, latent=latent,
                  nonfinite_validation_evaluations=len(nonfinite), **result)
    record['validation_rmse'] = summary['validation']['rmse']
    record['validation_direction_accuracy'] = endpoint_summary['validation_direction_accuracy']
    record['history_tail'] = history.tail(3).to_dict('records')
    (out/f'{name}.json').write_text(json.dumps(diagnostics.finite(record), indent=2, allow_nan=False)+'\n')
    print(json.dumps(diagnostics.finite({k: record[k] for k in [
        'name', 'seconds', 'epochs_executed', 'stopped', 'validation_rmse',
        'validation_direction_accuracy']})), flush=True)


if __name__ == '__main__':
    main()
