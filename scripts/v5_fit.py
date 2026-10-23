"""Fit one v5 chemCPA arm and seed on Tahoe-100M prepared inputs; never reads test outcomes.

Arms, hyperparameters, objectives and the A2 cap are v4's (configs/protocol_v4.yaml
arms; scripts/v4_fit.py arm_spec). v5 changes only what protocol_v5 prespecifies
or the data require: descriptors drop training columns with std <= NOISE_STD_TOLERANCE;
contexts are the 30 (cell line, dose) contexts with the cell line as covariate;
a key's contrast is the equal-weight mean over its one to three replicates, and
the pseudobulk loss weights replicates the same way (drugperturb.v5.chemcpa).
Writes training, validation and test counterfactual deltas, the training
history, training/validation diagnostics and the descriptor scaling summary
that the seal gate checks.
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
from drugperturb.v5.selection import ARMS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def fit_seeds(protocol, arm):
    seeds = protocol['training']['seeds']
    return seeds['A0'] if arm == 'A0_v3_reference' else seeds['candidates']


def read_tsv(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'replicate': str})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-epochs', type=int, help='Engineering smoke runs only; recorded in the fit record')
    parser.add_argument('--manifests', type=Path, default=ROOT/'manifests/v5')
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/protocol_v5.yaml')
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = yaml.safe_load(args.protocol.read_text())
    if protocol['status'] != 'PRESPECIFIED_BEFORE_V5_FITTING':
        raise ValueError('v5 protocol is not prespecified')
    if args.seed not in fit_seeds(protocol, args.arm):
        raise ValueError('seed is not prespecified for this arm')
    v4 = yaml.safe_load((ROOT/'configs/protocol_v4.yaml').read_text())
    training, common, arm = protocol['training'], v4['arms']['common'], arm_spec(v4, args.arm)
    name = f'{args.arm}_v5_s{args.seed}'
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
    scaling = scaling_diagnostics(descriptors, dict(zip(compounds.compound_id, compounds.split)))  # gated by v5_seal.py
    vocabulary = [VEHICLE_ID, *compounds.compound_id.tolist()]
    X = normalize_counts(sub_counts, sub_library)
    train_index = np.flatnonzero(train_mask[keep])
    additional = dict(doser_type=common['doser_type'], decoder_activation=arm['decoder_activation'])
    target = None
    if arm['pseudobulk_weight']:
        keys = sorted(queries['train'])
        target = v5_chemcpa.WeightedPseudobulkTarget(
            pairs=counterfactual.pseudobulk_pairs(queries['train'], keys),
            truth=np.vstack([truth['train'][k] for k in keys]), vehicle_cells=vehicles,
            pairs_per_update=arm['pseudobulk']['pairs_per_update'],
            cells_per_well=arm['pseudobulk']['cells_per_vehicle_well'], seed=args.seed+1, device=args.device)
    offset = (chemcpa_v4.gene_offsets(sub_counts[train_index], sub_library[train_index])
              if arm['likelihood'] == 'nb' else None)
    model = v5_chemcpa.ChemCPAv5(
        likelihood=arm['likelihood'], gene_offset=offset, pseudobulk_weight=arm['pseudobulk_weight'],
        pseudobulk_target=target, hparams=common['hparams'], additional=additional, n_genes=width,
        drug_vocabulary=vocabulary, drug_matrix=np.vstack([descriptors[c] for c in vocabulary]),
        cell_types=lines, use_cell_type=True, device=args.device, seed=args.seed)
    predict = lambda well, condition: model.predict_mean(well['values'], condition)
    deltas = lambda q: counterfactual.counterfactual_deltas(predict, q, vehicles, gene_count=width)
    nonfinite = []

    def validate(m):
        """Validation RMSE; an evaluation with non-finite predictions scores +inf and is never selected (v4 A2)."""
        try:
            return v3_runner.mean_query_rmse(deltas(queries['validation'])[0], truth['validation'])
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
                            counts=sub_counts[train_index] if arm['likelihood'] == 'nb' else None,
                            library=sub_library[train_index])
    history = pd.DataFrame(result.pop('history'))
    predictions = {s: deltas(q) for s, q in queries.items()}
    train_means = diagnostics.context_means(predictions['train'][0])
    summary = {s: diagnostics.summarize(predictions[s][0], truth[s], reference, train_means)
               for s in ['train', 'validation']}
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
    np.savez_compressed(out/f'{name}.npz', **arrays)
    history.to_csv(out/f'{name}_history.tsv', sep='\t', index=False)
    record = dict(name=name, analysis='v5', arm=args.arm, seed=args.seed,
                  arm_spec={k: v for k, v in arm.items() if k != 'pseudobulk'}, max_epochs_override=args.max_epochs,
                  engineering_smoke=args.max_epochs is not None, n_training_cells=int(train_index.size),
                  n_validation_cells=int(validation_mask.sum()), n_train_queries=len(queries['train']),
                  n_validation_queries=len(queries['validation']), n_test_queries=len(queries['test']),
                  n_vehicle_units=len(vehicles), cell_lines=lines, descriptor_columns=len(kept),
                  descriptor_min_std=NOISE_STD_TOLERANCE, descriptor_scaling=scaling,
                  seconds=time.perf_counter()-started, device=args.device,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                  test_outcomes_loaded=False, diagnostics=summary, latent=latent,
                  nonfinite_validation_evaluations=len(nonfinite), **result)
    record['validation_rmse'] = summary['validation']['rmse']
    record['history_tail'] = history.tail(3).to_dict('records')
    (out/f'{name}.json').write_text(json.dumps(diagnostics.finite(record), indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: record[k] for k in ['name', 'seconds', 'epochs_executed', 'stopped']}), flush=True)


if __name__ == '__main__':
    main()
