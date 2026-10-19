"""Fit one v3 cell-level model (method, protocol, candidate, seed); never reads test outcomes.

Inputs are the response-free prepare outputs: model-cell archive (training,
validation and vehicle cells only), counterfactual query table, vehicle-well
contexts and validation-compound truth. Writes validation/test counterfactual
deltas and a fit record for later selection and sealing by v3_seal.py.
"""
from pathlib import Path
import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import torch
import yaml

from drugperturb.v3 import runner
from drugperturb.v3.cells import load_model_archive
from drugperturb.v3.descriptors import VEHICLE_ID, compound_table
from drugperturb.v3.pseudobulk import normalize_counts

ROOT = Path(__file__).resolve().parents[1]


def load_inputs(prepared):
    obs, counts, library = load_model_archive(prepared/'model_cells.npz')
    queries = runner.read_query_table(prepared/'queries.tsv')
    vehicle_contexts = pd.read_csv(prepared/'vehicle_contexts.tsv', sep='\t')
    with np.load(prepared/'validation_truth.npz', allow_pickle=False) as z:
        truth = {tuple(k.split('|')): z['delta'][i] for i, k in enumerate(z['keys'])}
    return obs, counts, library, queries, vehicle_contexts, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['chemcpa', 'biolord'], required=True)
    parser.add_argument('--protocol', choices=['Q1', 'Q2'], required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-epochs', type=int, help='Engineering smoke runs only; recorded in the fit record')
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = yaml.safe_load((ROOT/'configs/protocol_v3.yaml').read_text())
    if protocol['status'] != 'PRESPECIFIED_BEFORE_V3_CELL_MODEL_FITTING':
        raise ValueError('v3 protocol is not prespecified')
    spec = protocol['models']['chemCPA' if args.method == 'chemcpa' else 'biolord']
    if args.candidate not in spec['candidates'] or args.seed not in protocol['models']['seeds']:
        raise ValueError('candidate/seed is not prespecified')
    candidate = spec['candidates'][args.candidate]
    name = f'{args.method}_{args.protocol}_{args.candidate}_s{args.seed}'
    out = args.output_root/'fits'
    out.mkdir(parents=True, exist_ok=True)
    if (out/f'{name}.json').exists():
        raise ValueError(f'fit {name} already exists')

    obs, counts, library, query_table, vehicle_contexts, truth = load_inputs(args.prepared)
    train_mask, validation_mask, cell_types = runner.protocol_cells(obs, args.protocol, vehicle_contexts)
    keep = np.flatnonzero(train_mask | validation_mask)
    sub = obs.iloc[keep].reset_index(drop=True)
    sub_counts, sub_library = counts[keep], library[keep]
    contexts = runner.PROTOCOL_CONTEXTS[args.protocol]
    val_queries = runner.build_queries(query_table, contexts, ['validation'])
    val_queries = {k: v for k, v in val_queries.items() if k in truth}
    val_truth = {k: truth[k] for k in val_queries}
    test_queries = runner.build_queries(query_table, contexts, ['test'])
    wells = [r['vehicle_well'] for q in [*val_queries.values(), *test_queries.values()] for r in q]
    vehicles = runner.vehicle_wells(sub, sub_counts, sub_library, wells)
    if not set(sub.loc[sub.Combo.isin(wells), 'role']) <= {'vehicle'}:
        raise ValueError('counterfactual source wells must be vehicle wells')
    width = sub_counts.shape[1]

    split = pd.read_csv(ROOT/'manifests/split_manifest.tsv', sep='\t', keep_default_na=False)
    descriptors, kept = compound_table(split.compound_id.tolist(), split.canonical_smiles.tolist(),
                                       split.loc[split.split.eq('train'), 'compound_id'].tolist())
    use_cell_type = args.protocol == 'Q2'
    X = normalize_counts(sub_counts, sub_library)
    train_index = np.flatnonzero(train_mask[keep])
    if args.method == 'chemcpa':
        from drugperturb.v3 import chemcpa
        vocabulary = [VEHICLE_ID, *split.compound_id.tolist()]
        model = chemcpa.ChemCPA(hparams=candidate['hparams'], additional=candidate['additional'], n_genes=width,
                                drug_vocabulary=vocabulary, drug_matrix=np.vstack([descriptors[c] for c in vocabulary]),
                                cell_types=cell_types, use_cell_type=use_cell_type, device=args.device, seed=args.seed)
        predict = lambda m: (lambda well, condition: m.predict_mean(well['values'], condition))
        validate = lambda m: runner.mean_query_rmse(
            runner.predict_queries(predict(m), val_queries, vehicles, width)[0], val_truth)
        t = sub.iloc[train_index]
        result = chemcpa.fit(model, X[train_index], t.compound_id.tolist(), t.dose_nM.fillna(0).tolist(),
                             t.cell_type.tolist(), batch_size=candidate['hparams']['batch_size'],
                             max_epochs=args.max_epochs or spec['max_epochs'],
                             evaluate_every=spec['evaluate_every_epochs'],
                             patience=spec['early_stopping_patience_evaluations'],
                             max_minutes=spec['max_minutes_per_fit'], seed=args.seed, validate=validate)
        history = pd.DataFrame(result.pop('history'))
        predictor = predict(model)
    else:
        from drugperturb.v3 import biolord_model
        adata = biolord_model.build_adata(X, sub, descriptors, cell_types, use_cell_type)
        model = biolord_model.Biolord(adata, candidate=candidate, seed=args.seed, use_cell_type=use_cell_type,
                                      cell_types=cell_types, model_name=name)
        settings = dict(candidate)
        if args.max_epochs:
            settings['train'] = dict(candidate['train'], max_epochs=args.max_epochs)
        history = model.fit(settings, 'gpu' if args.device.startswith('cuda') else 'cpu')
        predictor = lambda well, condition: model.predict_mean(well, condition, descriptors)
        result = dict(epochs_executed=int(history.epoch.max()) if 'epoch' in history else None)
    val_contrast, val_anchored = runner.predict_queries(predictor, val_queries, vehicles, width)
    result['validation_rmse'] = runner.mean_query_rmse(val_contrast, val_truth)
    result['validation_rmse_anchored'] = runner.mean_query_rmse(val_anchored, val_truth)
    test_contrast, test_anchored = runner.predict_queries(predictor, test_queries, vehicles, width)
    val_keys, test_keys = sorted(val_contrast), sorted(test_contrast)
    np.savez_compressed(out/f'{name}.npz',
                        **{f'validation_{k}': v for k, v in runner.prediction_arrays(val_contrast, val_anchored, val_keys).items()},
                        **{f'test_{k}': v for k, v in runner.prediction_arrays(test_contrast, test_anchored, test_keys).items()})
    history.to_csv(out/f'{name}_history.tsv', sep='\t', index=False)
    record = dict(name=name, method=args.method, protocol=args.protocol, candidate=args.candidate, seed=args.seed,
                  max_epochs_override=args.max_epochs, engineering_smoke=args.max_epochs is not None,
                  n_training_cells=int(train_index.size), n_validation_cells=int(validation_mask.sum()),
                  n_validation_queries=len(val_keys), n_test_queries=len(test_keys),
                  n_vehicle_wells=len(vehicles), descriptor_columns=len(kept), seconds=time.perf_counter()-started,
                  device=args.device, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                  test_outcomes_loaded=False, **result)
    (out/f'{name}.json').write_text(json.dumps(record, indent=2, allow_nan=False, default=float)+'\n')
    print(json.dumps({k: record[k] for k in ['name', 'validation_rmse', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
