"""Fit one v4 chemCPA arm (setting, arm, seed); never reads test outcomes.

Inputs are response-free prepare outputs (v3 K2 for sci-Plex3, v4 O2 for OP3):
model-cell archive without test cells, counterfactual query table, vehicle-well
contexts and training/validation truth. Writes training, validation and test
counterfactual deltas, the training history, and training/validation collapse
diagnostics for sealing by v4_seal.py.
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

from drugperturb.v3 import runner as v3_runner
from drugperturb.v3.cells import load_model_archive
from drugperturb.v3.descriptors import VEHICLE_ID, compound_table
from drugperturb.v3.pseudobulk import normalize_counts
from drugperturb.v4 import chemcpa_v4, counterfactual, diagnostics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['A0_v3_reference', 'I1_nb_likelihood', 'I2_counterfactual_pseudobulk', 'I3_nb_plus_pseudobulk']


def load_truth(path):
    with np.load(path, allow_pickle=False) as z:
        return {tuple(k.split('|')): z['delta'][i] for i, k in enumerate(z['keys'])}


def read(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'replicate': str, 'dose_pattern': str})


def arm_spec(protocol, arm):
    """Resolve likelihood, decoder activation and pseudobulk weight of an arm."""
    arms = protocol['arms']
    if arm == 'I3_nb_plus_pseudobulk':
        nb, pb = arms['I1_nb_likelihood'], arms['I2_counterfactual_pseudobulk']
        return dict(likelihood='nb', decoder_activation=nb['decoder_activation'],
                    pseudobulk_weight=pb['pseudobulk_weight'], pseudobulk=pb)
    spec = arms[arm]
    return dict(likelihood='nb' if spec['likelihood'].startswith('negative_binomial') else 'gaussian',
                decoder_activation=spec['decoder_activation'], pseudobulk_weight=spec['pseudobulk_weight'],
                pseudobulk=arms['I2_counterfactual_pseudobulk'])


def load_setting(setting, prepared):
    spec = counterfactual.SETTINGS[setting]
    obs, counts, library = load_model_archive(prepared/'model_cells.npz')
    query_table = v3_runner.read_query_table(prepared/'queries.tsv')
    vehicle_contexts = pd.read_csv(prepared/'vehicle_contexts.tsv', sep='\t')
    truth = {s: load_truth(prepared/f'{s}_truth.npz') for s in ['train', 'validation']}
    if spec['dataset'] == 'sciplex3':
        compounds = read(ROOT/'manifests/split_manifest.tsv')
        train_rows = counterfactual.sciplex_training_queries(read(ROOT/'manifests/v2/well_mapping.tsv'), spec['contexts'])
    else:
        compounds = read(ROOT/'manifests/v4/op3_compounds.tsv')
        train_rows = query_table[query_table.split.eq('train')]
    return obs, counts, library, query_table, train_rows, vehicle_contexts, truth, compounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=list(counterfactual.SETTINGS), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-epochs', type=int, help='Engineering smoke runs only; recorded in the fit record')
    args = parser.parse_args()
    started = time.perf_counter()
    protocol = yaml.safe_load((ROOT/'configs/protocol_v4.yaml').read_text())
    if protocol['status'] != 'PRESPECIFIED_BEFORE_V4_FITTING':
        raise ValueError('v4 protocol is not prespecified')
    if args.seed not in protocol['training']['seeds'][args.setting]:
        raise ValueError('seed is not prespecified for this setting')
    spec, training, common = counterfactual.SETTINGS[args.setting], protocol['training'], protocol['arms']['common']
    arm = arm_spec(protocol, args.arm)
    name = f'{args.arm}_{args.setting}_s{args.seed}'
    out = args.output_root/'fits'
    out.mkdir(parents=True, exist_ok=True)
    if (out/f'{name}.json').exists():
        raise ValueError(f'fit {name} already exists')

    obs, counts, library, query_table, train_rows, vehicle_contexts, truth, compounds = load_setting(
        args.setting, args.prepared)
    contexts, R = spec['contexts'], spec['replicates']
    train_mask, validation_mask, cell_types = counterfactual.setting_cells(obs, contexts, vehicle_contexts)
    keep = np.flatnonzero(train_mask | validation_mask)
    sub = obs.iloc[keep].reset_index(drop=True)
    sub_counts, sub_library = counts[keep], library[keep]
    queries = {}
    for split, rows in [('train', train_rows), ('validation', query_table), ('test', query_table)]:
        built = counterfactual.build_queries(rows, contexts, [split], R)
        queries[split] = built if split == 'test' else {k: v for k, v in built.items() if k in truth[split]}
    if set(truth['validation']) & set(queries['validation']) != set(k for k in truth['validation'] if k[1] in contexts):
        raise ValueError('validation queries do not cover the validation truth')
    wells = sorted({r['vehicle_well'] for q in queries.values() for v in q.values() for r in v})
    vehicles = v3_runner.vehicle_wells(sub, sub_counts, sub_library, wells)
    if not set(sub.loc[sub.Combo.isin(wells), 'role']) <= {'vehicle'}:
        raise ValueError('counterfactual source wells must be vehicle wells')
    width = sub_counts.shape[1]
    reference = diagnostics.context_means({k: v for k, v in truth['train'].items() if k[1] in contexts})

    train_ids = compounds.loc[compounds.split.eq('train'), 'compound_id'].tolist()
    descriptors, kept = compound_table(compounds.compound_id.tolist(), compounds.canonical_smiles.tolist(), train_ids)
    vocabulary = [VEHICLE_ID, *compounds.compound_id.tolist()]
    X = normalize_counts(sub_counts, sub_library)
    train_index = np.flatnonzero(train_mask[keep])
    additional = dict(doser_type=common['doser_type'], decoder_activation=arm['decoder_activation'])
    target = None
    if arm['pseudobulk_weight']:
        keys = sorted(queries['train'])
        pairs = [dict(compound=k[0], dose_nM=queries['train'][k][0]['treated']['dose_nM'],
                      cell_type=queries['train'][k][0]['treated']['cell_type'],
                      wells=[r['vehicle_well'] for r in queries['train'][k]]) for k in keys]
        target = chemcpa_v4.PseudobulkTarget(
            pairs=pairs, truth=np.vstack([truth['train'][k] for k in keys]), vehicle_cells=vehicles, n_replicates=R,
            pairs_per_update=arm['pseudobulk']['pairs_per_update'],
            cells_per_well=arm['pseudobulk']['cells_per_vehicle_well'], seed=args.seed+1, device=args.device)
    offset = (chemcpa_v4.gene_offsets(sub_counts[train_index], sub_library[train_index])
              if arm['likelihood'] == 'nb' else None)
    model = chemcpa_v4.ChemCPAv4(
        likelihood=arm['likelihood'], gene_offset=offset, pseudobulk_weight=arm['pseudobulk_weight'],
        pseudobulk_target=target, hparams=common['hparams'], additional=additional, n_genes=width,
        drug_vocabulary=vocabulary, drug_matrix=np.vstack([descriptors[c] for c in vocabulary]),
        cell_types=cell_types, use_cell_type=spec['use_cell_type'], device=args.device, seed=args.seed)
    predict = lambda well, condition: model.predict_mean(well['values'], condition)
    deltas = lambda q: counterfactual.counterfactual_deltas(predict, q, vehicles, gene_count=width, n_replicates=R)
    validate = lambda m: v3_runner.mean_query_rmse(deltas(queries['validation'])[0], truth['validation'])
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
        per_type = {}
        for cell in cell_types:
            keys = [k for k in queries[split] if queries[split][k][0]['treated']['cell_type'] == cell]
            source = sorted({r['vehicle_well'] for k in keys for r in queries[split][k]})
            if not keys:
                continue
            values = np.vstack([vehicles[w]['values'] for w in source])
            values = values[rng.choice(len(values), size=min(2000, len(values)), replace=False)]
            conditions = [(k[0], queries[split][k][0]['treated']['dose_nM']) for k in keys]
            per_type[cell] = model.latent_summary(values, conditions, cell)
        latent[split] = dict(per_cell_type=per_type, **{
            k: float(np.mean([v[k] for v in per_type.values()])) for k in next(iter(per_type.values()))})
    arrays = {}
    for split, (contrast, anchored) in predictions.items():
        keys = sorted(contrast)
        for k, v in v3_runner.prediction_arrays(contrast, anchored, keys).items():
            arrays[f'{split}_{k}'] = v
    np.savez_compressed(out/f'{name}.npz', **arrays)
    history.to_csv(out/f'{name}_history.tsv', sep='\t', index=False)
    record = dict(name=name, setting=args.setting, arm=args.arm, seed=args.seed, arm_spec={k: v for k, v in arm.items()
                  if k != 'pseudobulk'}, max_epochs_override=args.max_epochs, engineering_smoke=args.max_epochs is not None,
                  n_training_cells=int(train_index.size), n_validation_cells=int(validation_mask.sum()),
                  n_train_queries=len(queries['train']), n_validation_queries=len(queries['validation']),
                  n_test_queries=len(queries['test']), n_vehicle_wells=len(vehicles), descriptor_columns=len(kept),
                  seconds=time.perf_counter()-started, device=args.device,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                  test_outcomes_loaded=False, diagnostics=summary, latent=latent, **result)
    record['validation_rmse'] = summary['validation']['rmse']
    record['history_tail'] = history.tail(3).to_dict('records')
    (out/f'{name}.json').write_text(json.dumps(diagnostics.finite(record), indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: record[k] for k in ['name', 'seconds', 'epochs_executed', 'stopped']}), flush=True)


if __name__ == '__main__':
    main()
