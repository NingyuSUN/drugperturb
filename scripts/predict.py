"""Infer a fixed A549/24h/1000nM response from trusted, sealed project artifacts.

No training or test outcomes are loaded. Only this project's hash-verified
transform and selected checkpoints can be deserialized; arbitrary model paths
are deliberately absent from the public interface.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import numpy as np
import pandas as pd
import joblib
import torch
import yaml
from drugperturb.chemistry import standardize_smiles, morgan_fingerprint, molecular_graph
from drugperturb.models import MLP, MolecularGNN, predict_network

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _verify_hashes(receipt):
    verified = {}
    for relative, expected in receipt['artifact_hashes'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError('Receipt artifact path escapes the project')
        if sha(path) != expected:
            raise ValueError(f'Artifact hash mismatch: {relative}')
        verified[relative] = expected
    return verified


def predict_smiles(smiles, *, cell_type='A549', time_hours=24, dose_nM=1000, model='M3'):
    """Return selected-gene prediction table and provenance, without fitting."""
    if cell_type != 'A549' or time_hours != 24 or dose_nM != 1000:
        raise ValueError('Supported context only: A549, 24 hours, 1000 nM')
    if model not in ('M3', 'M2', 'M1'):
        raise ValueError('model must be M3, M2, or M1')
    structure = standardize_smiles(smiles)  # Reject invalid/ambiguous input first.
    seal_path = ROOT / 'receipts/prediction_seal.json'
    freeze_path = ROOT / 'receipts/dataset_freeze.json'
    seal = json.loads(seal_path.read_text())
    if sha(freeze_path) != seal['dataset_freeze_sha256']:
        raise ValueError('Frozen dataset receipt hash mismatch')
    sealed_hashes = _verify_hashes(seal)
    freeze = json.loads(freeze_path.read_text())
    # These are the only dataset artifacts needed for inference; no truth arrays.
    required = ['configs/protocol_v1.yaml', 'data/processed/target_transform.joblib',
                'data/processed/development.npz', 'manifests/selected_genes.tsv',
                'manifests/split_manifest.tsv']
    for relative in required:
        expected = freeze['artifact_hashes'][relative]
        if sha(ROOT / relative) != expected:
            raise ValueError(f'Frozen artifact hash mismatch: {relative}')
    selection_path = 'receipts/model_selection.json'
    if selection_path not in sealed_hashes:
        raise ValueError('Model selection is absent from prediction seal')
    selection = json.loads((ROOT / selection_path).read_text())
    protocol = yaml.safe_load((ROOT / 'configs/protocol_v1.yaml').read_text())
    if protocol['status'] != 'FROZEN_BEFORE_TRAINING':
        raise ValueError('Training protocol is not frozen')
    cfg = protocol['models']
    hp = selection['selected'][model]
    candidate = int(hp['candidate'])
    if candidate < 0:
        raise ValueError('Selected candidate must be nonnegative')
    seeds = list(cfg['training_seeds']) if model != 'M1' else []
    if model != 'M1' and (len(seeds) != 3 or len(set(seeds)) != 3 or
            selection['prediction_policy'] != 'three_seed_arithmetic_mean_for_M2_M3' or
            cfg['neural_final_prediction'] != 'arithmetic_mean_of_three_seed_predictions'):
        raise ValueError('Expected the frozen three-seed arithmetic-mean prediction policy')
    checkpoints = ([f'models/run_v1/M1_c{candidate}.joblib'] if model == 'M1' else
                   [f'models/run_v1/{model}_c{candidate}_s{seed}.pt' for seed in seeds])
    if any(path not in sealed_hashes for path in checkpoints):
        raise ValueError('Selected checkpoint is absent from prediction seal')
    # Deserialization follows all required provenance and content checks.
    transform = joblib.load(ROOT / 'data/processed/target_transform.joblib')
    genes = pd.read_csv(ROOT / 'manifests/selected_genes.tsv', sep='\t')
    if (genes.gene_id.isna().any() or genes.gene_id.duplicated().any() or
            genes.output_index.tolist() != list(range(len(genes))) or
            len(genes) != len(transform.gene_indices)):
        raise ValueError('Selected gene identity/order differs from the frozen transform')
    fingerprint = morgan_fingerprint(structure['canonical_smiles'])[None, :]
    with np.load(ROOT / 'data/processed/development.npz', allow_pickle=False) as development:
        train_fp = development['train_fp'].astype(np.float64)
        train_ids = development['train_ids'].copy()
    split = pd.read_csv(ROOT / 'manifests/split_manifest.tsv', sep='\t')
    if train_ids.tolist() != split.loc[split.split.eq('train'), 'compound_id'].tolist():
        raise ValueError('Training fingerprint identity/order differs from frozen split')
    if (train_fp.ndim != 2 or len(train_fp) == 0 or train_fp.shape[1] != fingerprint.shape[1] or
            len(train_fp) != len(train_ids) or not np.isin(train_fp, [0, 1]).all()):
        raise ValueError('Training fingerprints must be finite aligned binary rows')
    if model == 'M1':
        fitted = joblib.load(ROOT / checkpoints[0])
        coordinates = fitted.predict(fingerprint)
    else:
        inputs = fingerprint if model == 'M2' else [molecular_graph(structure['canonical_smiles'])]
        predictions = []
        torch.set_num_threads(2)
        for checkpoint in checkpoints:
            network = (MLP(fingerprint.shape[1], transform.n_components, hp['hidden_dim'], hp['dropout'])
                       if model == 'M2' else MolecularGNN(transform.n_components,
                       hidden_dim=hp['hidden_dim'], layers=cfg['gnn_layers'], dropout=hp['dropout']))
            state = torch.load(ROOT / checkpoint, map_location='cpu', weights_only=True)
            network.load_state_dict(state)
            predictions.append(predict_network(network, inputs))
        coordinates = np.mean(predictions, axis=0)
    delta = transform.inverse_transform(coordinates)
    if delta.shape != (1, len(genes)) or not np.isfinite(delta).all():
        raise ValueError('Predicted delta must be finite and match the selected gene order')
    intersection = train_fp @ fingerprint[0]
    union = train_fp.sum(axis=1) + fingerprint.sum() - intersection
    similarity = np.divide(intersection, union, out=np.ones_like(intersection), where=union != 0)
    nearest = int(similarity.argmax())
    table = pd.DataFrame({'gene_id': genes.gene_id, 'predicted_delta': delta[0]})
    metadata = {'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'model': model, 'candidate': candidate, 'training_seeds': seeds,
                'prediction_policy': 'three_seed_arithmetic_mean' if seeds else 'selected_ridge',
                'input_smiles': smiles, 'canonical_parent_smiles': structure['canonical_smiles'],
                'standardization_policy': structure['standardization_policy'],
                'cell_type': cell_type, 'time_hours': time_hours, 'dose_nM': dose_nM,
                'n_selected_genes': len(genes), 'max_training_tanimoto': float(similarity[nearest]),
                'nearest_training_compound': str(train_ids[nearest]), 'n_training_compounds': len(train_ids),
                'test_outcomes_loaded': False, 'source_commit': seal['source_commit'],
                'scope': 'Predicted treated-minus-matched-vehicle log1p(CPM) delta in the training-selected genes; A549 24h 1000nM only. Chemical similarity is training-set support, not calibrated uncertainty. No efficacy, clinical, target-engagement or cell-phase claim.',
                'prediction_seal_sha256': sha(seal_path), 'dataset_freeze_sha256': sha(freeze_path),
                'inference_source_sha256': sha(Path(__file__)),
                'checkpoint_sha256': {p: sealed_hashes[p] for p in checkpoints},
                'input_artifact_sha256': {p: freeze['artifact_hashes'][p] for p in required}}
    return table, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smiles', required=True)
    parser.add_argument('--cell-type', default='A549')
    parser.add_argument('--time-hours', type=float, default=24)
    parser.add_argument('--dose-nM', type=float, default=1000, dest='dose_nM')
    parser.add_argument('--model', choices=['M3', 'M2', 'M1'], default='M3')
    parser.add_argument('--output-prefix', type=Path, default=ROOT / 'reports/prediction')
    args = parser.parse_args()
    try:
        table, metadata = predict_smiles(args.smiles, cell_type=args.cell_type,
            time_hours=args.time_hours, dose_nM=args.dose_nM, model=args.model)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    prefix = args.output_prefix
    tsv, receipt = Path(str(prefix) + '.tsv'), Path(str(prefix) + '.json')
    if tsv.exists() or receipt.exists():
        parser.error('Output exists; choose a new --output-prefix')
    tsv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(tsv, sep='\t', index=False)
    metadata['prediction_tsv_sha256'] = sha(tsv)
    receipt.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'predictions': str(tsv.resolve()), 'receipt': str(receipt.resolve()),
        'model': metadata['model'], 'n_genes': len(table),
        'max_training_tanimoto': metadata['max_training_tanimoto']}))


if __name__ == '__main__':
    main()
