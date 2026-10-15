#!/usr/bin/env python3
"""Exploratory development-only CV/learning curves, never final-test inputs."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import time

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from drugperturb.models import MolecularGNN, predict_network
from drugperturb.response import TrainingTargetTransform

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FRACTIONS = (0.25, 0.5, 1.0)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def selected_epoch_budget(runs, candidate, expected_seeds):
    rows = runs.loc[runs.method.eq('M3') & runs.candidate.eq(candidate)]
    if len(rows) != 3 or set(rows.seed.astype(int)) != set(expected_seeds) or len(set(expected_seeds)) != 3:
        raise ValueError('Selected M3 candidate must have exactly the three primary training seeds')
    epochs = rows.best_epoch.to_numpy(dtype=float)
    if not np.isfinite(epochs).all() or (epochs < 1).any() or not np.equal(epochs, np.floor(epochs)).all():
        raise ValueError('Selected main best_epoch values must be positive integers')
    return max(1, int(np.floor(np.median(epochs) + 0.5))), [
        {'seed': int(row.seed), 'best_epoch': int(row.best_epoch)}
        for row in rows.sort_values('seed').itertuples()]

def nested_subsets(n, seed=SEED):
    if n < 8:
        raise ValueError('At least eight training compounds required for nondegenerate 25% subset')
    order = np.random.default_rng(seed).permutation(n)
    return [(fraction, order[:max(2, int(math.ceil(n * fraction)))].copy()) for fraction in FRACTIONS]

def read_graphs(path):
    return [{k: np.asarray(v, dtype=np.int64 if k == 'edge_index' else np.float32)
             for k, v in graph.items()} for graph in json.loads(path.read_text())]

def load_development(path):
    with np.load(path, allow_pickle=False) as archive:
        if not {'y', 'fp', 'ids'}.issubset(archive.files):
            raise ValueError(f'{path.name} must contain raw all-gene y, fp and ids')
        data = {k: archive[k].copy() for k in ('y', 'fp', 'ids')}
    y, fp, ids = data['y'], data['fp'], data['ids']
    if y.ndim != 2 or fp.ndim != 2 or len(y) != len(fp) or len(ids) != len(y):
        raise ValueError('Development arrays have inconsistent dimensions')
    if not np.isfinite(y).all() or not np.isfinite(fp).all() or not np.isin(fp, [0, 1]).all():
        raise ValueError('Development arrays must be finite with binary fingerprints')
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or any(not str(v) for v in ids):
        raise ValueError('Compound IDs must be unique nonempty prediction units')
    return data

def fit_fixed_gnn(graphs, targets, hp, epochs, batch_size, seed=SEED):
    """Only fitting data enter optimization: no evaluation labels or early stop."""
    if epochs < 1 or batch_size < 1 or len(graphs) != len(targets):
        raise ValueError('Invalid fixed training budget or graph/target count')
    torch.set_num_threads(min(2, max(1, os.cpu_count() or 1)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = MolecularGNN(targets.shape[1], hidden_dim=hp['hidden_dim'], layers=2,
                         dropout=hp['dropout']).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=1e-4)
    order_rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for start in range(0, len(targets), batch_size):
            if start == 0:
                order = order_rng.permutation(len(targets))
            index = order[start:start + batch_size]
            target = torch.as_tensor(targets[index], dtype=torch.float32, device='cpu')
            optimizer.zero_grad(set_to_none=True)
            pred = model([graphs[int(i)] for i in index])
            loss = torch.mean((pred - target) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite fixed-training loss at epoch {epoch}')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'Nonfinite fixed-training gradient at epoch {epoch}')
            optimizer.step()
            total_loss += float(loss.detach()) * len(index)
        history.append({'epoch': epoch, 'online_minibatch_training_MSE': total_loss / len(targets)})
    model.eval()
    return model, history

def fit_models(train_y, train_fp, train_graphs, query_fp, query_graphs, settings):
    """Fit transformations/models without any query response argument."""
    transform = TrainingTargetTransform(settings['max_genes'], settings['max_components']).fit(train_y)
    z = transform.transform(train_y)
    ridge = Ridge(alpha=settings['ridge_alpha'], solver='cholesky').fit(train_fp, z)
    gnn, history = fit_fixed_gnn(train_graphs, z, settings['gnn_hp'], settings['gnn_epochs'],
                                 settings['batch_size'], settings['seed'])
    ridge_coordinates = np.asarray(ridge.predict(query_fp)).reshape(len(query_fp), transform.n_components)
    predictions = {
        'M1': transform.inverse_transform(ridge_coordinates),
        'M3': transform.inverse_transform(predict_network(gnn, query_graphs, batch_size=settings['batch_size'])),
    }
    return transform, predictions, ridge, gnn, history

def measured_metrics(truth, prediction):
    if truth.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('Predictions and measured selected-gene truth must align and be finite')
    residual = prediction - truth
    return np.sqrt(np.mean(residual ** 2, axis=1)), np.mean(np.abs(residual), axis=1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    selection_path = ROOT / 'receipts/model_selection.json'
    if not selection_path.exists():
        raise FileNotFoundError('Main model_selection.json must exist before diagnostics; no data loaded')
    allowed = ['receipts/model_selection.json', 'receipts/training_runs.tsv', 'configs/protocol_v1.yaml',
               'receipts/dataset_freeze.json', 'data/processed/training_allgenes.npz',
               'data/processed/validation_allgenes.npz', 'data/processed/graphs_train.json',
               'data/processed/graphs_validation.json', 'data/processed/gene_metadata.tsv',
               'scripts/development_diagnostics.py', 'src/drugperturb/models.py', 'src/drugperturb/response.py']
    input_hashes = {name: sha(ROOT / name) for name in allowed}
    receipt_path = ROOT / 'receipts/development_diagnostics.json'
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous['input_hashes'] != input_hashes:
            raise ValueError('Completed diagnostic inputs changed; refuse silent rerun')
        for name, digest in previous['artifact_hashes'].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f'Completed diagnostic artifact changed: {name}')
        print('Existing diagnostic receipt and artifacts verified; no fitting repeated.')
        return
    freeze = json.loads((ROOT / 'receipts/dataset_freeze.json').read_text())
    for name in ['data/processed/training_allgenes.npz', 'data/processed/validation_allgenes.npz',
                 'data/processed/graphs_train.json', 'data/processed/graphs_validation.json']:
        if freeze['artifact_hashes'].get(name) != input_hashes[name]:
            raise ValueError(f'Development input not verified by dataset freeze: {name}')
    selection = json.loads(selection_path.read_text())
    protocol = yaml.safe_load((ROOT / 'configs/protocol_v1.yaml').read_text())
    cfg = protocol['models']
    if protocol['status'] != 'FROZEN_BEFORE_TRAINING':
        raise ValueError('Primary protocol is not frozen')
    if cfg['max_development_folds'] < 3 or list(cfg['learning_curve_fractions']) != list(FRACTIONS):
        raise ValueError('Diagnostics require the predeclared three folds and 25/50/100% fractions')
    runs = pd.read_csv(ROOT / 'receipts/training_runs.tsv', sep='\t')
    chosen = selection['selected']
    epochs, epoch_sources = selected_epoch_budget(runs, chosen['M3']['candidate'], cfg['training_seeds'])
    hp = {k: chosen['M3'][k] for k in ('hidden_dim', 'dropout', 'lr')}
    if hp != cfg['gnn_candidates'][chosen['M3']['candidate']]:
        raise ValueError('Selected GNN hyperparameters disagree with primary protocol')
    if chosen['M1']['alpha'] != cfg['ridge_alphas'][chosen['M1']['candidate']]:
        raise ValueError('Selected Ridge alpha disagrees with primary protocol')
    settings = {'seed': SEED, 'ridge_alpha': chosen['M1']['alpha'], 'gnn_hp': hp,
                'gnn_epochs': epochs, 'epoch_sources': epoch_sources, 'batch_size': cfg['batch_size'],
                'max_genes': protocol['response']['max_training_selected_genes'],
                'max_components': protocol['response']['max_training_target_pcs'],
                'optimizer': 'AdamW', 'weight_decay': 1e-4, 'device': 'cpu', 'torch_threads': 2,
                'early_stopping': False, 'hyperparameter_search': False, 'diagnostic_seed_count': 1}
    output = ROOT / 'models/diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    frozen_settings = {'settings': settings, 'input_hashes': input_hashes, 'source_commit': args.source_commit}
    settings_path = output / 'frozen_settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != frozen_settings:
        raise ValueError('Partial diagnostic settings/inputs changed; preserve previous artifacts')
    save_json(settings_path, frozen_settings)
    train = load_development(ROOT / 'data/processed/training_allgenes.npz')
    validation = load_development(ROOT / 'data/processed/validation_allgenes.npz')
    if set(train['ids']) & set(validation['ids']):
        raise ValueError('Primary training/validation IDs overlap')
    genes = pd.read_csv(ROOT / 'data/processed/gene_metadata.tsv', sep='\t').gene_id.to_numpy(dtype=str)
    if train['y'].shape[1] != len(genes) or validation['y'].shape[1] != len(genes):
        raise ValueError('All-gene responses do not match gene metadata dimensions')
    gt = read_graphs(ROOT / 'data/processed/graphs_train.json')
    gv = read_graphs(ROOT / 'data/processed/graphs_validation.json')
    if len(gt) != len(train['ids']) or len(gv) != len(validation['ids']):
        raise ValueError('Graph order inputs have inconsistent row counts')
    jobs = []
    for fold, (fit_idx, query_idx) in enumerate(KFold(n_splits=3, shuffle=True, random_state=SEED).split(train['ids']), 1):
        jobs.append((f'cv_fold_{fold}', 'compound_cv', fold, None, fit_idx, train, gt, query_idx))
    for fraction, fit_idx in nested_subsets(len(train['ids'])):
        jobs.append((f'learning_{int(fraction * 100)}', 'learning_curve', None, fraction,
                     fit_idx, validation, gv, np.arange(len(validation['ids']))))
    summaries, all_rows = [], []
    for name, kind, fold, fraction, fit_idx, query, graphs_query, query_idx in jobs:
        work = output / name
        work.mkdir(exist_ok=True)
        complete = work / 'result.json'
        if complete.exists():
            result = json.loads(complete.read_text())
            if result['input_hashes'] != input_hashes:
                raise ValueError(f'Cached diagnostic input mismatch: {name}')
            for artifact, digest in result['artifact_hashes'].items():
                if sha(ROOT / artifact) != digest:
                    raise ValueError(f'Cached diagnostic artifact mismatch: {artifact}')
            summaries.extend(result['summaries'])
            all_rows.extend(result['compound_metrics'])
            continue
        task_start = time.perf_counter()
        fit_graphs = [gt[int(i)] for i in fit_idx]
        eval_graphs = [graphs_query[int(i)] for i in query_idx]
        transform, predictions, ridge, gnn, history = fit_models(
            train['y'][fit_idx], train['fp'][fit_idx], fit_graphs,
            query['fp'][query_idx], eval_graphs, settings)
        # Labels are used only after transforms, epochs, weights and predictions are fixed.
        truth = transform.select_truth(query['y'][query_idx])
        selected_ids = genes[transform.gene_indices]
        joblib.dump(transform, work / 'transform.joblib')
        joblib.dump(ridge, work / 'M1.joblib')
        torch.save(gnn.state_dict(), work / 'M3.pt')
        pd.DataFrame(history).to_csv(work / 'M3_history.tsv', sep='\t', index=False)
        np.savez_compressed(work / 'predictions.npz', ids=query['ids'][query_idx], train_ids=train['ids'][fit_idx],
                            gene_ids=selected_ids, truth=truth, **predictions)
        job_summaries, job_rows = [], []
        for method, prediction in predictions.items():
            rmse, mae = measured_metrics(truth, prediction)
            metadata = {'diagnostic': kind, 'run': name, 'fold': fold, 'fraction': fraction, 'method': method,
                        'seed': SEED, 'n_train_compounds': len(fit_idx), 'n_evaluation_compounds': len(query_idx),
                        'n_selected_genes': len(selected_ids), 'n_target_pcs': transform.n_components,
                        'fixed_epochs': epochs if method == 'M3' else 0, 'ridge_alpha': settings['ridge_alpha'] if method == 'M1' else None}
            summary = {**metadata, 'mean_compound_RMSE': float(rmse.mean()), 'median_compound_RMSE': float(np.median(rmse)),
                       'mean_compound_MAE': float(mae.mean()), 'gene_set_comparison': 'fold_or_subset_training_selected_genes; not a fixed gene-space comparison',
                       'exploratory_single_seed': True, 'run_seconds_including_both_models': time.perf_counter() - task_start}
            job_summaries.append(summary)
            job_rows.extend({**metadata, 'compound_id': str(cid), 'rmse': float(r), 'mae': float(m)}
                            for cid, r, m in zip(query['ids'][query_idx], rmse, mae))
        artifacts = {str(p.relative_to(ROOT)): sha(p) for p in sorted(work.iterdir()) if p.is_file() and p.name != 'result.json'}
        save_json(complete, {'input_hashes': input_hashes, 'artifact_hashes': artifacts,
                             'summaries': job_summaries, 'compound_metrics': job_rows,
                             'train_ids': train['ids'][fit_idx].tolist(), 'evaluation_ids': query['ids'][query_idx].tolist(),
                             'selected_gene_ids': selected_ids.tolist(), 'epoch_sources': epoch_sources})
        summaries.extend(job_summaries)
        all_rows.extend(job_rows)
        print(json.dumps({'finished': name, 'summaries': job_summaries}), flush=True)
    table = ROOT / 'receipts/development_diagnostics.tsv'
    pd.DataFrame(summaries).to_csv(table, sep='\t', index=False)
    artifact_files = [table, *sorted(p for p in output.rglob('*') if p.is_file())]
    save_json(receipt_path, {'status': 'COMPLETE_EXPLORATORY_DEVELOPMENT_ONLY', 'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                             'source_commit': args.source_commit, 'input_hashes': input_hashes,
                             'artifact_hashes': {str(p.relative_to(ROOT)): sha(p) for p in artifact_files},
                             'settings': settings, 'summaries': summaries, 'compound_metrics': all_rows,
                             'test_truth_loaded': False, 'test_features_loaded': False,
                             'runtime': {'host': socket.gethostname(), 'python': platform.python_version(), 'torch': torch.__version__,
                                         'numpy': np.__version__, 'device': 'cpu', 'torch_num_threads': torch.get_num_threads()},
                             'elapsed_seconds_this_invocation': time.perf_counter() - started,
                             'limitations': ['One exploratory seed; no diagnostic hyperparameter search.',
                                             'Compound folds are not scaffold-held-out evaluation.',
                                             'Genes change with fold/subset; scores are not directly comparable in a fixed gene space.',
                                             'Primary validation set previously selected hyperparameters and epoch budget; learning curves are development diagnostics.',
                                             'Shared controls and limited biological replicates remain correlated uncertainty.']})
    print(f'Diagnostics saved: {receipt_path}', flush=True)

if __name__ == '__main__':
    main()
