"""Training/validation-only diagnostic of chemCPA effect sizes (never reads test outcomes).

Refits one prespecified chemCPA candidate on Q1 and reports, for training and
validation compounds, the size of predicted counterfactual deltas, their RMSE
against measured deltas and zero change, the per-compound gene-wise Pearson r,
and the size of the drug term relative to the basal latent of vehicle cells.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
import yaml

from drugperturb.v3 import chemcpa, runner
from drugperturb.v3.cells import load_model_archive
from drugperturb.v3.descriptors import VEHICLE_ID, compound_table
from drugperturb.v3.pseudobulk import normalize_counts

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 'A549_24h_1000nM'


def truth_table(path):
    with np.load(path, allow_pickle=False) as z:
        return {tuple(k.split('|')): z['delta'][i] for i, k in enumerate(z['keys'])}


def summarize(pred, truth):
    keys = sorted(set(pred) & set(truth))
    p = np.vstack([pred[k] for k in keys]); t = np.vstack([truth[k] for k in keys])
    r = [np.corrcoef(a, b)[0, 1] for a, b in zip(p, t)]
    scale = float((p*t).sum()/(p*p).sum()) if (p*p).sum() > 0 else float('nan')
    return dict(n=len(keys), mean_abs_pred=float(np.abs(p).mean()), mean_abs_true=float(np.abs(t).mean()),
                rmse=float(np.sqrt(((p-t)**2).mean(axis=1)).mean()), rmse_zero=float(np.sqrt((t**2).mean(axis=1)).mean()),
                mean_gene_pearson_r=float(np.nanmean(r)), least_squares_scale=scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--candidate', default='C1_repository_default')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load((ROOT/'configs/protocol_v3.yaml').read_text())
    spec = protocol['models']['chemCPA']
    candidate = spec['candidates'][args.candidate]
    obs, counts, library = load_model_archive(args.prepared/'model_cells.npz')
    vehicle_contexts = pd.read_csv(args.prepared/'vehicle_contexts.tsv', sep='\t')
    train_mask, val_mask, cell_types = runner.protocol_cells(obs, 'Q1', vehicle_contexts)
    keep = np.flatnonzero(train_mask | val_mask)
    sub, sub_counts, sub_library = obs.iloc[keep].reset_index(drop=True), counts[keep], library[keep]
    X = normalize_counts(sub_counts, sub_library)
    split = pd.read_csv(ROOT/'manifests/split_manifest.tsv', sep='\t', keep_default_na=False)
    wells = pd.read_csv(ROOT/'manifests/v2/well_mapping.tsv', sep='\t', keep_default_na=False, dtype={'replicate': str})
    rows = wells[wells.context_id.eq(CONTEXT) & wells.row_kind.eq('drug') & wells.split.isin(['train', 'validation'])]
    query_table = pd.DataFrame(dict(compound_id=rows.treatment, context_id=CONTEXT, split=rows.split, replicate=rows.replicate,
                                    cell_type='A549', dose_nM=1000.0, vehicle_Combo=rows.control_Combo))
    truth = {**truth_table(args.prepared/'train_truth.npz'), **truth_table(args.prepared/'validation_truth.npz')}
    queries = {s: {k: v for k, v in runner.build_queries(query_table, [CONTEXT], [s]).items() if k in truth}
               for s in ['train', 'validation']}
    wells_needed = [r['vehicle_well'] for q in queries.values() for v in q.values() for r in v]
    vehicles = runner.vehicle_wells(sub, sub_counts, sub_library, wells_needed)
    descriptors, _ = compound_table(split.compound_id.tolist(), split.canonical_smiles.tolist(),
                                    split.loc[split.split.eq('train'), 'compound_id'].tolist())
    vocabulary = [VEHICLE_ID, *split.compound_id.tolist()]
    model = chemcpa.ChemCPA(hparams=candidate['hparams'], additional=candidate['additional'], n_genes=X.shape[1],
                            drug_vocabulary=vocabulary, drug_matrix=np.vstack([descriptors[c] for c in vocabulary]),
                            cell_types=cell_types, use_cell_type=False, device='cpu', seed=args.seed)
    predict = lambda well, condition: model.predict_mean(well['values'], condition)
    val_truth = {k: truth[k] for k in queries['validation']}
    validate = lambda m: runner.mean_query_rmse(runner.predict_queries(predict, queries['validation'], vehicles,
                                                                       X.shape[1])[0], val_truth)
    t_index = np.flatnonzero(train_mask[keep])
    t = sub.iloc[t_index]
    result = chemcpa.fit(model, X[t_index], t.compound_id.tolist(), t.dose_nM.fillna(0).tolist(), t.cell_type.tolist(),
                         batch_size=candidate['hparams']['batch_size'], max_epochs=args.epochs, evaluate_every=10,
                         patience=99, max_minutes=60, seed=args.seed, validate=validate)
    report = dict(candidate=args.candidate, seed=args.seed, epochs=args.epochs, best_epoch=result['best_epoch'],
                  test_outcomes_loaded=False)
    for s in ['train', 'validation']:
        contrast, _ = runner.predict_queries(predict, queries[s], vehicles, X.shape[1])
        report[s] = summarize(contrast, truth)
    # Drug term versus basal latent on one vehicle well.
    well = next(iter(vehicles.values()))
    with torch.no_grad():
        model.model.eval()
        genes = torch.as_tensor(well['values'])
        basal = model.model.encoder(genes)
        some = [c for c, _ in list(queries['train'])[:20]]
        drugs, dose, _ = model.encode_conditions(some, [1000.0]*len(some), ['A549']*len(some))
        drug_term = model.model.compute_drug_embeddings_(drugs_idx=drugs, dosages=dose)
    report['latent'] = dict(basal_norm=float(basal.norm(dim=1).mean()), drug_term_norm=float(drug_term.norm(dim=1).mean()))
    report['history'] = result['history']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float)+'\n')
    print(json.dumps({k: report[k] for k in ['best_epoch', 'train', 'validation', 'latent']}, indent=2))


if __name__ == '__main__':
    main()
