"""Shared v3 machinery: counterfactual queries, validation scoring and sealing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pseudobulk import counterfactual_deltas, normalize_counts

PROTOCOL_CONTEXTS = {
    'Q1': ['A549_24h_1000nM'],
    'Q2': [f'{c}_24h_{d}nM' for c in ['A549', 'K562', 'MCF7'] for d in [10, 100, 1000, 10000]],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def context_ids(obs):
    """Context ID of each drug cell; vehicle cells get an empty string."""
    dose = obs.dose_nM.fillna(0).round().astype(int).astype(str)
    return np.where(obs.role.eq('vehicle'), '', obs.cell_type+'_24h_'+dose+'nM')


def protocol_cells(obs, protocol, vehicle_contexts):
    """Training and validation masks; vehicles only from wells serving the contexts.

    Q1 therefore sees exactly the A549 1 uM drug and vehicle wells used by v1.
    """
    contexts = PROTOCOL_CONTEXTS[protocol]
    drug_in = obs.role.isin(['train', 'validation']).to_numpy() & np.isin(context_ids(obs), contexts)
    serving = set(vehicle_contexts.loc[vehicle_contexts.context_id.isin(contexts), 'Combo'])
    vehicle_in = obs.role.eq('vehicle').to_numpy() & obs.Combo.isin(serving).to_numpy()
    train = (drug_in & obs.role.eq('train').to_numpy()) | vehicle_in
    validation = drug_in & obs.role.eq('validation').to_numpy()
    cell_types = sorted({c.split('_')[0] for c in contexts})
    return train, validation, cell_types


def build_queries(query_table, contexts, split_names):
    """Two replicate counterfactual records per (compound, context)."""
    rows = query_table[query_table.context_id.isin(contexts) & query_table.split.isin(split_names)]
    queries = {}
    for (cid, context), group in rows.groupby(['compound_id', 'context_id'], sort=True):
        group = group.sort_values('replicate')
        if group.replicate.tolist() != ['rep1', 'rep2']:
            continue
        records = []
        for row in group.itertuples(index=False):
            records.append(dict(
                treated=dict(compound=cid, dose_nM=float(row.dose_nM), cell_type=row.cell_type),
                vehicle=dict(compound='DMSO', dose_nM=0.0, cell_type=row.cell_type),
                vehicle_well=row.vehicle_Combo))
        queries[(cid, context)] = records
    return queries


def vehicle_wells(obs, counts, library, wells):
    """Model inputs, counts and libraries for each requested vehicle well."""
    output = {}
    vehicle = obs.role.eq('vehicle').to_numpy()
    for well in sorted(set(wells)):
        index = np.flatnonzero(vehicle & obs.Combo.eq(well).to_numpy())
        if index.size == 0:
            raise ValueError(f'vehicle well {well} has no exported cells')
        output[well] = dict(values=normalize_counts(counts[index], library[index]), index=index,
                            counts=counts[index], library=library[index], n_cells=int(index.size))
    return output


def mean_query_rmse(predicted, truth):
    """Equal-weight mean RMSE over (compound, context) keys present in truth."""
    keys = sorted(truth)
    if not keys or set(keys) - set(predicted):
        raise ValueError('validation predictions are missing truth keys')
    return float(np.mean([np.sqrt(np.mean((predicted[k]-truth[k])**2)) for k in keys]))


def predict_queries(predict, queries, vehicle_cells, gene_count):
    return counterfactual_deltas(predict, queries, vehicle_cells, gene_count=gene_count)


def stack(deltas, keys, width):
    if not keys:
        return np.zeros((0, width))
    return np.vstack([deltas[k] for k in keys])


def write_seal(output_root, protocol_name, method, arrays, metadata, inputs):
    """Write the prediction archive, then a JSON seal hashing it and its inputs."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root/f'{method}_{protocol_name}_predictions.npz'
    if archive.exists():
        raise ValueError(f'refusing to overwrite sealed predictions {archive}')
    np.savez_compressed(archive, **arrays)
    seal = dict(method=method, protocol=protocol_name, status='SEALED_BEFORE_TEST_SCORING',
                test_outcomes_loaded=False, predictions=dict(path=archive.name, sha256=sha256(archive)),
                inputs={k: dict(path=str(v), sha256=sha256(v)) for k, v in inputs.items()}, **metadata)
    (root/f'{method}_{protocol_name}_seal.json').write_text(json.dumps(seal, indent=2, allow_nan=False)+'\n')
    return seal


def prediction_arrays(contrast, anchored, keys):
    """Flat arrays keyed by '<compound>|<context>' rows."""
    width = len(next(iter(contrast.values()))) if contrast else 0
    return dict(keys=np.asarray([f'{c}|{x}' for c, x in keys]),
                contrast=stack(contrast, keys, width), anchored=stack(anchored, keys, width))


def read_query_table(path):
    table = pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'replicate': str})
    required = {'compound_id', 'context_id', 'split', 'replicate', 'cell_type', 'dose_nM', 'vehicle_Combo'}
    if not required.issubset(table):
        raise ValueError(f'query table requires {sorted(required)}')
    return table
