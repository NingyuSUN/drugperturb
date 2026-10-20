"""Counterfactual queries and contrasts for the v4 settings.

v3 required exactly two replicates; OP3 has three donors. These functions take
the replicate count as an argument and, for two replicates, reproduce the v3
contrast exactly (same pseudobulk, same equal-weight mean).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v3 import runner as v3_runner
from ..v3.pseudobulk import MODEL_SCALE, measured_log_cpm, pseudobulk_log_cpm

# A gene holding a cell's whole library has model value log1p(1e4); no count can exceed it.
# Capping at this bound is a no-op for any attainable output (so the identity model still
# reproduces zero change) and keeps divergent decoder outputs from overflowing expm1.
MAX_MODEL_VALUE = float(np.log1p(MODEL_SCALE))


def capped_pseudobulk(values, library):
    return pseudobulk_log_cpm(np.minimum(np.asarray(values, dtype=np.float64), MAX_MODEL_VALUE), library)

SETTINGS = {
    'sciplex_Q1': dict(dataset='sciplex3', contexts=v3_runner.PROTOCOL_CONTEXTS['Q1'], replicates=2,
                       use_cell_type=False),
    'sciplex_Q2': dict(dataset='sciplex3', contexts=v3_runner.PROTOCOL_CONTEXTS['Q2'], replicates=2,
                       use_cell_type=True),
    'op3': dict(dataset='op3', contexts=['Tcells_24h_1000nM', 'NKcells_24h_1000nM', 'Bcells_24h_1000nM',
                                         'Myeloid_24h_1000nM'], replicates=3, use_cell_type=True),
}


def load_truth(path):
    """(compound, context) -> delta row, decompressing the matrix once.

    Indexing ``z['delta']`` per key would re-read the whole array for every
    row and keep each copy alive through the row view.
    """
    with np.load(path, allow_pickle=False) as z:
        keys, delta = z['keys'], z['delta']
    return {tuple(k.split('|')): delta[i] for i, k in enumerate(keys)}


def replicate_mean(deltas, n_replicates):
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != n_replicates or not np.isfinite(values).all():
        raise ValueError(f'expected {n_replicates} finite replicate delta vectors')
    return values.mean(axis=0)


def setting_cells(obs, contexts, vehicle_contexts):
    """Training and validation masks; vehicles only from wells serving the contexts (v3 rule)."""
    drug_in = obs.role.isin(['train', 'validation']).to_numpy() & np.isin(v3_runner.context_ids(obs), contexts)
    serving = set(vehicle_contexts.loc[vehicle_contexts.context_id.isin(contexts), 'Combo'])
    vehicle_in = obs.role.eq('vehicle').to_numpy() & obs.Combo.isin(serving).to_numpy()
    train = (drug_in & obs.role.eq('train').to_numpy()) | vehicle_in
    validation = drug_in & obs.role.eq('validation').to_numpy()
    return train, validation, sorted({c.split('_')[0] for c in contexts})


def build_queries(query_table, contexts, splits, n_replicates):
    """Replicate counterfactual records per (compound, context); incomplete keys are skipped."""
    rows = query_table[query_table.context_id.isin(contexts) & query_table.split.isin(splits)]
    queries = {}
    for (cid, context), group in rows.groupby(['compound_id', 'context_id'], sort=True):
        group = group.sort_values('replicate')
        if len(group) != n_replicates or group.replicate.nunique() != n_replicates:
            continue
        queries[(cid, context)] = [dict(
            treated=dict(compound=cid, dose_nM=float(row.dose_nM), cell_type=row.cell_type),
            vehicle=dict(compound='DMSO', dose_nM=0.0, cell_type=row.cell_type),
            vehicle_well=row.vehicle_Combo) for row in group.itertuples(index=False)]
    return queries


def counterfactual_deltas(predict, queries, vehicle_cells, *, gene_count, n_replicates):
    """Counterfactual contrast and anchored sensitivity, as v3, for any replicate count."""
    contrast, anchored, cache = {}, {}, {}
    for key, replicates in queries.items():
        if len(replicates) != n_replicates:
            raise ValueError(f'{key}: expected {n_replicates} replicate queries')
        c_rows, a_rows = [], []
        for record in replicates:
            well = vehicle_cells[record['vehicle_well']]
            treated = capped_pseudobulk(predict(well, record['treated']), well['library'])
            cache_key = (record['vehicle_well'], tuple(sorted(record['vehicle'].items())))
            if cache_key not in cache:
                cache[cache_key] = capped_pseudobulk(predict(well, record['vehicle']), well['library'])
            if treated.shape != (gene_count,):
                raise ValueError('prediction width differs from the frozen gene space')
            c_rows.append(treated-cache[cache_key])
            a_rows.append(treated-measured_log_cpm(well['counts'], well['library']))
        contrast[key] = replicate_mean(c_rows, n_replicates)
        anchored[key] = replicate_mean(a_rows, n_replicates)
    return contrast, anchored


def sciplex_training_queries(well_mapping, contexts):
    """Training-compound query rows for sci-Plex3 contexts from the committed v2 well mapping."""
    rows = well_mapping[well_mapping.row_kind.eq('drug') & well_mapping.split.eq('train')
                        & well_mapping.context_id.isin(contexts)]
    return pd.DataFrame(dict(compound_id=rows.treatment.to_numpy(), context_id=rows.context_id.to_numpy(),
                             split='train', replicate=rows.replicate.to_numpy(),
                             cell_type=rows.cell_type.to_numpy(), dose_nM=rows.context_dose_nM.astype(float).to_numpy(),
                             vehicle_Combo=rows.control_Combo.to_numpy()))
