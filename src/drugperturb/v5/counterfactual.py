"""v5 counterfactual queries and contrasts for keys with a variable number of replicates.

A Tahoe-100M (compound, context) has one to three replicate samples, each with
its own matched vehicle unit. The counterfactual contrast of a key is the
equal-weight mean over its replicates, the same estimand as the measured truth
(drugperturb.v5.prepare). Contexts are named '<cell line>_<dose>nM'.
"""
from __future__ import annotations

import numpy as np

from ..v3.pseudobulk import measured_log_cpm
from ..v4.counterfactual import capped_pseudobulk


def context_ids(obs):
    """Context ID of each drug cell; vehicle cells get an empty string."""
    dose = obs.dose_nM.fillna(0).round().astype(int).astype(str)
    return np.where(obs.role.eq('vehicle'), '', obs.cell_type+'_'+dose+'nM')


def cell_lines(contexts):
    return sorted({c.rsplit('_', 1)[0] for c in contexts})


def setting_cells(obs, contexts, vehicle_contexts):
    """Training and validation masks; vehicle cells only from units serving the contexts."""
    drug_in = obs.role.isin(['train', 'validation']).to_numpy() & np.isin(context_ids(obs), contexts)
    serving = set(vehicle_contexts.loc[vehicle_contexts.context_id.isin(contexts), 'Combo'])
    vehicle_in = obs.role.eq('vehicle').to_numpy() & obs.Combo.isin(serving).to_numpy()
    train = (drug_in & obs.role.eq('train').to_numpy()) | vehicle_in
    validation = drug_in & obs.role.eq('validation').to_numpy()
    return train, validation


def build_queries(query_table, contexts, splits):
    """Replicate counterfactual records per (compound, context); one record per replicate sample."""
    rows = query_table[query_table.context_id.isin(contexts) & query_table.split.isin(splits)]
    queries = {}
    for (cid, context), group in rows.groupby(['compound_id', 'context_id'], sort=True):
        group = group.sort_values('replicate')
        if group.replicate.duplicated().any():
            raise ValueError(f'{(cid, context)}: a replicate appears twice')
        queries[(cid, context)] = [dict(
            treated=dict(compound=cid, dose_nM=float(row.dose_nM), cell_type=row.cell_type),
            vehicle=dict(compound='DMSO', dose_nM=0.0, cell_type=row.cell_type),
            vehicle_well=row.vehicle_Combo) for row in group.itertuples(index=False)]
    return queries


def counterfactual_deltas(predict, queries, vehicle_cells, *, gene_count):
    """Counterfactual contrast and anchored sensitivity: equal-weight mean over each key's replicates."""
    contrast, anchored, cache = {}, {}, {}
    for key, replicates in queries.items():
        if not replicates:
            raise ValueError(f'{key}: no replicate query')
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
        c_rows, a_rows = np.asarray(c_rows), np.asarray(a_rows)
        if not np.isfinite(c_rows).all() or not np.isfinite(a_rows).all():
            raise ValueError(f'{key}: non-finite replicate deltas')
        contrast[key], anchored[key] = c_rows.mean(axis=0), a_rows.mean(axis=0)
    return contrast, anchored


def pseudobulk_pairs(queries, keys):
    """Pseudobulk-loss pairs (compound, dose, cell line, one vehicle well per replicate) for ``keys``."""
    return [dict(compound=k[0], dose_nM=queries[k][0]['treated']['dose_nM'],
                 cell_type=queries[k][0]['treated']['cell_type'], wells=[r['vehicle_well'] for r in queries[k]])
            for k in keys]
