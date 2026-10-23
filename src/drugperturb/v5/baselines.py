"""v5 vehicle- and plate-anchored comparators (B1a, M1a, B1p); never reads a test response.

As in v4 amendment A1, a delta baseline is anchored by adding the context's mean
training vehicle level and subtracting the query's own vehicle level:

    anchored(q) = delta_baseline(q) + Vbar_train(context) - mean_r V_r(q)

V_r is the all-cell log1p CPM of replicate r's pooled vehicle unit (plate and
line), so the anchor uses every vehicle cell, not the capped model cells.

B1p (secondary) averages, per replicate, the measured deltas of the context's
training compounds on that replicate's plate. Those deltas share the replicate's
vehicle unit, so B1p carries its vehicle term without an anchoring term. Where a
plate holds fewer than ``min_compounds`` such training compounds, that replicate
falls back to the anchored training mean B1c + Vbar_train - V_r. B1p is the mean
over the query's replicates.
"""
from __future__ import annotations

import numpy as np

B1P_MIN_COMPOUNDS = 5


def records(queries, split):
    """(compound, context) -> replicate rows (replicate, plate, vehicle_unit) of one split of the query table."""
    rows = queries[queries.split.eq(split)].sort_values('replicate').rename(columns={'vehicle_Combo': 'vehicle_unit'})
    return {k: g[['replicate', 'plate', 'vehicle_unit']].to_dict('records')
            for k, g in rows.groupby(['compound_id', 'context_id'], sort=True)}


def query_vehicle_level(recs, levels):
    return {k: np.mean([levels[r['vehicle_unit']] for r in reps], axis=0) for k, reps in recs.items()}


def training_vehicle_means(train_recs, levels):
    """Per context: mean over training keys of their replicate-mean vehicle level."""
    groups = {}
    for (_, context), value in query_vehicle_level(train_recs, levels).items():
        groups.setdefault(context, []).append(value)
    return {context: np.mean(values, axis=0) for context, values in groups.items()}


def anchor(delta_predictions, recs, levels, vbar):
    vehicle = query_vehicle_level({k: recs[k] for k in delta_predictions}, levels)
    return {k: v+vbar[k[1]]-vehicle[k] for k, v in delta_predictions.items()}


def plate_means(train_replicates, train_delta, *, min_compounds=B1P_MIN_COMPOUNDS):
    """(context, plate) -> mean over the training compounds on that plate of their per-plate deltas.

    A compound with two samples on one plate is averaged first, so each compound
    counts once. Plates with fewer than ``min_compounds`` compounds are omitted.
    """
    table = train_replicates.assign(row=np.arange(len(train_replicates)))
    per_compound = {k: train_delta[g.row.to_numpy()].mean(axis=0)
                    for k, g in table.groupby(['context_id', 'plate', 'compound_id'], sort=True)}
    grouped = {}
    for (context, plate, _), value in per_compound.items():
        grouped.setdefault((context, plate), []).append(value)
    return {k: np.mean(v, axis=0) for k, v in grouped.items() if len(v) >= min_compounds}, \
        {k: len(v) for k, v in grouped.items()}


def plate_anchored(recs, b1c, levels, vbar, means):
    """B1p for each key of ``recs``; returns predictions and the number of replicates that fell back."""
    output, fallback = {}, 0
    for key, reps in recs.items():
        rows = []
        for r in reps:
            mean = means.get((key[1], r['plate']))
            if mean is None:
                fallback += 1
                mean = b1c[key]+vbar[key[1]]-levels[r['vehicle_unit']]
            rows.append(mean)
        output[key] = np.mean(rows, axis=0)
    return output, fallback


def stack(predictions, keys):
    return np.vstack([predictions[k] for k in keys]) if keys else np.zeros((0, 0))


def key_strings(keys):
    return np.asarray([f'{c}|{x}' for c, x in keys])


def frame_keys(frame):
    return sorted(set(map(tuple, frame[['compound_id', 'context_id']].to_numpy())))
