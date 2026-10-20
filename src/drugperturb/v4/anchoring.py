"""Vehicle-anchored simple baselines for models that see the matched vehicle cells.

The measured response of a query is T - V, where V is the log1p CPM of its
matched vehicle well(s). A counterfactual model receives those vehicle cells,
so it can predict roughly E[T] - V and cancel V's sampling noise, which no
delta-only baseline can do (the v3 report's anchored-sensitivity argument).
The information-matched simple versions add the same vehicle information to a
delta baseline:

    anchored(q) = delta_baseline(q) + Vbar_train(context) - mean_r V_r(q)

where Vbar_train is the mean over the context's training pairs of their
replicate-mean vehicle level. With delta_baseline = B1c this is the training
mean of treated levels minus this query's vehicle level (B1a); with M1c, M1a.
Only training deltas and vehicle cells enter; no validation or test response.
"""
from __future__ import annotations

import numpy as np

from ..v3.pseudobulk import measured_log_cpm


def vehicle_levels(vehicle_cells):
    """Measured log1p CPM of each vehicle well from its cells (equals the v1/v2 well normalization)."""
    return {well: measured_log_cpm(cells['counts'], cells['library']) for well, cells in vehicle_cells.items()}


def query_vehicle_level(queries, levels):
    """Replicate-mean measured vehicle level of each query key."""
    return {key: np.mean([levels[r['vehicle_well']] for r in records], axis=0) for key, records in queries.items()}


def training_vehicle_means(train_queries, levels, keys):
    """Per context: mean over training pairs of their replicate-mean vehicle level."""
    per_key = query_vehicle_level({k: train_queries[k] for k in keys}, levels)
    groups = {}
    for (_, context), value in per_key.items():
        groups.setdefault(context, []).append(value)
    return {context: np.mean(values, axis=0) for context, values in groups.items()}


def anchor(delta_predictions, queries, levels, vbar_train):
    """Add Vbar_train(context) - mean_r V_r(q) to delta predictions keyed like ``queries``."""
    vehicle = query_vehicle_level({k: queries[k] for k in delta_predictions}, levels)
    return {k: v+vbar_train[k[1]]-vehicle[k] for k, v in delta_predictions.items()}
