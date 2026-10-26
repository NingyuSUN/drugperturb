"""v6 preparation: half-split truth, DE sets, the responder gate and leave-one-out residual targets.

v5's pair building, count QC, gene selection and model inputs are reused
unchanged (``drugperturb.v5.prepare``); v6 adds what the new endpoint needs:

- the two measurement halves of every unit, each with its pooled vehicle rows;
- the vehicle-versus-vehicle null that calibrates the responder threshold;
- each key's DE set and responder status, from the selection half only;
- leave-one-out plate means, the training target of the residual arm M2.

Nothing here fits a model. The evaluation half of a test key is read only by the
evaluation kernel, after every prediction is sealed; this module computes test
keys' selection-half quantities, which the protocol allows and the endpoint
needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..v4.design import log_cpm
from ..v5.baselines import B1P_MIN_COMPOUNDS
from ..v5.prepare import MIN_DETECTION, STUDY_GATE, with_pooled_vehicles  # noqa: F401
from . import endpoint
from .design import MIN_HALF_CELLS, MIN_VEHICLE_CELLS, MIN_WELL_CELLS  # noqa: F401
from .extract import HALVES, split_halves

MIN_CELLS = MIN_HALF_CELLS  # the per-half threshold; the per-well gates live in v6.design
MIN_TEST_COMPOUNDS = 20
MIN_TEST_GROUPS = 10
MIN_TEST_RESPONDERS = 10  # protocol_v6 count_qc context_gate


def half_tables(ids, counts, library, n_cells, units):
    """Per half, the pooled-vehicle-extended arrays and a unit-id to row map (v5 layout)."""
    halves = split_halves(ids, counts, library, n_cells)
    output = {}
    for half in HALVES:
        part = halves[half]
        order = pd.Series(part['row_of'])
        reindexed = order.reindex(units.unit_id).to_numpy(np.int64)
        if np.isnan(order.reindex(units.unit_id)).any():
            raise ValueError(f'half {half} is missing a unit of the design')
        pooled_ids, pooled_counts, pooled_library, pooled_cells = with_pooled_vehicles(
            units, part['counts'][reindexed], part['library'][reindexed], part['n_cells'][reindexed])
        output[half] = dict(ids=pooled_ids, counts=pooled_counts, library=pooled_library, n_cells=pooled_cells,
                            row_of={u: i for i, u in enumerate(pooled_ids)})
    return output


def key_cells(pairs, keys, table, *, column='treatment_unit'):
    """Cells behind each key's rows of ``column`` in one half, summed over replicates."""
    row_of, cells = table['row_of'], table['n_cells']
    grouped = pairs.groupby(['compound_id', 'context_id'])[column].apply(list)
    return np.asarray([sum(int(cells[row_of[u]]) for u in grouped[k]) for k in keys], np.int64)


def half_count_qc(pairs, half_cells_of, *, minimum_cells=MIN_CELLS):
    """Replicate-level QC on both halves: every half of every replicate well and vehicle unit.

    ``half_cells_of`` maps a unit id to ``{'S': n, 'E': n}``. A replicate passes
    when all four cell sets behind its two deltas hold ``minimum_cells`` cells.
    """
    table = pairs.copy()
    for half in HALVES:
        table[f'treatment_cells_{half}'] = [half_cells_of.get(u, {}).get(half, 0) for u in table.treatment_unit]
        table[f'vehicle_cells_{half}'] = [half_cells_of.get(u, {}).get(half, 0) for u in table.vehicle_unit]
    columns = [f'{side}_cells_{half}' for side in ('treatment', 'vehicle') for half in HALVES]
    table['replicate_passes'] = table[columns].ge(minimum_cells).all(axis=1)
    table['min_half_cells'] = table[columns].min(axis=1)
    return table


def context_gate(compound, *, min_test_compounds=MIN_TEST_COMPOUNDS, min_test_groups=MIN_TEST_GROUPS,
                 min_test_responders=MIN_TEST_RESPONDERS):
    """Per-context counts and gate G3, which v6 extends with a minimum number of responders."""
    summary = []
    for context, group in compound.groupby('context_id'):
        row = dict(context_id=context)
        for split in ('train', 'validation', 'test'):
            chosen = group[group.split.eq(split) & group.eligible]
            row[f'eligible_{split}_compounds'] = len(chosen)
            row[f'eligible_{split}_groups'] = chosen.scaffold_group.nunique()
            row[f'responder_{split}_compounds'] = int(chosen.responder.sum()) if 'responder' in chosen else 0
        row['test_count_gate_met'] = bool(row['eligible_test_compounds'] >= min_test_compounds
                                          and row['eligible_test_groups'] >= min_test_groups
                                          and row['responder_test_compounds'] >= min_test_responders)
        summary.append(row)
    return pd.DataFrame(summary)


def half_delta(pairs, keys, table, columns):
    """Equal-weight replicate-mean strict delta of each key in one half, over ``columns``."""
    row_of, counts, library = table['row_of'], table['counts'], table['library']
    columns = np.asarray(columns)
    if not keys:
        return np.zeros((0, len(columns)))
    grouped = pairs.groupby(['compound_id', 'context_id'])
    rows = []
    for key in keys:
        group = grouped.get_group(key)
        t = np.asarray([row_of[u] for u in group.treatment_unit], np.int64)
        v = np.asarray([row_of[u] for u in group.vehicle_unit], np.int64)
        delta = log_cpm(counts[t][:, columns].toarray(), library[t])-log_cpm(counts[v][:, columns].toarray(), library[v])
        rows.append(delta.mean(axis=0))
    return np.vstack(rows)


def half_standard_error(pairs, keys, table, columns):
    """Replicate-mean standard error of each key's delta in one half; replicates average independently."""
    row_of, counts, library = table['row_of'], table['counts'], table['library']
    columns = np.asarray(columns)
    if not keys:
        return np.zeros((0, len(columns)))
    grouped = pairs.groupby(['compound_id', 'context_id'])
    rows = []
    for key in keys:
        group = grouped.get_group(key)
        t = np.asarray([row_of[u] for u in group.treatment_unit], np.int64)
        v = np.asarray([row_of[u] for u in group.vehicle_unit], np.int64)
        error = endpoint.delta_standard_error(counts[t][:, columns].toarray(), library[t],
                                              counts[v][:, columns].toarray(), library[v])
        rows.append(np.sqrt((error**2).sum(axis=0))/len(error))
    return np.vstack(rows)


def vehicle_null(units, table, columns):
    """Between-well vehicle statistics: one per (plate, line) with two vehicle samples.

    A key's delta compares a treated well with a pooled vehicle unit, so its
    noise includes well-to-well variation. The null therefore compares the two
    DMSO wells of the same plate and line with each other, in the selection
    half, and never touches an evaluation half (amendment A1).
    """
    row_of, counts, library = table['row_of'], table['counts'], table['library']
    columns = np.asarray(columns)
    rows = []
    vehicle = units[units.role.eq('vehicle')]
    for (plate, line), group in vehicle.groupby(['plate', 'cell_line_id']):
        samples = sorted(group.unit_id)
        if len(samples) != 2:
            continue
        a, b = (row_of[s] for s in samples)
        delta = (log_cpm(counts[[a]][:, columns].toarray(), library[[a]])
                 -log_cpm(counts[[b]][:, columns].toarray(), library[[b]]))
        error = endpoint.delta_standard_error(counts[[a]][:, columns].toarray(), library[[a]],
                                              counts[[b]][:, columns].toarray(), library[[b]])
        rows.append(dict(plate=plate, cell_line_id=line, unit_a=samples[0], unit_b=samples[1],
                         statistic=float(endpoint.response_statistic(delta, error)[0]),
                         cells_a=int(table['n_cells'][a]), cells_b=int(table['n_cells'][b])))
    if not rows:
        raise ValueError('no (plate, line) has two vehicle samples, so the null cannot be built')
    return pd.DataFrame(rows)


def leave_one_out_plate_means(replicates, delta, *, min_compounds=B1P_MIN_COMPOUNDS):
    """Per (context, plate), the plate mean and each training compound's leave-one-out mean.

    A compound with two samples on one plate is averaged first, so each compound
    counts once. ``means`` holds the plain plate mean where the plate carries at
    least ``min_compounds`` compounds; ``loo`` holds a compound's mean over the
    others where at least ``min_compounds`` remain after it is removed.
    """
    table = replicates.assign(row=np.arange(len(replicates)))
    per_compound = {k: delta[g.row.to_numpy()].mean(axis=0)
                    for k, g in table.groupby(['context_id', 'plate', 'compound_id'], sort=True)}
    grouped, members = {}, {}
    for (context, plate, compound), value in per_compound.items():
        grouped.setdefault((context, plate), []).append(value)
        members.setdefault((context, plate), []).append(compound)
    means, counts, loo = {}, {}, {}
    for key, values in grouped.items():
        stacked = np.vstack(values)
        counts[key] = len(values)
        if len(values) >= min_compounds:
            means[key] = stacked.mean(axis=0)
        if len(values)-1 >= min_compounds:
            total = stacked.sum(axis=0)
            for compound, value in zip(members[key], values):
                loo[(*key, compound)] = (total-value)/(len(values)-1)
    return means, loo, counts


def residual_targets(keys, delta, reference):
    """Measured delta minus its reference prediction, for the keys that have one.

    Returns the residual rows and the keys that had no reference, which the
    caller falls back to the anchored training mean for (protocol_v6
    residual_secondary fallback).
    """
    rows, kept, missing = [], [], []
    for key, measured in zip(keys, delta):
        value = reference.get(key)
        if value is None:
            missing.append(key)
            continue
        rows.append(measured-value)
        kept.append(key)
    return (np.vstack(rows) if rows else np.zeros((0, delta.shape[1]))), kept, missing
