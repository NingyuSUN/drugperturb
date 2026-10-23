"""v5 preparation: response pairs, count QC, training-only gene selection, truth and model inputs.

Nothing here fits a model or computes a test-compound response. A (compound,
context) has one replicate per sample of that compound at the context's dose;
each replicate's strict matched vehicle is the pooled vehicle unit of its plate
and line (drugperturb.v5.extract.vehicle_unit). Count QC uses cell numbers only;
gene selection uses training-compound deltas and vehicle-cell detection only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ..v4.design import log_cpm

# protocol_v5 count_qc, gates G3/G4 and gene_space (prose values; tests check them against the protocol text)
MIN_CELLS = 30
MIN_TEST_COMPOUNDS = 20
MIN_TEST_GROUPS = 10
STUDY_GATE = dict(min_contexts=6, min_lines=3, min_doses=2)
MIN_DETECTION = 0.05


def response_pairs(units, compounds, contexts):
    """One row per (compound, context, replicate sample) for compound units of the protocol contexts."""
    info = compounds.set_index('compound_id')
    context_of = {(r.cell_line_id, int(r.dose_nM)): r.context_id for r in contexts.itertuples(index=False)}
    rows = []
    for u in units[~units.role.eq('vehicle')].itertuples(index=False):
        context = context_of.get((u.cell_line_id, int(u.dose_nM)))
        if context is None:
            continue
        rows.append(dict(compound_id=u.compound_id, context_id=context, split=u.role, replicate=u.sample,
                         cell_type=u.cell_line_id, dose_nM=float(u.dose_nM), plate=u.plate,
                         treatment_unit=u.unit_id, vehicle_unit=u.vehicle_unit,
                         scaffold_group=info.loc[u.compound_id, 'scaffold_group']))
    columns = ['compound_id', 'context_id', 'split', 'replicate', 'cell_type', 'dose_nM', 'plate', 'treatment_unit',
               'vehicle_unit', 'scaffold_group']
    return pd.DataFrame(rows, columns=columns).sort_values(['context_id', 'compound_id', 'replicate']).reset_index(drop=True)


def with_pooled_vehicles(units, counts, library, n_cells):
    """Append one row per vehicle unit (plate, line) pooling its vehicle samples; returns ids and arrays."""
    vehicle = units[units.role.eq('vehicle')]
    pooled = sorted(vehicle.vehicle_unit.unique())
    position = {u: i for i, u in enumerate(pooled)}
    rows = vehicle.index.to_numpy()
    indicator = sparse.csr_matrix((np.ones(len(rows), np.int64), ([position[v] for v in vehicle.vehicle_unit], rows)),
                                  shape=(len(pooled), len(units)))
    ids = np.r_[units.unit_id.to_numpy(str), np.asarray(pooled, str)]
    return (ids, sparse.vstack([counts, indicator@counts]).tocsr(), np.r_[library, indicator@library],
            np.r_[n_cells, indicator@n_cells])


def count_qc(pairs, cells_of, *, minimum_cells=MIN_CELLS, min_test_compounds=MIN_TEST_COMPOUNDS,
             min_test_groups=MIN_TEST_GROUPS):
    """Replicate-, compound- and context-level count QC from cell numbers only.

    A (compound, context) is eligible when every one of its replicate samples has
    at least ``minimum_cells`` cells of the line and so does each matched vehicle unit.
    """
    table = pairs.copy()
    lookup = pd.Series(cells_of, dtype=float)
    table['treatment_cells'] = table.treatment_unit.map(lookup).fillna(0).astype(int)
    table['vehicle_cells'] = table.vehicle_unit.map(lookup).fillna(0).astype(int)
    table['replicate_passes'] = (table.treatment_cells >= minimum_cells) & (table.vehicle_cells >= minimum_cells)
    compound = (table.groupby(['context_id', 'compound_id', 'split', 'scaffold_group'], as_index=False)
                .agg(replicates=('replicate', 'nunique'), passing=('replicate_passes', 'sum'),
                     min_treatment_cells=('treatment_cells', 'min'), min_vehicle_cells=('vehicle_cells', 'min')))
    compound['eligible'] = compound.replicates.ge(1) & compound.passing.eq(compound.replicates)
    summary = []
    for context, group in compound.groupby('context_id'):
        row = dict(context_id=context)
        for split in ('train', 'validation', 'test'):
            chosen = group[group.split.eq(split) & group.eligible]
            row[f'eligible_{split}_compounds'] = len(chosen)
            row[f'eligible_{split}_groups'] = chosen.scaffold_group.nunique()
        row['test_count_gate_met'] = (row['eligible_test_compounds'] >= min_test_compounds
                                      and row['eligible_test_groups'] >= min_test_groups)
        summary.append(row)
    return table, compound, pd.DataFrame(summary)


def study_gate(summary, contexts, *, min_contexts, min_lines, min_doses):
    """Gate G4: enough gate-passing contexts, cell lines and doses for a confirmatory result."""
    passing = contexts[contexts.context_id.isin(summary.loc[summary.test_count_gate_met, 'context_id'])]
    result = dict(gate_passing_contexts=sorted(passing.context_id), n_contexts=len(passing),
                  n_lines=int(passing.cell_line_id.nunique()), n_doses=int(passing.dose_nM.nunique()))
    result['met'] = (result['n_contexts'] >= min_contexts and result['n_lines'] >= min_lines
                     and result['n_doses'] >= min_doses)
    return result


def replicate_deltas(pairs, keys, row_of, counts, library, columns):
    """Per-replicate strict deltas, treated minus matched pooled vehicle log1p CPM, for the given keys.

    Returns a table of (compound_id, context_id, replicate, plate) and the delta matrix (rows in that order).
    """
    chosen = pairs.set_index(['compound_id', 'context_id']).loc[list(keys)].reset_index() if keys else pairs.iloc[:0]
    t = np.asarray([row_of[u] for u in chosen.treatment_unit], np.int64)
    v = np.asarray([row_of[u] for u in chosen.vehicle_unit], np.int64)
    columns = np.asarray(columns)
    if len(chosen) == 0:
        return chosen, np.zeros((0, len(columns)))
    delta = log_cpm(counts[t][:, columns].toarray(), library[t])-log_cpm(counts[v][:, columns].toarray(), library[v])
    return chosen[['compound_id', 'context_id', 'replicate', 'plate']].reset_index(drop=True), delta


def key_means(table, delta, keys):
    """Equal-weight mean over each key's replicates, in ``keys`` order."""
    if not keys:
        return np.zeros((0, delta.shape[1]))
    groups = table.groupby(['compound_id', 'context_id']).indices
    return np.vstack([delta[groups[k]].mean(axis=0) for k in keys])


def model_obs(cells, units):
    """v3 model-archive fields for model cells; the Combo of a vehicle cell is its pooled vehicle unit."""
    info = units.set_index('unit_id').loc[cells.unit_id]
    vehicle = info.role.eq('vehicle').to_numpy()
    return pd.DataFrame(dict(cell_id=cells.barcode.to_numpy(str),
                             Combo=np.where(vehicle, info.vehicle_unit, info.index).astype(str),
                             cell_type=info.cell_line_id.to_numpy(str), replicate=info['sample'].to_numpy(str),
                             plate_oligo=info.plate.to_numpy(str), compound_id=info.compound_id.to_numpy(str),
                             role=info.role.to_numpy(str), dose_nM=np.where(vehicle, 0.0, info.dose_nM.astype(float))))
