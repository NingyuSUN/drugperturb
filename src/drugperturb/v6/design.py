"""v6 cell-line draw and split count gate; reads the committed v5 vehicle-unit audit, never expression.

v5 scored 10 of Tahoe-100M's 50 lines, so their responses are open. v6 draws its
lines from the v5-eligible lines v5 did not select, which keeps every v6 response
unread by this project. The draw mechanism is v5's - the pool sorted by
Cellosaurus ID, then ``numpy.random.default_rng(seed).choice`` without
replacement - with the pool restricted and a different seed.

The v6 endpoint splits each well's cells into a selection and an evaluation half
(protocol_v6 measurement_split), so every count gate doubles v5's: a replicate
well needs ``MIN_WELL_CELLS`` cells of the line and a matched vehicle unit needs
``MIN_VEHICLE_CELLS``, which leaves ``MIN_HALF_CELLS`` - v5's own threshold - in
each half.
"""
from __future__ import annotations

import numpy as np

from ..v5.design import context_table  # noqa: F401  (re-exported; the v6 context rule is v5's)

MIN_HALF_CELLS = 30
MIN_WELL_CELLS = 2*MIN_HALF_CELLS
MIN_VEHICLE_CELLS = 4*MIN_HALF_CELLS


def select_new_cell_lines(units, *, exclude, min_median_cells=200, n_lines=10, seed=6):
    """Draw ``n_lines`` from the lines whose median vehicle unit is >= min_median_cells and
    that are not in ``exclude``. Returns the full 50-line table and the sorted chosen IDs."""
    excluded = set(exclude)
    median = units.groupby('cell_line_id').vehicle_cells.median().sort_index()
    eligible = [c for c in median.index if median[c] >= min_median_cells]
    unknown = excluded-set(median.index)
    if unknown:
        raise ValueError(f'excluded lines absent from the vehicle units: {sorted(unknown)}')
    pool = [c for c in eligible if c not in excluded]
    if len(pool) < n_lines:
        raise ValueError(f'only {len(pool)} unscored lines meet the vehicle-size rule; {n_lines} are required')
    draw = np.random.default_rng(seed).choice(len(pool), size=n_lines, replace=False)
    chosen = [pool[i] for i in draw]
    table = median.rename('median_vehicle_cells').reset_index()
    table['eligible'] = table.cell_line_id.isin(eligible)
    table['scored_in_v5'] = table.cell_line_id.isin(excluded)
    table['in_pool'] = table.cell_line_id.isin(pool)
    table['draw_order'] = table.cell_line_id.map({c: i+1 for i, c in enumerate(chosen)}).fillna(0).astype(int)
    table['selected'] = table.draw_order.gt(0)
    return table, sorted(chosen)


def vehicle_gate(units, lines, *, min_vehicle_cells=MIN_VEHICLE_CELLS):
    """The drawn lines' vehicle units with a flag for the split gate; one row per (plate, line)."""
    table = units[units.cell_line_id.isin(set(lines))].sort_values(['cell_line_id', 'plate']).copy()
    table['clears_split_gate'] = table.vehicle_cells.ge(min_vehicle_cells)
    table['cells_per_half'] = table.vehicle_cells//2
    return table.reset_index(drop=True)


def gate_feasibility(gate_table, *, min_vehicle_cells=MIN_VEHICLE_CELLS):
    """Per-line summary of the vehicle split gate; a feasibility report, not the gate itself.

    The protocol's count gate is computed by the preparation kernel from the extracted
    cells, which this audit never reads. This summary says only whether the committed
    vehicle-unit sizes leave room for the split.
    """
    rows = []
    for line, group in gate_table.groupby('cell_line_id', sort=True):
        rows.append(dict(cell_line_id=line, units=len(group),
                         units_clearing_gate=int(group.clears_split_gate.sum()),
                         min_vehicle_cells=int(group.vehicle_cells.min()),
                         median_vehicle_cells=float(group.vehicle_cells.median())))
    return rows, dict(threshold=min_vehicle_cells, units=len(gate_table),
                      units_clearing_gate=int(gate_table.clears_split_gate.sum()),
                      lines_with_every_unit_clearing=sum(r['units'] == r['units_clearing_gate'] for r in rows))
