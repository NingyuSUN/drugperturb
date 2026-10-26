"""Synthetic checks of the v6 cell-line draw and split count gate, plus the committed manifests.

Fixtures establish software behaviour only; they say nothing about Tahoe-100M.
"""
import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from drugperturb.v6 import design

ROOT = Path(__file__).resolve().parents[1]


def units_frame(sizes):
    """One (plate, line) vehicle unit per entry of ``sizes``: {line: [cells per plate]}."""
    rows = [dict(plate=f'plate{i+1}', cell_line_id=line, vehicle_cells=n)
            for line, counts in sizes.items() for i, n in enumerate(counts)]
    return pd.DataFrame(rows)


def test_draw_excludes_scored_lines_and_is_seeded():
    units = units_frame({'A': [500, 500], 'B': [400, 400], 'C': [300, 300],
                         'D': [10, 20], 'E': [900, 900], 'F': [201, 199]})
    table, chosen = design.select_new_cell_lines(units, exclude=['A', 'B'], min_median_cells=200, n_lines=2, seed=6)
    flags = table.set_index('cell_line_id')
    assert flags.eligible.to_dict() == {'A': True, 'B': True, 'C': True, 'D': False, 'E': True, 'F': True}
    assert flags.scored_in_v5.to_dict() == {'A': True, 'B': True, 'C': False, 'D': False, 'E': False, 'F': False}
    assert flags.in_pool.to_dict() == {'A': False, 'B': False, 'C': True, 'D': False, 'E': True, 'F': True}
    assert set(chosen) <= {'C', 'E', 'F'} and len(chosen) == 2
    assert design.select_new_cell_lines(units, exclude=['A', 'B'], min_median_cells=200,
                                        n_lines=2, seed=6)[1] == chosen
    assert sorted(table.loc[table.selected, 'draw_order']) == [1, 2]


def test_draw_refuses_an_unscored_pool_that_is_too_small_or_unknown():
    units = units_frame({'A': [500], 'B': [400], 'C': [10]})
    with pytest.raises(ValueError, match='unscored lines meet the vehicle-size rule'):
        design.select_new_cell_lines(units, exclude=['A'], min_median_cells=200, n_lines=2)
    with pytest.raises(ValueError, match='absent from the vehicle units'):
        design.select_new_cell_lines(units, exclude=['Z'], min_median_cells=200, n_lines=1)


def test_split_gate_thresholds_leave_v5s_threshold_in_each_half():
    assert design.MIN_WELL_CELLS == 2*design.MIN_HALF_CELLS
    assert design.MIN_VEHICLE_CELLS == 4*design.MIN_HALF_CELLS
    units = units_frame({'A': [500, 119], 'B': [120, 2000]})
    gate = design.vehicle_gate(units, ['A', 'B'])
    assert gate.clears_split_gate.tolist() == [True, False, True, True]
    assert gate.cells_per_half.tolist() == [250, 59, 60, 1000]
    assert design.vehicle_gate(units, ['A']).cell_line_id.unique().tolist() == ['A']
    per_line, overall = design.gate_feasibility(gate)
    assert overall == dict(threshold=120, units=4, units_clearing_gate=3, lines_with_every_unit_clearing=1)
    assert [r['cell_line_id'] for r in per_line] == ['A', 'B']
    assert per_line[0]['min_vehicle_cells'] == 119


def test_audit_script_reads_no_tahoe_file():
    source = (ROOT/'scripts/v6_design_audit.py').read_text()
    tree = ast.parse(source)
    constants = {node.targets[0].id: node.value for node in tree.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert ast.literal_eval(constants['INPUTS']) == ('cell_lines', 'vehicle_units')
    assert ast.literal_eval(constants['SEED']) == 6
    assert ast.literal_eval(constants['N_LINES']) == 10
    for forbidden in ('parquet', 'huggingface', 'obs_metadata', 'sample_metadata', 'requests', 'urlopen'):
        assert forbidden not in source.lower(), f'the v6 audit must not reach for {forbidden}'


def test_committed_manifests_match_the_protocol_and_the_v5_audit():
    lines = pd.read_csv(ROOT/'manifests/v6/cell_lines.tsv', sep='\t')
    contexts = pd.read_csv(ROOT/'manifests/v6/contexts.tsv', sep='\t')
    gate = pd.read_csv(ROOT/'manifests/v6/vehicle_units.tsv', sep='\t')
    v5 = pd.read_csv(ROOT/'manifests/v5/cell_lines.tsv', sep='\t')
    scored_in_v5 = set(v5.loc[v5.selected, 'cell_line_id'])
    assert len(lines) == 50 and lines.eligible.sum() == 48
    assert lines.in_pool.sum() == 38 and lines.selected.sum() == 10
    assert set(lines.loc[lines.scored_in_v5, 'cell_line_id']) == scored_in_v5
    assert not set(lines.loc[lines.selected, 'cell_line_id']) & scored_in_v5, 'a v6 line was scored in v5'
    assert (lines.loc[lines.selected, 'median_vehicle_cells'] >= 200).all()
    assert len(contexts) == 30 and contexts.dose_nM.nunique() == 3
    assert set(contexts.cell_line_id) == set(lines.loc[lines.selected, 'cell_line_id'])
    assert set(gate.cell_line_id) == set(contexts.cell_line_id) and len(gate) == 140
    receipt = json.loads((ROOT/'receipts/v6/design_audit.json').read_text())
    assert receipt['expression_read'] is False and receipt['compound_sample_cell_counts_read'] is False
    assert receipt['draw'] == dict(seed=6, n_lines=10,
                                   rule='v5 mechanism over the v5-eligible lines v5 did not select')
    assert receipt['selected_lines'] == sorted(lines.loc[lines.selected, 'cell_line_id'])
