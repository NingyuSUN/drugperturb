"""Synthetic checks of the v5 Tahoe-100M design audit: samples, compounds, split, vehicle units, lines.

Fixtures establish software behaviour only; they say nothing about Tahoe-100M.
"""
import ast
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip('rdkit')
from drugperturb.v5 import design  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tahoe_like_samples(vehicles_per_plate=2):
    rows = []
    drugs = ['Aspirin', 'Phenol ', 'Caffeine']
    for plate, dose in [('plate1', 0.05), ('plate2', 0.5), ('plate3', 5.0)]:
        for i, drug in enumerate(drugs):
            rows.append(dict(sample=f'{plate}_d{i}', plate=plate, drug=drug,
                             drugname_drugconc=f"[('{drug.strip()}', {dose}, 'uM')]"))
        rows.append(dict(sample=f'{plate}_ref', plate=plate, drug='Adagrasib',
                         drugname_drugconc="[('Adagrasib', 0.05, 'uM')]"))
        for v in range(vehicles_per_plate):
            rows.append(dict(sample=f'{plate}_v{v}', plate=plate, drug=design.VEHICLE_DRUG,
                             drugname_drugconc=f"[('{design.VEHICLE_DRUG}', 0.0, 'uM')]"))
    return pd.DataFrame(rows)


def test_parse_drugconc_accepts_one_agent_in_uM_only():
    assert design.parse_drugconc("[('Erdafitinib ', 0.05, 'uM')]") == ('Erdafitinib', 0.05)
    with pytest.raises(ValueError):
        design.parse_drugconc("[('A', 0.05, 'uM'), ('B', 0.5, 'uM')]")
    with pytest.raises(ValueError):
        design.parse_drugconc("[('A', 50.0, 'nM')]")


def test_build_samples_assigns_roles_vehicles_and_normalized_names():
    samples = design.build_samples(tahoe_like_samples())
    assert samples.role.value_counts().to_dict() == {'compound': 9, 'vehicle': 6, 'reference_control': 3}
    phenol = samples[samples.drug_source.eq('Phenol ')]
    assert phenol.drug.eq('Phenol').all()
    assert samples.loc[samples['sample'].eq('plate2_d0'), 'vehicle_samples'].item() == 'plate2_v0;plate2_v1'
    assert samples.loc[samples.role.eq('compound'), 'dose_nM'].isin([50, 500, 5000]).all()
    assert samples.loc[samples.role.eq('vehicle'), 'dose_nM'].eq(0).all()


def test_build_samples_rejects_unexpected_layouts():
    with pytest.raises(ValueError, match='vehicle samples'):
        design.build_samples(tahoe_like_samples(vehicles_per_plate=1))
    bad = tahoe_like_samples()
    bad.loc[0, 'drugname_drugconc'] = "[('Aspirin', 1.0, 'uM')]"
    with pytest.raises(ValueError, match='dose'):
        design.build_samples(bad)
    bad = tahoe_like_samples()
    bad.loc[0, 'drugname_drugconc'] = "[('Ibuprofen', 0.05, 'uM')]"
    with pytest.raises(ValueError, match='match the drug column'):
        design.build_samples(bad)


def drug_metadata():
    return pd.DataFrame(dict(
        drug=['Aspirin', 'Phenol', 'Caffeine', 'Adagrasib'],
        canonical_smiles=['CC(=O)Oc1ccccc1C(=O)O', 'Oc1ccccc1', None, 'C'],
        pubchem_cid=[2244.0, 996.0, None, 1.0]))


def test_compound_table_excludes_missing_structures_and_keeps_one_parent():
    samples = design.build_samples(tahoe_like_samples())
    meta = drug_metadata()
    meta.loc[len(meta)] = ['Sodium phenolate', '[O-]c1ccccc1.[Na+]', 2.0]
    extra = pd.DataFrame([dict(sample=f'plate{p}_sp', plate=f'plate{p}', drug='Sodium phenolate',
                               drugname_drugconc=f"[('Sodium phenolate', {d}, 'uM')]")
                          for p, d in [(1, 0.05), (2, 0.5), (3, 5.0)]])
    samples = design.build_samples(pd.concat([tahoe_like_samples(), extra], ignore_index=True))
    table = design.compound_table(meta, samples).set_index('compound_id')
    assert 'Adagrasib' not in table.index
    assert table.loc['Caffeine', 'exclusion_reason'] == 'no SMILES in drug_metadata'
    assert table.loc['Phenol', 'status'] == 'accepted'
    assert table.loc['Sodium phenolate', 'exclusion_reason'] == design.DUPLICATE_PREFIX+'Phenol'
    assert table.loc['Sodium phenolate', 'removed_fragments'] == '[Na+]'


def test_compound_table_rejects_drugs_without_metadata():
    samples = design.build_samples(tahoe_like_samples())
    with pytest.raises(ValueError, match='without metadata'):
        design.compound_table(drug_metadata().iloc[:2], samples)


def test_split_compounds_splits_accepted_whole_groups_only():
    smiles = ['c1ccccc1C', 'c1ccccc1CC', 'C1CCCCC1C', 'c1ccncc1', 'c1ccoc1', 'c1ccsc1', 'C1CCNCC1', 'CCO']
    meta = pd.DataFrame(dict(drug=[f'd{i}' for i in range(len(smiles))]+['bad'],
                             canonical_smiles=smiles+['CCO.CCN'], pubchem_cid=range(len(smiles)+1)))
    rows = [dict(sample=f's{i}', plate='plate1', drug=d, drugname_drugconc=f"[('{d}', 0.5, 'uM')]")
            for i, d in enumerate(meta.drug)]
    rows += [dict(sample=f'v{i}', plate='plate1', drug=design.VEHICLE_DRUG,
                  drugname_drugconc=f"[('{design.VEHICLE_DRUG}', 0.0, 'uM')]") for i in range(2)]
    samples = design.build_samples(pd.DataFrame(rows))
    spec = dict(seed=42, test_fraction=0.25, validation_fraction=0.15, min_test_compounds=2, min_test_groups=2)
    table, report = design.split_compounds(design.compound_table(meta, samples), spec)
    assert table.loc[table.compound_id.eq('bad'), 'split'].item() == ''
    accepted = table[table.status.eq('accepted')]
    assert accepted.split.isin(['train', 'validation', 'test']).all()
    assert accepted.groupby('scaffold_group').split.nunique().eq(1).all()
    assert report['test_minimum_met']


def test_vehicle_units_zero_fill_and_refuse_compound_cells():
    samples = design.build_samples(tahoe_like_samples())
    cells = pd.DataFrame(dict(sample=['plate1_v0']*3+['plate1_v1']*2+['plate2_v0'],
                              cell_line_id=['L1', 'L1', 'L2', 'L1', 'L2', 'L1']))
    units = design.vehicle_units(cells, samples, ['L1', 'L2']).set_index(['plate', 'cell_line_id']).vehicle_cells
    assert units[('plate1', 'L1')] == 3 and units[('plate1', 'L2')] == 2
    assert units[('plate2', 'L1')] == 1 and units[('plate3', 'L2')] == 0
    assert len(units) == 6
    with pytest.raises(ValueError, match='vehicle samples only'):
        design.vehicle_units(pd.DataFrame(dict(sample=['plate1_d0'], cell_line_id=['L1'])), samples, ['L1'])
    with pytest.raises(ValueError, match='unknown cell lines'):
        design.vehicle_units(cells, samples, ['L1'])


def test_select_cell_lines_is_a_seeded_draw_among_eligible_lines():
    units = pd.DataFrame(dict(plate=['p1', 'p2']*6, cell_line_id=sorted(['A', 'B', 'C', 'D', 'E', 'F']*2),
                              vehicle_cells=[500, 400, 300, 300, 10, 20, 250, 150, 900, 900, 199, 201]))
    table, chosen = design.select_cell_lines(units, min_median_cells=200, n_lines=3, seed=42)
    eligible = table.set_index('cell_line_id').eligible
    assert eligible.to_dict() == {'A': True, 'B': True, 'C': False, 'D': True, 'E': True, 'F': True}
    assert len(chosen) == 3 and set(chosen) <= set(eligible[eligible].index)
    assert design.select_cell_lines(units, min_median_cells=200, n_lines=3, seed=42)[1] == chosen
    assert table.selected.sum() == 3 and sorted(table.loc[table.selected, 'draw_order']) == [1, 2, 3]
    with pytest.raises(ValueError, match='meet the vehicle-size rule'):
        design.select_cell_lines(units, min_median_cells=200, n_lines=6)


def test_context_ids_cover_every_dose_of_each_line():
    contexts = design.context_table(['CVCL_2', 'CVCL_1'])
    assert contexts.context_id.tolist() == ['CVCL_1_50nM', 'CVCL_1_500nM', 'CVCL_1_5000nM',
                                            'CVCL_2_50nM', 'CVCL_2_500nM', 'CVCL_2_5000nM']


def test_audit_script_reads_design_columns_only():
    tree = ast.parse((ROOT/'scripts/v5_design_audit.py').read_text())
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and isinstance(node.value, ast.Tuple)}
    assert constants['OBS_COLUMNS'] == ('sample', 'cell_line', 'pass_filter')
    assert not any(c.startswith('mean_') for c in design.SAMPLE_COLUMNS)
    forbidden = {'gene_count', 'tscp_count', 'mread_count', 'pcnt_mito', 'S_score', 'G2M_score', 'phase'}
    assert not forbidden & set(constants['OBS_COLUMNS'])
