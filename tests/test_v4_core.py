"""Synthetic checks of v4 design, count QC, truth, gene selection, contrasts and diagnostics.

Fixtures establish software behaviour only; they say nothing about OP3 or sci-Plex3.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from drugperturb.v3 import pseudobulk as v3_pseudobulk
from drugperturb.v3.pseudobulk import normalize_counts
from drugperturb.v4 import counterfactual, design, diagnostics


def op3_like_design(cells_per_well=3, drop=None):
    """Three plates (one donor each), two rows each; columns 1-3 controls and DMSO, 4-5 compounds."""
    rows = []
    compounds = {('A', 4): ('LSM-1', 'cmpA', 'CCO'), ('A', 5): ('LSM-2', 'cmpB', 'c1ccccc1O'),
                 ('B', 4): ('LSM-3', 'cmpC', 'CC(=O)Nc1ccc(O)cc1'), ('B', 5): ('LSM-4', 'cmpD', 'CCN')}
    for p, donor in enumerate(['Donor 1', 'Donor 2', 'Donor 3']):
        plate = f'P{p}'
        for row in 'AB':
            for col in range(1, 6):
                if col == 1:
                    lincs, name, smiles, dose, control = 'LSM-B', 'Belinostat', 'O', 0.1, True
                elif col == 2:
                    lincs, name, smiles, dose, control = 'LSM-D', 'Dabrafenib', 'N', 1.0, True
                elif col == 3:
                    lincs, name, smiles, dose, control = 'LSM-V', design.VEHICLE_NAME, 'CS(C)=O', 14.1, True
                else:
                    (lincs, name, smiles), dose, control = compounds[(row, col)], 1.0, False
                if drop == (plate, f'{row}{col}'):
                    continue
                for _ in range(cells_per_well):
                    rows.append(dict(plate_name=plate, well=f'{row}{col}', row=row, col=col, donor_id=donor,
                                     library_id=f'{plate}{row}', sm_name=name, sm_lincs_id=lincs, SMILES=smiles,
                                     dose_uM=dose, timepoint_hr=24.0, control=control))
    return pd.DataFrame(rows)


def test_split_by_structure_fractions_is_structure_only_and_whole_group():
    records = [dict(compound_id=f'C{i}', canonical_smiles=f'S{i}', scaffold_group=f'G{i//2}') for i in range(80)]
    out, report = design.split_by_structure_fractions(records, seed=42, test_fraction=0.25, validation_fraction=0.15)
    again, _ = design.split_by_structure_fractions(list(reversed(records)), seed=42, test_fraction=0.25,
                                                   validation_fraction=0.15)
    assignment = {r['compound_id']: r['split'] for r in out}
    assert assignment == {r['compound_id']: r['split'] for r in again}
    groups = pd.DataFrame(out).groupby('scaffold_group').split.nunique()
    assert (groups == 1).all()
    assert report['test']['n_compounds'] >= 20 and report['test']['n_compounds'] <= 22
    assert report['validation']['n_compounds'] >= 12 and report['train']['n_compounds'] > 0
    with pytest.raises(ValueError):
        design.split_by_structure_fractions(records[:6], seed=42)


def test_build_wells_matches_same_row_vehicle_and_rejects_bad_layouts():
    wells = design.build_wells(op3_like_design())
    drugs = wells[wells.role.eq('drug')]
    assert len(drugs) == 12 and set(drugs.vehicle_well) == {'A3', 'B3'}
    assert (drugs.vehicle_well.str[0] == drugs.well.str[0]).all()
    assert set(wells.loc[wells.role.eq('vehicle'), 'compound_id']) == {'DMSO'}
    assert drugs.dose_nM.eq(1000.0).all()
    with pytest.raises(ValueError, match='one well per donor'):
        design.build_wells(op3_like_design(drop=('P1', 'A4')))
    with pytest.raises(ValueError, match='vehicle well'):
        design.build_wells(op3_like_design(drop=('P2', 'B3')))
    bad = op3_like_design()
    bad.loc[bad.well.eq('A4'), 'dose_uM'] = 10.0
    with pytest.raises(ValueError, match='1 uM'):
        design.build_wells(bad)


def compounds_frame():
    return pd.DataFrame(dict(compound_id=['LSM-1', 'LSM-2', 'LSM-3', 'LSM-4'], split=['train', 'train', 'validation', 'test'],
                             scaffold_group=['g1', 'g2', 'g3', 'g4']))


def test_response_pairs_and_count_qc_require_all_three_donors():
    wells = design.build_wells(op3_like_design())
    pairs = design.response_pairs(wells, compounds_frame())
    assert len(pairs) == 4*3*len(design.CELL_CODES)
    cells = {u: 40 for u in set(pairs.treatment_unit) | set(pairs.vehicle_unit)}
    first = pairs[(pairs.compound_id == 'LSM-1') & (pairs.cell_type == 'NKcells')].treatment_unit.iloc[0]
    cells[first] = 29
    _, compound, summary = design.count_qc(pairs, cells, min_test_compounds=1, min_test_groups=1)
    flags = compound.set_index(['compound_id', 'context_id']).eligible
    assert not flags[('LSM-1', 'NKcells_24h_1000nM')] and flags[('LSM-1', 'Tcells_24h_1000nM')]
    assert summary.set_index('context_id').test_count_gate_met.all()


def test_strict_deltas_are_donor_means_of_log_cpm_differences():
    wells = design.build_wells(op3_like_design())
    pairs = design.response_pairs(wells, compounds_frame())
    units = sorted(set(pairs.treatment_unit) | set(pairs.vehicle_unit))
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 50, size=(len(units), 6)).astype(float)
    library = counts.sum(axis=1)+100
    key = ('LSM-2', 'Bcells_24h_1000nM')
    got = design.strict_deltas(counts, library, units, pairs, [key])[0]
    rows = pairs[(pairs.compound_id == key[0]) & (pairs.context_id == key[1])]
    position = {u: i for i, u in enumerate(units)}
    manual = np.mean([np.log1p(counts[position[t]]/library[position[t]]*1e6)
                      - np.log1p(counts[position[v]]/library[position[v]]*1e6)
                      for t, v in zip(rows.treatment_unit, rows.vehicle_unit)], axis=0)
    assert np.allclose(got, manual, atol=0, rtol=0)
    sub = design.strict_deltas(counts, library, units, pairs, [key], gene_columns=[4, 1])[0]
    assert np.array_equal(sub, got[[4, 1]])


def test_select_genes_filters_on_vehicle_detection_then_ranks_training_variance():
    deltas = np.array([[0.0, 5.0, 1.0, 2.0], [0.0, -5.0, -1.0, 0.0]])
    detection = np.array([0.9, 0.01, 0.5, 0.5])
    assert design.select_genes(deltas, detection, n_genes=2, min_detection=0.05).tolist() == [2, 3]
    with pytest.raises(ValueError):
        design.select_genes(deltas, detection, n_genes=3, min_detection=0.05)


def vehicle_fixture(n_wells=3, genes=5, seed=1):
    rng = np.random.default_rng(seed)
    obs, blocks = [], []
    for w in range(n_wells):
        blocks.append(rng.poisson(3.0, size=(20, genes)))
        obs += [dict(Combo=f'v{w}', role='vehicle')]*20
    counts = sparse.csr_matrix(np.vstack(blocks))
    library = np.asarray(counts.sum(axis=1)).ravel()+50.0
    obs = pd.DataFrame(obs)
    index = {f'v{w}': np.flatnonzero(obs.Combo.eq(f'v{w}')) for w in range(n_wells)}
    cells = {w: dict(values=normalize_counts(counts[i], library[i]), counts=counts[i], library=library[i],
                     n_cells=len(i), index=i) for w, i in index.items()}
    return cells


def query_rows(n_replicates):
    return pd.DataFrame([dict(compound_id='X', context_id='K_24h_1000nM', split='test', replicate=f'r{i}',
                              cell_type='K', dose_nM=1000.0, vehicle_Combo=f'v{i}') for i in range(n_replicates)])


def test_two_replicate_contrast_equals_v3_and_identity_is_zero_change():
    cells = vehicle_fixture(2)
    table = query_rows(2).assign(replicate=['rep1', 'rep2'])
    from drugperturb.v3 import runner
    v3_queries = runner.build_queries(table, ['K_24h_1000nM'], ['test'])
    v4_queries = counterfactual.build_queries(table, ['K_24h_1000nM'], ['test'], 2)
    assert v3_queries == v4_queries
    shift = lambda well, c: well['values']+(0.3 if c['compound'] != 'DMSO' else 0.0)
    a = v3_pseudobulk.counterfactual_deltas(shift, v3_queries, cells, gene_count=5)
    b = counterfactual.counterfactual_deltas(shift, v4_queries, cells, gene_count=5, n_replicates=2)
    for x, y in zip(a, b):
        assert all(np.array_equal(x[k], y[k]) for k in x)


def test_three_donor_identity_model_reproduces_zero_change_exactly():
    cells = vehicle_fixture(3)
    queries = counterfactual.build_queries(query_rows(3), ['K_24h_1000nM'], ['test'], 3)
    contrast, anchored = counterfactual.counterfactual_deltas(lambda well, c: well['values'], queries, cells,
                                                              gene_count=5, n_replicates=3)
    assert np.array_equal(contrast[('X', 'K_24h_1000nM')], np.zeros(5))
    assert np.allclose(anchored[('X', 'K_24h_1000nM')], 0.0, atol=1e-6)  # float32 model inputs
    assert counterfactual.build_queries(query_rows(2), ['K_24h_1000nM'], ['test'], 3) == {}


def test_load_truth_decompresses_once(tmp_path):
    delta = np.arange(12.0).reshape(3, 4)
    np.savez_compressed(tmp_path/'t.npz', keys=np.asarray(['a|K', 'b|K', 'c|L']), delta=delta)
    truth = counterfactual.load_truth(tmp_path/'t.npz')
    assert np.array_equal(truth[('c', 'L')], delta[2])
    bases = {id(v.base) for v in truth.values()}
    assert len(bases) == 1  # every row is a view of one decompressed matrix


def test_diagnostics_flag_collapse_and_select_primary_by_rule():
    rng = np.random.default_rng(3)
    truth = {(f'c{i}', 'K'): rng.normal(0.5, 1.0, 20) for i in range(10)}
    reference = {'K': np.full(20, 0.5)}
    collapsed = diagnostics.summarize({k: np.full(20, 1e-4) for k in truth}, truth, reference)
    assert collapsed['effect_size_ratio'] < 0.01 and collapsed['mean_gene_pearson_r'] == 0.0
    assert not diagnostics.non_collapse(collapsed)['non_collapsed']
    informed = diagnostics.summarize({k: 0.8*v for k, v in truth.items()}, truth, reference,
                                     {'K': np.mean(list(truth.values()), axis=0)})
    assert diagnostics.non_collapse(informed)['non_collapsed'] and informed['compound_specific_r'] > 0.5
    arms = {'I1': dict(validation_rmse=1.0, non_collapse=dict(non_collapsed=False)),
            'I2': dict(validation_rmse=1.2, non_collapse=dict(non_collapsed=True)),
            'I3': dict(validation_rmse=1.1, non_collapse=dict(non_collapsed=True))}
    assert diagnostics.select_primary(arms, ['I1', 'I2', 'I3'])['selected'] == 'I3'
    for a in arms.values():
        a['non_collapse']['non_collapsed'] = False
    chosen = diagnostics.select_primary(arms, ['I1', 'I2', 'I3'])
    assert chosen['selected'] == 'I1' and not chosen['any_passing']
    assert diagnostics.finite({'a': np.float64('nan'), 'b': [np.int64(2)]}) == {'a': None, 'b': [2]}
