"""Synthetic engineering checks for v3 response mapping, cell roles and vendored code."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from drugperturb.v3 import runner
from drugperturb.v3.cells import assign_roles, build_model_archive, load_model_archive, save_model_archive
from drugperturb.v3.pseudobulk import (compound_delta, counterfactual_deltas, measured_log_cpm,
                                       normalize_counts, pseudobulk_log_cpm)

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def well(n=7, genes=5):
    counts = sparse.csr_matrix(rng.poisson(3, size=(n, genes)))
    library = np.asarray(counts.sum(axis=1)).ravel()+rng.integers(5, 50, size=n)  # other genes
    return counts, library.astype(float)


def test_identity_cells_reproduce_measured_well_cpm():
    counts, library = well()
    values = normalize_counts(counts, library)
    measured = measured_log_cpm(counts, library)
    # Float32 model inputs: identity agrees to single-precision rounding.
    assert np.allclose(pseudobulk_log_cpm(values, library), measured, atol=1e-5)
    exact = np.log1p(counts.toarray()/library[:, None]*1e4)
    assert np.allclose(pseudobulk_log_cpm(exact, library), measured, atol=1e-12)
    expected = np.log1p(np.asarray(counts.sum(axis=0)).ravel()/library.sum()*1e6)
    assert np.allclose(measured, expected)


def test_negative_predictions_are_clipped_and_bad_libraries_rejected():
    values = np.full((3, 2), -0.5)
    assert np.array_equal(pseudobulk_log_cpm(values, np.ones(3)), np.zeros(2))
    with pytest.raises(ValueError):
        pseudobulk_log_cpm(values, np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        compound_delta(np.zeros((3, 2)))


def test_identity_model_gives_exact_zero_contrast():
    wells = {}
    for name in ['v1', 'v2']:
        counts, library = well()
        wells[name] = dict(values=normalize_counts(counts, library), counts=counts, library=library)
    queries = {('S1', 'ctx'): [dict(treated=dict(compound='S1', dose_nM=1000.0, cell_type='A549'),
                                    vehicle=dict(compound='DMSO', dose_nM=0.0, cell_type='A549'), vehicle_well=w)
                               for w in ['v1', 'v2']]}
    identity = lambda w, condition: w['values']
    contrast, anchored = counterfactual_deltas(identity, queries, wells, gene_count=5)
    assert np.array_equal(contrast[('S1', 'ctx')], np.zeros(5))
    assert np.allclose(anchored[('S1', 'ctx')], 0, atol=1e-5)
    shift = lambda w, condition: w['values']+(np.log(2) if condition['compound'] != 'DMSO' else 0)
    doubled, _ = counterfactual_deltas(shift, queries, wells, gene_count=5)
    assert np.all(doubled[('S1', 'ctx')] > 0)


def fixture_tables():
    wells = pd.DataFrame(dict(
        Combo=['d1', 'd2', 'd3', 'd4', 'v1'], row_kind=['drug']*4+['vehicle'],
        treatment=['S1', 'S2', 'S3', 'S9', 'S0000'], context_dose_nM=[1000]*4+[1000],
        context_id=['A549_24h_1000nM']*5))
    split = pd.DataFrame(dict(compound_id=['S1', 'S2', 'S3'], split=['train', 'validation', 'test']))
    table = pd.DataFrame(dict(cell_id=[f'c{i}' for i in range(6)], Combo=['d1', 'd2', 'd3', 'd4', 'v1', 'v1'],
                              cell_type='A549', replicate='rep1', plate_oligo='p1'))
    return wells, split, table


def test_roles_drop_test_and_excluded_cells_before_archiving(tmp_path):
    wells, split, table = fixture_tables()
    labelled = assign_roles(table, wells, split)
    assert labelled.role.tolist() == ['train', 'validation', 'test', 'excluded', 'vehicle', 'vehicle']
    assert labelled.compound_id.tolist()[-2:] == ['DMSO', 'DMSO']
    counts = sparse.csr_matrix(np.arange(12).reshape(6, 2))
    obs, merged, library, dropped = build_model_archive([(labelled, counts, np.full(6, 10.0))])
    assert dropped == {'test': 1, 'excluded': 1} and set(obs.role) == {'train', 'validation', 'vehicle'}
    assert merged.shape == (4, 2) and 'S3' not in set(obs.compound_id)
    save_model_archive(tmp_path/'m.npz', obs, merged, library)
    obs2, merged2, library2 = load_model_archive(tmp_path/'m.npz')
    assert obs2.equals(obs.astype({'dose_nM': float})) and (merged2 != merged).nnz == 0


def test_protocol_masks_use_only_serving_vehicles():
    obs = pd.DataFrame(dict(cell_id=list('abcdef'), Combo=['d1', 'd2', 'v1', 'v2', 'd3', 'd4'],
                            cell_type=['A549']*4+['MCF7']*2, role=['train', 'validation', 'vehicle', 'vehicle', 'train', 'train'],
                            compound_id=['S1', 'S2', 'DMSO', 'DMSO', 'S1', 'S1'],
                            dose_nM=[1000.0, 1000.0, 0.0, 0.0, 1000.0, 10.0]))
    serving = pd.DataFrame(dict(Combo=['v1', 'v2'], context_id=['A549_24h_1000nM', 'A549_24h_10nM']))
    train, val, cells = runner.protocol_cells(obs, 'Q1', serving)
    assert train.tolist() == [True, False, True, False, False, False] and val.tolist() == [False, True]+[False]*4
    train2, _, cells2 = runner.protocol_cells(obs, 'Q2', serving)
    assert train2.tolist() == [True, False, True, True, True, True] and cells2 == ['A549', 'K562', 'MCF7']


def test_vendored_chemcpa_file_is_byte_identical():
    path = ROOT/'src/drugperturb/third_party/_chemcpa_model_upstream.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        '481fe1f447f1318e0446764d4dcfc8b7e6d6c21efe21cd156fceb8e693f75ae6'
