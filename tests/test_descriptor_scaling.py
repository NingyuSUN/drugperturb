"""Descriptor column tolerance (min_std): v3/v4 default unchanged, noise columns dropped.

The manifest checks compute label-free descriptors for the committed compound
structures; they check software behaviour, not any response or score.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drugperturb.v3 import descriptors
from drugperturb.v3.descriptors import NOISE_STD_TOLERANCE, VEHICLE_SMILES, TrainingDescriptorScaler

ROOT = Path(__file__).resolve().parents[1]
# OP3 compounds with v3 z-scores near 1e16 (reports/v4/report.md): ixabepilone, alvocidib,
# dasatinib (test) and CEP-37440 (validation).
OP3_AFFECTED = ['LSM-43293', 'LSM-1011', 'LSM-1020', 'LSM-45849']
# Ten copies of 0.1 average to 0.09999999999999999, so np.std of the column is ~1e-17, not 0.
NOISY = np.column_stack([np.linspace(0.0, 1.0, 10), np.full(10, 0.1)])


def test_tolerance_drops_constant_columns_with_float_noise_std():
    query = np.array([[0.5, 0.2]])
    v3 = TrainingDescriptorScaler().fit(NOISY)
    assert v3.keep.tolist() == [True, True] and np.abs(v3.transform(query)).max() > 1e12
    tolerant = TrainingDescriptorScaler(NOISE_STD_TOLERANCE).fit(NOISY)
    assert tolerant.keep.tolist() == [True, False]
    assert np.array_equal(tolerant.transform(query), v3.transform(query)[:, :1])
    with pytest.raises(ValueError):
        TrainingDescriptorScaler(-1.0)


def test_compound_table_defaults_to_v3_and_passes_min_std(monkeypatch):
    raw = np.vstack([NOISY, [0.5, 0.1]])  # ten compounds, then DMSO
    monkeypatch.setattr(descriptors, 'rdkit2d_normalized', lambda smiles: (raw[:len(smiles)], ['a', 'b']))
    ids = [f'C{i}' for i in range(10)]
    assert descriptors.compound_table(ids, ['C']*10, ids)[1] == ['a', 'b']
    table, kept = descriptors.compound_table(ids, ['C']*10, ids, min_std=NOISE_STD_TOLERANCE)
    assert kept == ['a'] and table[descriptors.VEHICLE_ID].shape == (1,)


@pytest.fixture(scope='module', params=['sciplex3', 'op3'])
def manifest_descriptors(request):
    pytest.importorskip('descriptastorus')
    path = {'sciplex3': 'manifests/split_manifest.tsv', 'op3': 'manifests/v4/op3_compounds.tsv'}[request.param]
    table = pd.read_csv(ROOT/path, sep='\t', keep_default_na=False)
    raw, names = descriptors.rdkit2d_normalized([*table.canonical_smiles, VEHICLE_SMILES])
    return request.param, [*table.compound_id, 'DMSO'], raw, raw[:-1][table.split.eq('train').to_numpy()], names


def test_tolerance_keeps_exactly_the_nonconstant_training_columns(manifest_descriptors):
    dataset, ids, raw, train, names = manifest_descriptors
    v3 = TrainingDescriptorScaler().fit(train)
    tolerant = TrainingDescriptorScaler(NOISE_STD_TOLERANCE).fit(train)
    filled = np.where(np.isnan(train), np.nanmean(train, axis=0), train)
    assert np.array_equal(tolerant.keep, np.ptp(filled, axis=0) > 0)
    dropped = [n for n, old, new in zip(names, v3.keep, tolerant.keep) if old and not new]
    assert len(dropped) == {'sciplex3': 25, 'op3': 28}[dataset]  # the 25-28 columns in the v4 report
    # Every column both rules keep is scaled identically.
    v3_z, tolerant_z = v3.transform(raw), tolerant.transform(raw)
    shared = tolerant.keep[v3.keep]
    assert np.array_equal(tolerant_z, v3_z[:, shared])
    # Under v3, a compound equal to training in a dropped column gets the same value there
    # as every other such compound (±1: the offset and std are the same rounding error).
    affected = np.isin(ids, OP3_AFFECTED)
    noise = v3_z[:, ~shared]
    assert (noise[~affected] == noise[~affected][0]).all() and np.array_equal(np.abs(noise[0]), np.ones(len(dropped)))
    v3_max, tolerant_max = np.abs(v3_z).max(axis=1), np.abs(tolerant_z).max(axis=1)
    if dataset == 'sciplex3':
        assert not affected.any() and np.array_equal(v3_max, tolerant_max) and v3_max.max() < 50
    else:
        assert affected.sum() == len(OP3_AFFECTED)
        assert (v3_max[affected] > 1e15).all() and (v3_max[~affected] < 50).all()
        assert tolerant_max.max() < 10
