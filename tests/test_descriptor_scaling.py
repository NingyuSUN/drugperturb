"""Descriptor column tolerance (min_std) and the seal-time |z| gate.

The v3/v4 default is unchanged and noise columns are dropped under the tolerance;
the gate refuses scaled descriptors above DESCRIPTOR_Z_LIMIT. The manifest checks
compute label-free descriptors for the committed compound structures; they check
software behaviour, not any response or score.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from drugperturb.v3 import descriptors
from drugperturb.v3.descriptors import (DESCRIPTOR_Z_LIMIT, NOISE_STD_TOLERANCE, VEHICLE_SMILES,
                                        TrainingDescriptorScaler, descriptor_gate, scaling_diagnostics)

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


def noisy_fit(monkeypatch, min_std):
    """A fit record's descriptor_scaling: ten NOISY training compounds, validation compound V, then DMSO."""
    raw = np.vstack([NOISY, [0.5, 0.2], [0.5, 0.1]])  # V differs from training in the constant column
    monkeypatch.setattr(descriptors, 'rdkit2d_normalized', lambda smiles: (raw[:len(smiles)], ['a', 'b']))
    ids = [*(f'C{i}' for i in range(10)), 'V']
    table, _ = descriptors.compound_table(ids, ['C']*11, ids[:10], min_std=min_std)
    return dict(name='fit', descriptor_scaling=scaling_diagnostics(table, {**dict.fromkeys(ids[:10], 'train'),
                                                                           'V': 'validation'}))


def test_gate_refuses_a_training_constant_noisy_column_and_names_the_compound(monkeypatch):
    scaling = noisy_fit(monkeypatch, 0.0)['descriptor_scaling']
    assert scaling['z_limit'] == DESCRIPTOR_Z_LIMIT and list(scaling['above_limit']) == ['validation']
    assert list(scaling['above_limit']['validation']) == ['V'] and scaling['max_abs_z']['validation'] > 1e12
    assert scaling['max_abs_z']['train'] < 10 and scaling['max_abs_z']['vehicle'] < 10
    with pytest.raises(ValueError, match=r'descriptor gate: .*compounds above 1000: V \(validation, \|z\| = '):
        descriptor_gate({'fit': dict(descriptor_scaling=scaling)})


def test_gate_passes_under_the_noise_tolerance(monkeypatch):
    gate = descriptor_gate({'fit': noisy_fit(monkeypatch, NOISE_STD_TOLERANCE)})
    assert gate['passed'] and not gate['override'] and gate['above_limit'] == {} and not gate['fits_without_diagnostics']
    assert set(gate['max_abs_z']) == {'train', 'validation', 'vehicle'} and max(gate['max_abs_z'].values()) < 10


def test_gate_override_is_recorded_and_unchecked_fits_are_refused(monkeypatch):
    gate = descriptor_gate({'fit': noisy_fit(monkeypatch, 0.0)}, override=True)
    assert gate['override'] and not gate['passed'] and list(gate['above_limit']['validation']) == ['V']
    json.dumps(gate, allow_nan=False)  # stored as is in the seal
    # A record written before the gate has no diagnostics, so it cannot be certified.
    with pytest.raises(ValueError, match='fits without descriptor diagnostics: old'):
        descriptor_gate({'old': dict(name='old')})
    old = descriptor_gate({'old': dict(name='old')}, override=True)
    assert old['fits_without_diagnostics'] == ['old'] and not old['passed'] and old['override']


def test_gate_treats_a_z_beyond_float32_as_above_the_limit():
    table = {'V': np.array([np.inf, 0.0], dtype=np.float32), descriptors.VEHICLE_ID: np.zeros(2, dtype=np.float32)}
    scaling = scaling_diagnostics(table, {'V': 'test'})
    assert scaling['max_abs_z'] == {'test': None, 'vehicle': 0.0} and scaling['above_limit'] == {'test': {'V': None}}
    json.dumps(scaling, allow_nan=False)
    with pytest.raises(ValueError, match=r'V \(test, \|z\| = inf\)'):
        descriptor_gate({'fit': dict(descriptor_scaling=scaling)})


@pytest.fixture(scope='module', params=['sciplex3', 'op3'])
def manifest_descriptors(request):
    pytest.importorskip('descriptastorus')
    path = {'sciplex3': 'manifests/split_manifest.tsv', 'op3': 'manifests/v4/op3_compounds.tsv'}[request.param]
    table = pd.read_csv(ROOT/path, sep='\t', keep_default_na=False)
    raw, names = descriptors.rdkit2d_normalized([*table.canonical_smiles, VEHICLE_SMILES])
    return (request.param, [*table.compound_id, 'DMSO'], raw, raw[:-1][table.split.eq('train').to_numpy()], names,
            dict(zip(table.compound_id, table.split)))


def test_tolerance_keeps_exactly_the_nonconstant_training_columns(manifest_descriptors):
    dataset, ids, raw, train, names, _ = manifest_descriptors
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


def test_gate_names_exactly_the_v4_op3_compounds_and_passes_the_tolerance(manifest_descriptors):
    """On the committed structures the gate would have refused the v4 OP3 scaling before any fit was sealed."""
    dataset, ids, raw, train, _, splits = manifest_descriptors
    records = {}
    for rule, min_std in [('v3', 0.0), ('tolerance', NOISE_STD_TOLERANCE)]:
        scaling = scaling_diagnostics(dict(zip(ids, TrainingDescriptorScaler(min_std).fit(train).transform(raw))), splits)
        assert scaling['max_abs_z']['train'] < 11 and scaling['max_abs_z']['vehicle'] < 5
        records[rule] = {'fit': dict(descriptor_scaling=scaling)}
    tolerant = descriptor_gate(records['tolerance'])
    assert tolerant['passed'] and max(tolerant['max_abs_z'].values()) < 11  # 9.2 in OP3, 10.9 in sci-Plex3
    # Values lie in [0, 1], so |z| <= 1/std: under the tolerance no compound can reach the limit on these training sets.
    assert np.nanmin(raw) >= 0 and np.nanmax(raw) <= 1
    assert 1/TrainingDescriptorScaler(NOISE_STD_TOLERANCE).fit(train).std.min() < DESCRIPTOR_Z_LIMIT  # 416 and 485
    if dataset == 'sciplex3':
        assert descriptor_gate(records['v3'])['passed']
    else:
        with pytest.raises(ValueError, match='LSM-43293 \\(test'):
            descriptor_gate(records['v3'])
        above = descriptor_gate(records['v3'], override=True)['above_limit']
        assert {s: set(found) for s, found in above.items()} == {'test': set(OP3_AFFECTED[:3]), 'validation': {'LSM-45849'}}
        assert min(z for found in above.values() for z in found.values()) > 1e15
