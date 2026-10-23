"""Synthetic checks of the v5 evaluator on the synthetic pipeline (tests/test_v5_pipeline.py).

Arm seals are fabricated: an oracle arm that predicts the measured test deltas,
an arm equal to B1a, and two zero arms. The synthetic data have only three test
compounds, so the context and study gates are switched on in a copy of the
prepared outputs. Fixtures establish software behaviour only.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip('rdkit')
pytest.importorskip('sklearn')
from test_v5_pipeline import REPO, make_dataset, pipeline  # noqa: E402
from drugperturb.v3 import runner  # noqa: E402
from drugperturb.v5 import prepare  # noqa: E402
from drugperturb.v5.selection import ARMS  # noqa: E402


def measured_test_deltas(root, prepared):
    part = next((root/'extraction').glob('part_*'))
    with np.load(part/'unit_counts.npz') as z:
        counts = sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
        library, n_cells = z['library'], z['n_cells']
    units = pd.read_csv(part/'units.tsv', sep='\t', keep_default_na=False)
    manifests = root/'manifests'
    pairs = prepare.response_pairs(units, pd.read_csv(manifests/'compounds.tsv', sep='\t', keep_default_na=False),
                                   pd.read_csv(manifests/'contexts.tsv', sep='\t'))
    ids, matrix, unit_library, _ = prepare.with_pooled_vehicles(units, counts, library, n_cells)
    qc = pd.read_csv(prepared/'count_qc/compound_qc.tsv', sep='\t')
    keys = sorted(map(tuple, qc.loc[qc.split.eq('test') & qc.eligible, ['compound_id', 'context_id']].to_numpy()))
    frozen = pd.read_csv(prepared/'frozen_genes.tsv', sep='\t').source_index.to_numpy()
    table, delta = prepare.replicate_deltas(pairs, keys, {u: i for i, u in enumerate(ids)}, matrix, unit_library, frozen)
    return dict(zip(keys, prepare.key_means(table, delta, keys)))


def seal_arms(sealed, comparators, truth, primary, level):
    keys = sorted(comparators['B1a'])
    width = len(next(iter(truth.values())))
    predictions = {'A0_v3_reference': {k: np.zeros(width) for k in keys},
                   'I1_nb_likelihood': {k: truth.get(k, np.zeros(width)) for k in keys},
                   'I2_counterfactual_pseudobulk': {k: comparators['B1a'][k] for k in keys},
                   'I3_nb_plus_pseudobulk': {k: np.zeros(width) for k in keys}}
    arms = {}
    for arm in ARMS:
        values = np.vstack([predictions[arm][k] for k in keys])
        final = dict(test_keys=np.asarray([f'{c}|{x}' for c, x in keys]), test_contrast=values, test_anchored=values)
        seal = runner.write_seal(sealed, 'v5', arm, final, dict(analysis='v5'), {})
        arms[arm] = dict(sha256=seal['predictions']['sha256'])
    record = dict(status='SELECTED_ON_VALIDATION_BEFORE_TEST_SCORING', arms=arms, primary_arms=primary,
                  interval_level=level, near_tie=len(primary) > 1, any_passing=True, selected=primary[0])
    (sealed/'selection_v5.json').write_text(json.dumps(record))


def evaluate(root, prepared, sealed, out, check=True):
    env = {**os.environ, 'PYTHONPATH': str(REPO/'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, str(REPO/'scripts/v5_evaluate.py'), '--extraction-root', root/'extraction',
                             '--prepared', prepared, '--sealed-root', sealed, root/'sealed', '--output-root', out,
                             '--manifests', root/'manifests', '--protocol', root/'protocol.yaml'],
                            cwd=REPO, env=env, capture_output=True, text=True)
    if check:
        assert result.returncode == 0, result.stdout[-3000:]+result.stderr[-3000:]
    return result


@pytest.fixture(scope='module')
def evaluated(tmp_path_factory):
    root = tmp_path_factory.mktemp('v5eval')/'d'
    make_dataset(root)
    prepared, baseline_root = pipeline(root, [(0, 3)])
    gated = root/'prepared_gated'
    shutil.copytree(prepared, gated)
    summary = pd.read_csv(gated/'count_qc/context_summary.tsv', sep='\t')
    summary['test_count_gate_met'] = True
    summary.to_csv(gated/'count_qc/context_summary.tsv', sep='\t', index=False)
    receipt = json.loads((gated/'prepare_receipt.json').read_text())
    receipt['study_gate_G4']['met'] = True
    (gated/'prepare_receipt.json').write_text(json.dumps(receipt))
    with np.load(baseline_root/'baselines_v5_predictions.npz') as z:
        comparators = {m: {tuple(k.split('|')): v for k, v in zip(z['keys'], z[m])} for m in ['B1a']}
    truth = measured_test_deltas(root, gated)
    sealed = root/'arms'
    sealed.mkdir()
    seal_arms(sealed, comparators, truth, ['I1_nb_likelihood', 'I2_counterfactual_pseudobulk'], 0.975)
    evaluate(root, gated, sealed, root/'evaluation')
    return root, gated, sealed, json.loads((root/'evaluation/summary.json').read_text())


def test_the_oracle_arm_meets_p1_and_the_b1a_copy_does_not(evaluated):
    _, _, _, summary = evaluated
    p1 = summary['endpoints']['P1_beyond_vehicle_anchoring']
    oracle, copy = p1['per_arm']['I1_nb_likelihood'], p1['per_arm']['I2_counterfactual_pseudobulk']
    assert oracle['supported'] and oracle['estimates']['B1a']['level'] == 0.975
    assert copy['estimates']['B1a']['point_estimate'] == 0 and not copy['supported']
    assert p1['supported'] and summary['test_truth_first_computed_here']


def test_the_oracle_has_zero_error_and_secondaries_are_reported(evaluated):
    root, _, _, summary = evaluated
    table = pd.read_csv(root/'evaluation/summary.tsv', sep='\t')
    assert np.allclose(table.loc[table.method.eq('I1_nb_likelihood'), 'mean_compound_rmse'], 0)
    assert summary['endpoints']['S3_structure_beyond_anchoring'] is not None
    secondary = pd.read_csv(root/'evaluation/secondary.tsv', sep='\t')
    assert {'pooled', 'dose_5000nM', 'line_L1'} <= set(secondary.subset)
    assert len(pd.read_csv(root/'evaluation/leave_one_plate_out.tsv', sep='\t')) == 2*2*4


def test_the_evaluator_refuses_a_tampered_seal_or_a_mismatched_selection(evaluated, tmp_path):
    root, gated, sealed, _ = evaluated
    tampered = tmp_path/'tampered'
    shutil.copytree(sealed, tampered)
    archive = tampered/'I1_nb_likelihood_v5_predictions.npz'
    archive.write_bytes(archive.read_bytes()+b'x')
    assert evaluate(root, gated, tampered, tmp_path/'e1', check=False).returncode != 0
    mismatched = tmp_path/'mismatched'
    shutil.copytree(sealed, mismatched)
    record = json.loads((mismatched/'selection_v5.json').read_text())
    record['arms']['I1_nb_likelihood']['sha256'] = '0'*64
    (mismatched/'selection_v5.json').write_text(json.dumps(record))
    result = evaluate(root, gated, mismatched, tmp_path/'e2', check=False)
    assert result.returncode != 0 and 'selection record' in result.stderr
