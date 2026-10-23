"""End-to-end software check of the OP3 path on a synthetic h5ad with the OP3 layout.

Design -> extraction -> preparation -> baselines run in any environment; fits,
sealing and evaluation run where descriptastorus and torch are installed. The
scripts run unchanged from a temporary copy of the repository whose protocol
lowers the test-set minimums and gene count to fit the fixture. Nothing here
touches real OP3 data.
"""
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
SMILES = ['Cc1ccccc1', 'Cc1ccncc1', 'NC1CCCCC1', 'c1ccc2ccccc2c1', 'CC1CCNCC1', 'Cc1ccoc1', 'Cc1ccsc1', 'CC1CCOC1',
          'Cc1cnccn1', 'CC1CC1', 'Cc1ccc2[nH]ccc2c1', 'CN1CCCC1', 'Oc1ccc(cc1)C1CC1', 'CC1=CCCCC1']
TYPES = {'T cells': 60, 'NK cells': 36, 'B cells': 36, 'Myeloid cells': 36}
GENES = 90


def write_h5ad(path, obs, counts, genes):
    text = h5py.string_dtype()
    with h5py.File(path, 'w') as h5:
        group = h5.create_group('obs')
        group.attrs['_index'] = 'obs_id'
        group.attrs['column-order'] = np.array(list(obs.columns), dtype=object)
        group.create_dataset('obs_id', data=np.array(obs.index, dtype=object), dtype=text)
        for name in obs.columns:
            values = obs[name]
            if values.dtype == object:
                categories = sorted(set(values))
                node = group.create_group(name)
                node.create_dataset('categories', data=np.array(categories, dtype=object), dtype=text)
                node.create_dataset('codes', data=pd.Categorical(values, categories).codes.astype(np.int16))
            else:
                group.create_dataset(name, data=values.to_numpy())
        normalized = counts.multiply(1e4/np.asarray(counts.sum(axis=1))).tocsr()
        normalized.data = np.log1p(normalized.data)
        for key, matrix in [('X', normalized), ('raw/X', counts)]:
            node = h5.create_group(key)
            node.attrs['encoding-type'] = 'csr_matrix'
            node.attrs['shape'] = np.asarray(matrix.shape)
            for part in ['data', 'indices', 'indptr']:
                node.create_dataset(part, data=getattr(matrix, part).astype(np.float64 if part == 'data' else np.int64))
        var = h5.create_group('raw/var')
        var.attrs['_index'] = '_index'
        var.create_dataset('_index', data=np.array(genes, dtype=object), dtype=text)


def synthetic_op3(path):
    rng = np.random.default_rng(7)
    effect = {i: rng.normal(0, 0.6, GENES)*(rng.random(GENES) < 0.3) for i in range(len(SMILES))}
    base = rng.gamma(2.0, 0.6, GENES)
    rows, blocks = [], []
    for p, donor in enumerate(['Donor 1', 'Donor 2', 'Donor 3']):
        plate = f'PL-{p}'
        for r, row in enumerate('AB'):
            for col in range(1, 11):
                if col == 1:
                    lincs, name, smiles, dose, control, k = 'LSM-B', 'Belinostat', 'O', 0.1, True, None
                elif col == 2:
                    lincs, name, smiles, dose, control, k = 'LSM-D', 'Dabrafenib', 'N', 1.0, True, None
                elif col == 3:
                    lincs, name, smiles, dose, control, k = 'LSM-V', 'Dimethyl Sulfoxide', 'CS(C)=O', 14.1, True, None
                else:
                    k = r*7+col-4
                    lincs, name, smiles, dose, control = f'LSM-{100+k}', f'cmp{k}', SMILES[k], 1.0, False
                for t, (cell_type, n) in enumerate(TYPES.items()):
                    n = n if k is None or k % 5 else n//3  # a few compound wells lose cells
                    rate = base*np.exp(0 if k is None else effect[k])*(1+0.3*t)
                    blocks.append(rng.poisson(rate, size=(n, GENES)))
                    for i in range(n):
                        rows.append(dict(plate_name=plate, well=f'{row}{col}', row=row, col=col, donor_id=donor,
                                         library_id=f'{plate}{row}', sm_name=name, sm_lincs_id=lincs, SMILES=smiles,
                                         dose_uM=dose, timepoint_hr=24.0, control=control, cell_type=cell_type,
                                         cell_type_orig=cell_type))
    obs = pd.DataFrame(rows)
    obs.index = [f'cell{i}' for i in range(len(obs))]
    key = obs.plate_name+'_'+obs.well+'_'+obs.cell_type
    obs['cell_count_by_well_celltype'] = key.map(key.value_counts()).astype(np.int64)
    counts = sparse.csr_matrix(np.vstack(blocks).astype(np.float64))
    write_h5ad(path, obs, counts, [f'G{i}' for i in range(GENES)])


def run(repo, script, *args, python=sys.executable):
    env = {'PYTHONPATH': str(REPO/'src'), 'PYTHONDONTWRITEBYTECODE': '1', 'PATH': '/usr/bin:/bin'}
    result = subprocess.run([python, str(repo/'scripts'/script), *map(str, args)], cwd=repo, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:]+result.stderr[-3000:]
    return result.stdout


@pytest.fixture(scope='module')
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('op3')
    repo = tmp/'repo'
    shutil.copytree(REPO/'scripts', repo/'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    (repo/'configs').mkdir(parents=True)
    shutil.copy(REPO/'configs/protocol_v3.yaml', repo/'configs/protocol_v3.yaml')
    protocol = yaml.safe_load((REPO/'configs/protocol_v4.yaml').read_text())
    protocol['datasets']['op3']['split'].update(min_test_compounds=2, min_test_groups=2)
    protocol['datasets']['op3']['gene_space']['n_genes'] = 40
    (repo/'configs/protocol_v4.yaml').write_text(yaml.safe_dump(protocol, sort_keys=False))
    h5ad = tmp/'op3.h5ad'
    synthetic_op3(h5ad)
    run(repo, 'v4_op3_design.py', '--h5ad', h5ad)
    run(repo, 'v4_op3_design.py', '--h5ad', h5ad, '--check')
    run(repo, 'v4_op3_extract.py', '--h5ad', h5ad, '--output-root', tmp/'ext', '--rows-per-block', 1000)
    run(repo, 'v4_op3_prepare.py', '--extraction-root', tmp/'ext', '--output-root', tmp/'prep')
    run(repo, 'v4_context_baselines.py', '--prepared', tmp/'prep', '--output-root', tmp/'sealed_base')
    return tmp, repo


def test_extraction_and_preparation_are_consistent_and_test_free(pipeline):
    tmp, repo = pipeline
    receipt = json.loads((tmp/'ext/extract_receipt.json').read_text())
    assert receipt['checks']['design_equals_committed_manifests'] and receipt['test_outcomes_computed'] is False
    assert receipt['checks']['cell_frozen_counts_sum_to_unit_pseudobulk']
    assert receipt['checks']['unit_cell_counts_equal_authors']['cell_type']
    assert set(receipt['dropped_cells']) == {'test', 'control'}
    compounds = pd.read_csv(repo/'manifests/v4/op3_compounds.tsv', sep='\t')
    cells = pd.read_csv(tmp/'ext/cells/cells.tsv.gz', sep='\t')
    test_ids = set(compounds.loc[compounds.split.eq('test'), 'compound_id'])
    assert test_ids and not set(cells.compound_id) & test_ids
    prepared = json.loads((tmp/'prep/prepare_receipt.json').read_text())
    assert prepared['test_outcomes_computed'] is False and prepared['training_truth_pairs'] > 0
    with np.load(tmp/'prep/validation_truth.npz') as z:
        assert not {k.split('|')[0] for k in z['keys']} & test_ids
    frozen = pd.read_csv(tmp/'ext/frozen_genes.tsv', sep='\t')
    assert len(frozen) == 40 and frozen.vehicle_detection.min() >= 0.05
    queries = pd.read_csv(tmp/'prep/queries.tsv', sep='\t')
    assert set(queries.split) == {'train', 'validation', 'test'} and queries.groupby(
        ['compound_id', 'context_id']).replicate.nunique().eq(3).all()
    seal = json.loads((tmp/'sealed_base/context_baselines_op3_seal.json').read_text())
    assert seal['status'] == 'SEALED_BEFORE_TEST_SCORING'


def test_fits_seal_select_and_evaluate(pipeline):
    pytest.importorskip('descriptastorus')
    pytest.importorskip('torch')
    tmp, repo = pipeline
    protocol = yaml.safe_load((repo/'configs/protocol_v4.yaml').read_text())
    for arm in ['A0_v3_reference', 'I1_nb_likelihood', 'I2_counterfactual_pseudobulk', 'I3_nb_plus_pseudobulk']:
        for seed in protocol['training']['seeds']['op3']:
            run(repo, 'v4_fit.py', '--setting', 'op3', '--arm', arm, '--seed', seed, '--prepared', tmp/'prep',
                '--output-root', tmp/'gpu', '--device', 'cpu', '--max-epochs', 2)
    record = json.loads((tmp/'gpu/fits/I3_nb_plus_pseudobulk_op3_s42.json').read_text())
    assert record['test_outcomes_loaded'] is False and record['engineering_smoke']
    assert {'train', 'validation'} <= set(record['diagnostics']) and record['latent']['validation']['basal_norm'] > 0
    with pytest.raises(AssertionError):
        run(repo, 'v4_seal.py', '--setting', 'op3', '--fit-root', tmp/'gpu/fits', '--prepared', tmp/'prep',
            '--output-root', tmp/'gpu/refused')
    # Under the v4 descriptor rule the fixture's training-constant columns give some validation/test compounds
    # and DMSO a |z| of order 1e16 or more, like the v4 OP3 bug: the seal refuses before writing anything.
    scaling = record['descriptor_scaling']
    flagged = {c for found in scaling['above_limit'].values() for c in found}
    assert flagged and {'validation', 'test'} & set(scaling['above_limit']) and scaling['max_abs_z']['train'] < 10
    with pytest.raises(AssertionError, match='descriptor gate') as refused:
        run(repo, 'v4_seal.py', '--setting', 'op3', '--fit-root', tmp/'gpu/fits', '--prepared', tmp/'prep',
            '--output-root', tmp/'gpu/gated', '--allow-smoke')
    assert all(c in str(refused.value) for c in flagged) and not (tmp/'gpu/gated').exists()
    run(repo, 'v4_seal.py', '--setting', 'op3', '--fit-root', tmp/'gpu/fits', '--prepared', tmp/'prep',
        '--output-root', tmp/'gpu/sealed', '--allow-smoke', '--override-descriptor-gate')
    selection = json.loads((tmp/'gpu/sealed/selection_op3.json').read_text())
    assert selection['selected'] in protocol['primary_arm_selection']['candidates']
    for arm in ['A0_v3_reference', 'I1_nb_likelihood', 'I2_counterfactual_pseudobulk', 'I3_nb_plus_pseudobulk']:
        gate = json.loads((tmp/f'gpu/sealed/{arm}_op3_seal.json').read_text())['descriptor_gate']
        assert gate == selection['descriptor_gate'] and gate['override'] and not gate['passed']
        assert {c for found in gate['above_limit'].values() for c in found} == flagged
    run(repo, 'v4_anchored_baselines.py', '--setting', 'op3', '--prepared', tmp/'prep', '--baselines-root',
        tmp/'sealed_base', '--output-root', tmp/'sealed_anchored')
    run(repo, 'v4_diagnostics_report.py', '--sealed-root', tmp/'gpu/sealed', '--prepared', f'op3={tmp/"prep"}',
        '--output-root', tmp/'diag')
    diag = pd.read_csv(tmp/'diag/diagnostics.tsv', sep='\t')
    assert {'B1a', 'M1a'} <= set(diag.arm) and diag.loc[diag.arm.eq('B1a'), 'r_vs_B1a'].abs().max() < 1e-9
    run(repo, 'v4_evaluate_op3.py', '--extraction-root', tmp/'ext', '--prepared', tmp/'prep',
        '--sealed-root', tmp/'sealed_base', '--sealed-root', tmp/'gpu/sealed', '--sealed-root', tmp/'sealed_anchored',
        '--output-root', tmp/'eval')
    summary = json.loads((tmp/'eval/summary.json').read_text())
    assert summary['endpoints']['selected_arm'] == selection['selected'] and summary['test_truth_first_computed_here']
    assert set(summary['endpoints']['E3_beyond_vehicle_anchoring']['estimates']) == {'B1a', 'M1a'}
    per = pd.read_csv(tmp/'eval/per_compound.tsv', sep='\t')
    zero = per[per.method.eq('B0')]
    assert len(zero) and np.isfinite(per.rmse).all()


def test_corrected_descriptor_path_is_opt_in_labelled_and_never_mixed(pipeline):
    """--descriptor-tolerance (configs/v4_corrected_posthoc.yaml) reaches the scaler and labels every output."""
    pytest.importorskip('descriptastorus')
    pytest.importorskip('torch')
    from drugperturb.v3.descriptors import DESCRIPTOR_Z_LIMIT, NOISE_STD_TOLERANCE, compound_table
    tmp, repo = pipeline
    protocol = yaml.safe_load((repo/'configs/protocol_v4.yaml').read_text())
    labels = dict(analysis='v4_corrected_posthoc', descriptor_min_std=NOISE_STD_TOLERANCE)
    for arm in ['A0_v3_reference', 'I1_nb_likelihood', 'I2_counterfactual_pseudobulk', 'I3_nb_plus_pseudobulk']:
        for seed in protocol['training']['seeds']['op3']:
            run(repo, 'v4_fit.py', '--setting', 'op3', '--arm', arm, '--seed', seed, '--prepared', tmp/'prep',
                '--output-root', tmp/'corr', '--device', 'cpu', '--max-epochs', 1, '--descriptor-tolerance')
    run(repo, 'v4_fit.py', '--setting', 'op3', '--arm', 'A0_v3_reference', '--seed', 42, '--prepared', tmp/'prep',
        '--output-root', tmp/'default', '--device', 'cpu', '--max-epochs', 1)
    compounds = pd.read_csv(repo/'manifests/v4/op3_compounds.tsv', sep='\t', keep_default_na=False)
    args = (compounds.compound_id.tolist(), compounds.canonical_smiles.tolist(),
            compounds.loc[compounds.split.eq('train'), 'compound_id'].tolist())
    corrected = json.loads((tmp/'corr/fits/A0_v3_reference_op3_s42.json').read_text())
    default = json.loads((tmp/'default/fits/A0_v3_reference_op3_s42.json').read_text())
    assert {k: corrected[k] for k in labels} == labels and not set(labels) & set(default)
    assert corrected['descriptor_columns'] == len(compound_table(*args, min_std=NOISE_STD_TOLERANCE)[1])
    assert default['descriptor_columns'] == len(compound_table(*args)[1])
    assert corrected['descriptor_columns'] < default['descriptor_columns']  # the fixture has noise columns too

    seal = ['v4_seal.py', '--setting', 'op3', '--prepared', tmp/'prep', '--allow-smoke']
    with pytest.raises(AssertionError, match='descriptor rule'):
        run(repo, *seal, '--fit-root', tmp/'corr/fits', '--output-root', tmp/'refused_default')
    shutil.copytree(tmp/'corr/fits', tmp/'mixed')
    shutil.copy(tmp/'default/fits/A0_v3_reference_op3_s42.json', tmp/'mixed/A0_v3_reference_op3_s42.json')
    with pytest.raises(AssertionError, match='descriptor rule'):
        run(repo, *seal, '--fit-root', tmp/'mixed', '--output-root', tmp/'refused_mixed', '--descriptor-tolerance')
    run(repo, *seal, '--fit-root', tmp/'corr/fits', '--output-root', tmp/'corr/sealed', '--descriptor-tolerance')
    selection = json.loads((tmp/'corr/sealed/selection_op3.json').read_text())
    arm_seal = json.loads((tmp/'corr/sealed/I1_nb_likelihood_op3_seal.json').read_text())
    assert {k: selection[k] for k in labels} == labels and {k: arm_seal[k] for k in labels} == labels
    assert selection['status'] == 'SELECTED_ON_VALIDATION_BEFORE_TEST_SCORING' and selection['test_outcomes_loaded'] is False
    # The tolerance removes the noise columns, so the same compounds pass the descriptor gate without an override.
    gate = arm_seal['descriptor_gate']
    assert gate == selection['descriptor_gate'] and gate['passed'] and not gate['override'] and gate['above_limit'] == {}
    assert max(gate['max_abs_z'].values()) <= DESCRIPTOR_Z_LIMIT
    assert corrected['descriptor_scaling']['above_limit'] == {} and default['descriptor_scaling']['above_limit']

    run(repo, 'v4_anchored_baselines.py', '--setting', 'op3', '--prepared', tmp/'prep', '--baselines-root',
        tmp/'sealed_base', '--output-root', tmp/'corr/anchored')
    evaluate = ['v4_evaluate_op3.py', '--extraction-root', tmp/'ext', '--prepared', tmp/'prep', '--sealed-root',
                tmp/'sealed_base', '--sealed-root', tmp/'corr/sealed', '--sealed-root', tmp/'corr/anchored']
    with pytest.raises(AssertionError, match='descriptor rule'):
        run(repo, *evaluate, '--output-root', tmp/'corr/eval_refused')
    run(repo, *evaluate, '--output-root', tmp/'corr/eval', '--descriptor-tolerance')
    summary = json.loads((tmp/'corr/eval/summary.json').read_text())
    assert {k: summary[k] for k in labels} == labels and summary['test_truth_first_computed_here'] is False
    assert summary['endpoints']['selected_arm'] == selection['selected']
