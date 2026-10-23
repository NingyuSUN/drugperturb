"""Synthetic engineering checks for the v5 chemCPA fits and seals (cell-model env only).

A tiny prepared directory in the v5 layout (variable replicates, pooled vehicle
units, v5 context names) runs through scripts/v5_fit.py and scripts/v5_seal.py.
Fixtures establish software behaviour only; they say nothing about Tahoe-100M.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse

pytest.importorskip('descriptastorus')
torch = pytest.importorskip('torch')

from drugperturb.v3.cells import save_model_archive  # noqa: E402
from drugperturb.v5 import chemcpa, counterfactual  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GENES = 12
SMILES = {'T1': 'CCO', 'T2': 'c1ccccc1O', 'T3': 'CC(=O)Nc1ccc(O)cc1', 'T4': 'CCN', 'V1': 'Cn1cnc2c1c(=O)n(C)c(=O)n2C',
          'V2': 'c1ccncc1', 'X1': 'CC(C)O', 'X2': 'c1ccsc1'}
SPLIT = dict(T1='train', T2='train', T3='train', T4='train', V1='validation', V2='validation', X1='test', X2='test')
LINES, PLATES = ['CVCL_A', 'CVCL_B'], {'p1': 50, 'p2': 5000, 'p3': 5000}


def prepared_dir(root):
    """A tiny prepared directory and manifests in the v5 layout; returns (prepared, manifests)."""
    rng = np.random.default_rng(3)
    prepared, manifests = root/'prepared', root/'manifests'
    prepared.mkdir(parents=True)
    manifests.mkdir()
    rows, blocks, queries = [], [], []
    for plate, dose in PLATES.items():
        for line in LINES:
            unit = f'{plate}:vehicle:{line}'
            blocks.append(rng.poisson(4.0, size=(10, GENES)))
            rows += [dict(cell_id=f'{unit}:{i}', Combo=unit, cell_type=line, replicate=f'{plate}_v', plate_oligo=plate,
                          compound_id='DMSO', role='vehicle', dose_nM=0.0) for i in range(10)]
        for c, split in SPLIT.items():
            if plate == 'p3' and c not in ('T1', 'V1', 'X1'):
                continue
            sample = f'{plate}_{c}'
            for line in LINES:
                context = f'{line}_{dose}nM'
                queries.append(dict(compound_id=c, context_id=context, split=split, replicate=sample, cell_type=line,
                                    dose_nM=float(dose), plate=plate, treatment_Combo=f'{sample}:{line}',
                                    vehicle_Combo=f'{plate}:vehicle:{line}', scaffold_group=f'g:{c}', count_eligible=True))
                if split == 'test':
                    continue
                rate = np.full(GENES, 4.0)
                rate[:4] *= 1+0.3*list(SMILES).index(c)
                blocks.append(rng.poisson(rate, size=(8, GENES)))
                rows += [dict(cell_id=f'{sample}:{line}:{i}', Combo=f'{sample}:{line}', cell_type=line, replicate=sample,
                              plate_oligo=plate, compound_id=c, role=split, dose_nM=float(dose)) for i in range(8)]
    counts = sparse.csr_matrix(np.vstack(blocks).astype(np.int32))
    library = np.asarray(counts.sum(axis=1)).ravel().astype(float)+50.0
    save_model_archive(prepared/'model_cells.npz', pd.DataFrame(rows), counts, library)
    queries = pd.DataFrame(queries)
    queries.to_csv(prepared/'queries.tsv', sep='\t', index=False)
    (queries[['vehicle_Combo', 'context_id']].drop_duplicates().rename(columns={'vehicle_Combo': 'Combo'})
     .to_csv(prepared/'vehicle_contexts.tsv', sep='\t', index=False))
    for split in ['train', 'validation']:
        keys = sorted(set(map(tuple, queries.loc[queries.split.eq(split), ['compound_id', 'context_id']].to_numpy())))
        np.savez_compressed(prepared/f'{split}_truth.npz', keys=np.asarray([f'{c}|{x}' for c, x in keys]),
                            delta=rng.normal(0, 0.3, (len(keys), GENES)))
    pd.DataFrame([dict(context_id=f'{line}_{dose}nM', cell_line_id=line, dose_nM=dose)
                  for line in LINES for dose in (50, 5000)]).to_csv(manifests/'contexts.tsv', sep='\t', index=False)
    pd.DataFrame(dict(compound_id=list(SMILES), status='accepted', split=[SPLIT[c] for c in SMILES],
                      canonical_smiles=list(SMILES.values()))).to_csv(manifests/'compounds.tsv', sep='\t', index=False)
    protocol = yaml.safe_load((REPO/'configs/protocol_v5.yaml').read_text())
    protocol['training']['seeds'] = dict(candidates=[1, 2], A0=[1, 2])
    (root/'protocol.yaml').write_text(yaml.safe_dump(protocol))
    return prepared, manifests


def run(script, *args, check=True):
    env = {**os.environ, 'PYTHONPATH': str(REPO/'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, str(REPO/'scripts'/script), *map(str, args)], cwd=REPO, env=env,
                            capture_output=True, text=True)
    if check:
        assert result.returncode == 0, result.stdout[-3000:]+result.stderr[-3000:]
    return result


def test_weighted_target_pads_and_weights_replicates_equally():
    cells = {w: dict(values=np.ones((3, GENES), np.float32), library=np.ones(3), n_cells=3) for w in ['a', 'b', 'c']}
    pairs = [dict(compound='T1', dose_nM=50.0, cell_type='L', wells=['a']),
             dict(compound='T2', dose_nM=50.0, cell_type='L', wells=['b', 'c'])]
    target = chemcpa.WeightedPseudobulkTarget(pairs=pairs, truth=np.zeros((2, GENES)), vehicle_cells=cells,
                                              pairs_per_update=2, cells_per_well=4, seed=0, device='cpu')
    assert target.replicates == [1, 2] and target.n_replicates == 2
    assert np.allclose(target.weights.numpy(), [[1.0, 0.0], [0.5, 0.5]])
    assert target.pairs[0]['wells'] == ['a', 'a']


def test_counterfactual_deltas_average_a_variable_number_of_replicates():
    cells = {w: dict(values=np.zeros((2, 3)), counts=sparse.csr_matrix(np.ones((2, 3))), library=np.full(2, 10.0),
                     n_cells=2) for w in ['u', 'v']}
    shift = {'u': 0.1, 'v': 0.3}
    predict = lambda well, condition: np.full((2, 3), 0.0 if condition['compound'] == 'DMSO' else well['values'][0, 0])
    for w in cells:
        cells[w]['values'] = np.full((2, 3), shift[w])
    record = lambda w: dict(treated=dict(compound='C', dose_nM=50.0, cell_type='L'),
                            vehicle=dict(compound='DMSO', dose_nM=0.0, cell_type='L'), vehicle_well=w)
    contrast, _ = counterfactual.counterfactual_deltas(predict, {('C', 'x'): [record('u')], ('D', 'x'): [record('u'), record('v')]},
                                                       cells, gene_count=3)
    one = counterfactual.counterfactual_deltas(predict, {('C', 'x'): [record('v')]}, cells, gene_count=3)[0][('C', 'x')]
    assert np.allclose(contrast[('D', 'x')], (contrast[('C', 'x')]+one)/2)


def test_context_ids_and_cell_lines_use_the_v5_names():
    obs = pd.DataFrame(dict(role=['train', 'vehicle'], cell_type=['CVCL_0099', 'CVCL_0099'], dose_nM=[5000.0, 0.0]))
    assert counterfactual.context_ids(obs).tolist() == ['CVCL_0099_5000nM', '']
    assert counterfactual.cell_lines(['CVCL_0099_50nM', 'CVCL_1715_5000nM']) == ['CVCL_0099', 'CVCL_1715']


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp('v5fit')
    prepared, manifests = prepared_dir(root)
    common = ['--prepared', prepared, '--output-root', root/'out', '--device', 'cpu', '--max-epochs', '1',
              '--manifests', manifests, '--protocol', root/'protocol.yaml']
    for arm in ['A0_v3_reference', 'I1_nb_likelihood', 'I2_counterfactual_pseudobulk', 'I3_nb_plus_pseudobulk']:
        for seed in (1, 2):
            run('v5_fit.py', '--arm', arm, '--seed', seed, *common)
    return root, prepared


def test_fits_write_blind_records_with_descriptor_scaling_and_all_queries(fitted):
    root, _ = fitted
    record = json.loads((root/'out/fits/I3_nb_plus_pseudobulk_v5_s1.json').read_text())
    assert record['test_outcomes_loaded'] is False and record['analysis'] == 'v5' and record['engineering_smoke']
    assert record['descriptor_min_std'] == 1e-8 and record['descriptor_scaling']['z_limit'] == 1e3
    assert record['cell_lines'] == LINES and record['n_test_queries'] == 2*2*2
    with np.load(root/'out/fits/I3_nb_plus_pseudobulk_v5_s1.npz') as z:
        assert len(z['test_keys']) == 8 and np.isfinite(z['test_contrast']).all()


def test_seal_refuses_smoke_fits_and_seals_with_a_selection(fitted):
    root, prepared = fitted
    args = ['--fit-root', root/'out', '--prepared', prepared, '--protocol', root/'protocol.yaml']
    refused = run('v5_seal.py', *args, '--output-root', root/'refused', check=False)
    assert refused.returncode != 0 and 'engineering smoke' in refused.stderr
    run('v5_seal.py', *args, '--output-root', root/'sealed', '--allow-smoke')
    selection = json.loads((root/'sealed/selection_v5.json').read_text())
    assert selection['status'] == 'SELECTED_ON_VALIDATION_BEFORE_TEST_SCORING' and selection['descriptor_gate']['passed']
    assert selection['selected'] in selection['primary_arms'] and 1 <= selection['bonferroni_k'] <= 3
    assert len(list((root/'sealed').glob('*_v5_seal.json'))) == 4


def test_diagnostics_report_tabulates_arms_and_comparators_on_validation(fitted):
    root, prepared = fitted
    sealed = root/'sealed'
    if not (sealed/'selection_v5.json').exists():
        run('v5_seal.py', '--fit-root', root/'out', '--prepared', prepared, '--protocol', root/'protocol.yaml',
            '--output-root', sealed, '--allow-smoke')
    from drugperturb.v3 import runner
    from drugperturb.v4.counterfactual import load_truth
    val = load_truth(prepared/'validation_truth.npz')
    keys = sorted(val)
    rng = np.random.default_rng(5)
    arrays = dict(keys=np.asarray(['X1|CVCL_A_50nM']), validation_keys=np.asarray([f'{c}|{x}' for c, x in keys]),
                  **{f'validation_{m}': rng.normal(0, 0.2, (len(keys), GENES)) for m in ['B1c', 'M1c', 'B1a', 'M1a', 'B1p']})
    if not (sealed/'baselines_v5_seal.json').exists():
        runner.write_seal(sealed, 'v5', 'baselines', arrays, {}, {})
    run('v5_diagnostics_report.py', '--sealed-root', sealed, '--prepared', prepared, '--output-root', root/'diag')
    table = pd.read_csv(root/'diag/diagnostics.tsv', sep='\t')
    assert set(table.arm) == {'B1c', 'M1c', 'B1a', 'M1a', 'B1p', 'A0_v3_reference', 'I1_nb_likelihood',
                              'I2_counterfactual_pseudobulk', 'I3_nb_plus_pseudobulk'}
    assert np.allclose(table.loc[table.arm.eq('B1a'), 'r_vs_B1a'], 0)
    assert len(pd.read_csv(root/'diag/diagnostics_per_seed.tsv', sep='\t')) == 4*2*2


def test_seal_refuses_a_fit_above_the_descriptor_limit(fitted):
    root, prepared = fitted
    path = root/'out/fits/I1_nb_likelihood_v5_s2.json'
    record = json.loads(path.read_text())
    record['descriptor_scaling']['max_abs_z']['test'] = 5e15
    record['descriptor_scaling']['above_limit'] = {'test': {'X1': 5e15}}
    gated = root/'gated'
    (gated/'fits').mkdir(parents=True)
    for p in (root/'out/fits').iterdir():
        (gated/'fits'/p.name).write_bytes(p.read_bytes())
    (gated/'fits'/path.name).write_text(json.dumps(record))
    result = run('v5_seal.py', '--fit-root', gated, '--prepared', prepared, '--protocol', root/'protocol.yaml',
                 '--output-root', root/'gated_sealed', '--allow-smoke', check=False)
    assert result.returncode != 0 and 'descriptor gate' in result.stderr and 'X1' in result.stderr
