"""Synthetic engineering checks for the v6 fits and seal (cell-model env only).

A tiny prepared directory in the v6 layout - half-split validation truth, DE
sets, responder flags, leave-one-out residual targets and sealed B1p - runs
through scripts/v6_fit.py and scripts/v6_seal.py. The endpoint itself is checked
in tests/test_v6_endpoint.py and tests/test_v6_evaluate.py, which need neither
torch nor descriptastorus. Fixtures establish software behaviour only; they say
nothing about Tahoe-100M.
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
pytest.importorskip('torch')

from drugperturb.v3 import runner  # noqa: E402
from drugperturb.v3.cells import save_model_archive  # noqa: E402
from drugperturb.v6 import endpoint, selection  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GENES, DE_K = 12, 4
SMILES = {'T1': 'CCO', 'T2': 'c1ccccc1O', 'T3': 'CC(=O)Nc1ccc(O)cc1', 'T4': 'CCN', 'T5': 'CCCO', 'T6': 'CCCCO',
          'V1': 'Cn1cnc2c1c(=O)n(C)c(=O)n2C', 'V2': 'c1ccncc1', 'X1': 'CC(C)O', 'X2': 'c1ccsc1'}
SPLIT = {**{f'T{i}': 'train' for i in range(1, 7)}, 'V1': 'validation', 'V2': 'validation',
         'X1': 'test', 'X2': 'test'}
LINES, PLATES = ['CVCL_A', 'CVCL_B'], {'p1': 50, 'p2': 5000}
SEEDS = (1, 2)


def prepared_dir(root):
    """A tiny prepared directory, manifests and sealed B1p in the v6 layout."""
    rng = np.random.default_rng(3)
    prepared, manifests, sealed = root/'prepared', root/'manifests', root/'sealed'
    (prepared/'endpoint').mkdir(parents=True)
    manifests.mkdir()
    rows, blocks, queries = [], [], []
    for plate, dose in PLATES.items():
        for line in LINES:
            unit = f'{plate}:vehicle:{line}'
            blocks.append(rng.poisson(4.0, size=(10, GENES)))
            rows += [dict(cell_id=f'{unit}:{i}', Combo=unit, cell_type=line, replicate=f'{plate}_v',
                          plate_oligo=plate, compound_id='DMSO', role='vehicle', dose_nM=0.0) for i in range(10)]
        for c, split in SPLIT.items():
            sample = f'{plate}_{c}'
            for line in LINES:
                queries.append(dict(compound_id=c, context_id=f'{line}_{dose}nM', split=split, replicate=sample,
                                    cell_type=line, dose_nM=float(dose), plate=plate,
                                    treatment_Combo=f'{sample}:{line}', vehicle_Combo=f'{plate}:vehicle:{line}',
                                    scaffold_group=f'g:{c}', count_eligible=True, responder=c != 'X2'))
                if split == 'test':
                    continue
                rate = np.full(GENES, 4.0)
                rate[:4] *= 1+0.3*list(SMILES).index(c)
                blocks.append(rng.poisson(rate, size=(8, GENES)))
                rows += [dict(cell_id=f'{sample}:{line}:{i}', Combo=f'{sample}:{line}', cell_type=line,
                              replicate=sample, plate_oligo=plate, compound_id=c, role=split,
                              dose_nM=float(dose)) for i in range(8)]
    counts = sparse.csr_matrix(np.vstack(blocks).astype(np.int32))
    library = np.asarray(counts.sum(axis=1)).ravel().astype(float)+50.0
    save_model_archive(prepared/'model_cells.npz', pd.DataFrame(rows), counts, library)
    queries = pd.DataFrame(queries)
    queries.to_csv(prepared/'queries.tsv', sep='\t', index=False)
    (queries[['vehicle_Combo', 'context_id']].drop_duplicates().rename(columns={'vehicle_Combo': 'Combo'})
     .to_csv(prepared/'vehicle_contexts.tsv', sep='\t', index=False))

    keys_of = {s: sorted(set(map(tuple, queries.loc[queries.split.eq(s), ['compound_id', 'context_id']].to_numpy())))
               for s in ['train', 'validation', 'test']}
    truth = {}
    for split in ['train', 'validation']:
        keys = keys_of[split]
        delta = rng.normal(0, 0.3, (len(keys), GENES))
        truth[split] = dict(zip(keys, delta))
        payload = dict(keys=np.asarray([f'{c}|{x}' for c, x in keys]), delta=delta)
        if split == 'validation':
            payload['delta_evaluation'] = delta+rng.normal(0, 0.05, delta.shape)
            payload['delta_selection'] = delta+rng.normal(0, 0.05, delta.shape)
        np.savez_compressed(prepared/f'{split}_truth.npz', **payload)
    # DE sets and responder flags for every key, as preparation writes them.
    all_keys = sorted(set().union(*keys_of.values()))
    selection_delta = rng.normal(0, 0.4, (len(all_keys), GENES))
    np.savez_compressed(prepared/'endpoint'/'de_sets.npz',
                        keys=np.asarray([f'{c}|{x}' for c, x in all_keys]),
                        genes=endpoint.de_gene_sets(selection_delta, k=DE_K).astype(np.int32),
                        selection_delta=selection_delta,
                        selection_error=np.full_like(selection_delta, 0.1),
                        statistic=np.linalg.norm(selection_delta, axis=1),
                        responder=np.asarray([c != 'X2' for c, _ in all_keys]), threshold=0.0, k=DE_K)
    # Leave-one-out residual targets: every training key but the first, to exercise the fallback count.
    residual_keys = keys_of['train'][1:]
    np.savez_compressed(prepared/'train_residual_targets.npz',
                        keys=np.asarray([f'{c}|{x}' for c, x in residual_keys]),
                        residual=np.vstack([truth['train'][k]*0.5 for k in residual_keys]),
                        fallback_keys=np.asarray([f'{keys_of["train"][0][0]}|{keys_of["train"][0][1]}']))
    # Sealed B1p for the validation and test keys, which the residual arm adds back.
    arrays = {}
    for split, prefix in [('test', ''), ('validation', 'validation_')]:
        keys = keys_of[split]
        arrays[f'{prefix}keys'] = np.asarray([f'{c}|{x}' for c, x in keys])
        arrays[f'{prefix}B1p'] = rng.normal(0, 0.2, (len(keys), GENES))
    runner.write_seal(sealed, 'v6', 'baselines', arrays, dict(selection={}, residual={}),
                      {'queries': prepared/'queries.tsv'})

    pd.DataFrame([dict(context_id=f'{line}_{dose}nM', cell_line_id=line, dose_nM=dose)
                  for line in LINES for dose in PLATES.values()]).to_csv(manifests/'contexts.tsv', sep='\t',
                                                                          index=False)
    pd.DataFrame(dict(compound_id=list(SMILES), status='accepted', split=[SPLIT[c] for c in SMILES],
                      canonical_smiles=list(SMILES.values()))).to_csv(manifests/'compounds.tsv', sep='\t',
                                                                       index=False)
    protocol = yaml.safe_load((REPO/'configs/protocol_v6.yaml').read_text())
    protocol['training']['seeds'] = dict(candidates=list(SEEDS))
    protocol['response_endpoint']['de_gene_set']['k'] = DE_K
    (root/'protocol.yaml').write_text(yaml.safe_dump(protocol))
    return prepared, manifests, sealed


def run(script, *args, check=True):
    env = {**os.environ, 'PYTHONPATH': str(REPO/'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, str(REPO/'scripts'/script), *map(str, args)], cwd=REPO, env=env,
                            capture_output=True, text=True)
    if check:
        assert result.returncode == 0, result.stdout[-3000:]+result.stderr[-3000:]
    return result


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp('v6fit')
    prepared, manifests, sealed = prepared_dir(root)
    common = ['--prepared', prepared, '--baselines', sealed/'baselines_v6_predictions.npz',
              '--output-root', root/'out', '--device', 'cpu', '--max-epochs', '1',
              '--manifests', manifests, '--protocol', root/'protocol.yaml']
    for arm in selection.ARMS:
        for seed in SEEDS:
            run('v6_fit.py', '--arm', arm, '--seed', seed, *common)
    return root, prepared, sealed


def records(root):
    return {p.stem: json.loads(p.read_text()) for p in sorted((root/'out'/'fits').glob('*_v6_s*.json'))}


def test_both_arms_fit_and_only_the_residual_arm_uses_the_residual_target(fitted):
    root, _, _ = fitted
    found = records(root)
    assert len(found) == len(selection.ARMS)*len(SEEDS)
    for name, record in found.items():
        assert record['objective'] == 'I2_counterfactual_pseudobulk', 'both arms share the objective'
        assert record['residual_target'] is (record['arm'] == 'M2_pseudobulk_residual')
        assert record['test_outcomes_loaded'] is False
        assert record['evaluation_half_test_delta_loaded'] is False
        assert record['n_test_queries'] > 0 and record['n_validation_queries'] > 0
        assert np.isfinite(record['validation_rmse'])
        assert 0.0 <= record['validation_direction_accuracy'] <= 1.0


def test_the_residual_arm_drops_training_keys_without_a_leave_one_out_reference(fitted):
    root, _, _ = fitted
    found = records(root)
    raw = found['M1_pseudobulk_raw_v6_s1']
    residual = found['M2_pseudobulk_residual_v6_s1']
    assert raw['n_training_keys_without_a_target'] == 0
    assert residual['n_training_keys_without_a_target'] == 1
    assert residual['n_target_keys'] == raw['n_target_keys']-1


def test_the_fit_records_the_endpoint_quantities_selection_needs(fitted):
    root, _, _ = fitted
    e = records(root)['M1_pseudobulk_raw_v6_s1']['endpoint']
    for field in ['validation_direction_accuracy', 'validation_direction_accuracy_all_keys',
                  'b1p_direction_accuracy', 'split_noise_floor', 'residual_r', 'residual_share']:
        assert field in e, field
    assert 0 < e['n_validation_responders'] <= e['n_validation_keys']
    assert 0.0 <= e['split_noise_floor'] <= 1.0


def test_the_residual_arm_predicts_in_delta_space_by_adding_b1p(fitted):
    """The sealed test prediction must be the raw residual plus B1p, to floating-point tolerance."""
    root, _, sealed = fitted
    with np.load(root/'out'/'fits'/'M2_pseudobulk_residual_v6_s1.npz', allow_pickle=False) as z:
        keys = np.asarray(z['test_keys'], str)
        contrast, raw = z['test_contrast'], z['test_raw']
    with np.load(sealed/'baselines_v6_predictions.npz', allow_pickle=False) as z:
        b1p = dict(zip(np.asarray(z['keys'], str), z['B1p']))
    assert np.allclose(contrast, raw+np.vstack([b1p[k] for k in keys]), atol=1e-8)
    with np.load(root/'out'/'fits'/'M1_pseudobulk_raw_v6_s1.npz', allow_pickle=False) as z:
        assert np.allclose(z['test_contrast'], z['test_raw']), 'the raw arm adds no offset'


def test_seal_averages_seeds_selects_on_direction_and_refuses_smoke_by_default(fitted):
    root, prepared, _ = fitted
    refused = run('v6_seal.py', '--fit-root', root/'out', '--prepared', prepared,
                  '--output-root', root/'sealed_refused', '--protocol', root/'protocol.yaml', check=False)
    assert refused.returncode != 0 and 'engineering smoke fit' in refused.stderr
    run('v6_seal.py', '--fit-root', root/'out', '--prepared', prepared, '--output-root', root/'sealed_arms',
        '--allow-smoke', '--override-descriptor-gate', '--protocol', root/'protocol.yaml')
    record = json.loads((root/'sealed_arms'/'selection_v6.json').read_text())
    assert record['higher_is_better'] is True and record['selected'] in selection.ARMS
    assert record['test_outcomes_loaded'] is False and record['evaluation_half_test_delta_loaded'] is False
    means = record['mean_seed_validation_direction_accuracy']
    assert record['selected'] == max(means, key=means.get)
    for arm in selection.ARMS:
        seal = json.loads((root/'sealed_arms'/f'{arm}_v6_seal.json').read_text())
        assert seal['seeds'] == list(SEEDS) and len(seal['seed_direction_accuracy']) == len(SEEDS)
        assert seal['residual_target'] is (arm == 'M2_pseudobulk_residual')
        with np.load(root/'sealed_arms'/seal['predictions']['path'], allow_pickle=False) as z:
            assert np.allclose(z['test_contrast'],
                               np.mean([z[f'test_contrast_s{s}'] for s in SEEDS], axis=0))
