import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def api():
    from drugperturb import biology_v2
    return biology_v2


def fixture():
    mapping = pd.DataFrame({'treatment': ['b', 'b', 'a', 'a', 'test', 'test'],
        'replicate': ['rep1', 'rep2'] * 3,
        'Combo': ['b1', 'b2', 'a1', 'a2', 't1', 't2'],
        'control_Combo': ['c1', 'c2'] * 3, 'eligible': [True] * 6})
    counts = pd.DataFrame({'b1': [10, 90, 0], 'b2': [10, 90, 0],
        'a1': [20, 80, 0], 'a2': [20, 80, 0], 'c1': [0, 100, 0],
        'c2': [0, 100, 0], 't1': [-9, 0, 0], 't2': ['bad', 0, 0]},
        index=pd.Index(['g1', 'g2', 'zero'], name='gene_id'))
    return counts, mapping


def test_train_projection_ignores_nontrain_invalid_values_and_ids(tmp_path):
    b = api(); counts, mapping = fixture(); path = tmp_path/'counts.tsv'
    counts.to_csv(path, sep='\t')
    selected = b.project_mapping(mapping, ['a', 'b'])
    train = b.read_projected_counts(path, selected)
    assert list(train.columns) == ['a1', 'a2', 'b1', 'b2', 'c1', 'c2']
    before = b.training_prevalence(train, selected)[0]
    counts = counts.rename(columns={'t1': 'arbitrary_target'}); counts['t2'] = 999999
    mapping.loc[mapping.treatment.eq('test'), 'Combo'] = ['missing_x', 'missing_y']
    counts.to_csv(path, sep='\t')
    selected = b.project_mapping(mapping, ['b', 'a'])
    after = b.training_prevalence(b.read_projected_counts(path, selected), selected)[0]
    pd.testing.assert_series_equal(before, after)
    assert before.to_dict() == {'g1': True, 'g2': True, 'zero': False}


def test_unique_control_denominator_and_ceil_prevalence():
    b = api(); _, mapping = fixture(); mapping = b.project_mapping(mapping, ['a', 'b'])
    # Ten unique controls, only one expresses g1; repeated c0 must not upweight it.
    rows = []; data = {}
    for i in range(10):
        for rep in ['rep1', 'rep2']:
            k = f'd{i}_{rep}'; control = 'c0' if i < 6 else f'c{i}'
            rows.append(dict(treatment=f'x{i}', replicate=rep, Combo=k,
                             control_Combo=control, eligible=True))
            data[k] = [0, 100]
    for i in [0, 6, 7, 8, 9]: data[f'c{i}'] = [1 if i == 0 else 0, 100]
    mask, stats = b.training_prevalence(pd.DataFrame(data, index=['g1', 'g2']), pd.DataFrame(rows))
    assert stats['unique_control_wells'] == 5
    assert stats['control_required_wells'] == 1
    assert mask['g1']
    data['c10'] = [0, 100]
    rows[-1]['control_Combo'] = 'c10'
    mask, stats = b.training_prevalence(pd.DataFrame(data, index=['g1', 'g2']), pd.DataFrame(rows))
    assert stats['control_required_wells'] == 2
    assert not mask['g1']


def sets(ids, original=20):
    return {'sets': [{'set_id': f's{i}', 'source_member_count': original,
                     'mapped_unique_gene_ids': list(ids)} for i in range(5)]}


def test_five_set_gate_deduplicates_and_uses_original_denominator():
    b = api(); ids = [f'g{i}' for i in range(11)]
    coverage, members = b.freeze_members(pd.Series(True, index=ids), sets(ids + ['g0']))
    assert coverage.n_retained.tolist() == [11] * 5
    assert coverage.passed.all() and len(members) == 55
    source = sets(ids, original=23)
    coverage, _ = b.freeze_members(pd.Series(True, index=ids), source)
    assert not coverage.passed.any()
    source = sets(ids)
    source['sets'][-1]['source_member_count'] = 23
    coverage, _ = b.freeze_members(pd.Series(True, index=ids), source)
    assert coverage.passed.tolist() == [True, True, True, True, False]


def test_missing_gene_is_an_alignment_error_but_measured_zero_is_filter_failure():
    b = api(); mask = pd.Series([True, False], index=['g1', 'zero'])
    with pytest.raises(ValueError, match='missing.*gene'):
        b.freeze_members(mask, sets(['g1', 'absent']))
    coverage, members = b.freeze_members(mask, sets(['zero'], original=1))
    assert not coverage.passed.any() and members.empty


def test_labels_use_all_human_library_and_two_equal_deltas():
    b = api(); counts, mapping = fixture(); mapping = b.project_mapping(mapping, ['a'])
    projected = counts.loc[:, ['a1', 'a2', 'c1', 'c2']].astype(float)
    members = pd.DataFrame({'set_id': ['s1'], 'gene_id': ['g1']})
    ids, labels = b.gene_set_labels(projected, mapping, members, ['s1'])
    assert ids == ['a']
    np.testing.assert_allclose(labels, [[np.log1p(200000)]])
    with pytest.raises(ValueError, match='missing.*gene'):
        b.gene_set_labels(projected.drop(index='g1'), mapping, members, ['s1'])


def test_models_fit_four_unscaled_binary_candidates_and_ties_choose_first():
    b = api(); fp = np.zeros((2, 2048)); fp[0, 0] = 1; fp[1, 1] = 1
    y = np.ones((2, 5)) * np.arange(5)
    result = b.fit_candidates(fp, y, fp[:1], y[:1], ['b', 'a'], fp[:1])
    assert result['fit_count'] == 4 and result['selected_candidate'] == 0
    assert [m.alpha for m in result['models']] == [.1, 1., 10., 100.]
    for m in result['models']:
        assert m.fit_intercept
        np.testing.assert_allclose(m.intercept_, np.arange(5))
    np.testing.assert_array_equal(result['test_predictions']['B2'], [np.arange(5)])
    with pytest.raises(ValueError, match='binary'):
        b.fit_candidates(fp + .5, y, fp[:1], y[:1], ['b', 'a'], fp[:1])


def test_nearest_neighbor_tie_is_sorted_training_compound_id():
    b = api(); fp = np.ones((2, 2048)); query = np.zeros((1, 2048))
    y = np.array([[9] * 5, [2] * 5])
    result = b.fit_candidates(fp, y, fp[:1], y[:1], ['z', 'a'], query)
    np.testing.assert_array_equal(result['test_predictions']['B2'], [[2] * 5])


def test_failed_coverage_branch_never_reads_validation_or_fits(tmp_path):
    b = api(); counts, mapping = fixture(); counts.to_csv(tmp_path/'counts.tsv', sep='\t')
    split = pd.DataFrame({'compound_id': ['a', 'b', 'test'], 'split': ['train', 'train', 'validation'],
                          'canonical_smiles': ['CC', 'CCC', 'O']})
    # Validation columns contain invalid values; no validation read is allowed on failure.
    result = b.develop_head(tmp_path/'counts.tsv', mapping, split, sets(['g1', 'g2', 'zero']))
    assert result['status'] == 'FAILED_COVERAGE_ZERO_FITS'
    assert result['fit_count'] == 0 and not result['validation_counts_read']
    assert result['transfer_A_continues']


def test_input_digest_mutation_is_rejected_by_seal(tmp_path):
    b = api(); source = tmp_path/'input'; source.write_text('before')
    seal = {'source_inputs': {'counts': b.file_record(source)}, 'artifact_hashes': {}}
    b.verify_seal(seal, {'counts': source}, {}, {})
    source.write_text('after')
    with pytest.raises(ValueError, match='input.*changed'):
        b.verify_seal(seal, {'counts': source}, {}, {})


def cli_module():
    path = Path(__file__).resolve().parents[1]/'scripts/train_biology_v2.py'
    spec = importlib.util.spec_from_file_location('train_biology_v2', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def cli_inputs(tmp_path, *, passing):
    import yaml
    b = api(); ids = [f'g{i}' for i in range(10)] if passing else ['g0']
    index = pd.Index([*ids, 'background'], name='gene_id')
    data = {k: [1] * len(ids) + [100] for k in ['a1', 'a2', 'b1', 'b2', 'v1', 'v2', 'c1', 'c2']}
    data['test_only_invalid'] = ['bad'] * len(index)
    paths = {k: tmp_path/n for k, n in {'counts': 'counts.tsv', 'mapping': 'mapping.tsv',
        'gene_metadata': 'genes.tsv', 'split': 'split.tsv', 'biology': 'biology.json',
        'protocol': 'protocol.yaml', 'input_contract': 'contract.json', 'source_receipt': 'source.json'}.items()}
    pd.DataFrame(data, index=index).to_csv(paths['counts'], sep='\t')
    mapping = pd.DataFrame({'treatment': ['a', 'a', 'b', 'b', 'v', 'v'],
        'replicate': ['rep1', 'rep2'] * 3, 'Combo': ['a1', 'a2', 'b1', 'b2', 'v1', 'v2'],
        'control_Combo': ['c1', 'c2'] * 3, 'eligible': [True] * 6,
        'cell_type': ['A549'] * 6, 'time_point': [24] * 6, 'dose': [1000] * 6})
    mapping.to_csv(paths['mapping'], sep='\t', index=False)
    split = pd.DataFrame({'compound_id': ['a', 'b', 'v', 't'], 'split': ['train', 'train', 'validation', 'test'],
        'canonical_smiles': ['CC', 'CCC', 'CCN', 'CO'], 'scaffold_group': ['a', 'b', 'v', 't']})
    split.to_csv(paths['split'], sep='\t', index=False)
    pd.DataFrame({'gene_id': index, 'gene_symbol': index}).to_csv(paths['gene_metadata'], sep='\t', index=False)
    definition = sets(ids, original=20)
    definition['gene_metadata'] = {'sha256': b.file_record(paths['gene_metadata'])['sha256']}
    paths['biology'].write_text(json.dumps(definition))
    protocol = {'schema_version': 2, 'status': 'PRESPECIFIED_BEFORE_NEW_CONTEXT_EFFECT_EVALUATION',
        'scope': {'source_context': {'cell_type': 'A549', 'time_hours': 24, 'dose_nM': 1000}},
        'split': {'original_counts': {'train': 2, 'validation': 1, 'test': 1}},
        'biology_head': {'set_ids': [f's{i}' for i in range(5)], 'alpha_candidates': [.1, 1., 10., 100.],
          'minimum_unique_genes_per_set': 10, 'minimum_fraction_original_source_members': .5,
          'feature_scaling': 'none_binary_features', 'label_scaling': 'none', 'fit_intercept': True,
          'refit_train_plus_validation': False}}
    paths['protocol'].write_text(yaml.safe_dump(protocol))
    contract = {'schema_version': 1, 'expected_inputs': {k: b.file_record(paths[k]) for k in ['split', 'protocol']}}
    paths['input_contract'].write_text(json.dumps(contract))
    aliases = {'counts': 'data/processed/pseudobulk_counts.tsv.gz', 'mapping': 'manifests/qualified_well_mapping.tsv',
       'gene_metadata': 'data/processed/gene_metadata.tsv', 'split': 'manifests/split_manifest.tsv',
       'biology': 'configs/biology_v1.json'}
    paths['source_receipt'].write_text(json.dumps({'input_hashes': {v: b.file_record(paths[k])['sha256'] for k, v in aliases.items()}}))
    args = [value for k, p in paths.items() for value in [f'--{k.replace("_", "-")}', str(p)]]
    args += ['--input-contract-sha256', b.file_record(paths['input_contract'])['sha256'],
             '--source-receipt-sha256', b.file_record(paths['source_receipt'])['sha256'],
             '--model-dir', str(tmp_path/'models'), '--output-root', str(tmp_path/'public')]
    return args, paths


@pytest.mark.parametrize('passing', [False, True])
def test_cli_records_zero_or_four_fits_and_seals_without_paths_or_test_truth(tmp_path, passing):
    module = cli_module(); args, paths = cli_inputs(tmp_path, passing=passing)
    module.main(args)
    root = tmp_path/'public'
    receipt = json.loads((root/'receipts/v2_biology_prediction_seal.json').read_text())
    assert receipt['fit_count'] == (4 if passing else 0)
    assert receipt['test_outcomes_loaded'] is False
    assert str(tmp_path) not in json.dumps(receipt)
    assert receipt['validation_counts_read'] is passing
    # An identical rerun validates the seal without a fifth candidate or a refit.
    module.main(args)
    if passing:
        with np.load(tmp_path/'models/test_predictions.npz') as d:
            assert d['ids'].tolist() == ['t'] and d['Ridge'].shape == (1, 5)
        assert len(list((tmp_path/'models').glob('Ridge_c*.joblib'))) == 4
    else:
        assert not list((tmp_path/'models').glob('*.joblib'))
    paths['biology'].write_text('{}')
    with pytest.raises(ValueError, match='hash|changed'):
        module.main(args)


def test_cli_rejects_wrong_code_revision_and_source_receipt_mutation(tmp_path):
    module = cli_module(); args, paths = cli_inputs(tmp_path, passing=False)
    with pytest.raises(ValueError, match='Git HEAD'):
        module.main([*args, '--source-commit', '0' * 40])
    paths['source_receipt'].write_text('{}')
    with pytest.raises(ValueError, match='trusted hash'):
        module.main(args)


def test_source_human_gene_alignment_is_checked_before_fitting(tmp_path):
    b = api(); counts, mapping = fixture(); path = tmp_path/'counts.tsv'
    counts.to_csv(path, sep='\t')
    split = pd.DataFrame({'compound_id': ['a', 'b'], 'split': ['train', 'train'],
                          'canonical_smiles': ['CC', 'CCC']})
    with pytest.raises(ValueError, match='gene identities/order'):
        b.develop_head(path, mapping, split, sets(['g1']), expected_gene_ids=['wrong'])
