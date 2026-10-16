"""Synthetic scientific-boundary checks; no real target effects are opened."""
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def api():
    return importlib.import_module('drugperturb.transfer_v2')


def fixture():
    rows = []; qualified = []; qc = []
    for rep in ['rep1', 'rep2']:
        digit = rep[-1]; plate = 'p'+digit
        for drug in ['a', 'b']:
            combo = drug+digit
            rows.append(dict(context_id='MCF7_24h_1000nM', Combo=combo, cell_type='MCF7',
                time_point=24, plate_oligo=plate, replicate=rep, dose_pattern='2', dose=1000,
                treatment=drug, vehicle=False, row_kind='drug', control_Combo='c'+digit))
            qualified.append(dict(context_id='MCF7_24h_1000nM', treatment=drug, replicate=rep,
                Combo=combo, control_Combo='c'+digit, eligible=True, count_qc_done=True,
                treatment_cells=40, control_cells=40))
            qc.append(dict(Combo=combo, n_cells=40))
        for control, pattern in [('c', '2'), ('d', '4')]:
            rows.append(dict(context_id='MCF7_24h_1000nM', Combo=control+digit, cell_type='MCF7',
                time_point=24, plate_oligo=plate, replicate=rep, dose_pattern=pattern, dose=0,
                treatment='Vehicle', vehicle=True, row_kind='vehicle', control_Combo=''))
            qc.append(dict(Combo=control+digit, n_cells=40))
    counts = pd.DataFrame({'a1': [9, 1, 0], 'a2': [10, 90, 0],
        'b1': [1, 9, 0], 'b2': [90, 10, 0], 'c1': [20, 20, 0],
        'c2': [5, 5, 0], 'd1': [0, 100, 0], 'd2': [1, 9, 0]},
        index=['g1', 'g2', 'zero'])
    return counts, pd.DataFrame(rows), pd.DataFrame(qualified), pd.DataFrame(qc)


def test_truth_normalizes_all_genes_each_well_before_two_vehicle_mean():
    t = api(); counts, wells, qualified, qc = fixture()
    pairs = t.alternate_controls(wells, qualified, qc, ['a', 'b'])
    assert pairs.compound_eligible.all()
    strict, alternate = t.measured_responses(counts, qualified, pairs, ['a', 'b'], ['g1', 'zero'])
    expected = ((np.log1p(900000)-np.log1p(500000))
                +(np.log1p(100000)-np.log1p(500000))) / 2
    np.testing.assert_allclose(strict[0], [expected, 0])
    expected_alt = ((np.log1p(900000)-(np.log1p(500000)+np.log1p(0))/2)
                    +(np.log1p(100000)-(np.log1p(500000)+np.log1p(100000))/2))/2
    np.testing.assert_allclose(alternate[0], [expected_alt, 0])
    assert not np.isclose(alternate[0, 0], np.log1p(500000)-np.log1p(275000))


@pytest.mark.parametrize('change', ['one_vehicle_low', 'one_vehicle_missing', 'third_vehicle', 'wrong_pattern'])
def test_sensitivity_requires_both_two_vehicle_controls_without_fallback(change):
    t = api(); counts, wells, qualified, qc = fixture()
    if change == 'one_vehicle_low': qc.loc[qc.Combo.eq('d1'), 'n_cells'] = 29
    if change == 'one_vehicle_missing': qc = qc[~qc.Combo.eq('d1')]
    if change == 'third_vehicle':
        extra = wells[wells.Combo.eq('d1')].copy(); extra['Combo'] = 'extra'; wells = pd.concat([wells, extra])
    if change == 'wrong_pattern': wells.loc[wells.Combo.eq('d1'), 'dose_pattern'] = '2'
    pairs = t.alternate_controls(wells, qualified, qc, ['a', 'b'])
    assert not pairs.compound_eligible.any()
    strict, alternate = t.measured_responses(counts, qualified, pairs, ['a', 'b'], ['g1'])
    assert np.isfinite(strict).all() and np.isnan(alternate).all()


def test_missing_gene_is_rejected_while_measured_zero_is_retained():
    t = api(); counts, wells, qualified, qc = fixture()
    pairs = t.alternate_controls(wells, qualified, qc, ['a', 'b'])
    with pytest.raises(ValueError, match='missing.*gene'):
        t.measured_responses(counts.drop(index='zero'), qualified, pairs, ['a', 'b'], ['zero'])
    p, _ = t.measured_responses(counts, qualified, pairs, ['a', 'b'], ['zero'])
    np.testing.assert_array_equal(p, [[0], [0]])


def test_qualified_mapping_and_compound_identity_cannot_be_silently_duplicated():
    t = api(); _, wells, qualified, qc = fixture()
    with pytest.raises(ValueError, match='replicate|duplicate'):
        t.alternate_controls(wells, pd.concat([qualified, qualified.iloc[:1]]), qc, ['a', 'b'])
    with pytest.raises(ValueError, match='unique'):
        t.align_predictions({'ids': np.array(['a', 'a']), 'B0': np.zeros((2, 2))}, ['a'], ['B0'], 2)


def test_prediction_alignment_preserves_query_order_and_rejects_missing_id():
    t = api(); sealed = {'ids': np.array(['b', 'a']), 'M1': np.array([[9, 8], [1, 2]])}
    out = t.align_predictions(sealed, ['a', 'b'], ['M1'], 2)
    np.testing.assert_array_equal(out['M1'], [[1, 2], [9, 8]])
    with pytest.raises(ValueError, match='missing.*compound'):
        t.align_predictions(sealed, ['absent'], ['M1'], 2)


def test_metrics_equal_weight_compounds_and_keep_each_set_errors():
    t = api(); truth = np.zeros((2, 5))
    pred = {'B0': np.array([[0] * 5, [4] * 5])}
    per, summary, by_set = t.score_predictions(truth, pred, ['a', 'b'], ['x', 'y'],
                                             metric='mae', feature_ids=list('12345'))
    assert summary['B0']['mean_error'] == 2
    assert len(by_set) == 10 and set(by_set.feature_id) == set('12345')
    per, summary, _ = t.score_predictions(truth, pred, ['a', 'b'], ['x', 'y'], metric='rmse')
    assert summary['B0']['mean_error'] == 2
    assert not np.isclose(summary['B0']['mean_error'], np.sqrt(8))
    assert summary['B0']['n_spearman_undefined'] == 2


def test_bootstrap_uses_per_compound_mae_difference_and_group_multiplicity():
    t = api()
    r = t.bootstrap_error_difference([-1, -3, 2], ['a', 'b', 'c'], ['g1', 'g1', 'g2'])
    assert np.isclose(r['point_estimate'], -2/3)
    assert r['ci_lower'] == -2 and r['ci_upper'] == 2
    assert r['draws'] == 2000 and r['seed'] == 2026
    assert r['sampled_compounds_min'] == 2 and r['sampled_compounds_max'] == 4
    reverse = t.bootstrap_error_difference([2, -3, -1], ['c', 'b', 'a'], ['g2', 'g1', 'g1'])
    np.testing.assert_array_equal(r['bootstrap_differences'], reverse['bootstrap_differences'])


def test_gate_failed_has_only_descriptive_point_estimate_and_no_interval():
    t = api(); truth = np.zeros((2, 5)); prediction = {'Ridge': np.ones((2, 5)), 'B1': np.zeros((2, 5))}
    per, _, _ = t.score_predictions(truth, prediction, ['a', 'b'], ['x', 'y'], metric='mae')
    r = t.contrast_summary(per, 'Ridge', 'B1', gate_met=False)
    assert r['point_estimate'] == 1 and r['ci_lower'] is None and r['ci_upper'] is None
    assert r['status'] == 'DESCRIPTIVE_NO_COUNT_GATE'


def test_strict_alternative_compare_same_subset_not_full_baseline():
    t = api(); strict = np.array([[0.], [100.], [4.]])
    alternative = np.array([[2.], [np.nan], [6.]])
    predictions = {'B0': np.zeros((3, 1))}
    result = t.score_control_sensitivity(strict, alternative, predictions,
        ['a', 'b', 'c'], ['g1', 'g2', 'g3'], metric='rmse')
    assert result['compound_ids'] == ['a', 'c']
    assert result['strict_summary']['B0']['mean_error'] == 2
    assert result['alternative_summary']['B0']['mean_error'] == 4


def test_gene_sets_require_exact_fixed_members_and_preserve_protocol_order():
    t = api(); delta = np.array([[2., 4., 9.]])
    members = pd.DataFrame({'set_id': ['s2', 's1', 's1'], 'gene_id': ['g3', 'g2', 'g1']})
    result = t.project_sets(delta, ['g1', 'g2', 'g3'], members, ['s1', 's2'])
    np.testing.assert_array_equal(result, [[3, 9]])
    with pytest.raises(ValueError, match='missing.*gene'):
        t.project_sets(delta[:, :2], ['g1', 'g2'], members, ['s1', 's2'])


def test_target_manifest_hash_mutation_rejected_before_parsing_counts(tmp_path):
    t = api(); p = tmp_path/'counts.tsv'; p.write_text('malformed response that must not be parsed')
    entry = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': '0' * 64}
    with pytest.raises(ValueError, match='hash'):
        t.checked_file(entry, tmp_path)


def test_biology_readouts_are_never_labeled_as_five_genes():
    t = api()
    per, summary, _ = t.score_predictions(np.zeros((1, 5)), {'B0': np.zeros((1, 5))},
        ['a'], ['g'], metric='mae', feature_ids=list('12345'), analysis_unit='gene_set')
    assert per.n_outputs.tolist() == [5] and per.analysis_unit.tolist() == ['gene_set']
    assert summary['B0']['n_outputs'] == 5 and summary['B0']['analysis_unit'] == 'gene_set'
    assert 'n_genes' not in per and 'n_genes' not in summary['B0']


def test_cohort_uses_frozen_test_qualification_and_rejects_role_or_group_swap():
    t = api(); _, _, qualified, _ = fixture()
    split = pd.DataFrame({'compound_id': ['a', 'b'], 'split': ['test'] * 2,
                          'scaffold_group': ['x', 'y'], 'canonical_smiles': ['CC', 'CCC']})
    compounds = split.assign(context_id='MCF7_24h_1000nM', role='primary_test',
                             eligible=True, count_qc_done=True)
    summary = pd.DataFrame([dict(context_id='MCF7_24h_1000nM', count_qc_done=True,
        eligible_test_compounds=2, eligible_test_structure_groups=2, evaluation_gate_met=False)])
    result = t.frozen_cohort(compounds, qualified, summary, split, 'MCF7_24h_1000nM', 'primary_test')
    assert result['ids'] == ['a', 'b'] and not result['gate_met']
    compounds.loc[0, 'scaffold_group'] = 'y'
    with pytest.raises(ValueError, match='frozen.*identity|scaffold'):
        t.frozen_cohort(compounds, qualified, summary, split, 'MCF7_24h_1000nM', 'primary_test')


def test_cli_contract_change_prevents_any_target_processing(tmp_path):
    import importlib.util
    path = Path(__file__).resolve().parents[1]/'scripts/evaluate_transfer_v2.py'
    spec = importlib.util.spec_from_file_location('evaluate_transfer_v2', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    manifest = tmp_path/'sources.json'; manifest.write_text('{}')
    with pytest.raises(ValueError, match='manifest.*hash'):
        module.main(['--v1-root', str(tmp_path/'v1'), '--biology-model-dir', str(tmp_path/'models'),
            '--target-manifest', str(manifest), '--target-manifest-sha256', '0' * 64,
            '--input-contract-sha256', '0' * 64, '--biology-seal-sha256', '0' * 64,
            '--qc-receipt-sha256', '0' * 64,
            '--output-root', str(tmp_path/'public')])


def cli_module():
    import importlib.util
    path = Path(__file__).resolve().parents[1]/'scripts/evaluate_transfer_v2.py'
    spec = importlib.util.spec_from_file_location('evaluate_transfer_v2', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def source_manifest_fixture(tmp_path):
    module = cli_module(); t = api(); descriptors = {}
    from drugperturb.biology_v2 import file_record
    for key in ['counts', 'gene_metadata', 'well_qc', 'extraction_summary']:
        path = tmp_path/(key+'.txt'); path.write_text('synthetic metadata / counts not parsed here')
        descriptors[key] = {**file_record(path), 'path': str(path)}
    summary = pd.DataFrame([{'key': k, 'value': v} for k, v in dict(
        context_cell_type='MCF7', context_time_hours='24', source_human_genes='3',
        all_human_gene_rows_exported='TRUE', UMI_conserved='TRUE', source_hash_QC_all_pass='TRUE',
        counts_nonnegative_integer='TRUE', treatment_effects_computed='FALSE').items()])
    summary.to_csv(descriptors['extraction_summary']['path'], sep='\t', index=False)
    path = Path(descriptors['extraction_summary']['path'])
    descriptors['extraction_summary'] = {**file_record(path), 'path': str(path)}
    descriptors['extractor_code'] = {**file_record(module.ROOT/'scripts/export_context_rds.R'),
                                    'path': str(module.ROOT/'scripts/export_context_rds.R')}
    raw = {'basename': 'GSM4150378_sciPlex3_MCF7_24hrs.RDS.gz', 'bytes': 111, 'sha256': 'a' * 64}
    acquisition = tmp_path/'acquisition.json'
    acquisition.write_text(json.dumps({'sources': [{'filename': raw['basename'], 'bytes': 111,
                       'sha256': raw['sha256'], 'gzip_integrity_checked': True}]}))
    descriptors['acquisition_receipt'] = {**file_record(acquisition), 'path': str(acquisition)}
    output_names = {'counts': 'pseudobulk_counts.tsv.gz', 'gene_metadata': 'gene_metadata.tsv',
                    'well_qc': 'well_qc.tsv', 'extraction_summary': 'extraction_summary.tsv'}
    extraction = {'schema_version': 1, 'sources': {'MCF7': {'status': 'COMPLETE', 'source': raw,
        'outputs': {name: {k: descriptors[key][k] for k in ['bytes', 'sha256']} for key, name in output_names.items()},
        'code_input_hashes': {name: file_record(module.ROOT/name)['sha256'] for name in
          ['scripts/export_context_rds.R', 'scripts/csc_utils.R', 'manifests/v2/well_mapping.tsv', 'configs/protocol_v2.yaml']}}}}
    ep = tmp_path/'extraction.json'; ep.write_text(json.dumps(extraction))
    descriptors['extraction_receipt'] = {**file_record(ep), 'path': str(ep)}
    descriptors['raw_source'] = raw
    qc = {'well_qc_inputs': [{k: descriptors['well_qc'][k] for k in ['bytes', 'sha256']}]}
    return module, {'schema_version': 1, 'sources': {'MCF7': descriptors}}, qc, extraction, ep


@pytest.mark.parametrize('change', ['output', 'source', 'code'])
def test_target_source_requires_bound_extraction_source_code_and_output(tmp_path, change):
    module, manifest, qc, extraction, ep = source_manifest_fixture(tmp_path)
    checked, _ = module.verify_sources(manifest, tmp_path, qc)
    assert 'MCF7' in checked
    # Even a rehashed manifest descriptor cannot hide a source->output contradiction.
    executed = extraction['sources']['MCF7']
    if change == 'output':
        executed['outputs']['pseudobulk_counts.tsv.gz']['sha256'] = '0' * 64
    elif change == 'source':
        executed['source'] = {**executed['source'], 'sha256': '0' * 64}
    else:
        executed['code_input_hashes']['manifests/v2/well_mapping.tsv'] = '0' * 64
    ep.write_text(json.dumps(extraction))
    from drugperturb.biology_v2 import file_record
    manifest['sources']['MCF7']['extraction_receipt'].update(file_record(ep))
    with pytest.raises(ValueError, match='extraction'):
        module.verify_sources(manifest, tmp_path, qc)


def test_context_orchestration_keeps_fixed_five_set_units_and_descriptive_gate():
    t = api(); counts, wells, qualified, qc = fixture()
    members = pd.DataFrame({'set_id': list('12345'), 'gene_id': ['g1', 'g2', 'zero', 'g1', 'g2']})
    cohort = {'ids': ['a', 'b'], 'groups': ['x', 'y'], 'gate_met': False}
    a = {method: np.zeros((2, 2)) for method in t.A_METHODS}
    b = {method: np.zeros((2, 5)) for method in t.B_METHODS}
    result = t.evaluate_context(counts, wells, qualified, qc, cohort, a, ['g1', 'zero'],
        b_predictions=b, members=members, set_ids=list('12345'))
    table = result['per_compound']; biology = table[table.analysis.eq('B')]
    assert len(table) == 2 * (6 + 4) * 3
    assert set(biology.n_outputs) == {5} and set(biology.n_gene_sets) == {5}
    assert set(biology.n_unique_member_genes) == {3}
    assert set(biology.analysis_unit) == {'gene_set'} and set(biology.error_metric) == {'mae'}
    assert len(result['per_set']) == 2 * 4 * 5 * 3
    assert result['sensitivity_compounds'] == 2 and result['sensitivity_groups'] == 2
    assert set(result['contrasts'].status) == {'DESCRIPTIVE_NO_COUNT_GATE'}
    assert result['contrasts'].ci_lower.isna().all()
    # JSON serialization must retain undefined descriptive summaries as null.
    json.dumps(result['summary'], allow_nan=False)
