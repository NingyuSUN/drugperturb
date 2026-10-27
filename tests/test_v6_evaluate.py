"""Synthetic checks of the v6 seal selection, diagnostics report and evaluation.

The extraction, preparation and comparators run for real; the arms are sealed
synthetically, so the endpoint, its sign convention and its anchors are tested
without the cell-model environment. ``test_v6_cellmodels.py`` covers the fits
themselves. Fixtures establish software behaviour only; they say nothing about
Tahoe-100M.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('rdkit')
pytest.importorskip('sklearn')
from drugperturb.v3 import runner  # noqa: E402
from drugperturb.v6 import scoring, selection  # noqa: E402
from test_v5_pipeline import make_dataset, npz, run  # noqa: E402
from test_v6_pipeline import CELLS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SEEDS = (42, 123)
ACCURACY = {'M1_pseudobulk_raw': 0.62, 'M2_pseudobulk_residual': 0.55}


def seal_synthetic_arms(root, prepared, protocol):
    """Write arm seals in the shape v6_seal.py produces, plus the selection record.

    Each arm's test prediction is a signed fraction of the SELECTION-half delta,
    which is information the arms are entitled to, so the sealed archives behave
    like a model that partly recovers the right directions.
    """
    sealed = root/'sealed'
    with np.load(prepared/'endpoint/de_sets.npz', allow_pickle=False) as z:
        keys = np.asarray(z['keys'], str)
        delta = z['selection_delta']
        splits = {k: s for k, s in zip(keys, np.asarray(z['responder'], bool))}
    queries = pd.read_csv(prepared/'queries.tsv', sep='\t', keep_default_na=False)
    split_of = dict(zip(queries.compound_id+'|'+queries.context_id, queries.split))
    arms = {}
    for index, (arm, accuracy) in enumerate(ACCURACY.items()):
        arrays = {}
        for split in ['train', 'validation', 'test']:
            rows = [i for i, k in enumerate(keys) if split_of.get(k) == split]
            arrays[f'{split}_keys'] = keys[rows]
            # A weaker signal for the second arm, and a deterministic sign flip on some genes.
            scale = 0.6 if index == 0 else 0.3
            flip = np.ones(delta.shape[1])
            flip[index::3] = -1.0
            arrays[f'{split}_contrast'] = delta[rows]*scale*flip
            arrays[f'{split}_anchored'] = delta[rows]*scale*flip
            arrays[f'{split}_raw'] = delta[rows]*scale
        endpoint = dict(validation_direction_accuracy=accuracy,
                        validation_direction_accuracy_all_keys=accuracy-0.01,
                        b1p_direction_accuracy=0.58, split_noise_floor=0.72,
                        residual_r=0.2-0.1*index, residual_share=0.4)
        metadata = dict(analysis='v6', arm=arm, seeds=list(SEEDS), residual_target=index == 1,
                        final_prediction='mean over seeds',
                        mean_seed_validation_rmse=0.5+0.01*index,
                        seed_validation_rmse=[0.5+0.01*index]*len(SEEDS),
                        mean_seed_validation_direction_accuracy=accuracy,
                        seed_direction_accuracy=[accuracy-0.005, accuracy+0.005], endpoint=endpoint,
                        seed_averaged={'validation': dict(effect_size_ratio=0.8, rmse=0.5, mean_gene_pearson_r=0.3)},
                        non_collapse=dict(non_collapsed=True, effect_size_ratio=0.8,
                                          validation_direction_accuracy=accuracy),
                        latent={}, descriptor_gate=dict(passed=True), engineering_smoke=True, fits=[])
        runner.write_seal(sealed, 'v6', arm, arrays, metadata, {'protocol': protocol})
        arms[arm] = dict(seed_direction_accuracy=metadata['seed_direction_accuracy'],
                         seed_rmse=metadata['seed_validation_rmse'],
                         validation_direction_accuracy=accuracy, validation_rmse=metadata['mean_seed_validation_rmse'],
                         non_collapse=metadata['non_collapse'])
    rule = json.loads(json.dumps(dict(candidates=list(ACCURACY), metric='mean over seeds of the per-seed '
                                      'validation direction accuracy', rule='the highest', near_tie_rule='v5 rule')))
    chosen = selection.select(arms, rule['candidates'])
    record = dict(status='SELECTED_ON_VALIDATION_BEFORE_TEST_SCORING', test_outcomes_loaded=False,
                  evaluation_half_test_delta_loaded=False, metric=rule['metric'], rule=rule['rule'],
                  near_tie_rule=rule['near_tie_rule'], arms=arms, descriptor_gate=dict(passed=True), **chosen)
    (sealed/'selection_v6.json').write_text(json.dumps(record, indent=2)+'\n')
    assert splits  # the responder flags are used by the evaluation itself
    return sealed


def switch_gates_on(prepared, gated):
    """The fixture has three test compounds per context, far below the real gate, so a copy of the
    prepared outputs declares the gates met. Everything else is the real pipeline's output."""
    import shutil
    shutil.copytree(prepared, gated)
    summary = pd.read_csv(gated/'count_qc/context_summary.tsv', sep='\t')
    summary['test_count_gate_met'] = True
    summary.to_csv(gated/'count_qc/context_summary.tsv', sep='\t', index=False)
    receipt = json.loads((gated/'prepare_receipt.json').read_text())
    receipt['study_gate_G4']['met'] = True
    (gated/'prepare_receipt.json').write_text(json.dumps(receipt))
    return gated


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp('v6eval')/'study'
    make_dataset(root, cells=CELLS, protocol_path='configs/protocol_v6.yaml')
    common = ['--manifests', root/'manifests', '--protocol', root/'protocol.yaml']
    run('v6_extract.py', '--obs', root/'obs_metadata.parquet', '--gene-metadata', root/'gene_metadata.parquet',
        '--shard-dir', root/'shards', '--output-root', root/'extraction', '--shard-start', 0, '--shard-stop', 3,
        *common)
    run('v6_prepare.py', '--extraction-root', root/'extraction', '--output-root', root/'prepared', *common)
    run('v6_baselines.py', '--prepared', root/'prepared', '--output-root', root/'sealed', *common)
    run('v6_cheap_report.py', '--prepared', root/'prepared', '--sealed', root/'sealed',
        '--output', root/'reports'/'cheap_report.md', '--protocol', root/'protocol.yaml')
    seal_synthetic_arms(root, root/'prepared', root/'protocol.yaml')
    gated = switch_gates_on(root/'prepared', root/'prepared_gated')
    run('v6_diagnostics_report.py', '--prepared', gated, '--sealed', root/'sealed',
        '--output', root/'reports'/'diagnostics.md', '--protocol', root/'protocol.yaml')
    run('v6_evaluate.py', '--extraction-root', root/'extraction', '--prepared', gated,
        '--sealed', root/'sealed', '--output-root', root/'evaluation', *common)
    return root


def evaluation(study):
    return json.loads((study/'evaluation'/'evaluation_v6.json').read_text())


def test_the_seal_selects_on_direction_accuracy_and_higher_wins(study):
    record = json.loads((study/'sealed'/'selection_v6.json').read_text())
    assert record['higher_is_better'] is True
    means = record['mean_seed_validation_direction_accuracy']
    assert record['selected'] == max(means, key=means.get) == 'M1_pseudobulk_raw'
    assert record['test_outcomes_loaded'] is False and record['evaluation_half_test_delta_loaded'] is False
    assert 0 < record['bonferroni_k'] <= 2 and record['interval_level'] >= 0.95


def test_selection_prefers_the_higher_accuracy_and_honours_the_tie_rule():
    arms = {'M1_pseudobulk_raw': dict(seed_direction_accuracy=[0.60, 0.62],
                                      non_collapse=dict(non_collapsed=True)),
            'M2_pseudobulk_residual': dict(seed_direction_accuracy=[0.50, 0.52],
                                           non_collapse=dict(non_collapsed=True))}
    chosen = selection.select(arms, list(arms))
    assert chosen['selected'] == 'M1_pseudobulk_raw' and chosen['primary_arms'] == ['M1_pseudobulk_raw']
    assert chosen['near_tie'] is False and chosen['bonferroni_k'] == 1
    close = {'M1_pseudobulk_raw': dict(seed_direction_accuracy=[0.60, 0.61],
                                       non_collapse=dict(non_collapsed=True)),
             'M2_pseudobulk_residual': dict(seed_direction_accuracy=[0.598, 0.606],
                                            non_collapse=dict(non_collapsed=True))}
    tied = selection.select(close, list(close))
    assert tied['near_tie'] is True and len(tied['primary_arms']) == 2
    assert tied['bonferroni_k'] == 2 and tied['interval_level'] == pytest.approx(1-0.05/2)
    collapsed = {k: dict(v, non_collapse=dict(non_collapsed=False)) for k, v in arms.items()}
    fallback = selection.select(collapsed, list(collapsed))
    assert fallback['any_passing'] is False and fallback['selected'] == 'M1_pseudobulk_raw'
    with pytest.raises(ValueError, match='no fit summary'):
        selection.select(arms, [*arms, 'M3_missing'])


def test_the_diagnostics_report_leads_with_the_ceiling_and_hides_no_test_number(study):
    text = (study/'reports'/'diagnostics.md').read_text()
    assert 'before any test scoring' in text
    for heading in ['Split noise floor', 'B1p on its own', 'Residual share', 'Arm selection',
                    'Does chemical structure help any method', 'Context gates']:
        assert heading in text, heading
    for forbidden in ['TEST_SCORED', 'P1_direction', 'test accuracy']:
        assert forbidden not in text, f'the pre-test report must not mention {forbidden}'
    per_seed = pd.read_csv(study/'reports'/'per_seed_validation.tsv', sep='\t')
    assert len(per_seed) == len(ACCURACY)*len(SEEDS) and set(per_seed.arm) == set(ACCURACY)


def test_the_cheap_report_decides_g8_and_d7_without_any_arm(study):
    record = json.loads((study/'reports'/'cheap_report.json').read_text())
    assert record['status'] == 'CPU_ONLY_NO_ARM_FITTED'
    assert record['test_outcomes_computed'] is False
    assert record['evaluation_half_test_delta_computed'] is False
    assert record['gate_G8_met'] is (record['split_noise_floor'] >= record['g8_threshold'])
    assert record['accuracy']['B0'] == pytest.approx(0.5), 'zero change must sit exactly at chance'
    assert 'B1p' in record['accuracy'] and 'M1r' in record['accuracy']
    assert 'M1r_minus_B1p' in record['structure_contrasts']
    assert isinstance(record['structure_adds_over_reference'], bool)
    # D7 as committed compares each structure-using comparator against chance, not against what it
    # extends, so the two readings can disagree; the report must carry the committed one.
    margins = record['d7_margins_over_chance']
    assert set(margins) <= set(record['accuracy']) and 'M1r' in margins
    for name, margin in margins.items():
        assert margin == pytest.approx(record['accuracy'][name]-0.5)
    assert record['d7_fires'] is not any(m > record['d7_margin'] for m in margins.values())
    assert record['validation_responders'] <= record['validation_keys']
    text = (study/'reports'/'cheap_report.md').read_text()
    assert 'No arm has been fitted' in text
    assert 'Can the endpoint work at all' in text and 'Does chemical structure help' in text
    assert 'Decision D7 as committed' in text
    for forbidden in ['TEST_SCORED', 'P1_direction']:
        assert forbidden not in text, f'the CPU-only report must not mention {forbidden}'


def test_the_evaluation_anchors_pin_the_metric(study):
    anchors = evaluation(study)['anchors']
    assert anchors['zero_change_scores_chance'] is True, 'zero change must score exactly 0.5'
    assert anchors['truth_fed_as_prediction_scores_one'] is True
    assert anchors['b0_equals_zero'] is True
    assert anchors['halves_partition_each_unit'] is True


def test_the_primary_endpoint_uses_the_right_sign_and_comparators(study):
    record = evaluation(study)
    p1 = record['endpoints']['P1_direction_beyond_simple_baselines']
    assert p1['per_arm'], 'at least one primary arm must be scored'
    for info in p1['per_arm'].values():
        assert set(info['estimates']) == {'B1p', 'M1a'}, 'P1 compares against B1p and M1a'
        for estimate in info['estimates'].values():
            assert estimate['positive_favors'] == 'candidate'
            assert estimate['ci_lower'] <= estimate['point_estimate'] <= estimate['ci_upper']
        assert info['supported'] == all(e['ci_lower'] > 0 for e in info['estimates'].values())
    assert p1['supported'] is bool(p1['confirmatory'] and record['endpoints']['gate_G8_met']
                                   and any(v['supported'] for v in p1['per_arm'].values()))


def test_every_prespecified_secondary_is_reported(study):
    record = evaluation(study)
    endpoints = record['endpoints']
    for name in ['S1_above_chance', 'S2_all_keys', 'S3_structure_baselines', 'S5_rmse_on_de_genes',
                 'S6_replicates_v5']:
        assert name in endpoints and endpoints[name], name
    assert set(endpoints['S2_all_keys']) == {'B1p', 'M1a'}
    assert 'M1r_minus_B1p' in endpoints['S3_structure_baselines']
    for name in ['S5_rmse_on_de_genes', 'S6_replicates_v5']:
        assert 'an error' in endpoints[name]['note'], f'{name} must state its sign'
        for key, estimate in endpoints[name].items():
            if key != 'note':
                assert estimate['positive_favors'] == 'baseline', f'{name}.{key} is an error, not an accuracy'
    for key, estimate in endpoints['S3_structure_baselines'].items():
        assert estimate['positive_favors'] == 'candidate', key
    secondary = record['secondary']
    for name in ['every_pair', 'without_small_vehicle_line', 'leave_one_plate_out', 'de_overlap', 'per_context']:
        assert name in secondary, name
    assert secondary['accuracy_k20'], 'the k sensitivity must be reported'
    assert endpoints['split_noise_floor'] == pytest.approx(0.72)


def test_the_per_compound_table_covers_responders_and_every_method(study):
    frame = pd.read_csv(study/'evaluation'/'per_compound.tsv', sep='\t')
    assert {'method', 'compound_id', 'context_id', 'scaffold_group', 'responder', 'direction_accuracy',
            'de_rmse'} <= set(frame.columns)
    assert frame.direction_accuracy.between(0, 1).all()
    assert set(ACCURACY) <= set(frame.method)
    assert {'B0', 'B1c', 'B1p', 'M1a', 'M1r'} <= set(frame.method)
    assert np.allclose(frame[frame.method.eq('B0')].direction_accuracy, 0.5), 'zero change scores chance'
    summary = pd.read_csv(study/'evaluation'/'summary.tsv', sep='\t')
    assert summary.responder_direction_accuracy.is_monotonic_decreasing


def test_the_evaluation_refuses_a_tampered_seal(study, tmp_path):
    import shutil
    sealed = tmp_path/'sealed'
    shutil.copytree(study/'sealed', sealed)
    archive = sealed/'baselines_v6_predictions.npz'
    data = npz(archive)
    data['B1p'] = data['B1p']+1.0
    np.savez_compressed(archive, **data)
    with pytest.raises(AssertionError, match='do not match their seal'):
        run('v6_evaluate.py', '--extraction-root', study/'extraction', '--prepared', study/'prepared',
            '--sealed', sealed, '--output-root', tmp_path/'evaluation',
            '--manifests', study/'manifests', '--protocol', study/'protocol.yaml')


def test_scoring_helpers_agree_with_the_endpoint_module():
    truth = {('a', 'c'): np.array([1.0, -2.0, 0.5]), ('b', 'c'): np.array([-1.0, 2.0, 0.25])}
    prediction = {k: -v for k, v in truth.items()}
    sets = dict(keys=np.asarray(['a|c', 'b|c']), genes=np.array([[0, 1], [0, 1]]))
    keys = [('a', 'c'), ('b', 'c')]
    assert scoring.mean_accuracy(prediction, truth, sets, keys) == pytest.approx(0.0)
    assert scoring.mean_accuracy(truth, truth, sets, keys) == pytest.approx(1.0)
    reference = {k: np.zeros(3) for k in keys}
    assert scoring.residual_correlation(truth, truth, reference, keys) == pytest.approx(1.0)
    assert scoring.residual_correlation(prediction, truth, reference, keys) == pytest.approx(-1.0)
    assert np.isnan(scoring.residual_correlation(reference, truth, reference, keys)), 'a flat arm has no r'
    assert scoring.residual_share(truth, reference, keys) == pytest.approx(1.0)
    assert scoring.gene_sets(sets, keys).tolist() == [[0, 1], [0, 1]]
    with pytest.raises(ValueError, match='no DE set'):
        scoring.gene_sets(sets, [('z', 'c')])


def test_non_collapse_requires_beating_chance_by_the_margin():
    import importlib.util
    spec = importlib.util.spec_from_file_location('v6s', REPO/'scripts/v6_seal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    good = module.non_collapse(dict(validation_direction_accuracy=0.55), dict(effect_size_ratio=0.5))
    assert good['non_collapsed'] is True
    chance = module.non_collapse(dict(validation_direction_accuracy=0.505), dict(effect_size_ratio=0.5))
    assert chance['non_collapsed'] is False and chance['checks']['direction_above_chance'] is False
    flat = module.non_collapse(dict(validation_direction_accuracy=0.9), dict(effect_size_ratio=0.01))
    assert flat['non_collapsed'] is False and flat['checks']['effect_size_ratio'] is False
    assert module.non_collapse(dict(validation_direction_accuracy=float('nan')),
                               dict(effect_size_ratio=0.5))['non_collapsed'] is False
