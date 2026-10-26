"""Synthetic checks of the v6 extraction and preparation: the S/E split, DE sets and the responder gate.

The same Tahoe-like dataset as the v5 pipeline test, with more cells per unit
because v6 splits every unit in two, runs through the v6 scripts. Fixtures
establish software behaviour only; they say nothing about Tahoe-100M.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip('rdkit')
pytest.importorskip('sklearn')
from drugperturb.v4.design import log_cpm  # noqa: E402
from drugperturb.v5 import extract as v5_extract  # noqa: E402
from drugperturb.v6 import endpoint, extract, prepare  # noqa: E402
from test_v5_pipeline import CAP, make_dataset, npz, run  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CELLS = (70, 90)  # per (sample, line), so each half clears the 30-cell threshold


def pipeline(root, parts):
    for start, stop in parts:
        run('v6_extract.py', '--obs', root/'obs_metadata.parquet',
            '--gene-metadata', root/'gene_metadata.parquet', '--shard-dir', root/'shards',
            '--output-root', root/'extraction', '--shard-start', start, '--shard-stop', stop,
            '--manifests', root/'manifests', '--protocol', root/'protocol.yaml')
    run('v6_prepare.py', '--extraction-root', root/'extraction', '--output-root', root/'prepared',
        '--manifests', root/'manifests', '--protocol', root/'protocol.yaml')
    run('v6_baselines.py', '--prepared', root/'prepared', '--output-root', root/'sealed',
        '--manifests', root/'manifests', '--protocol', root/'protocol.yaml')
    return root/'prepared'


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('v6')
    truth = make_dataset(tmp/'two', cells=CELLS, protocol_path='configs/protocol_v6.yaml')
    two = pipeline(tmp/'two', [(0, 2), (2, 3)])
    make_dataset(tmp/'one', cells=CELLS, protocol_path='configs/protocol_v6.yaml')
    one = pipeline(tmp/'one', [(0, 3)])
    return dict(truth=truth, two=two, one=one, root=tmp)


def receipt(prepared):
    return json.loads((prepared/'prepare_receipt.json').read_text())


def part_receipts(root):
    return [json.loads(p.read_text()) for p in sorted((root/'extraction').rglob('part_receipt.json'))]


def test_halves_partition_every_unit_and_give_the_extra_cell_to_evaluation(runs):
    expected = pd.read_csv(runs['two'].parent/'extraction'/'part_00000_00002'/'expected_half_cells.tsv', sep='\t')
    truth = runs['truth']
    passing = truth[truth.passing.eq('full')].groupby(['sample', 'line']).size()
    wide = expected.pivot(index='parent_unit', columns='half', values='expected_cells')
    for (sample, line), total in passing.items():
        unit = f'{sample}:{line}'
        if unit not in wide.index:
            continue
        assert wide.loc[unit, 'S']+wide.loc[unit, 'E'] == total, f'{unit} halves must cover its cells'
        assert wide.loc[unit, 'E']-wide.loc[unit, 'S'] == total % 2, 'an odd unit gives the extra cell to E'


def test_the_split_is_deterministic_and_independent_of_partitioning(runs):
    one, two = part_receipts(runs['one'].parent), part_receipts(runs['two'].parent)
    assert len({r['half_label_sha256'] for r in one+two}) == 1, 'the half labels must not depend on shard ranges'
    assert {r['split_seed'] for r in one+two} == {extract.SPLIT_SEED}
    cells = pd.read_csv(runs['two'].parent/'extraction'/'part_00000_00002'/'expected_half_cells.tsv', sep='\t')
    assert set(cells.half) == set(extract.HALVES)


def test_prepared_outputs_do_not_depend_on_partitioning(runs):
    for name in ['frozen_genes.tsv', 'queries.tsv', 'endpoint/keys.tsv', 'endpoint/vehicle_null.tsv',
                 'count_qc/context_summary.tsv']:
        left = pd.read_csv(runs['one']/name, sep='\t')
        right = pd.read_csv(runs['two']/name, sep='\t')
        pd.testing.assert_frame_equal(left, right)
    for name in ['train_truth.npz', 'validation_truth.npz', 'endpoint/de_sets.npz']:
        a, b = npz(runs['one']/name), npz(runs['two']/name)
        assert a.keys() == b.keys()
        for key in a:
            assert np.array_equal(a[key], b[key]) or np.allclose(a[key], b[key]), f'{name}:{key} differs'


def test_de_sets_come_from_the_selection_half_and_stay_inside_the_frozen_genes(runs):
    sets = npz(runs['two']/'endpoint/de_sets.npz')
    frozen = pd.read_csv(runs['two']/'frozen_genes.tsv', sep='\t')
    k = int(sets['k'])
    assert sets['genes'].shape == (len(sets['keys']), k)
    assert sets['genes'].min() >= 0 and sets['genes'].max() < len(frozen)
    assert np.array_equal(sets['genes'], endpoint.de_gene_sets(sets['selection_delta'], k=k)), \
        'the stored sets must be the top-k of the stored selection-half delta'
    assert (np.diff(sets['genes'], axis=1) > 0).all(), 'gene columns are sorted and distinct'


def test_no_evaluation_half_test_delta_is_computed_in_preparation(runs):
    info = receipt(runs['two'])
    assert info['evaluation_half_test_delta_computed'] is False and info['test_outcomes_computed'] is False
    keys = pd.read_csv(runs['two']/'endpoint/keys.tsv', sep='\t')
    assert set(keys.split) <= {'train', 'validation', 'test'} and keys.split.eq('test').any()
    validation = npz(runs['two']/'validation_truth.npz')
    assert 'delta_evaluation' in validation, 'validation needs the evaluation half its endpoint uses'
    train = npz(runs['two']/'train_truth.npz')
    assert 'delta_evaluation' not in train, 'training targets use whole units'


def test_the_responder_null_is_between_wells_and_sets_the_threshold(runs):
    null = pd.read_csv(runs['two']/'endpoint/vehicle_null.tsv', sep='\t')
    info = receipt(runs['two'])['responder']
    assert len(null) == info['nulls'] >= 1
    assert (null.unit_a != null.unit_b).all(), 'the null compares two different vehicle wells'
    assert null.unit_a.str.contains('_v').all() and null.unit_b.str.contains('_v').all()
    assert info['threshold'] == pytest.approx(np.percentile(null.statistic.to_numpy(), 95.0))
    keys = pd.read_csv(runs['two']/'endpoint/keys.tsv', sep='\t')
    assert (keys.responder == (keys.statistic > info['threshold'])).all()


def test_a_perturbed_compound_is_called_a_responder_and_a_quiet_one_is_not(runs):
    """The synthetic effects differ per compound, so the gate must separate at least some of them."""
    keys = pd.read_csv(runs['two']/'endpoint/keys.tsv', sep='\t')
    assert keys.responder.any() and not keys.responder.all(), \
        f'the gate should separate compounds, got {keys.responder.sum()} of {len(keys)}'
    loud = keys[keys.responder].statistic.min()
    quiet = keys[~keys.responder].statistic.max()
    assert loud > quiet


def test_the_statistic_does_not_track_cell_number(runs):
    keys = pd.read_csv(runs['two']/'endpoint/keys.tsv', sep='\t')
    assert keys.selection_cells.nunique() > 5
    correlation = np.corrcoef(keys.selection_cells, keys.statistic)[0, 1]
    assert abs(correlation) < 0.6, f'standardizing should break the size dependence, got r={correlation:.2f}'


def test_count_qc_requires_both_halves_and_the_context_gate_counts_responders(runs):
    replicate = pd.read_csv(runs['two']/'count_qc/replicate_qc.tsv', sep='\t')
    columns = [f'{side}_cells_{half}' for side in ('treatment', 'vehicle') for half in extract.HALVES]
    assert set(columns) <= set(replicate.columns)
    assert (replicate.loc[replicate.replicate_passes, columns] >= prepare.MIN_CELLS).all().all()
    small = replicate[replicate.treatment_unit.eq('plate2_C12:L2')]  # the fixture's 12-cell unit
    assert not small.empty and not small.replicate_passes.any(), 'the 12-cell unit cannot pass a half gate'
    assert (small.min_half_cells < prepare.MIN_CELLS).all()
    summary = pd.read_csv(runs['two']/'count_qc/context_summary.tsv', sep='\t')
    assert 'responder_test_compounds' in summary
    assert (summary.responder_test_compounds <= summary.eligible_test_compounds).all()


def test_half_deltas_match_a_manual_computation(runs):
    """One key's selection-half delta, recomputed from the stored half sums."""
    sets = npz(runs['two']/'endpoint/de_sets.npz')
    keys = [tuple(k.split('|')) for k in sets['keys']]
    frozen = pd.read_csv(runs['two']/'frozen_genes.tsv', sep='\t')
    parts = sorted((runs['two'].parent/'extraction').glob('part_*'))
    halves = pd.read_csv(parts[0]/'half_units.tsv', sep='\t')
    units = pd.read_csv(parts[0]/'units.tsv', sep='\t')
    shards = pd.read_csv(runs['two'].parent/'manifests'/'expression_source.tsv', sep='\t')
    expected = pd.read_csv(parts[0]/'expected_half_cells.tsv', sep='\t')
    selection = pd.read_csv(parts[0]/'model_selection.tsv', sep='\t')
    merged = extract.merge_parts(parts, halves, expected_cells=expected.expected_cells.to_numpy(np.int64),
                                 model_barcodes=selection.barcode.to_numpy(str), cap=CAP,
                                 shard_manifest=shards[shards.role.eq('expression_shard')])
    tables = prepare.half_tables(merged['half_ids'], merged['half_counts'], merged['half_library'],
                                 merged['half_cells'], units)
    queries = pd.read_csv(runs['two']/'queries.tsv', sep='\t')
    pairs = queries.rename(columns=dict(treatment_Combo='treatment_unit', vehicle_Combo='vehicle_unit'))
    key = keys[0]
    rows = pairs[pairs.compound_id.eq(key[0]) & pairs.context_id.eq(key[1])]
    table = tables['S']
    columns = frozen.source_index.to_numpy()
    manual = np.mean([log_cpm(table['counts'][[table['row_of'][r.treatment_unit]]][:, columns].toarray(),
                              table['library'][[table['row_of'][r.treatment_unit]]])
                      -log_cpm(table['counts'][[table['row_of'][r.vehicle_unit]]][:, columns].toarray(),
                               table['library'][[table['row_of'][r.vehicle_unit]]])
                      for r in rows.itertuples(index=False)], axis=0)
    stored = sets['selection_delta'][keys.index(key)]
    assert np.allclose(manual.ravel(), stored, atol=1e-10)


def test_extraction_anchors_catch_a_half_unit_order_change(runs):
    parts = sorted((runs['two'].parent/'extraction').glob('part_*'))
    halves = pd.read_csv(parts[0]/'half_units.tsv', sep='\t')
    expected = pd.read_csv(parts[0]/'expected_half_cells.tsv', sep='\t')
    selection = pd.read_csv(parts[0]/'model_selection.tsv', sep='\t')
    shards = pd.read_csv(runs['two'].parent/'manifests'/'expression_source.tsv', sep='\t')
    with pytest.raises(ValueError, match='unit order differs'):
        extract.merge_parts(parts, halves.iloc[::-1].reset_index(drop=True),
                            expected_cells=expected.expected_cells.to_numpy(np.int64),
                            model_barcodes=selection.barcode.to_numpy(str), cap=CAP,
                            shard_manifest=shards[shards.role.eq('expression_shard')])


def test_model_cells_are_capped_per_unit_not_per_half(runs):
    selection = pd.read_csv(runs['two'].parent/'extraction'/'part_00000_00002'/'model_selection.tsv', sep='\t')
    parent = selection.unit_id.str.rsplit(':', n=1).str[0]
    assert parent.value_counts().max() <= CAP, 'the cap is a cap per unit'
    assert selection.unit_id.str.endswith(('S', 'E')).all(), 'a model cell records which half it fell in'
    roles = receipt(runs['two'])['model_cell_roles']
    assert set(roles) <= set(v5_extract.MODEL_ROLES) and 'test' not in roles


def test_leave_one_out_plate_means_remove_only_the_named_compound():
    replicates = pd.DataFrame(dict(context_id=['c']*6, plate=['p']*6,
                                   compound_id=['a', 'b', 'c', 'd', 'e', 'f']))
    delta = np.arange(6.0).reshape(6, 1)
    means, loo, counts = prepare.leave_one_out_plate_means(replicates, delta, min_compounds=5)
    assert counts[('c', 'p')] == 6 and means[('c', 'p')] == pytest.approx(2.5)
    assert loo[('c', 'p', 'a')] == pytest.approx(np.mean([1, 2, 3, 4, 5]))
    assert loo[('c', 'p', 'f')] == pytest.approx(np.mean([0, 1, 2, 3, 4]))
    thin = prepare.leave_one_out_plate_means(replicates.iloc[:5], delta[:5], min_compounds=5)
    assert thin[0] and not thin[1], 'with five compounds a plate mean exists but no leave-one-out mean does'


def test_residual_targets_report_the_keys_without_a_reference():
    keys = [('a', 'c'), ('b', 'c')]
    delta = np.array([[1.0, 2.0], [3.0, 4.0]])
    residual, kept, missing = prepare.residual_targets(keys, delta, {('a', 'c'): np.array([1.0, 1.0])})
    assert kept == [('a', 'c')] and missing == [('b', 'c')]
    assert residual.tolist() == [[0.0, 1.0]]


def sealed(prepared):
    return json.loads((prepared.parent/'sealed'/'baselines_v6_seal.json').read_text())


def test_sealed_comparators_cover_delta_and_residual_space(runs):
    archive = npz(runs['two'].parent/'sealed'/'baselines_v6_predictions.npz')
    for method in ['B0', 'B1c', 'B2c', 'M1c', 'B1a', 'M1a', 'B1p', 'B1r', 'B2r', 'M1r', 'M1r_unscaled']:
        assert method in archive, f'{method} must be sealed'
    # The diagnostics need a validation prediction from every method that has one; B0 and B2c have none, as in v5.
    for method in ['B1c', 'M1c', 'B1a', 'M1a', 'B1p', 'B1r', 'B2r', 'M1r']:
        assert f'validation_{method}' in archive, f'validation_{method} must be sealed'
    assert np.abs(archive['B0']).max() == 0.0, 'B0 is the zero prediction'
    assert len(archive['keys']) == len(archive['B1p'])


def test_residual_comparators_reduce_to_b1p_plus_a_residual(runs):
    archive = npz(runs['two'].parent/'sealed'/'baselines_v6_predictions.npz')
    keys = list(archive['keys'])
    b1p = {k: v for k, v in zip(keys, archive['B1p'])}
    for method in ['B1r', 'B2r', 'M1r']:
        present = list(archive[f'{method}_keys'])
        assert present, f'{method} should cover at least one key'
        for key, value in zip(present, archive[method]):
            assert not np.allclose(value, 0.0), 'a residual comparator is B1p plus a residual, not a residual'
            assert value.shape == b1p[key].shape
    info = sealed(runs['two'])['residual']
    assert info['reference'].startswith('B1p')
    scales = [v['scale'] for v in info['per_context'].values() if v['fitted']]
    assert scales and all(0.0 <= s <= 1.0 for s in scales), f'scales must be clipped to [0, 1], got {scales}'


def test_scale_factor_is_least_squares_clipped_to_the_unit_interval():
    import importlib.util
    spec = importlib.util.spec_from_file_location('v6b', REPO/'scripts/v6_baselines.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.scale_factor(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(1.0)
    assert module.scale_factor(np.array([1.0, 2.0]), np.array([0.5, 1.0])) == pytest.approx(0.5)
    assert module.scale_factor(np.array([1.0, 2.0]), np.array([-1.0, -2.0])) == 0.0, 'a wrong sign gives up'
    assert module.scale_factor(np.zeros(3), np.ones(3)) == 0.0


def test_prepare_constants_match_the_protocol_text():
    protocol = yaml.safe_load((REPO/'configs/protocol_v6.yaml').read_text())
    text = protocol['count_qc']['rule']
    assert str(prepare.MIN_CELLS) in text and str(prepare.MIN_WELL_CELLS) in text
    assert str(prepare.MIN_VEHICLE_CELLS) in text
    assert str(prepare.MIN_TEST_RESPONDERS) in protocol['count_qc']['context_gate']
    assert protocol['status'] == 'PRESPECIFIED_BEFORE_V6_FITTING'
    assert 'A1_2026-10-26' in protocol['amendments'], 'the responder-null amendment must be logged'
    assert 'A2_2026-10-26' in protocol['amendments'], 'the training-value amendment must be logged'


def test_the_training_schedule_equals_v5s():
    """Amendment A2 spelled these out as keys; a test pins them to v5's values."""
    v5 = yaml.safe_load((REPO/'configs/protocol_v5.yaml').read_text())['training']
    v6 = yaml.safe_load((REPO/'configs/protocol_v6.yaml').read_text())['training']
    for key in ['max_epochs', 'evaluate_every_epochs', 'early_stopping_patience_evaluations', 'checkpoint',
                'max_minutes_per_fit']:
        assert v6[key] == v5[key], f'{key} must equal v5: {v6[key]} != {v5[key]}'
    assert v6['seeds']['candidates'] == v5['seeds']['candidates']
