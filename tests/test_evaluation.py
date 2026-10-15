"""Hand-computable synthetic checks; not a scientific evaluation dataset."""
import numpy as np
import pytest

from drugperturb.evaluation import (
    compound_metrics, paired_scaffold_bootstrap, summarize_compound_metrics,
)


def test_main_rmse_is_mean_of_compound_rmse_not_pooled_gene_rmse():
    truth = np.zeros((2, 2))
    predicted = np.array([[0., 0.], [6., 8.]])
    table = compound_metrics(truth, predicted, ['a', 'b'], ['A', 'B'])
    np.testing.assert_allclose(table['rmse'], [0., np.sqrt(50)])
    np.testing.assert_allclose(table['mae'], [0., 7.])
    summary = summarize_compound_metrics(table)
    np.testing.assert_allclose(summary['mean_compound_rmse'], np.sqrt(50) / 2)
    assert not np.isclose(summary['mean_compound_rmse'], 5.)
    assert summary['n_compounds'] == 2
    assert summary['n_groups'] == 2


def test_spearman_ties_use_average_ranks_without_treating_ties_as_ordered():
    table = compound_metrics(
        [[1., 1., 2., 3.]], [[1., 2., 2., 3.]], ['a'], ['A'],
    )
    np.testing.assert_allclose(table.loc[0, 'spearman'], 5 / 6)


def test_zero_centered_cosine_does_not_subtract_vector_means():
    table = compound_metrics([[1., 2., 3.]], [[2., 3., 4.]], ['a'], ['A'])
    np.testing.assert_allclose(table.loc[0, 'cosine_zero'], 20 / np.sqrt(406))
    assert table.loc[0, 'cosine_zero'] < 1


def test_constant_spearman_zero_norm_cosine_and_empty_sign_have_explicit_na_counts():
    table = compound_metrics(
        [[0., 0., 0.], [1., 1., 1.]], [[1., -1., 0.], [2., 3., 4.]],
        ['zero_truth', 'constant_truth'], ['A', 'B'],
    )
    assert table['spearman'].isna().all()
    assert np.isnan(table.loc[0, 'cosine_zero'])
    assert np.isfinite(table.loc[1, 'cosine_zero'])
    assert np.isnan(table.loc[0, 'sign_agreement'])
    assert table.loc[0, 'n_sign_genes'] == 0
    summary = summarize_compound_metrics(table)
    assert summary['mean_spearman'] is None
    assert summary['n_spearman_valid'] == 0
    assert summary['n_spearman_undefined'] == 2
    assert summary['n_cosine_zero_valid'] == 1
    assert summary['n_cosine_zero_undefined'] == 1
    assert summary['n_sign_agreement_valid'] == 1
    assert summary['n_sign_agreement_undefined'] == 1


def test_sign_agreement_excludes_exact_zero_truth_unless_explicitly_requested():
    truth, predicted = [[-1., 0., 2.]], [[1., 0., -2.]]
    nonzero = compound_metrics(truth, predicted, ['a'], ['A'])
    all_genes = compound_metrics(truth, predicted, ['a'], ['A'], sign_nonzero_only=False)
    assert nonzero.loc[0, 'sign_agreement'] == 0
    assert nonzero.loc[0, 'n_sign_genes'] == 2
    np.testing.assert_allclose(all_genes.loc[0, 'sign_agreement'], 1 / 3)
    assert all_genes.loc[0, 'n_sign_genes'] == 3


def test_paired_scaffold_bootstrap_weights_compounds_and_retains_cluster_multiplicity():
    # A: three differences of -1; B: one difference of +3.
    # Point mean = 0. AA draws -> -1; AB/BA -> 0; BB -> +3.
    # Equal weighting of group means would incorrectly give point=1 and AB=1.
    truth = np.zeros((4, 2))
    candidate = np.array([[0., 0.], [0., 0.], [0., 0.], [4., 4.]])
    baseline = np.ones((4, 2))
    result = paired_scaffold_bootstrap(truth, candidate, baseline,
                                       ['a', 'b', 'c', 'd'], ['A', 'A', 'A', 'B'])
    assert result['point_estimate'] == 0
    np.testing.assert_array_equal(np.unique(result['bootstrap_differences']), [-1., 0., 3.])
    assert result['ci_lower'] == -1
    assert result['ci_upper'] == 3
    assert result['n_compounds'] == 4
    assert result['n_groups'] == 2
    assert result['draws'] == 2000
    assert result['seed'] == 2026
    assert result['comparison'] == 'candidate_minus_baseline_rmse'
    assert result['resampling_unit'] == 'scaffold_group'


def test_identical_models_have_exactly_zero_paired_difference_and_interval():
    truth = np.zeros((3, 2))
    prediction = np.array([[1., 1.], [3., 3.], [7., 7.]])
    result = paired_scaffold_bootstrap(truth, prediction, prediction,
                                       ['a', 'b', 'c'], ['A', 'B', 'C'])
    assert result['point_estimate'] == 0
    assert result['ci_lower'] == result['ci_upper'] == 0
    assert np.all(result['bootstrap_differences'] == 0)


def test_bootstrap_is_reproducible_and_independent_of_input_row_order():
    truth = np.zeros((4, 2))
    candidate = np.array([[0., 0.], [1., 1.], [4., 4.], [2., 2.]])
    baseline = np.array([[1., 1.], [4., 4.], [2., 2.], [3., 3.]])
    ids, groups = np.array(['a', 'b', 'c', 'd']), np.array(['A', 'A', 'B', 'C'])
    first = paired_scaffold_bootstrap(truth, candidate, baseline, ids, groups)
    second = paired_scaffold_bootstrap(truth, candidate, baseline, ids, groups)
    order = [3, 1, 0, 2]
    shuffled = paired_scaffold_bootstrap(truth[order], candidate[order], baseline[order],
                                         ids[order], groups[order])
    np.testing.assert_array_equal(first['bootstrap_differences'], second['bootstrap_differences'])
    np.testing.assert_array_equal(first['bootstrap_differences'], shuffled['bootstrap_differences'])
    assert first['point_estimate'] == shuffled['point_estimate']


@pytest.mark.parametrize('change', [
    'truth_nan', 'prediction_inf', 'shape', 'not_matrix', 'duplicate_ids',
    'id_length', 'group_length', 'missing_group', 'empty_rows', 'empty_genes',
])
def test_invalid_metric_inputs_are_rejected_instead_of_silently_dropping_rows(change):
    truth, predicted = np.zeros((2, 2)), np.ones((2, 2))
    ids, groups = ['a', 'b'], ['A', 'B']
    if change == 'truth_nan': truth[0, 0] = np.nan
    elif change == 'prediction_inf': predicted[0, 0] = np.inf
    elif change == 'shape': predicted = np.ones((2, 3))
    elif change == 'not_matrix': truth = [0., 0.]
    elif change == 'duplicate_ids': ids = ['a', 'a']
    elif change == 'id_length': ids = ['a']
    elif change == 'group_length': groups = ['A']
    elif change == 'missing_group': groups = ['A', None]
    elif change == 'empty_rows': truth = predicted = np.zeros((0, 2)); ids = groups = []
    elif change == 'empty_genes': truth = predicted = np.zeros((2, 0))
    with pytest.raises(ValueError):
        compound_metrics(truth, predicted, ids, groups)


def test_bootstrap_rejects_one_cluster_and_nonfinite_baseline():
    x = np.zeros((2, 2))
    with pytest.raises(ValueError, match='two'):
        paired_scaffold_bootstrap(x, x, x, ['a', 'b'], ['A', 'A'])
    bad = x.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        paired_scaffold_bootstrap(x, x, bad, ['a', 'b'], ['A', 'B'])
