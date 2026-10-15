"""Small synthetic fixtures test leakage boundaries, not scientific performance."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('development_diagnostics', Path(__file__).resolve().parents[1] / 'scripts/development_diagnostics.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_epoch_budget_uses_only_selected_candidate_and_three_expected_seeds():
    runs = pd.DataFrame([
        {'method': 'M3', 'candidate': 1, 'seed': s, 'best_epoch': e} for s, e in [(42, 9), (123, 2), (2026, 5)]
    ] + [{'method': 'M3', 'candidate': 0, 'seed': 42, 'best_epoch': 1000},
         {'method': 'M2', 'candidate': 1, 'seed': 42, 'best_epoch': 500}])
    assert diag.selected_epoch_budget(runs, 1, [42, 123, 2026])[0] == 5
    with pytest.raises(ValueError, match='exactly'):
        diag.selected_epoch_budget(runs.iloc[1:], 1, [42, 123, 2026])


def test_nested_learning_curve_is_reproducible_without_using_labels():
    first = diag.nested_subsets(17)
    second = diag.nested_subsets(17)
    assert [len(index) for _, index in first] == [5, 9, 17]
    for (_, a), (_, b) in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert set(first[0][1]) < set(first[1][1]) < set(first[2][1])


def test_model_fit_has_no_query_labels_and_truth_is_not_pca_smoothed():
    graph = {'x': np.ones((1, 10), dtype=np.float32), 'edge_index': np.empty((2, 0), dtype=np.int64), 'edge_attr': np.empty((0, 6), dtype=np.float32)}
    y = np.array([[0, 2, 7], [1, 0, 7], [3, 1, 7], [5, 4, 7]], dtype=float)
    fp = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=float)
    settings = {'max_genes': 2, 'max_components': 1, 'ridge_alpha': 1.0,
                'gnn_hp': {'hidden_dim': 4, 'dropout': 0.1, 'lr': 0.001}, 'gnn_epochs': 2, 'batch_size': 2, 'seed': 42}
    tr, pred, _, _, history = diag.fit_models(y, fp, [graph] * 4, fp[:1], [graph], settings)
    frozen_prediction = pred['M3'].copy()
    # A gene variable only in held-out data must not enter selection.
    query_y = np.array([[80, -30, 1000000]], dtype=float)
    assert 2 not in tr.gene_indices
    actual = tr.select_truth(query_y)
    np.testing.assert_array_equal(actual, query_y[:, tr.gene_indices])
    assert not np.allclose(actual, tr.inverse_transform(tr.transform(query_y)))
    diag.measured_metrics(actual, pred['M3'])
    diag.measured_metrics(actual * 100, pred['M3'])
    np.testing.assert_array_equal(pred['M3'], frozen_prediction)
    assert [h['epoch'] for h in history] == [1, 2]
    assert all(set(h) == {'epoch', 'online_minibatch_training_MSE'} for h in history)
