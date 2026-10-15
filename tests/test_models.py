"""Synthetic software tests only: no biological performance claims."""
import numpy as np
import pytest
import torch
from torch import nn

from drugperturb.models import MLP, MolecularGNN, fit_network, predict_network


def _graph(values=(0.06, 0.06, 0.08), bond_column=0):
    x = np.zeros((len(values), 10), dtype=np.float32)
    x[:, 0] = values
    if len(values) == 1:
        edges = np.empty((2, 0), dtype=np.int64)
    else:
        edges = np.asarray([(i, j) for i in range(len(values) - 1)
                            for j in (i + 1,)], dtype=np.int64).T
        edges = np.concatenate((edges, edges[::-1]), axis=1)
    attrs = np.zeros((edges.shape[1], 6), dtype=np.float32)
    attrs[:, bond_column] = 1
    return {'x': x, 'edge_index': edges, 'edge_attr': attrs}


def _synthetic():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(64, 6)).astype(np.float32)
    y = np.column_stack((2 * x[:, 0] - x[:, 1], x[:, 2] + 0.5 * x[:, 3]))
    return x[:48], y[:48], x[48:], y[48:]


def test_mlp_predicts_correct_shape_and_disables_dropout_for_prediction():
    torch.manual_seed(3)
    model = MLP(6, 2, hidden_dim=12, dropout=0.5)
    x = np.ones((5, 6), dtype=np.float32)
    first, second = predict_network(model, x), predict_network(model, x)
    assert first.shape == (5, 2)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_gnn_prediction_is_invariant_to_node_and_edge_order():
    torch.manual_seed(11)
    model = MolecularGNN(output_dim=2, hidden_dim=12, dropout=0)
    graph = _graph()
    permutation = np.array([2, 0, 1])
    inverse = np.argsort(permutation)
    reordered = {
        'x': graph['x'][permutation],
        'edge_index': inverse[graph['edge_index']][:, ::-1].copy(),
        'edge_attr': graph['edge_attr'][::-1].copy(),
    }
    np.testing.assert_allclose(
        predict_network(model, [graph]), predict_network(model, [reordered]),
        atol=1e-6, rtol=1e-6,
    )


def test_gnn_batch_prediction_matches_separate_graphs_and_handles_no_bonds():
    torch.manual_seed(12)
    model = MolecularGNN(output_dim=3, hidden_dim=10, dropout=0)
    graphs = [_graph(), _graph((0.07,)), _graph((0.08, 0.07))]
    batched = predict_network(model, graphs)
    separate = np.concatenate([predict_network(model, [g]) for g in graphs])
    assert batched.shape == (3, 3)
    assert np.isfinite(batched).all()
    np.testing.assert_allclose(batched, separate, atol=1e-6, rtol=1e-6)


def test_gnn_uses_bond_identity_in_message_passing():
    torch.manual_seed(13)
    model = MolecularGNN(output_dim=2, hidden_dim=12, dropout=0)
    single = predict_network(model, [_graph(bond_column=0)])
    double = predict_network(model, [_graph(bond_column=1)])
    assert not np.allclose(single, double, atol=1e-7)


def test_gnn_rejects_out_of_bounds_edge_instead_of_connecting_different_graphs():
    graph = _graph()
    graph['edge_index'][0, 0] = 3
    model = MolecularGNN(output_dim=1)
    with pytest.raises(ValueError, match='edge'):
        predict_network(model, [graph, _graph()])


def test_training_reduces_real_synthetic_regression_loss():
    x, y, vx, vy = _synthetic()
    result = fit_network(
        lambda: MLP(6, 2, hidden_dim=32, dropout=0), x, y, vx, vy,
        seed=7, lr=0.02, max_epochs=120, patience=25, batch_size=16,
    )
    assert result['best_val_mse'] < result['history'][0]['val_mse'] * 0.25
    assert result['best_val_mse'] < 0.2
    assert 1 <= result['best_epoch'] <= len(result['history'])
    predicted = predict_network(result['model'], vx)
    assert predicted.shape == vy.shape
    assert np.isfinite(predicted).all()
    np.testing.assert_allclose(np.mean((predicted - vy) ** 2), result['best_val_mse'], rtol=1e-5)
    assert all(np.isfinite(r['train_mse']) and np.isfinite(r['val_mse']) for r in result['history'])
    assert all(r['epoch_seconds'] >= 0 for r in result['history'])


def test_same_seed_repeats_initialization_shuffling_dropout_and_best_model():
    x, y, vx, vy = _synthetic()
    kwargs = dict(seed=23, lr=0.01, max_epochs=20, patience=6, batch_size=12)
    factory = lambda: MLP(6, 2, hidden_dim=12, dropout=0.2)
    first = fit_network(factory, x, y, vx, vy, **kwargs)
    torch.manual_seed(9876)
    np.random.seed(9876)
    second = fit_network(factory, x, y, vx, vy, **kwargs)
    np.testing.assert_array_equal(
        predict_network(first['model'], vx), predict_network(second['model'], vx),
    )
    assert first['best_epoch'] == second['best_epoch']
    assert [(r['train_mse'], r['val_mse']) for r in first['history']] == [
        (r['train_mse'], r['val_mse']) for r in second['history']
    ]


def test_validation_early_stop_restores_best_weights_instead_of_last_epoch():
    x = np.array([[-1.], [1.]], dtype=np.float32)
    def factory():
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        return model
    result = fit_network(
        factory, x, 2 * x, x, -2 * x,
        seed=3, lr=0.1, max_epochs=20, patience=3, batch_size=2,
    )
    assert result['best_epoch'] == 1
    assert len(result['history']) == 4
    assert result['history'][-1]['val_mse'] > result['best_val_mse']
    actual = np.mean((predict_network(result['model'], x) + 2 * x) ** 2)
    np.testing.assert_allclose(actual, result['best_val_mse'], rtol=1e-6)


def test_graph_network_can_train_with_shared_trainer_and_finite_gradients():
    graphs = [_graph((0.01 * i,)) for i in range(1, 13)]
    target = np.arange(1, 13, dtype=np.float32).reshape(-1, 1) / 10
    result = fit_network(
        lambda: MolecularGNN(output_dim=1, hidden_dim=12, dropout=0),
        graphs[:8], target[:8], graphs[8:], target[8:],
        seed=5, lr=0.02, max_epochs=80, patience=15, batch_size=4,
    )
    assert result['best_val_mse'] < result['history'][0]['val_mse']
    assert np.isfinite(predict_network(result['model'], graphs)).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in result['model'].parameters())


@pytest.mark.parametrize('bad', ['inputs', 'targets'])
def test_trainer_rejects_nonfinite_data_before_optimization(bad):
    x, y, vx, vy = _synthetic()
    if bad == 'inputs':
        x[0, 0] = np.nan
    else:
        y[0, 0] = np.inf
    with pytest.raises(ValueError, match='finite'):
        fit_network(lambda: MLP(6, 2), x, y, vx, vy, seed=2,
                    lr=0.01, max_epochs=2, patience=1, batch_size=16)


def test_custom_validation_score_controls_selection_even_when_coordinate_mse_improves():
    x = np.array([[-1.], [1.]], dtype=np.float32)
    def factory():
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        return model
    scorer = lambda predicted: float(np.mean((predicted + 2 * x) ** 2))
    result = fit_network(
        factory, x, 2 * x, x, 2 * x,
        seed=3, lr=0.1, max_epochs=20, patience=3, batch_size=2,
        validation_scorer=scorer,
    )
    assert result['best_epoch'] == 1
    assert len(result['history']) == 4
    assert result['history'][-1]['val_mse'] < result['history'][0]['val_mse']
    assert result['history'][-1]['validation_score'] > result['best_val_score']
    restored_score = scorer(predict_network(result['model'], x))
    np.testing.assert_allclose(restored_score, result['best_val_score'], rtol=1e-6)
