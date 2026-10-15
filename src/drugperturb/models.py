"""Small actual PyTorch models and validation-only checkpoint selection.

Pure torch avoids PyG/torch-scatter binary extension compatibility requirements.
The GNN still performs two rounds of bond-aware molecular message passing, then
mean pools atoms. Determinism is scoped to the same runtime/device; cross-device
bitwise equivalence is not promised. Inputs must already have frozen transforms.
This module never constructs splits or accepts final-test data during training.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import os
import platform
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    """Fingerprint -> Linear(hidden=64) -> ReLU -> Dropout(.1) -> Linear(out)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        if min(input_dim, output_dim, hidden_dim) < 1:
            raise ValueError('network dimensions must be positive')
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.architecture = dict(name='fingerprint_mlp', input_dim=input_dim,
                                 output_dim=output_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MolecularGNN(nn.Module):
    """Two bond-aware message-passing layers with neighbor and graph mean pools.

    h0 = ReLU(Linear(atom_features, hidden_dim)). In each of exactly two layers:
    message(u->v) = ReLU(Linear(concat(h_u, bond_features), hidden_dim));
    a_v = mean(incoming messages), or zero for an isolated atom;
    h_v = Dropout(ReLU(Linear(concat(h_v, a_v), hidden_dim))).
    Prediction = Linear(mean(all atom states), output_dim).

    forward accepts a list of chemistry.molecular_graph dictionaries. Node order
    has no meaning, graph boundaries are preserved, and no self loops are added.
    """

    def __init__(self, output_dim: int, atom_dim: int = 10, bond_dim: int = 6,
                 hidden_dim: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if layers != 2:
            raise ValueError('protocol requires exactly two message-passing layers')
        if min(atom_dim, bond_dim, hidden_dim, output_dim) < 1:
            raise ValueError('network dimensions must be positive')
        self.atom_dim, self.bond_dim, self.output_dim = atom_dim, bond_dim, output_dim
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.message_layers = nn.ModuleList([
            nn.Linear(hidden_dim + bond_dim, hidden_dim) for _ in range(layers)
        ])
        self.update_layers = nn.ModuleList([
            nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.architecture = dict(name='bond_aware_molecular_gnn', atom_dim=atom_dim,
                                 bond_dim=bond_dim, output_dim=output_dim,
                                 hidden_dim=hidden_dim, layers=layers, dropout=dropout,
                                 neighbor_pool='mean', graph_pool='mean')

    def forward(self, graphs: Sequence[dict[str, np.ndarray]]) -> torch.Tensor:
        device = next(self.parameters()).device
        x, edges, bonds, graph_ids = _collate_graphs(graphs, self.atom_dim, self.bond_dim, device)
        h = torch.relu(self.atom_projection(x))
        source, destination = edges
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            messages = torch.relu(message_layer(torch.cat((h[source], bonds), dim=1)))
            aggregated = torch.zeros_like(h)
            aggregated.index_add_(0, destination, messages)
            degree = h.new_zeros((len(h), 1))
            degree.index_add_(0, destination, h.new_ones((len(destination), 1)))
            mean_message = aggregated / degree.clamp_min(1)
            h = self.dropout(torch.relu(update_layer(torch.cat((h, mean_message), dim=1))))
        pooled = h.new_zeros((len(graphs), h.shape[1]))
        pooled.index_add_(0, graph_ids, h)
        n_atoms = torch.bincount(graph_ids, minlength=len(graphs)).to(h.dtype).unsqueeze(1)
        return self.readout(pooled / n_atoms)


def _collate_graphs(graphs, atom_dim, bond_dim, device):
    if not graphs:
        raise ValueError('graph input must be nonempty')
    atoms, edges, bonds, membership = [], [], [], []
    offset = 0
    integer_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    for graph_id, graph in enumerate(graphs):
        x = torch.as_tensor(graph['x'], dtype=torch.float32, device=device)
        edge = torch.as_tensor(graph['edge_index'], device=device)
        attr = torch.as_tensor(graph['edge_attr'], dtype=torch.float32, device=device)
        if x.ndim != 2 or x.shape[1] != atom_dim or len(x) == 0:
            raise ValueError(f'atom features must have nonempty shape (N,{atom_dim})')
        if edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype not in integer_types:
            raise ValueError('edge_index must have integer shape (2,E)')
        if attr.shape != (edge.shape[1], bond_dim):
            raise ValueError(f'edge attributes must have shape (E,{bond_dim})')
        if edge.numel() and (edge.min() < 0 or edge.max() >= len(x)):
            raise ValueError('edge index is outside its own graph')
        if not torch.isfinite(x).all() or not torch.isfinite(attr).all():
            raise ValueError('graph features must be finite')
        atoms.append(x)
        edges.append(edge.to(torch.int64) + offset)
        bonds.append(attr)
        membership.append(torch.full((len(x),), graph_id, dtype=torch.int64, device=device))
        offset += len(x)
    return torch.cat(atoms), torch.cat(edges, dim=1), torch.cat(bonds), torch.cat(membership)


def _validate_inputs(inputs, expected_n=None):
    if isinstance(inputs, np.ndarray):
        if inputs.ndim != 2 or not len(inputs):
            raise ValueError('fingerprints must be a nonempty two-dimensional array')
        if not np.isfinite(inputs).all():
            raise ValueError('fingerprints must be finite')
    elif isinstance(inputs, (list, tuple)):
        if not inputs or not all(isinstance(graph, dict) for graph in inputs):
            raise ValueError('graph inputs must be a nonempty list of dictionaries')
    else:
        raise ValueError('inputs must be a NumPy fingerprint array or list of graphs')
    if expected_n is not None and len(inputs) != expected_n:
        raise ValueError('input and target row counts do not match')


def _take_inputs(inputs, indices):
    return inputs[indices] if isinstance(inputs, np.ndarray) else [inputs[int(i)] for i in indices]


def _forward(model, inputs):
    if isinstance(inputs, np.ndarray):
        inputs = torch.as_tensor(inputs, dtype=torch.float32, device=next(model.parameters()).device)
    return model(inputs)


def predict_network(model: nn.Module, inputs, *, batch_size: int = 256) -> np.ndarray:
    """Return finite float32 predictions in input order, with dropout disabled.

    Preserve the caller's train/eval mode. Batching never adds cross-graph edges.
    """
    _validate_inputs(inputs)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    previous_mode = model.training
    model.eval()
    outputs = []
    try:
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch = _take_inputs(inputs, np.arange(start, min(start + batch_size, len(inputs))))
                predicted = _forward(model, batch)
                if predicted.ndim != 2 or predicted.shape[0] != len(batch):
                    raise ValueError('network predictions must have shape (N,outputs)')
                if not torch.isfinite(predicted).all():
                    raise FloatingPointError('network produced nonfinite predictions')
                outputs.append(predicted.detach().cpu().numpy())
    finally:
        model.train(previous_mode)
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _targets(values, name):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError(f'{name} targets must have nonempty shape (N,outputs)')
    if not np.isfinite(values).all():
        raise ValueError(f'{name} targets must be finite')
    return values


def _mse(predicted, target):
    if predicted.shape != target.shape:
        raise ValueError('network prediction and target shapes do not match')
    return float(np.mean((predicted.astype(np.float64) - target.astype(np.float64)) ** 2))


def fit_network(
    model_factory: Callable[[], nn.Module], train_inputs, train_targets,
    val_inputs, val_targets, *, seed: int = 42, lr: float = 1e-3,
    max_epochs: int = 200, patience: int = 20, batch_size: int = 32,
    validation_scorer: Callable[[np.ndarray], float] | None = None,
) -> dict[str, Any]:
    """Train with AdamW; select/restore only by validation score (lower is better).

    Set Python/NumPy/torch seeds BEFORE calling the factory. Cap torch threads at
    two, enable strict deterministic algorithms, shuffle with a seeded generator,
    and stop after patience epochs without strict validation-score improvement.
    Device follows the factory's model; default constructors use CPU.

    val_mse is coordinate-space MSE. If targets are standardized/PCA coordinates,
    supply validation_scorer(predicted_validation_coordinates) whose closure
    inverses the frozen transforms and computes RMSE against actual validation
    gene-space delta. This callback controls both early stopping and checkpoint
    selection; final-test responses must never be included in that closure.

    History records train/val MSE after each epoch in eval mode, validation_score,
    and epoch time. best_val_mse is MSE AT THE SELECTED checkpoint, not necessarily
    the minimum observed coordinate MSE when a custom scorer is supplied.
    """
    y = _targets(train_targets, 'training')
    vy = _targets(val_targets, 'validation')
    _validate_inputs(train_inputs, len(y))
    _validate_inputs(val_inputs, len(vy))
    if y.shape[1] != vy.shape[1]:
        raise ValueError('training and validation output dimensions differ')
    if min(max_epochs, patience, batch_size) < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('epochs, patience, batch_size and learning rate must be positive and finite')
    torch.set_num_threads(min(2, max(1, os.cpu_count() or 1)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = model_factory()
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    order_rng = np.random.default_rng(seed)
    history = []
    best_score, best_val_mse, best_epoch = float('inf'), float('inf'), 0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        model.train()
        order = order_rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            indices = order[start:start + batch_size]
            batch = _take_inputs(train_inputs, indices)
            target = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            predicted = _forward(model, batch)
            if predicted.shape != target.shape:
                raise ValueError('network prediction and target shapes do not match')
            loss = torch.mean((predicted - target) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite optimization loss at epoch {epoch}')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'nonfinite gradient at epoch {epoch}')
            optimizer.step()
        train_prediction = predict_network(model, train_inputs, batch_size=batch_size)
        val_prediction = predict_network(model, val_inputs, batch_size=batch_size)
        train_mse, val_mse = _mse(train_prediction, y), _mse(val_prediction, vy)
        score = float(validation_scorer(val_prediction.copy())) if validation_scorer else val_mse
        if not np.isfinite(score):
            raise FloatingPointError(f'nonfinite validation score at epoch {epoch}')
        history.append(dict(epoch=epoch, train_mse=train_mse, val_mse=val_mse,
                            validation_score=score, epoch_seconds=time.perf_counter() - started))
        if score < best_score:
            best_score, best_val_mse, best_epoch = score, val_mse, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return dict(
        model=model, best_epoch=best_epoch, history=history,
        best_val_mse=best_val_mse, best_val_score=best_score,
        selection_metric='custom_validation_score' if validation_scorer else 'val_mse',
        parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        config=dict(seed=seed, lr=lr, max_epochs=max_epochs, patience=patience,
                    batch_size=batch_size, optimizer='AdamW', weight_decay=1e-4),
        runtime=dict(python=platform.python_version(), numpy=np.__version__,
                     torch=torch.__version__, device=str(device),
                     torch_num_threads=torch.get_num_threads(), deterministic_algorithms=True),
    )
