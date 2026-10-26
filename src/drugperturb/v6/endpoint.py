"""v6 endpoint: DE gene sets, the responder statistic and direction accuracy.

The protocol scores each (compound, context) key on its own differentially
expressed genes, chosen from the selection half and scored against the
evaluation half, so the two share no measurement noise (protocol_v6
measurement_split).

- ``de_gene_sets`` picks the ``k`` frozen genes with the largest selection-half
  |delta| per key.
- ``response_statistic`` standardizes a pooled delta by its own sampling
  standard error, so its size does not depend on how many cells the key holds.
  The threshold is calibrated empirically against vehicle-versus-vehicle pairs
  (amendment A1), which carry real well-to-well variation rather than Poisson
  theory alone.
- ``direction_accuracy`` is the primary metric: over a key's DE genes, the
  fraction whose predicted sign equals the evaluation-half sign. A predicted
  zero counts as a half, so zero change scores exactly 0.5.

Nothing here reads a test response: the caller passes the arrays it is entitled
to, and the evaluation half of a test key reaches this module only inside the
evaluation kernel, after every prediction is sealed.
"""
from __future__ import annotations

import numpy as np

# protocol_v6 response_endpoint (prose values; tests check them against the protocol text)
DE_K = 50
DE_K_SENSITIVITY = (20, 100)
RESPONDER_PERCENTILE = 95.0
COUNT_FLOOR = 1.0


def log_cpm_standard_error(counts, library, *, floor=COUNT_FLOOR):
    """Delta-method Poisson standard error of log1p CPM, per unit row and gene.

    With ``y = log1p(c/L*1e6)`` and ``Var(c) ~ c`` the delta method gives
    ``sd(y) = (1e6/L) * sqrt(max(c, floor)) / (1 + c/L*1e6)``. The floor keeps a
    gene with no observed count from claiming zero uncertainty; it does not make
    the error exact, which is why the responder threshold is calibrated on
    measured vehicle pairs rather than taken from this formula.
    """
    counts = np.asarray(counts, dtype=np.float64)
    library = np.asarray(library, dtype=np.float64)
    if counts.ndim != 2 or library.shape != (counts.shape[0],):
        raise ValueError('counts must be (units, genes) with one library per unit')
    if (library <= 0).any() or not np.isfinite(library).all():
        raise ValueError('libraries must be finite and positive')
    scale = (1e6/library)[:, None]
    return scale*np.sqrt(np.maximum(counts, floor))/(1.0+counts*scale)


def delta_standard_error(treated_counts, treated_library, vehicle_counts, vehicle_library, *, floor=COUNT_FLOOR):
    """Standard error of a treated-minus-vehicle log1p CPM delta; the halves are disjoint cell sets."""
    treated = log_cpm_standard_error(treated_counts, treated_library, floor=floor)
    vehicle = log_cpm_standard_error(vehicle_counts, vehicle_library, floor=floor)
    return np.sqrt(treated**2+vehicle**2)


def response_statistic(delta, standard_error):
    """Root-mean-square standardized deviation of a delta; one value per row.

    Standardizing by each gene's own sampling error makes the statistic
    approximately free of the cell numbers behind the delta, which is what the
    protocol's cell-count matching was for. Pure sampling noise gives about 1.
    """
    delta = np.atleast_2d(np.asarray(delta, dtype=np.float64))
    standard_error = np.atleast_2d(np.asarray(standard_error, dtype=np.float64))
    if delta.shape != standard_error.shape:
        raise ValueError('delta and standard error must have the same shape')
    if (standard_error <= 0).any():
        raise ValueError('standard errors must be positive')
    return np.sqrt(np.mean((delta/standard_error)**2, axis=1))


def responder_threshold(null_statistics, *, percentile=RESPONDER_PERCENTILE):
    """The responder cut: a percentile of the vehicle-versus-vehicle statistics."""
    null_statistics = np.asarray(null_statistics, dtype=np.float64)
    if null_statistics.size == 0:
        raise ValueError('the responder threshold needs at least one null statistic')
    return float(np.percentile(null_statistics, percentile))


def de_gene_sets(delta, *, k=DE_K):
    """Per row, the ``k`` gene columns with the largest |delta|; ties break by column order.

    Returns an integer array of shape (rows, k) with each row's columns sorted
    ascending, so a set is a stable, readable list of gene positions.
    """
    delta = np.atleast_2d(np.asarray(delta, dtype=np.float64))
    if k <= 0 or k > delta.shape[1]:
        raise ValueError(f'k must be in 1..{delta.shape[1]}, got {k}')
    order = np.lexsort((np.arange(delta.shape[1])[None, :].repeat(len(delta), axis=0), -np.abs(delta)), axis=1)
    return np.sort(order[:, :k], axis=1)


def direction_accuracy(prediction, truth, gene_sets):
    """Per row, the fraction of its DE genes whose predicted sign matches the truth's.

    A predicted zero scores 0.5, so a zero-change prediction scores exactly 0.5
    wherever the truth is nonzero. A gene whose truth is exactly zero carries no
    direction and is dropped; the count of dropped genes is returned. A row left
    with no gene scores NaN.
    """
    prediction = np.atleast_2d(np.asarray(prediction, dtype=np.float64))
    truth = np.atleast_2d(np.asarray(truth, dtype=np.float64))
    gene_sets = np.atleast_2d(np.asarray(gene_sets, dtype=np.int64))
    if prediction.shape != truth.shape or len(gene_sets) != len(truth):
        raise ValueError('prediction, truth and gene sets must describe the same rows')
    rows = np.arange(len(truth))[:, None]
    predicted, measured = prediction[rows, gene_sets], truth[rows, gene_sets]
    scored = np.sign(measured) != 0
    agreement = np.where(np.sign(predicted) == 0, 0.5, (np.sign(predicted) == np.sign(measured)).astype(float))
    counts = scored.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        accuracy = np.where(counts > 0, np.where(scored, agreement, 0.0).sum(axis=1)/counts, np.nan)
    return accuracy, int((~scored).sum())


def restricted_rmse(prediction, truth, gene_sets):
    """Per row, RMSE over its DE genes only; the secondary metric."""
    prediction = np.atleast_2d(np.asarray(prediction, dtype=np.float64))
    truth = np.atleast_2d(np.asarray(truth, dtype=np.float64))
    gene_sets = np.atleast_2d(np.asarray(gene_sets, dtype=np.int64))
    if prediction.shape != truth.shape or len(gene_sets) != len(truth):
        raise ValueError('prediction, truth and gene sets must describe the same rows')
    rows = np.arange(len(truth))[:, None]
    return np.sqrt(np.mean((prediction[rows, gene_sets]-truth[rows, gene_sets])**2, axis=1))


def de_overlap(prediction, gene_sets, *, k=None):
    """Jaccard overlap between each row's measured DE set and its predicted top-k genes."""
    gene_sets = np.atleast_2d(np.asarray(gene_sets, dtype=np.int64))
    k = gene_sets.shape[1] if k is None else k
    predicted = de_gene_sets(prediction, k=k)
    return np.asarray([len(set(a) & set(b))/len(set(a) | set(b)) for a, b in zip(gene_sets, predicted)])
