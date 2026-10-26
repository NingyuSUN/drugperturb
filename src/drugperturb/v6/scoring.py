"""v6 scoring helpers shared by the fit, the diagnostics report and the evaluation.

The endpoint lives in :mod:`drugperturb.v6.endpoint`; this module adapts it to the
pipeline's key-dictionary layout and adds the two quantities the diagnostics need:
the correlation between a predicted and a measured residual, and the split noise
floor - the accuracy ``delta_S`` itself reaches against ``delta_E``, which is the
ceiling any model could reach given the split's measurement noise.
"""
from __future__ import annotations

import numpy as np

from . import endpoint


def stack(mapping, keys):
    return np.vstack([mapping[k] for k in keys]) if keys else np.zeros((0, 0))


def gene_sets(de_sets, keys):
    """Rows of a prepared DE-set archive for the given keys, in that order."""
    position = {tuple(k.split('|')): i for i, k in enumerate(np.asarray(de_sets['keys'], str))}
    missing = [k for k in keys if k not in position]
    if missing:
        raise ValueError(f'no DE set for {len(missing)} keys, first {missing[:3]}')
    return de_sets['genes'][[position[k] for k in keys]].astype(np.int64)


def accuracy_by_key(prediction, truth, de_sets, keys):
    """Direction accuracy per key, as a dict; keys with no scorable gene are dropped."""
    if not keys:
        return {}, 0
    values, dropped = endpoint.direction_accuracy(stack(prediction, keys), stack(truth, keys),
                                                  gene_sets(de_sets, keys))
    return {k: float(v) for k, v in zip(keys, values) if np.isfinite(v)}, dropped


def restricted_rmse_by_key(prediction, truth, de_sets, keys):
    if not keys:
        return {}
    values = endpoint.restricted_rmse(stack(prediction, keys), stack(truth, keys), gene_sets(de_sets, keys))
    return {k: float(v) for k, v in zip(keys, values)}


def mean_accuracy(prediction, truth, de_sets, keys):
    """Mean direction accuracy over keys; NaN when no key is scorable."""
    scored, _ = accuracy_by_key(prediction, truth, de_sets, keys)
    return float(np.mean(list(scored.values()))) if scored else float('nan')


def residual_correlation(prediction, truth, reference, keys):
    """Pearson r between predicted and measured residual over the reference, pooled over keys.

    Returns NaN where either side is constant, which is what a collapsed arm
    produces; the caller treats NaN as no signal rather than as an error.
    """
    if not keys:
        return float('nan')
    predicted = np.concatenate([np.ravel(prediction[k]-reference[k]) for k in keys])
    measured = np.concatenate([np.ravel(truth[k]-reference[k]) for k in keys])
    if predicted.std() == 0 or measured.std() == 0:
        return float('nan')
    return float(np.corrcoef(predicted, measured)[0, 1])


def residual_share(truth, reference, keys):
    """Variance of the measured residual as a fraction of the variance of the measured delta."""
    if not keys:
        return float('nan')
    measured = np.concatenate([np.ravel(truth[k]) for k in keys])
    residual = np.concatenate([np.ravel(truth[k]-reference[k]) for k in keys])
    total = measured.var()
    return float(residual.var()/total) if total > 0 else float('nan')


def split_noise_floor(selection_delta, evaluation_delta, de_sets, keys):
    """Direction accuracy of the selection half against the evaluation half: the endpoint's ceiling."""
    return mean_accuracy(selection_delta, evaluation_delta, de_sets, keys)
