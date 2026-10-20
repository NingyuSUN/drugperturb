"""Collapse diagnostics on training/validation pairs and the prespecified non-collapse criteria.

Inputs are dictionaries keyed by (compound_id, context_id). Every quantity is
computed from predictions and measured training/validation deltas only.
"""
from __future__ import annotations

import numpy as np

CRITERIA = dict(effect_size_ratio=0.10, mean_gene_pearson_r=0.10)


def finite(value):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def pearson(a, b):
    """Gene-wise Pearson r; 0 when either vector is constant (a collapsed prediction)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a-a.mean(), b-b.mean()
    denominator = np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/denominator) if denominator > 0 else 0.0


def context_means(truth):
    """B1c: mean measured delta over each context's training pairs."""
    groups = {}
    for (_, context), value in truth.items():
        groups.setdefault(context, []).append(value)
    return {context: np.mean(values, axis=0) for context, values in groups.items()}


def summarize(predicted, truth, reference_means, training_prediction_means=None):
    """Effect size, direction, accuracy and compound-specific signal over shared keys.

    ``reference_means`` maps context -> B1c vector (the scale of a typical
    effect); ``training_prediction_means`` maps context -> the arm's mean
    prediction over that context's training pairs, for the compound-specific r.
    """
    keys = sorted(set(predicted) & set(truth))
    if not keys:
        raise ValueError('no shared prediction/truth keys')
    p = np.vstack([predicted[k] for k in keys])
    t = np.vstack([truth[k] for k in keys])
    reference = np.vstack([reference_means[k[1]] for k in keys])
    rmse = np.sqrt(((p-t)**2).mean(axis=1))
    rmse_zero = np.sqrt((t**2).mean(axis=1))
    denominator = (p*p).sum()
    result = dict(
        n=len(keys), contexts=len({k[1] for k in keys}),
        mean_abs_pred=float(np.abs(p).mean()), mean_abs_true=float(np.abs(t).mean()),
        rmse=float(rmse.mean()), rmse_zero=float(rmse_zero.mean()),
        rmse_minus_zero=float((rmse-rmse_zero).mean()),
        mean_gene_pearson_r=float(np.mean([pearson(a, b) for a, b in zip(p, t)])),
        least_squares_scale=float((p*t).sum()/denominator) if denominator > 0 else float('nan'),
        effect_size_ratio=float(np.linalg.norm(p, axis=1).mean()/np.linalg.norm(reference, axis=1).mean()),
        reference_r=float(np.mean([pearson(a, b) for a, b in zip(reference, t)])))
    if training_prediction_means is not None:
        centre = np.vstack([training_prediction_means[k[1]] for k in keys])
        result['compound_specific_r'] = float(np.mean([pearson(a, b) for a, b in zip(p-centre, t-reference)]))
    return result


def non_collapse(summary, criteria=None):
    """The three prespecified criteria and their conjunction."""
    criteria = CRITERIA if criteria is None else criteria
    checks = dict(pass_effect_size_ratio=summary['effect_size_ratio'] >= criteria['effect_size_ratio'],
                  pass_mean_gene_pearson_r=summary['mean_gene_pearson_r'] >= criteria['mean_gene_pearson_r'],
                  pass_rmse_below_zero_change=summary['rmse'] < summary['rmse_zero'])
    return dict(checks, non_collapsed=all(checks.values()))


def select_primary(arms, candidates):
    """Prespecified OP3 rule: lowest validation RMSE among non-collapsed candidates, else among all."""
    passing = [a for a in candidates if arms[a]['non_collapse']['non_collapsed']]
    pool = passing or list(candidates)
    chosen = min(pool, key=lambda a: (arms[a]['validation_rmse'], list(candidates).index(a)))
    return dict(selected=chosen, passing=passing, any_passing=bool(passing),
                validation_rmse={a: arms[a]['validation_rmse'] for a in candidates})
