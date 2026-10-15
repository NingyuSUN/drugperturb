"""Compound-level errors and paired scaffold-cluster uncertainty, no p-values.

Supply actual measured delta-expression truth in the same frozen gene space as
predictions; reconstructed/PCA-smoothed truth is not a substitute. Rows must be
unique prediction-unit compounds, not cells, replicate wells or formulation aliases.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _matrix(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f'{name} must be real-valued')
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f'{name} must have nonempty shape (compounds,genes)')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite; rows are never silently dropped')
    return array


def _labels(values, n, name, unique=False):
    labels = list(values)
    if len(labels) != n:
        raise ValueError(f'{name} length must match the number of compounds')
    if not all(isinstance(label, str) and label.strip() for label in labels):
        raise ValueError(f'{name} must contain nonempty strings without missing values')
    if unique and len(set(labels)) != n:
        raise ValueError('compound IDs must be unique prediction units')
    return labels


def _validated(truth, prediction, compound_ids, scaffold_groups):
    truth, prediction = _matrix(truth, 'truth'), _matrix(prediction, 'prediction')
    if truth.shape != prediction.shape:
        raise ValueError('truth and prediction shapes must match exactly')
    ids = _labels(compound_ids, len(truth), 'compound_ids', unique=True)
    groups = _labels(scaffold_groups, len(truth), 'scaffold_groups')
    return truth, prediction, ids, groups


def _errors(truth, prediction):
    # Scaling avoids overflow when squaring finite, large errors or summing them.
    with np.errstate(over='raise', invalid='raise'):
        try:
            error = prediction - truth
            scale = np.max(np.abs(error), axis=1)
            normalized = np.divide(error, scale[:, None], out=np.zeros_like(error),
                                   where=scale[:, None] != 0)
            rmse = scale * np.sqrt(np.mean(normalized ** 2, axis=1))
            mae = scale * np.mean(np.abs(normalized), axis=1)
        except FloatingPointError as exc:
            raise ValueError('finite inputs produced unrepresentable error metrics') from exc
    return rmse, mae


def _spearman(truth, prediction):
    if np.all(truth == truth[0]) or np.all(prediction == prediction[0]):
        return float('nan')
    # Average ranks handle ties. Only the coefficient is computed, never a p-value.
    a, b = rankdata(truth, method='average'), rankdata(prediction, method='average')
    a, b = a - a.mean(), b - b.mean()
    return float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))


def _cosine_zero(truth, prediction):
    # The origin is delta=0; do NOT subtract gene-vector means (that is Pearson).
    tscale, pscale = np.max(np.abs(truth)), np.max(np.abs(prediction))
    if tscale == 0 or pscale == 0:
        return float('nan')
    a, b = truth / tscale, prediction / pscale
    return float(np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1, 1))


def compound_metrics(
    truth, prediction, compound_ids: Sequence[str], scaffold_groups: Sequence[str],
    *, sign_nonzero_only: bool = True,
) -> pd.DataFrame:
    """Compute metrics per compound, retaining input order and every valid row.

    RMSE/MAE use every supplied gene. Spearman is undefined for constant vectors;
    cosine_zero is undefined if either delta vector has zero norm. Sign agreement
    defaults to truth != 0 exactly, with no effect-size filtering; prediction zero
    disagrees with any nonzero truth. n_sign_genes records each denominator.
    Undefined metrics are NaN, never replaced by zero or omitted without counts.
    """
    truth, prediction, ids, groups = _validated(truth, prediction, compound_ids, scaffold_groups)
    rmse, mae = _errors(truth, prediction)
    rows = []
    for index, (actual, predicted) in enumerate(zip(truth, prediction)):
        mask = actual != 0 if sign_nonzero_only else np.ones(len(actual), dtype=bool)
        n_sign = int(mask.sum())
        agreement = float(np.mean(np.sign(actual[mask]) == np.sign(predicted[mask]))) if n_sign else np.nan
        rows.append(dict(
            compound_id=ids[index], scaffold_group=groups[index], n_genes=len(actual),
            rmse=float(rmse[index]), mae=float(mae[index]),
            spearman=_spearman(actual, predicted), cosine_zero=_cosine_zero(actual, predicted),
            sign_agreement=agreement, n_sign_genes=n_sign,
            n_truth_nonzero=int(np.count_nonzero(actual)), sign_nonzero_only=sign_nonzero_only,
        ))
    return pd.DataFrame(rows)


def summarize_compound_metrics(table: pd.DataFrame) -> dict[str, Any]:
    """Equal-weight compounds, reporting valid/undefined counts for every NA metric.

    The main metric is mean(sqrt(mean_gene(error**2))) across compounds. It is
    NOT sqrt(mean_all_compound_gene(error**2)). Undefined-only means are None,
    allowing strict JSON null serialization; per-compound tables retain NaN.
    """
    if table.empty or table['compound_id'].duplicated().any():
        raise ValueError('summary requires nonempty unique compound rows')
    if table['n_genes'].nunique() != 1 or table['sign_nonzero_only'].nunique() != 1:
        raise ValueError('summary requires a consistent gene count and sign definition')
    if not np.isfinite(table[['rmse', 'mae']].to_numpy(dtype=float)).all():
        raise ValueError('compound error metrics must be finite')
    result: dict[str, Any] = dict(
        n_compounds=len(table), n_groups=table['scaffold_group'].nunique(),
        n_genes=int(table['n_genes'].iloc[0]),
        mean_compound_rmse=float(table['rmse'].mean()),
        mean_compound_mae=float(table['mae'].mean()),
        sign_nonzero_only=bool(table['sign_nonzero_only'].iloc[0]),
    )
    for metric in ('spearman', 'cosine_zero', 'sign_agreement'):
        values = table[metric].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError('metric values must be finite or explicitly undefined (NaN)')
        valid = np.isfinite(values)
        result[f'n_{metric}_valid'] = int(valid.sum())
        result[f'n_{metric}_undefined'] = int((~valid).sum())
        result[f'mean_{metric}'] = float(values[valid].mean()) if valid.any() else None
    return result


def paired_scaffold_bootstrap(
    truth, candidate, baseline, compound_ids: Sequence[str], scaffold_groups: Sequence[str],
    *, draws: int = 2000, seed: int = 2026,
) -> dict[str, Any]:
    """Paired percentile cluster bootstrap of candidate RMSE minus baseline RMSE.

    Compute each compound's RMSE difference first. Sample K scaffold groups with
    replacement from K unique observed groups; include ALL compounds in a sampled
    group, repeated with its multiplicity. Each draw averages compounds, not group
    means, so draw denominators vary with scaffold size. Negative favors candidate.
    The observed point estimate is the mean compound difference (not mean draws).

    The 95% interval uses empirical 2.5/97.5 percentiles with linear interpolation.
    This is conditional on these compounds/genes and already fitted models; it
    does not establish independent scaffold biology, account for shared-control
    uncertainty or repeated fitting, or validate external generalization/efficacy.
    """
    truth, candidate, ids, groups = _validated(truth, candidate, compound_ids, scaffold_groups)
    baseline = _matrix(baseline, 'baseline')
    if baseline.shape != truth.shape:
        raise ValueError('baseline and truth shapes must match exactly')
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError('cluster bootstrap requires at least two scaffold groups')
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError('draws must be a positive integer')
    # Sort compounds before reductions: a row permutation must not change RNG
    # group order or floating-point summation order within any scaffold group.
    order = np.argsort(np.asarray(ids))
    groups_array = np.asarray(groups)[order]
    candidate_rmse, _ = _errors(truth[order], candidate[order])
    baseline_rmse, _ = _errors(truth[order], baseline[order])
    difference = candidate_rmse - baseline_rmse
    group_sums = np.array([difference[groups_array == group].sum() for group in unique_groups])
    group_sizes = np.array([np.count_nonzero(groups_array == group) for group in unique_groups])
    sampled = np.random.default_rng(seed).integers(0, len(unique_groups), size=(draws, len(unique_groups)))
    sampled_n = group_sizes[sampled].sum(axis=1)
    bootstrap = group_sums[sampled].sum(axis=1) / sampled_n
    if not np.isfinite(bootstrap).all():
        raise ValueError('bootstrap differences are not finite')
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], method='linear')
    return dict(
        point_estimate=float(difference.mean()), ci_lower=float(lower), ci_upper=float(upper),
        bootstrap_differences=bootstrap, n_compounds=len(ids), n_genes=truth.shape[1],
        n_groups=len(unique_groups), scaffold_order=unique_groups,
        sampled_compounds_min=int(sampled_n.min()), sampled_compounds_max=int(sampled_n.max()),
        draws=int(draws), seed=int(seed), confidence_level=0.95,
        comparison='candidate_minus_baseline_rmse', negative_favors='candidate',
        method='paired_scaffold_cluster_percentile_bootstrap',
        resampling_unit='scaffold_group', aggregation_unit='compound', quantile_method='linear',
        limitation='Conditional cluster interval; scaffold independence, shared-control and refit uncertainty are not established.',
    )
