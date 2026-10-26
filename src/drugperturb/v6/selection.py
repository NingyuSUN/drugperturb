"""v6 primary-arm selection on validation direction accuracy (protocol_v6 primary_arm_selection).

v5's rule with the sign flipped, because the v6 metric is an accuracy and higher
is better. Validation compounds only. Among non-collapsed candidates (all
candidates if none qualifies), the leader has the HIGHEST mean over seeds of the
per-seed validation direction accuracy over responder keys; ties break by listed
order. Let d be the leader's mean minus a candidate's and se the Welch standard
error of that difference from the per-seed values. If the runner-up has
d < z*se, every pooled candidate within z*se of the leader is co-primary, and
each co-primary arm is tested at the Bonferroni level 0.05/k.
"""
from __future__ import annotations

import numpy as np

from ..v5.selection import ALPHA, Z, welch_se  # noqa: F401

ARMS = ['M1_pseudobulk_raw', 'M2_pseudobulk_residual']
METRIC = 'seed_direction_accuracy'


def select(arms, candidates, *, z=Z, alpha=ALPHA, max_arms=2, metric=METRIC):
    """``arms[a]`` holds ``metric`` (per-seed validation accuracy) and ``non_collapse`` (with ``non_collapsed``)."""
    candidates = list(candidates)
    missing = [a for a in candidates if a not in arms]
    if missing:
        raise ValueError(f'no fit summary for {missing}')
    passing = [a for a in candidates if arms[a]['non_collapse']['non_collapsed']]
    pool = passing or candidates
    mean = {a: float(np.mean(arms[a][metric])) for a in candidates}
    order = sorted(pool, key=lambda a: (-mean[a], candidates.index(a)))
    leader = order[0]
    comparisons = {a: dict(d=mean[leader]-mean[a], se=welch_se(arms[leader][metric], arms[a][metric]))
                   for a in order[1:]}
    for c in comparisons.values():
        c['within'] = bool(c['d'] < z*c['se'])
    tie = bool(order[1:] and comparisons[order[1]]['within'])
    primary = ([leader]+[a for a in order[1:] if comparisons[a]['within']] if tie else [leader])[:max_arms]
    k = len(primary)
    return dict(selected=leader, primary_arms=primary, near_tie=tie, bonferroni_k=k,
                interval_level=1-alpha/k, passing=passing, any_passing=bool(passing),
                mean_seed_validation_direction_accuracy=mean, comparisons_to_leader=comparisons, z=z,
                higher_is_better=True)
