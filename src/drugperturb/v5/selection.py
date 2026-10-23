"""v5 primary-arm selection with the prespecified near-tie rule (protocol_v5 primary_arm_selection).

Validation compounds only. Among non-collapsed candidates (all candidates if
none qualifies), the leader has the lowest mean over seeds of the per-seed
validation RMSE (ties by listed order). Let d be a candidate's mean minus the
leader's and se the Welch standard error of that difference from the per-seed
values. If the runner-up has d < z*se, every pooled candidate with d < z*se is a
co-primary arm together with the leader (at most ``max_arms``), and each
co-primary arm is tested at the Bonferroni level 0.05/k.
"""
from __future__ import annotations

import numpy as np

Z = 1.96
ALPHA = 0.05
ARMS = ['A0_v3_reference', 'I1_nb_likelihood', 'I2_counterfactual_pseudobulk', 'I3_nb_plus_pseudobulk']


def welch_se(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError('the near-tie rule needs at least two seeds per arm')
    return float(np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b)))


def select(arms, candidates, *, z=Z, alpha=ALPHA, max_arms=3):
    """``arms[a]`` holds ``seed_rmse`` (per-seed validation RMSE) and ``non_collapse`` (with ``non_collapsed``)."""
    passing = [a for a in candidates if arms[a]['non_collapse']['non_collapsed']]
    pool = passing or list(candidates)
    mean = {a: float(np.mean(arms[a]['seed_rmse'])) for a in candidates}
    order = sorted(pool, key=lambda a: (mean[a], list(candidates).index(a)))
    leader = order[0]
    comparisons = {a: dict(d=mean[a]-mean[leader], se=welch_se(arms[leader]['seed_rmse'], arms[a]['seed_rmse']))
                   for a in order[1:]}
    for c in comparisons.values():
        c['within'] = bool(c['d'] < z*c['se'])
    tie = bool(order[1:] and comparisons[order[1]]['within'])
    primary = [leader]+([a for a in order[1:] if comparisons[a]['within']] if tie else [])
    primary = primary[:max_arms]
    k = len(primary)
    return dict(selected=leader, primary_arms=primary, near_tie=tie, bonferroni_k=k,
                interval_level=1-alpha/k, passing=passing, any_passing=bool(passing),
                mean_seed_validation_rmse=mean, comparisons_to_leader=comparisons, z=z)
