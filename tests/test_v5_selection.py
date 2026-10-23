"""The v5 primary-arm selection and its near-tie rule (validation RMSEs only; synthetic numbers)."""
import numpy as np
import pytest

from drugperturb.v5 import selection

CANDIDATES = ['I1_nb_likelihood', 'I2_counterfactual_pseudobulk', 'I3_nb_plus_pseudobulk']


def arms(rmse, collapsed=()):
    return {a: dict(seed_rmse=list(v), non_collapse=dict(non_collapsed=a not in collapsed)) for a, v in zip(CANDIDATES, rmse)}


def test_a_clear_leader_is_the_only_primary_arm():
    chosen = selection.select(arms([[0.50, 0.501, 0.499], [0.46, 0.461, 0.459], [0.48, 0.481, 0.479]]), CANDIDATES)
    assert chosen['selected'] == CANDIDATES[1] and chosen['primary_arms'] == [CANDIDATES[1]]
    assert not chosen['near_tie'] and chosen['bonferroni_k'] == 1 and chosen['interval_level'] == pytest.approx(0.95)


def test_a_near_tie_makes_every_arm_within_the_margin_co_primary():
    rng = np.random.default_rng(0)
    noisy = lambda m: (m+rng.normal(0, 0.005, 10)).tolist()
    chosen = selection.select(arms([noisy(0.4680), noisy(0.4676), noisy(0.60)]), CANDIDATES)
    assert chosen['near_tie'] and set(chosen['primary_arms']) == set(CANDIDATES[:2])
    assert chosen['bonferroni_k'] == 2 and chosen['interval_level'] == pytest.approx(0.975)
    assert not chosen['comparisons_to_leader'][CANDIDATES[2]]['within']


def test_collapsed_candidates_are_excluded_unless_none_qualifies():
    rmse = [[0.40, 0.41], [0.46, 0.47], [0.50, 0.51]]
    chosen = selection.select(arms(rmse, collapsed={CANDIDATES[0]}), CANDIDATES)
    assert chosen['selected'] == CANDIDATES[1] and chosen['passing'] == CANDIDATES[1:]
    chosen = selection.select(arms(rmse, collapsed=set(CANDIDATES)), CANDIDATES)
    assert chosen['selected'] == CANDIDATES[0] and not chosen['any_passing']


def test_the_rule_needs_two_seeds():
    with pytest.raises(ValueError, match='two seeds'):
        selection.select(arms([[0.5], [0.5], [0.5]]), CANDIDATES)


def test_selection_constants_match_the_protocol_text():
    from pathlib import Path
    text = ' '.join((Path(__file__).resolve().parents[1]/'configs/protocol_v5.yaml').read_text().split())
    assert 'If d < 1.96 se' in text and '(at most three)' in text and selection.Z == 1.96
