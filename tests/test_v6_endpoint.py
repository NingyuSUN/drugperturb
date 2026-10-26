"""Synthetic checks of the v6 endpoint: DE sets, the responder statistic and direction accuracy.

Fixtures establish software behaviour only; they say nothing about Tahoe-100M.
"""
import numpy as np
import pytest
import yaml

from pathlib import Path

from drugperturb.v6 import endpoint

ROOT = Path(__file__).resolve().parents[1]


def test_constants_match_the_protocol():
    protocol = yaml.safe_load((ROOT/'configs/protocol_v6.yaml').read_text())
    gene_set = protocol['response_endpoint']['de_gene_set']
    assert endpoint.DE_K == int(gene_set['k'])
    assert list(endpoint.DE_K_SENSITIVITY) == list(gene_set['k_sensitivity'])
    assert endpoint.RESPONDER_PERCENTILE == 95.0


def test_de_sets_take_the_largest_absolute_deltas_and_break_ties_by_column():
    delta = np.array([[0.0, -3.0, 1.0, 2.0], [5.0, 5.0, 5.0, -1.0]])
    assert endpoint.de_gene_sets(delta, k=2).tolist() == [[1, 3], [0, 1]]
    assert endpoint.de_gene_sets(delta, k=4).tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    with pytest.raises(ValueError, match='k must be in'):
        endpoint.de_gene_sets(delta, k=5)


def test_zero_change_scores_exactly_chance_and_a_perfect_prediction_scores_one():
    truth = np.array([[1.0, -2.0, 0.5, -0.1]])
    sets = endpoint.de_gene_sets(truth, k=3)
    chance, dropped = endpoint.direction_accuracy(np.zeros_like(truth), truth, sets)
    assert chance.tolist() == [0.5] and dropped == 0
    perfect, _ = endpoint.direction_accuracy(truth, truth, sets)
    assert perfect.tolist() == [1.0]
    flipped, _ = endpoint.direction_accuracy(-truth, truth, sets)
    assert flipped.tolist() == [0.0]


def test_direction_accuracy_ignores_magnitude_and_drops_genes_with_no_measured_direction():
    truth = np.array([[2.0, -4.0, 0.0]])
    sets = np.array([[0, 1, 2]])
    scaled, _ = endpoint.direction_accuracy(np.array([[0.001, -99.0, 7.0]]), truth, sets)
    assert scaled.tolist() == [1.0], 'only the sign should matter'
    accuracy, dropped = endpoint.direction_accuracy(np.array([[1.0, 1.0, 1.0]]), truth, sets)
    assert dropped == 1 and accuracy.tolist() == [0.5]
    blank, dropped = endpoint.direction_accuracy(np.zeros((1, 3)), np.zeros((1, 3)), sets)
    assert dropped == 3 and np.isnan(blank).all()


def test_the_statistic_is_about_one_under_poisson_noise_and_grows_with_a_real_effect():
    rng = np.random.default_rng(0)
    genes, library = 400, 2_000_000
    base = rng.gamma(2.0, 50.0, size=genes)
    null_pairs = np.array([rng.poisson(base) for _ in range(2)], dtype=float)
    libraries = null_pairs.sum(axis=1)
    from drugperturb.v4.design import log_cpm
    delta = log_cpm(null_pairs[[0]], libraries[[0]])-log_cpm(null_pairs[[1]], libraries[[1]])
    error = endpoint.delta_standard_error(null_pairs[[0]], libraries[[0]], null_pairs[[1]], libraries[[1]])
    quiet = endpoint.response_statistic(delta, error)[0]
    assert 0.5 < quiet < 2.0, f'pure sampling noise should sit near 1, got {quiet}'
    loud = endpoint.response_statistic(delta+0.5, error)[0]
    assert loud > 3*quiet
    assert library > 0


def test_the_statistic_barely_moves_with_cell_number_while_a_raw_norm_does():
    """Standardizing is what replaces the committed rule's cell-count matching (amendment A1)."""
    rng = np.random.default_rng(1)
    from drugperturb.v4.design import log_cpm
    base = rng.gamma(2.0, 20.0, size=300)
    statistics, norms = [], []
    for depth in (1, 16):
        pair = np.array([rng.poisson(base*depth) for _ in range(2)], dtype=float)
        libraries = pair.sum(axis=1)
        delta = log_cpm(pair[[0]], libraries[[0]])-log_cpm(pair[[1]], libraries[[1]])
        error = endpoint.delta_standard_error(pair[[0]], libraries[[0]], pair[[1]], libraries[[1]])
        statistics.append(endpoint.response_statistic(delta, error)[0])
        norms.append(float(np.linalg.norm(delta)))
    assert max(statistics)/min(statistics) < 1.5, f'the statistic should be size-robust, got {statistics}'
    assert max(norms)/min(norms) > 2.0, f'a raw norm should move with depth, got {norms}'


def test_responder_threshold_is_a_percentile_of_the_nulls():
    nulls = np.arange(1.0, 101.0)
    assert endpoint.responder_threshold(nulls) == pytest.approx(95.05)
    assert endpoint.responder_threshold(nulls, percentile=50) == pytest.approx(50.5)
    with pytest.raises(ValueError, match='at least one null'):
        endpoint.responder_threshold(np.zeros(0))


def test_restricted_rmse_and_overlap_use_only_the_de_genes():
    truth = np.array([[1.0, 2.0, 0.0, 0.0]])
    sets = np.array([[0, 1]])
    assert endpoint.restricted_rmse(np.zeros_like(truth), truth, sets)[0] == pytest.approx(np.sqrt(2.5))
    assert endpoint.restricted_rmse(truth+np.array([[3.0, 0.0, 99.0, 99.0]]), truth, sets)[0] == pytest.approx(np.sqrt(4.5))
    assert endpoint.de_overlap(np.array([[5.0, 4.0, 0.0, 0.0]]), sets)[0] == pytest.approx(1.0)
    assert endpoint.de_overlap(np.array([[0.0, 0.0, 5.0, 4.0]]), sets)[0] == pytest.approx(0.0)


def test_standard_errors_refuse_bad_libraries_and_floor_empty_genes():
    counts = np.array([[0.0, 10.0]])
    error = endpoint.log_cpm_standard_error(counts, np.array([1000.0]))
    assert (error > 0).all(), 'an unobserved gene must not claim zero uncertainty'
    with pytest.raises(ValueError, match='libraries must be finite and positive'):
        endpoint.log_cpm_standard_error(counts, np.array([0.0]))
    with pytest.raises(ValueError, match='one library per unit'):
        endpoint.log_cpm_standard_error(counts, np.array([1000.0, 1000.0]))
