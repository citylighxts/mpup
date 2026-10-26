import numpy as np
import pytest

from experiments.critical_difference import (
    average_ranks, cliques, friedman_pvalue, nemenyi_cd, to_tikz,
)


def test_average_ranks_gives_rank_one_to_the_highest_score_and_averages_ties():
    scores = {"a": np.array([3.0, 3.0, 1.0]), "b": np.array([2.0, 3.0, 2.0]), "c": np.array([1.0, 1.0, 3.0])}
    ranks = average_ranks(scores)
    # block 1: a, b, c = 1, 2, 3; block 2: a and b tie for first = 1.5, 1.5, c = 3; block 3: c, b, a = 1, 2, 3
    assert ranks == pytest.approx({"a": (1 + 1.5 + 3) / 3, "b": (2 + 1.5 + 2) / 3, "c": (3 + 3 + 1) / 3})


def test_nemenyi_critical_difference_matches_the_formula_of_demsar():
    # CD = q_alpha * sqrt(k (k + 1) / (6 N)); k = 5 methods, N = 10 blocks, alpha = 0.05, q = 2.728
    assert nemenyi_cd(5, 10, 0.05) == pytest.approx(2.728 * np.sqrt(5 * 6 / (6 * 10)), rel=1e-6)
    assert nemenyi_cd(4, 1641, 0.05) == pytest.approx(2.569 * np.sqrt(4 * 5 / (6 * 1641)), rel=1e-6)
    assert nemenyi_cd(3, 20, 0.10) < nemenyi_cd(3, 20, 0.05)  # a looser level gives a smaller CD


def test_nemenyi_rejects_a_number_of_methods_it_has_no_critical_value_for():
    for k in (1, 11):
        with pytest.raises(ValueError):
            nemenyi_cd(k, 10)


def test_cliques_group_methods_that_are_not_significantly_different():
    ranks = {"mpup": 1.0, "ce": 1.3, "dense": 3.0, "bm25": 3.2}
    assert cliques(ranks, cd=0.5) == [["mpup", "ce"], ["dense", "bm25"]]
    assert cliques(ranks, cd=0.1) == []  # nothing is within 0.1 of anything
    assert cliques(ranks, cd=5.0) == [["mpup", "ce", "dense", "bm25"]]  # one group contains the others


def test_friedman_detects_a_consistent_order_and_not_noise():
    rng = np.random.default_rng(0)
    n = 60
    ordered = {"a": rng.normal(2, 0.3, n), "b": rng.normal(1, 0.3, n), "c": rng.normal(0, 0.3, n)}
    noise = {k: rng.normal(0, 1, n) for k in "abc"}
    assert friedman_pvalue(ordered) < 1e-10
    assert friedman_pvalue(noise) > 0.01


def test_tikz_diagram_names_every_method_and_draws_the_cd_bar():
    ranks = {"MPUP": 1.0, "CE": 2.1, "Dense": 3.0, "BM25": 3.9}
    tex = to_tikz(ranks, cd=0.5)
    for name in ranks:
        assert name in tex
    assert "CD = 0.50" in tex and tex.startswith("\\begin{tikzpicture}") and tex.rstrip().endswith("\\end{tikzpicture}")
