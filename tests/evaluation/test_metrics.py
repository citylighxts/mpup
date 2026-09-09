import numpy as np
import pytest
from src.evaluation.metrics import (
    exact_match_score, token_f1_score, combined_score,
    spearman_rho, ndcg_at_k, mrr,
)


def test_exact_match_perfect():
    assert exact_match_score("Paris", ["Paris", "paris"]) == 1.0

def test_exact_match_case_insensitive():
    assert exact_match_score("PARIS", ["paris"]) == 1.0

def test_exact_match_article_stripped():
    assert exact_match_score("the Eiffel Tower", ["Eiffel Tower"]) == 1.0

def test_exact_match_no_match():
    assert exact_match_score("London", ["Paris"]) == 0.0

def test_token_f1_partial():
    score = token_f1_score("Paris is beautiful", ["Paris France"])
    assert 0.0 < score < 1.0

def test_token_f1_perfect():
    assert token_f1_score("Paris", ["Paris"]) == 1.0

def test_token_f1_no_overlap():
    assert token_f1_score("London", ["Berlin"]) == 0.0

def test_combined_score_is_average():
    # perfect EM + perfect F1 → 1.0
    assert combined_score("Paris", ["Paris"]) == 1.0
    # EM=0, F1=0 → 0.0
    assert combined_score("London", ["Berlin"]) == 0.0

def test_spearman_perfect_correlation():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman_rho(y, y) == pytest.approx(1.0)

def test_spearman_negative_correlation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])
    assert spearman_rho(y_true, y_pred) == pytest.approx(-1.0)

def test_ndcg_at_k_perfect():
    y_true = np.array([3.0, 2.0, 1.0, 0.0])
    y_pred = np.array([3.0, 2.0, 1.0, 0.0])
    assert ndcg_at_k(y_true, y_pred, k=4) == pytest.approx(1.0)

def test_ndcg_at_k_less_than_k_items():
    # tidak crash ketika len(y) < k
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([1.0, 0.0])
    score = ndcg_at_k(y_true, y_pred, k=10)
    assert 0.0 <= score <= 1.0

def test_mrr_basic():
    assert mrr([1]) == pytest.approx(1.0)
    assert mrr([2]) == pytest.approx(0.5)
    assert mrr([1, 2]) == pytest.approx(0.75)
