import numpy as np
import pytest
from src.evaluation.metrics import (
    exact_match_score, token_f1_score, combined_score,
    spearman_rho, ndcg_at_k, mrr,
    exact_match, token_f1, evaluate_all,
    relaxed_match_score, relaxed_combined_score,
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

def test_mrr_empty():
    assert mrr([]) == 0.0

def test_token_f1_score_empty_pred():
    assert token_f1_score("", ["something"]) == 0.0

def test_ndcg_at_k_empty():
    assert ndcg_at_k(np.array([]), np.array([]), k=5) == 0.0

def test_exact_match_list():
    assert exact_match(["Paris", "London"], ["paris", "london"]) == 1.0
    assert exact_match(["Paris"], ["London"]) == 0.0

def test_token_f1_standalone():
    assert token_f1("Paris France", "Paris France") == 1.0
    assert token_f1("", "Paris") == 0.0
    assert token_f1("Paris", "") == 0.0

def test_evaluate_all_basic():
    y = np.array([1.0, 2.0, 3.0])
    result = evaluate_all(y, y, k=3)
    assert "spearman_rho" in result
    assert "ndcg_at_k" in result
    assert result["spearman_rho"] == pytest.approx(1.0)

def test_evaluate_all_with_qa():
    y = np.array([1.0, 2.0])
    result = evaluate_all(y, y, qa_predictions=["Paris", "London"],
                          qa_references=["Paris", "Berlin"], k=2)
    assert "em" in result
    assert "f1" in result
    assert result["em"] == pytest.approx(0.5)


# ── Relaxed matching — ported from the UtilityTransfer effort ────────────────
# Instruction-tuned models answer in sentences, which strict EM scores as wrong.
# Because utility is a *difference* of correctness, that collapses labels to zero.

def test_relaxed_match_accepts_sentence_answer():
    assert relaxed_match_score("The answer is Paris.", ["Paris"]) == 1.0

def test_strict_em_rejects_what_relaxed_accepts():
    assert exact_match_score("The answer is Paris.", ["Paris"]) == 0.0

def test_relaxed_match_rejects_wrong_answer():
    assert relaxed_match_score("London is the capital", ["Paris"]) == 0.0

def test_relaxed_match_is_token_level_not_substring():
    # "it" must not match inside "withering"
    assert relaxed_match_score("withering", ["it"]) == 0.0

def test_relaxed_match_multi_token_gold():
    assert relaxed_match_score("It was David Seville who did it", ["David Seville"]) == 1.0

def test_relaxed_match_empty_prediction():
    assert relaxed_match_score("", ["Paris"]) == 0.0


# ── token_f1 uses multiset overlap, not set overlap ──────────────────────────

def test_token_f1_counts_repeats_once_per_occurrence():
    # With set-based overlap the repeated "paris" would be collapsed and precision
    # would read 1.0; multiset overlap gives the correct 2*(1/3 * 1)/(1/3 + 1) = 0.5.
    assert token_f1_score("paris paris paris", ["paris"]) == pytest.approx(0.5)

def test_relaxed_combined_score_between_components():
    score = relaxed_combined_score("The answer is Paris.", ["Paris"])
    assert score == pytest.approx(0.5 * 1.0 + 0.5 * token_f1_score("The answer is Paris.", ["Paris"]))
