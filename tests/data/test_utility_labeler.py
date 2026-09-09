"""Tests for UtilityLabeler — ground-truth u_i_true computation."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.data.utility_labeler import (
    exact_match_score,
    token_f1_score,
    combined_score,
    UtilityLabeler,
    LabeledPassage,
)


def test_exact_match_normalizes_articles():
    """Test that EM normalizes article words (a, an, the)."""
    assert exact_match_score("the Eiffel Tower", ["Eiffel Tower"]) == 1.0


def test_token_f1_partial_overlap():
    """Test that F1 gives partial credit for token overlap."""
    score = token_f1_score("Paris France city", ["Paris capital France"])
    assert 0.0 < score < 1.0


def test_combined_score_range():
    """Test that combined score is in valid range [0, 1]."""
    s = combined_score("Paris", ["London"])
    assert 0.0 <= s <= 1.0


def test_utility_positive_when_passage_helps():
    """Test that u_true > 0 when passage improves answer."""
    mock_answerer = MagicMock()
    mock_answerer.answer_no_context.return_value = "I don't know"
    mock_answerer.answer_with_context.return_value = "Paris"

    labeler = UtilityLabeler(mock_answerer)
    passages = [{"id": "p0", "text": "Paris is the capital of France."}]
    results = labeler.label_query("What is the capital of France?", ["Paris"], passages)

    assert len(results) == 1
    assert results[0].u_true > 0.0


def test_utility_negative_when_passage_misleads():
    """Test that u_true < 0 when passage makes answer worse."""
    mock_answerer = MagicMock()
    mock_answerer.answer_no_context.return_value = "Paris"  # baseline already correct
    mock_answerer.answer_with_context.return_value = "London"  # passage misleads

    labeler = UtilityLabeler(mock_answerer)
    passages = [{"id": "p0", "text": "London is the capital."}]
    results = labeler.label_query("What is the capital of France?", ["Paris"], passages)

    assert results[0].u_true < 0.0


def test_utility_clipped_to_minus_one_plus_one():
    """Test that u_true is clipped to [-1.0, 1.0]."""
    mock_answerer = MagicMock()
    mock_answerer.answer_no_context.return_value = "wrong answer completely different"
    mock_answerer.answer_with_context.return_value = "Paris"

    labeler = UtilityLabeler(mock_answerer)
    passages = [{"id": "p0", "text": "Paris."}]
    results = labeler.label_query("q", ["Paris"], passages)
    assert -1.0 <= results[0].u_true <= 1.0


def test_no_context_called_once_per_query():
    """Test that answer_no_context is called exactly once per label_query."""
    mock_answerer = MagicMock()
    mock_answerer.answer_no_context.return_value = "wrong"
    mock_answerer.answer_with_context.return_value = "Paris"

    labeler = UtilityLabeler(mock_answerer)
    passages = [{"id": f"p{i}", "text": f"passage {i}"} for i in range(5)]
    labeler.label_query("q", ["Paris"], passages)

    # answer_no_context should be called only once per query, even with 5 passages
    assert mock_answerer.answer_no_context.call_count == 1
    # answer_with_context should be called once per passage
    assert mock_answerer.answer_with_context.call_count == 5
