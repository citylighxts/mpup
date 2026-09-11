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


# ── BatchedUtilityLabeler — ported from the UtilityTransfer effort ───────────

class _ScriptedLLM:
    """Returns queued answers in order; records how many generate() calls were made."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = 0

    def generate(self, prompts, max_new_tokens=None):
        self.calls += 1
        taken, self._answers = self._answers[: len(prompts)], self._answers[len(prompts) :]
        return taken


def _items():
    return [
        {
            "query_id": "q1",
            "query": "What is the capital of France?",
            "gold_answers": ["Paris"],
            "passages": [{"id": "p1", "text": "..."}, {"id": "p2", "text": "..."}],
        }
    ]


def test_batched_labeler_scores_helpful_passage_positive():
    from src.data.utility_labeler import BatchedUtilityLabeler

    # no-context answer is wrong; first passage fixes it, second does not.
    llm = _ScriptedLLM(["London", "The answer is Paris.", "Berlin"])
    rows = BatchedUtilityLabeler(llm).label_corpus(_items())

    assert [r["doc_id"] for r in rows] == ["p1", "p2"]
    assert rows[0]["u_true"] > 0
    assert rows[1]["u_true"] <= 0


def test_batched_labeler_detects_harmful_passage():
    from src.data.utility_labeler import BatchedUtilityLabeler

    # already correct without context; the passage breaks it.
    llm = _ScriptedLLM(["Paris", "London", "Paris"])
    rows = BatchedUtilityLabeler(llm).label_corpus(_items())
    assert rows[0]["u_true"] < 0
    assert rows[1]["u_true"] == 0.0


def test_batched_labeler_generates_no_context_answer_once_per_query():
    """Two generate() calls total: one for all queries, one for all passages."""
    from src.data.utility_labeler import BatchedUtilityLabeler

    llm = _ScriptedLLM(["London", "Paris", "Paris"])
    BatchedUtilityLabeler(llm).label_corpus(_items())
    assert llm.calls == 2
