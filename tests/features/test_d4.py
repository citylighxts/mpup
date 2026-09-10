"""Tests for D4: Faithfulness features (NLI cross-encoder)."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.features.d4_faithfulness import extract_d4, d4_to_array, _softmax


class _FakeNLI:
    """Mimics sentence_transformers.CrossEncoder for (premise, hypothesis) pairs."""

    def __init__(self, logits):
        self._logits = logits
        self.config = MagicMock()
        self.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    def predict(self, pairs):
        return np.array([self._logits])


def test_softmax_sums_to_one():
    out = _softmax(np.array([2.0, 1.0, 0.1]))
    assert out.sum() == pytest.approx(1.0)


def test_d4_shape():
    with patch("src.features.d4_faithfulness._get_nli", return_value=_FakeNLI([-4.0, 5.0, -0.5])):
        arr = d4_to_array(extract_d4("France's capital is Paris.", "Paris is the capital of France."))
        assert arr.shape == (2,)


def test_d4_entailment_dominates_for_supported_claim():
    with patch("src.features.d4_faithfulness._get_nli", return_value=_FakeNLI([-4.0, 5.0, -0.5])):
        feat = extract_d4("France's capital is Paris.", "Paris is the capital of France.")
        assert feat.entailment_prob > 0.9
        assert feat.contradiction_score < 0.05


def test_d4_contradiction_dominates_for_conflicting_claim():
    with patch("src.features.d4_faithfulness._get_nli", return_value=_FakeNLI([5.0, -3.0, -2.0])):
        feat = extract_d4("The sky is green.", "The sky is blue.")
        assert feat.contradiction_score > 0.9
        assert feat.entailment_prob < 0.05


def test_d4_probs_are_valid():
    with patch("src.features.d4_faithfulness._get_nli", return_value=_FakeNLI([1.0, 1.0, 1.0])):
        feat = extract_d4("h", "p")
        assert 0.0 <= feat.entailment_prob <= 1.0
        assert 0.0 <= feat.contradiction_score <= 1.0


def test_d4_respects_label_order():
    """Label indices are read from id2label, not positional assumptions."""
    fake = _FakeNLI([5.0, -3.0, -2.0])
    fake.config.id2label = {0: "entailment", 1: "contradiction", 2: "neutral"}
    with patch("src.features.d4_faithfulness._get_nli", return_value=fake):
        feat = extract_d4("h", "p")
        assert feat.entailment_prob > 0.9  # index 0 is now entailment


def test_d4_array_dtype():
    with patch("src.features.d4_faithfulness._get_nli", return_value=_FakeNLI([0.5, 0.3, 0.2])):
        arr = d4_to_array(extract_d4("h", "p"))
        assert isinstance(arr, np.ndarray)
