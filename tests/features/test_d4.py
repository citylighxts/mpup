"""Tests for D4: Faithfulness features (NLI-based)."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.features.d4_faithfulness import extract_d4, d4_to_array


def _mock_nli_result(entailment: float, contradiction: float):
    """Mock zero-shot-classification pipeline result format."""
    return {
        "labels": ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"],
        "scores": [entailment, 1 - entailment - contradiction, contradiction],
    }


def test_d4_shape():
    with patch("src.features.d4_faithfulness._get_nli") as mock_nli:
        mock_pipe = MagicMock()
        mock_pipe.return_value = _mock_nli_result(0.8, 0.1)
        mock_nli.return_value = mock_pipe
        arr = d4_to_array(extract_d4("Paris is capital", "Paris is the capital of France."))
        assert arr.shape == (2,)


def test_d4_entailment_extracted():
    with patch("src.features.d4_faithfulness._get_nli") as mock_nli:
        mock_pipe = MagicMock()
        mock_pipe.return_value = _mock_nli_result(0.9, 0.05)
        mock_nli.return_value = mock_pipe
        feat = extract_d4("Paris is capital", "Paris is the capital of France.")
        assert feat.entailment_prob == pytest.approx(0.9)
        assert feat.contradiction_score == pytest.approx(0.05)


def test_d4_contradiction_extracted():
    with patch("src.features.d4_faithfulness._get_nli") as mock_nli:
        mock_pipe = MagicMock()
        mock_pipe.return_value = _mock_nli_result(0.1, 0.8)
        mock_nli.return_value = mock_pipe
        feat = extract_d4("sky is red", "The sky is blue.")
        assert feat.entailment_prob == pytest.approx(0.1)
        assert feat.contradiction_score == pytest.approx(0.8)


def test_d4_neutral_case():
    with patch("src.features.d4_faithfulness._get_nli") as mock_nli:
        mock_pipe = MagicMock()
        mock_pipe.return_value = _mock_nli_result(0.3, 0.2)
        mock_nli.return_value = mock_pipe
        feat = extract_d4("hypothesis", "premise")
        assert feat.entailment_prob == pytest.approx(0.3)
        assert feat.contradiction_score == pytest.approx(0.2)


def test_d4_array_values():
    with patch("src.features.d4_faithfulness._get_nli") as mock_nli:
        mock_pipe = MagicMock()
        mock_pipe.return_value = _mock_nli_result(0.7, 0.15)
        mock_nli.return_value = mock_pipe
        feat = extract_d4("hyp", "prem")
        arr = d4_to_array(feat)
        np.testing.assert_array_almost_equal(arr, [0.7, 0.15])


def test_d4_array_dtype():
    with patch("src.features.d4_faithfulness._get_nli") as mock_nli:
        mock_pipe = MagicMock()
        mock_pipe.return_value = _mock_nli_result(0.5, 0.3)
        mock_nli.return_value = mock_pipe
        arr = d4_to_array(extract_d4("h", "p"))
        assert isinstance(arr, np.ndarray)
