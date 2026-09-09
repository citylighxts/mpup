"""Tests for D2: Document-Quality features."""
import numpy as np
from src.features.d2_document_quality import extract_d2, d2_to_array


def test_d2_shape():
    arr = d2_to_array(extract_d2("This is a simple test sentence."))
    assert arr.shape == (3,)


def test_d2_token_length():
    feat = extract_d2("one two three four five")
    assert feat.token_length == 5


def test_d2_credibility_passthrough():
    feat = extract_d2("text", credibility_score=0.9)
    assert feat.source_credibility == 0.9


def test_d2_credibility_default():
    feat = extract_d2("some text")
    assert feat.source_credibility == 0.5


def test_d2_flesch_kincaid_grade_computed():
    feat = extract_d2("The quick brown fox jumps over the lazy dog.")
    assert feat.flesch_kincaid_grade >= 0.0


def test_d2_array_values():
    feat = extract_d2("word1 word2 word3", credibility_score=0.7)
    arr = d2_to_array(feat)
    assert arr[0] == 3  # token length
    assert arr[2] == 0.7  # credibility


def test_d2_array_dtype():
    arr = d2_to_array(extract_d2("sample text"))
    assert isinstance(arr, np.ndarray)
