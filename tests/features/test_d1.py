"""Tests for D1: Retriever-Centric features."""
import numpy as np
from src.retrieval.retriever import RetrievedPassage
from src.features.d1_retriever_centric import extract_d1, d1_to_array


def test_d1_shape():
    p = RetrievedPassage("d0", "text", bm25_score=3.5, dense_score=0.7, rank=1, score_gap=0.5)
    arr = d1_to_array(extract_d1(p))
    assert arr.shape == (4,)


def test_d1_values_correct():
    p = RetrievedPassage("d0", "text", bm25_score=3.5, dense_score=0.7, rank=2, score_gap=0.3)
    arr = d1_to_array(extract_d1(p))
    np.testing.assert_array_almost_equal(arr, [3.5, 0.7, 2, 0.3])


def test_d1_extract_returns_d1features():
    p = RetrievedPassage("doc1", "sample text", bm25_score=2.1, dense_score=0.5, rank=0, score_gap=0.8)
    feat = extract_d1(p)
    assert feat.bm25_score == 2.1
    assert feat.dense_score == 0.5
    assert feat.rank == 0
    assert feat.score_gap == 0.8


def test_d1_array_dtype():
    p = RetrievedPassage("d0", "text", bm25_score=1.0, dense_score=0.9, rank=5, score_gap=0.1)
    arr = d1_to_array(extract_d1(p))
    assert isinstance(arr, np.ndarray)
    assert arr.dtype in [np.float64, np.float32]
