import pytest
import numpy as np
from unittest.mock import MagicMock
from src.retrieval.retriever import RetrievedPassage
from src.probing.logit_probe import LogitProbeResult


@pytest.fixture
def sample_passages() -> list[RetrievedPassage]:
    return [
        RetrievedPassage(doc_id="p0", text="The capital of France is Paris.", bm25_score=4.5, dense_score=0.82, rank=0, score_gap=1.2),
        RetrievedPassage(doc_id="p1", text="Python is a programming language.", bm25_score=3.3, dense_score=0.61, rank=1, score_gap=0.5),
        RetrievedPassage(doc_id="p2", text="Unrelated noise document about cats.", bm25_score=0.8, dense_score=0.12, rank=2, score_gap=0.3),
    ]


@pytest.fixture
def sample_probe_results() -> list[LogitProbeResult]:
    return [
        LogitProbeResult(h_base=3.2, h_ctx=1.1, delta_h=2.1, base_perplexity=24.5, ctx_perplexity=3.0),
        LogitProbeResult(h_base=3.2, h_ctx=3.1, delta_h=0.1, base_perplexity=24.5, ctx_perplexity=22.0),
        LogitProbeResult(h_base=3.2, h_ctx=4.0, delta_h=-0.8, base_perplexity=24.5, ctx_perplexity=54.6),
    ]


@pytest.fixture
def tiny_feature_matrix() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    X = np.random.randn(50, 14)
    y = X[:, 13] * 0.5 + np.random.randn(50) * 0.1  # D5 delta_h → utility
    return X, y
