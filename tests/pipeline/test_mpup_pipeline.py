import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.mpup_pipeline import MPUPPipeline, RankedPassage
from src.predictor.mpup_predictor import MPUPPredictor
from src.predictor.calibration import LinearCalibrator
from src.probing.logit_probe import MockProber


@pytest.fixture
def trained_pipeline(tiny_feature_matrix):
    X, y = tiny_feature_matrix
    predictor = MPUPPredictor(algorithm="xgboost", n_estimators=10, max_depth=3)
    predictor.fit(X, y)
    prober = MockProber(seed=0)
    return MPUPPipeline(predictor=predictor, prober=prober, top_m=2)


def _mock_nli():
    fake = MagicMock()
    fake.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    fake.predict.return_value = np.array([[-2.0, 3.5, -0.5]])
    return fake


def test_rank_returns_sorted_descending(trained_pipeline, sample_passages):
    with patch("src.features.d4_faithfulness._get_nli", return_value=_mock_nli()):
        ranked = trained_pipeline.rank("What is the capital of France?", sample_passages)
    scores = [r.utility_score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_length_matches_input(trained_pipeline, sample_passages):
    with patch("src.features.d4_faithfulness._get_nli", return_value=_mock_nli()):
        ranked = trained_pipeline.rank("query", sample_passages)
    assert len(ranked) == len(sample_passages)


def test_build_rag_prompt_contains_query(trained_pipeline, sample_passages):
    with patch("src.features.d4_faithfulness._get_nli", return_value=_mock_nli()):
        ranked = trained_pipeline.rank("What is the capital?", sample_passages)
    prompt = trained_pipeline.build_rag_prompt("What is the capital?", ranked)
    assert "What is the capital?" in prompt
    assert "Context:" in prompt


def test_build_rag_prompt_respects_top_m(trained_pipeline, sample_passages):
    with patch("src.features.d4_faithfulness._get_nli", return_value=_mock_nli()):
        ranked = trained_pipeline.rank("query", sample_passages)
    prompt = trained_pipeline.build_rag_prompt("query", ranked)
    assert prompt.count("[1]") == 1
    assert "[3]" not in prompt


class _FakeDense:
    def enrich(self, query, passages):
        for i, p in enumerate(passages):
            p.dense_score = 0.9 - 0.1 * i
        return passages


def test_dense_retriever_fills_the_dense_feature_before_prediction(tiny_feature_matrix, sample_passages):
    X, y = tiny_feature_matrix
    predictor = MPUPPredictor(algorithm="xgboost", n_estimators=10, max_depth=3)
    predictor.fit(X, y)
    pipeline = MPUPPipeline(predictor=predictor, prober=MockProber(seed=0), dense_retriever=_FakeDense())
    with patch("src.features.d4_faithfulness._get_nli", return_value=_mock_nli()):
        ranked = pipeline.rank("What is the capital of France?", sample_passages)
    dense_col = {r.passage.doc_id: r.feature_vector[1] for r in ranked}
    assert dense_col == pytest.approx({"p0": 0.9, "p1": 0.8, "p2": 0.7})


def test_pipeline_works_with_a_query_centred_predictor(tiny_feature_matrix, sample_passages):
    X, y = tiny_feature_matrix
    groups = np.repeat(np.arange(10), 5)
    predictor = MPUPPredictor(algorithm="ridge", center_by_query=True, alpha=1.0)
    predictor.fit(X, y, groups=groups)
    pipeline = MPUPPipeline(predictor=predictor, prober=MockProber(seed=0), top_m=2)
    with patch("src.features.d4_faithfulness._get_nli", return_value=_mock_nli()):
        ranked = pipeline.rank("query", sample_passages)
    scores = [r.utility_score for r in ranked]
    assert scores == sorted(scores, reverse=True) and len(ranked) == 3
    assert sum(scores) == pytest.approx(0.0 + sum(scores))  # finite
    assert all(np.isfinite(scores))
