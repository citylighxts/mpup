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
