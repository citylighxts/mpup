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


def test_no_threshold_by_default_keeps_top_m_even_with_negative_scores(sample_passages):
    ranked = [RankedPassage(passage=p, utility_score=s) for p, s in zip(sample_passages, (-0.1, -0.2, -0.3))]
    pipeline = MPUPPipeline(predictor=MagicMock(), top_m=2)
    prompt = pipeline.build_rag_prompt("query", ranked)
    assert "[2]" in prompt and "[3]" not in prompt


def test_explicit_threshold_drops_passages_at_or_below_it(sample_passages):
    ranked = [RankedPassage(passage=p, utility_score=s) for p, s in zip(sample_passages, (0.3, 0.0, -0.3))]
    pipeline = MPUPPipeline(predictor=MagicMock(), top_m=5, utility_threshold=0.0)
    prompt = pipeline.build_rag_prompt("query", ranked)
    assert "[1]" in prompt and "[2]" not in prompt


# ------------------------------------------------------------------------ MPUP+

from src.predictor.mpup_plus import MPUPPlusPredictor


class _FakeCrossEncoder:
    def score(self, query, passages):
        return np.array([0.9, 0.2, -1.0][: len(passages)])


class _FakeHidden:
    def hidden_batch(self, query, passages):
        rng = np.random.default_rng(len(query))
        return rng.normal(size=(len(passages), 12)).astype(np.float32)


def _plus_model():
    rng = np.random.default_rng(0)
    q = np.repeat([f"q{i}" for i in range(30)], 5)
    X = rng.normal(size=(len(q), 14))
    ce = rng.normal(size=len(q))
    H = rng.normal(size=(len(q), 12)).astype(np.float32)
    y = X[:, 13] + 0.5 * ce + 0.3 * H[:, 0] + rng.normal(0, 0.1, len(q))
    return MPUPPlusPredictor(n_estimators=20, max_depth=2).fit(X, ce, H, y, q)


def test_mpup_plus_appends_the_cross_encoder_and_probe_scores_to_the_14_features(sample_passages):
    model = _plus_model()
    pipeline = MPUPPipeline(predictor=model, prober=MockProber(seed=0), cross_encoder=_FakeCrossEncoder(),
                            hidden_extractor=_FakeHidden(), top_m=2)
    query = "What is the capital of France?"
    with patch("src.features.d4_faithfulness._get_nli", return_value=_mock_nli()):
        ranked = pipeline.rank(query, sample_passages)
    by_id = {r.passage.doc_id: r for r in ranked}
    H = _FakeHidden().hidden_batch(query, [p.text for p in sample_passages])
    probe = model.probe.predict(H)
    X16 = np.stack([by_id[p.doc_id].feature_vector for p in sample_passages])  # in the order of the candidates
    assert X16.shape == (3, 16)
    np.testing.assert_allclose(X16[:, 14], [0.9, 0.2, -1.0])
    np.testing.assert_allclose(X16[:, 15], probe)
    # scores are relative to the other candidates of the query, so they are the regressor's output on all three rows
    expected = model.predict_features(X16)
    for i, p in enumerate(sample_passages):
        assert by_id[p.doc_id].utility_score == pytest.approx(float(expected[i]))
    assert [r.utility_score for r in ranked] == sorted([r.utility_score for r in ranked], reverse=True)


def test_mpup_plus_without_its_two_extra_models_is_refused(sample_passages):
    model = _plus_model()
    for kwargs in ({}, {"cross_encoder": _FakeCrossEncoder()}, {"hidden_extractor": _FakeHidden()}):
        pipeline = MPUPPipeline(predictor=model, prober=MockProber(seed=0), **kwargs)
        with patch("src.features.d4_faithfulness._get_nli", return_value=_mock_nli()):
            with pytest.raises(ValueError, match="MPUP\\+"):
                pipeline.rank("query", sample_passages)
