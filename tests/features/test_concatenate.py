import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.retriever import RetrievedPassage
from src.probing.logit_probe import LogitProbeResult, MockProber
from src.features.concatenate import (
    build_feature_vector, FEATURE_DIM, FEATURE_GROUPS,
)


def make_passage(**kwargs):
    defaults = dict(doc_id="d0", text="Paris is the capital of France.",
                    bm25_score=4.5, dense_score=0.8, rank=0, score_gap=1.2)
    return RetrievedPassage(**{**defaults, **kwargs})


def make_probe_result(**kwargs):
    defaults = dict(h_base=3.2, h_ctx=1.1, delta_h=2.1,
                    base_perplexity=24.5, ctx_perplexity=3.0)
    return LogitProbeResult(**{**defaults, **kwargs})


def _mock_nli():
    mock_pipe = MagicMock()
    mock_pipe.return_value = {
        "labels": ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"],
        "scores": [0.8, 0.1, 0.1],
    }
    return mock_pipe


def test_output_shape():
    with patch("src.features.d4_faithfulness._get_nli", return_value=_mock_nli()):
        vec = build_feature_vector("What is Paris?", make_passage(), make_probe_result())
    assert vec.shape == (FEATURE_DIM,)
    assert FEATURE_DIM == 14


def test_feature_groups_cover_all_dims():
    total = sum(s.stop - s.start for s in FEATURE_GROUPS.values())
    assert total == FEATURE_DIM


def test_d5_values_at_correct_slice():
    with patch("src.features.d4_faithfulness._get_nli", return_value=_mock_nli()):
        probe = make_probe_result(base_perplexity=10.0, ctx_perplexity=2.0, delta_h=8.0)
        vec = build_feature_vector("q", make_passage(), probe)
    d5_slice = FEATURE_GROUPS["d5"]
    assert vec[d5_slice][2] == pytest.approx(8.0)


def test_no_probe_result_gives_zero_d5():
    with patch("src.features.d4_faithfulness._get_nli", return_value=_mock_nli()):
        vec = build_feature_vector("q", make_passage(), probe_result=None)
    d5_slice = FEATURE_GROUPS["d5"]
    np.testing.assert_array_equal(vec[d5_slice], np.zeros(3))
