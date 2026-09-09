import pytest
import numpy as np
from src.probing.logit_probe import LogitProbeResult, MockProber


def test_delta_h_positive_means_high_utility():
    result = LogitProbeResult(h_base=3.5, h_ctx=1.2, delta_h=2.3,
                               base_perplexity=33.0, ctx_perplexity=3.3)
    assert result.delta_h > 0


def test_delta_h_negative_means_noise():
    result = LogitProbeResult(h_base=3.5, h_ctx=4.1, delta_h=-0.6,
                               base_perplexity=33.0, ctx_perplexity=60.0)
    assert result.delta_h < 0


def test_mock_prober_returns_logit_probe_result():
    prober = MockProber(seed=42)
    result = prober.probe("What is Paris?", "Paris is the capital of France.")
    assert isinstance(result, LogitProbeResult)
    assert isinstance(result.delta_h, float)
    assert isinstance(result.h_base, float)


def test_mock_prober_batch_length():
    prober = MockProber(seed=42)
    passages = ["passage one", "passage two", "passage three"]
    results = prober.probe_batch("query", passages)
    assert len(results) == 3


def test_mock_prober_deterministic():
    p1 = MockProber(seed=0).probe("q", "p")
    p2 = MockProber(seed=0).probe("q", "p")
    assert p1.delta_h == p2.delta_h


def test_delta_h_equals_h_base_minus_h_ctx():
    prober = MockProber(seed=7)
    result = prober.probe("q", "p")
    assert result.delta_h == pytest.approx(result.h_base - result.h_ctx)
