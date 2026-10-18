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


# ------------------------------------------------------------------ MLXProber

mx = pytest.importorskip("mlx.core")


class _FakeTokenizer:
    def encode(self, prompt):
        return [1] + [ord(c) % 7 + 2 for c in prompt]  # token 1 plays BOS


class _FakeModel:
    """Next-token logits depend on the last token, so different prompts give different entropy."""

    def __init__(self, vocab=8):
        self.vocab = vocab
        self.lengths = []

    def __call__(self, x):
        self.lengths.append(x.shape[1])
        last = int(x[0, -1].item())
        logits = np.zeros((1, x.shape[1], self.vocab), dtype=np.float32)
        logits[0, -1, last % self.vocab] = float(last)  # sharper distribution for larger tokens
        return mx.array(logits)


def _prober(model=None):
    from src.probing.logit_probe import MLXProber

    model = model or _FakeModel()
    llm = type("L", (), {"model": model, "tokenizer": _FakeTokenizer()})()
    return MLXProber(llm=llm), model


def test_mlx_prober_entropy_and_perplexity_are_consistent():
    prober, _ = _prober()
    h, ppl = prober._entropy("Query: hello")
    assert h > 0
    assert ppl == pytest.approx(float(np.exp(h)), rel=1e-4)
    uniform = type("M", (), {"__call__": lambda self, x: mx.zeros((1, x.shape[1], 8))})()
    prober2, _ = _prober(uniform)
    h_u, ppl_u = prober2._entropy("anything")
    assert h_u == pytest.approx(float(np.log(8)), rel=1e-4) and ppl_u == pytest.approx(8.0, rel=1e-3)


def test_mlx_prober_delta_h_is_base_minus_context_entropy():
    prober, _ = _prober()
    r = prober.probe("q", "some passage")
    assert r.delta_h == pytest.approx(r.h_base - r.h_ctx)


def test_mlx_prober_batch_computes_the_base_prompt_once_per_query():
    prober, model = _prober()
    results = prober.probe_batch("what?", ["p1", "p2", "p3"])
    assert len(results) == 3
    assert len(model.lengths) == 1 + 3  # one base pass plus one pass per passage
    assert len({r.h_base for r in results}) == 1
    assert prober.probe_batch("what?", []) == []


def test_mlx_prober_truncates_long_prompts_but_keeps_the_end():
    prober, model = _prober()
    prober._entropy("x" * 2000, max_length=64)
    assert model.lengths[-1] == 64
