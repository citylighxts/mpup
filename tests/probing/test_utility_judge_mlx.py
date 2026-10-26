from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from src.probing.utility_judge import JUDGE_PROMPT, MLXUtilityJudge


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        if text.strip() in ("Yes", "No"):
            return [3] if text.strip() == "Yes" else [4]
        return [5 + ord(c) % 7 for c in text]  # ids 5..11


def _tiny_qwen():
    from mlx_lm.models.qwen2 import Model, ModelArgs

    mx.random.seed(1)
    args = ModelArgs(model_type="qwen2", hidden_size=16, num_hidden_layers=2, intermediate_size=32,
                     num_attention_heads=2, num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=16)
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _llm():
    tok = _Tokenizer()
    return SimpleNamespace(model=_tiny_qwen(), tokenizer=tok, _encode=lambda msg: tok.encode(msg))


def _direct(llm, query, passage):
    ids = llm._encode(JUDGE_PROMPT.format(query=query, passage=passage))
    last = llm.model(mx.array([ids]))[0, -1].astype(mx.float32)
    return float(last[3] - last[4])


def test_the_score_is_yes_minus_no_at_the_last_token_of_each_prompt_whatever_the_padding():
    llm = _llm()
    passages = ["a much longer passage than the others", "short", "medium length one"]
    got = MLXUtilityJudge(llm, batch_size=2).score("who?", passages)
    np.testing.assert_allclose(got, [_direct(llm, "who?", p) for p in passages], rtol=1e-4, atol=1e-4)


def test_the_batch_size_does_not_change_the_scores():
    llm = _llm()
    passages = ["alpha", "beta beta", "gamma gamma gamma", "delta"]
    a = MLXUtilityJudge(llm, batch_size=1).score("q", passages)
    b = MLXUtilityJudge(llm, batch_size=4).score("q", passages)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
    assert MLXUtilityJudge(llm).score("q", []).shape == (0,)
