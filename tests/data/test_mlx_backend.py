from types import SimpleNamespace

from src.data.mlx_backend import MLXLLM


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return f"<chat>{messages[0]['content']}</chat>"

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False  # the template already contains BOS
        return [ord(c) for c in text]


def _llm(batch_size=2):
    calls = []

    def fake_batch_generate(model, tokenizer, prompts, max_tokens):
        calls.append((len(prompts), max_tokens))
        texts = ["  answer:" + "".join(chr(t) for t in p)[6:-7] + " " for p in prompts]
        return SimpleNamespace(texts=texts)

    llm = MLXLLM(_loaded=(object(), FakeTokenizer()), _batch_generate=fake_batch_generate,
                 batch_size=batch_size, max_new_tokens=7)
    return llm, calls


def test_generate_returns_answers_in_the_callers_order_even_though_it_sorts_by_length():
    llm, _ = _llm()
    prompts = ["a much longer prompt than the others", "tiny", "medium sized one"]
    assert llm.generate(prompts) == [f"answer:{p}" for p in prompts]


def test_generate_batches_and_forwards_the_token_budget():
    llm, calls = _llm(batch_size=2)
    llm.generate(["a", "bb", "ccc", "dddd", "eeeee"])
    assert [c[0] for c in calls] == [2, 2, 1]
    assert {c[1] for c in calls} == {7}
    calls.clear()
    llm.generate(["x"], max_new_tokens=3)
    assert calls == [(1, 3)]


def test_prompts_are_chat_formatted_once():
    llm, _ = _llm()
    assert llm.chat_format("hi") == "<chat>hi</chat>"
    assert llm._encode("hi") == [ord(c) for c in "<chat>hi</chat>"]


def test_empty_input_makes_no_model_call():
    llm, calls = _llm()
    assert llm.generate([]) == []
    assert calls == []


# ----------------------------------------------------------------- answer_logprobs

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


class _AFavouringModel:
    """Always puts most of its mass on token 'A' (id 65), whatever the context."""

    def __call__(self, x):
        logits = np.zeros((1, x.shape[1], 128), dtype=np.float32)
        logits[..., ord("A")] = 6.0
        return mx.array(logits)


def _scoring_llm():
    return MLXLLM(_loaded=(_AFavouringModel(), FakeTokenizer()), _batch_generate=lambda *a, **k: None)


def test_answer_logprobs_prefers_the_likely_answer():
    lp = _scoring_llm().answer_logprobs(["q1", "q1"], ["AAA", "BBB"])
    assert lp[0] > lp[1]
    assert lp[0] == pytest.approx(float(np.log(np.exp(6) / (np.exp(6) + 127))), rel=1e-4)


def test_answer_logprobs_is_the_mean_over_answer_tokens_and_keeps_order():
    lp = _scoring_llm().answer_logprobs(["p", "p", "p"], ["A", "AB", "B"])
    assert lp[0] > lp[1] > lp[2]
    assert lp[1] == pytest.approx((lp[0] + lp[2]) / 2, rel=1e-4)
