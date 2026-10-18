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
