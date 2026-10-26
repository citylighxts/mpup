from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.utility_labeler import WITH_CONTEXT_PROMPT
from src.probing.hidden_state import HiddenStateExtractor

HIDDEN, LAYERS = 4, 3


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) for c in text])


class _Model(torch.nn.Module):
    """Hidden state of layer l at position t: token id * (l + 1) + 0.01 * t + feature index, padding or not."""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.calls = []

    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        self.calls.append(tuple(input_ids.shape))
        pos = torch.arange(input_ids.shape[1], dtype=torch.float32)[None, :, None]
        feat = torch.arange(HIDDEN, dtype=torch.float32)
        states = tuple(input_ids.float()[..., None] * (l + 1) + 0.01 * pos + feat for l in range(LAYERS + 1))
        return SimpleNamespace(hidden_states=states)


def _llm():
    return SimpleNamespace(tokenizer=_Tokenizer(), model=_Model(), chat_format=lambda m: m + "\nA")


def _expected(query, passage, layer):
    ids = [ord(c) for c in WITH_CONTEXT_PROMPT.format(query=query, passage=passage) + "\nA"]
    return ids[-1] * (layer + 1) + 0.01 * (len(ids) - 1) + np.arange(HIDDEN)


def test_the_last_prompt_token_of_the_utility_prompt_is_read_whatever_the_padding():
    llm = _llm()
    passages = ["a much longer passage than the others", "short", "medium length one"]
    H = HiddenStateExtractor(llm, layer=2, batch_size=2).hidden_batch("who?", passages)
    assert H.shape == (3, HIDDEN) and H.dtype == np.float32
    for row, p in zip(H, passages):
        np.testing.assert_allclose(row, _expected("who?", p, 2), rtol=1e-5)


def test_the_batch_size_does_not_change_the_result():
    passages = ["alpha", "beta beta", "gamma gamma gamma", "delta"]
    a = HiddenStateExtractor(_llm(), layer=1, batch_size=1).hidden_batch("q", passages)
    b = HiddenStateExtractor(_llm(), layer=1, batch_size=4).hidden_batch("q", passages)
    np.testing.assert_allclose(a, b)


def test_no_passages_means_no_model_call():
    llm = _llm()
    assert HiddenStateExtractor(llm, layer=1).hidden_batch("q", []).shape[0] == 0
    assert llm.model.calls == []
