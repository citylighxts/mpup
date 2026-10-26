from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.probing.utility_judge import JUDGE_PROMPT, UtilityJudge

YES, NO, VOCAB = 5, 6, 10


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[1 + ord(c) % 3 for c in text])  # ids 1..3, never the yes/no ids

    def encode(self, text, add_special_tokens=False):
        return {"Yes": [YES], "No": [NO]}[text.strip()]


class _Model(torch.nn.Module):
    """The logit of 'Yes' at the last real position is 0.1 * (number of real tokens); 'No' is always 0."""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask):
        n = attention_mask.sum(dim=1).float()
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], VOCAB)
        for r in range(input_ids.shape[0]):
            logits[r, int(n[r]) - 1, YES] = 0.1 * n[r]
        return SimpleNamespace(logits=logits)


def _llm():
    return SimpleNamespace(tokenizer=_Tokenizer(), model=_Model(), chat_format=lambda m: m)


def test_the_score_is_the_yes_minus_no_logit_at_the_last_real_token_whatever_the_padding():
    passages = ["a much longer passage than the others", "short", "medium length one"]
    got = UtilityJudge(_llm(), batch_size=2).score("who?", passages)
    expected = [0.1 * len(JUDGE_PROMPT.format(query="who?", passage=p)) for p in passages]
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_the_batch_size_does_not_change_the_scores():
    passages = ["alpha", "beta beta", "gamma gamma gamma", "delta"]
    a = UtilityJudge(_llm(), batch_size=1).score("q", passages)
    b = UtilityJudge(_llm(), batch_size=4).score("q", passages)
    np.testing.assert_allclose(a, b)


def test_no_passages_means_no_model_call():
    assert UtilityJudge(_llm()).score("q", []).shape == (0,)


def test_the_prompt_asks_about_usefulness_for_answering_and_shows_no_gold_answer():
    p = JUDGE_PROMPT.format(query="Q?", passage="P")
    assert "Q?" in p and "P" in p and "useful" in p.lower() and "Yes" in p and "No" in p
