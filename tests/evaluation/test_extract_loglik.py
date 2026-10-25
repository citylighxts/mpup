import numpy as np
import pytest

from experiments.extract_loglik import smooth_utility
from src.data.utility_labeler import NO_CONTEXT_PROMPT


class FakeLLM:
    """log-prob of the answer is -4 without a passage and -1 with one, except the passage 'bad' (-9) and 'great' (+3)."""

    def answer_logprobs(self, prompts, answers):
        out = []
        for p, a in zip(prompts, answers):
            assert a  # only queries with a gold answer are scored
            if p == NO_CONTEXT_PROMPT.format(query="who?"):
                out.append(-4.0)
            elif "bad" in p:
                out.append(-9.0)
            elif "great" in p:
                out.append(3.0)
            else:
                out.append(-1.0)
        return out


ITEMS = [{"query_id": "q", "query": "who?", "gold_answers": ["Paris", "paris"],
          "passages": [{"id": "q:a", "text": "ok text"}, {"id": "q:b", "text": "bad text"}, {"id": "q:c", "text": "great text"}]}]


def test_rows_follow_the_requested_pair_order_and_are_clipped():
    pairs = [("q", "q:c"), ("q", "q:a"), ("q", "q:b")]
    u = smooth_utility(FakeLLM(), ITEMS, pairs)
    assert u.tolist() == pytest.approx([5.0, 3.0, -5.0])  # 7 and -5 clipped to [-5, 5]; ok text: -1 - (-4) = 3


def test_unknown_pair_raises():
    with pytest.raises(KeyError):
        smooth_utility(FakeLLM(), ITEMS, [("q", "q:zzz")])
