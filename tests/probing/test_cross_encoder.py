from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.probing.cross_encoder import CrossEncoderScorer


class _Tokenizer:
    def __call__(self, queries, passages, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        seqs = [[ord(c) for c in q + p] for q, p in zip(queries, passages)]
        m = max(map(len, seqs))
        ids = torch.zeros((len(seqs), m), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        return {"input_ids": ids}


class _Model(torch.nn.Module):
    def forward(self, input_ids):
        return SimpleNamespace(logits=input_ids.float().sum(dim=1, keepdim=True))


def test_scores_follow_the_order_of_the_passages_although_batches_are_formed_by_length():
    scorer = CrossEncoderScorer(_loaded=(_Model(), _Tokenizer()), batch_size=2)
    passages = ["a much longer passage", "x", "mid sized", "yy"]
    got = scorer.score("q", passages)
    expected = [sum(ord(c) for c in "q" + p) for p in passages]
    np.testing.assert_allclose(got, expected)
    assert got.dtype == np.float32


def test_no_passages_gives_an_empty_array():
    assert CrossEncoderScorer(_loaded=(_Model(), _Tokenizer())).score("q", []).shape == (0,)
