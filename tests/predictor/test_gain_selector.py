from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.evaluation.protocol import per_query_scores
from src.predictor.gain_selector import GainSelector, kd_loss


class _Tokenizer:
    def __call__(self, queries, passages, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        seqs = [[ord(c) % 64 + 1 for c in q + p] for q, p in zip(queries, passages)]
        m = max(map(len, seqs))
        ids = torch.zeros((len(seqs), m), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        return {"input_ids": ids}


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(65, 8, padding_idx=0)
        self.head = torch.nn.Linear(8, 1)
        self.device = torch.device("cpu")

    def forward(self, input_ids):
        mask = (input_ids > 0).float()[..., None]
        pooled = (self.emb(input_ids) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return SimpleNamespace(logits=self.head(pooled))


def _selector():
    return GainSelector(_loaded=(_Model(), _Tokenizer()))


def test_kd_loss_is_zero_when_the_scores_reproduce_the_target_distribution_and_positive_otherwise():
    t = torch.tensor([[1.0, 0.0, -1.0, 0.5]])
    assert float(kd_loss(t, t, temperature=1.0)) == pytest.approx(0.0, abs=1e-6)
    assert float(kd_loss(-t, t, temperature=1.0)) > 0.1
    # the loss ignores a constant shift of the scores, since only the order within a query matters
    assert float(kd_loss(t + 5.0, t, temperature=1.0)) == pytest.approx(0.0, abs=1e-6)


def _data(n_queries=40, per_query=6, seed=0):
    rng = np.random.default_rng(seed)
    queries, groups, targets = [], [], []
    for i in range(n_queries):
        texts = ["".join(rng.choice(list("ab"), size=rng.integers(5, 12))) for _ in range(per_query)]
        queries.append(f"q{i}")
        groups.append(texts)
        targets.append(np.array([t.count("a") / len(t) for t in texts]))  # utility = share of 'a' in the passage
    return queries, groups, targets


def test_fitting_on_utilities_teaches_the_order_of_passages_within_a_query():
    queries, groups, targets = _data()
    sel = _selector()
    before = np.concatenate([sel.score(q, g) for q, g in zip(queries, groups)])
    sel.fit(queries, groups, targets, epochs=30, lr=5e-2, temperature=0.2)
    q_ids = np.repeat(queries, [len(g) for g in groups])
    y = np.concatenate(targets)
    after = np.concatenate([sel.score(q, g) for q, g in zip(queries, groups)])
    rho_before = per_query_scores(y, before, q_ids)["spearman"].mean()
    rho_after = per_query_scores(y, after, q_ids)["spearman"].mean()
    assert rho_after > 0.95 and rho_after > rho_before + 0.05


def test_scores_follow_the_order_of_the_passages_whatever_the_batch_size():
    q, g = "query", ["aaa", "b", "abab abab", "bb"]
    a = GainSelector(_loaded=(_Model(), _Tokenizer()), batch_size=1).score(q, g)
    b = GainSelector(_loaded=(_Model(), _Tokenizer()), batch_size=4).score(q, g)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    assert GainSelector(_loaded=(_Model(), _Tokenizer())).score(q, []).shape == (0,)


def test_the_order_that_is_learned_comes_from_the_targets_not_from_the_text():
    queries, groups, targets = _data()
    sel = _selector().fit(queries, groups, [-t for t in targets], epochs=30, lr=5e-2, temperature=0.2)
    q_ids = np.repeat(queries, [len(g) for g in groups])
    scores = np.concatenate([sel.score(q, g) for q, g in zip(queries, groups)])
    assert per_query_scores(np.concatenate(targets), scores, q_ids)["spearman"].mean() < -0.9  # reversed targets, reversed order
