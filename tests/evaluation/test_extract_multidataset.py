from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from experiments.extract_multidataset import candidate_bm25, item_features
from src.features.concatenate import FEATURE_DIM
from src.probing.logit_probe import LogitProbeResult
from src.retrieval.stream_bm25 import CorpusStats


STATS = CorpusStats(n_docs=1000, avgdl=20.0, df={"capital": 50, "france": 40})


def _nli():
    fake = MagicMock()
    fake.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    fake.predict.return_value = np.array([[-2.0, 3.0, -0.5]])
    return fake


class FakeDense:
    def score_passages(self, query, texts):
        return [0.9 - 0.1 * i for i in range(len(texts))]


class FakeProber:
    def probe_batch(self, query, texts):
        return [LogitProbeResult(h_base=3.0, h_ctx=2.0 + 0.1 * i, delta_h=1.0 - 0.1 * i,
                                 base_perplexity=20.0, ctx_perplexity=7.0 + i) for i in range(len(texts))]


def test_retrieved_passages_keep_the_bm25_from_the_corpus_scan():
    item = {"query": "capital of France", "passages": [{"id": "a", "text": "x", "bm25": 7.5}, {"id": "b", "text": "y", "bm25": 3.0}]}
    assert candidate_bm25(item, STATS) == [7.5, 3.0]


def test_provided_passages_are_scored_with_the_same_corpus_statistics():
    item = {"query": "capital of France", "passages": [
        {"id": "a", "text": '"Paris"\nParis is the capital of France.'},
        {"id": "b", "text": '"Cats"\nCats are small mammals.'},
    ]}
    s = candidate_bm25(item, STATS)
    assert s[0] > 0 and s[1] == 0.0


def test_item_features_have_the_trained_scale_and_the_right_shape():
    item = {"query": "capital of France", "passages": [
        {"id": "a", "text": '"Cats"\nCats are small mammals.', "bm25": 2.0},
        {"id": "b", "text": '"Paris"\nParis is the capital of France.', "bm25": 8.0},
    ]}
    with patch("src.features.d4_faithfulness._get_nli", return_value=_nli()):
        X = item_features(item, "The capital of France is Paris.", STATS, FakeDense(), FakeProber())
    assert X.shape == (2, FEATURE_DIM) and np.isfinite(X).all()
    # D1: bm25 divided by the query's best, rank 0 = best, gap to the next-ranked / best
    assert X[1, 0] == pytest.approx(1.0) and X[0, 0] == pytest.approx(0.25)
    assert X[1, 2] == 0 and X[0, 2] == 1
    assert X[1, 3] == pytest.approx(0.75) and X[0, 3] == 0.0
    assert X[0, 1] == pytest.approx(0.9) and X[1, 1] == pytest.approx(0.8)  # dense cosine, passage order
    # D5 comes from the probe: delta_h = h_base - h_ctx
    assert X[0, 13] == pytest.approx(1.0) and X[1, 13] == pytest.approx(0.9)
