import numpy as np

import src.retrieval.dense_retriever as dense
from src.retrieval.retriever import RetrievedPassage


class FakeModel:
    """Embeds a string as a unit vector chosen by its first letter, and records what it saw."""

    def __init__(self, name):
        self.name = name
        self.seen = []

    def encode(self, text, normalize_embeddings, batch_size=None):
        texts = [text] if isinstance(text, str) else list(text)
        self.seen.extend(texts)
        vecs = []
        for t in texts:
            body = t.split(": ", 1)[-1]
            v = np.zeros(3)
            v["abc".index(body[0])] = 1.0
            vecs.append(v)
        return vecs[0] if isinstance(text, str) else np.array(vecs)


def _make(monkeypatch, name):
    monkeypatch.setattr(dense, "SentenceTransformer", FakeModel)
    return dense.DenseRetriever(name)


def test_cosine_scores_follow_passage_order(monkeypatch):
    r = _make(monkeypatch, "all-MiniLM-L6-v2")
    assert r.score_passages("apple", ["avocado", "banana", "cherry"]) == [1.0, 0.0, 0.0]


def test_e5_models_get_query_and_passage_prefixes(monkeypatch):
    r = _make(monkeypatch, "intfloat/multilingual-e5-large")
    r.score_passages("apple", ["banana"])
    assert r.model.seen == ["query: apple", "passage: banana"]


def test_other_models_get_no_prefix(monkeypatch):
    r = _make(monkeypatch, "all-MiniLM-L6-v2")
    r.score_passages("apple", ["banana"])
    assert r.model.seen == ["apple", "banana"]


def test_enrich_fills_dense_score_in_place_and_empty_input_is_safe(monkeypatch):
    r = _make(monkeypatch, "all-MiniLM-L6-v2")
    ps = [RetrievedPassage(doc_id="1", text="avocado"), RetrievedPassage(doc_id="2", text="banana")]
    r.enrich("apple", ps)
    assert [p.dense_score for p in ps] == [1.0, 0.0]
    assert r.score_passages("apple", []) == []
