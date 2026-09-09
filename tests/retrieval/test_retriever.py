import pytest
from src.retrieval.retriever import BM25Retriever, RetrievedPassage


CORPUS = [
    {"id": "d0", "text": "The capital of France is Paris."},
    {"id": "d1", "text": "Python is a high-level programming language."},
    {"id": "d2", "text": "Machine learning uses statistical methods."},
]


def test_bm25_top_result_relevant():
    retriever = BM25Retriever(CORPUS)
    results = retriever.retrieve("capital France Paris", top_k=3)
    assert results[0].doc_id == "d0"


def test_bm25_returns_correct_count():
    retriever = BM25Retriever(CORPUS)
    results = retriever.retrieve("python", top_k=2)
    assert len(results) == 2


def test_bm25_rank_is_zero_indexed():
    retriever = BM25Retriever(CORPUS)
    results = retriever.retrieve("python", top_k=3)
    ranks = [r.rank for r in results]
    assert ranks == list(range(len(results)))


def test_bm25_score_gap_nonnegative():
    retriever = BM25Retriever(CORPUS)
    results = retriever.retrieve("machine learning", top_k=3)
    for r in results:
        assert r.score_gap >= 0.0


def test_bm25_single_passage_no_crash():
    retriever = BM25Retriever([{"id": "d0", "text": "only one"}])
    results = retriever.retrieve("one", top_k=1)
    assert len(results) == 1
    assert results[0].score_gap == 0.0
