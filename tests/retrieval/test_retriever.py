import numpy as np
import pytest
from src.retrieval.retriever import BM25Retriever, RetrievedPassage, candidate_scores


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


def test_candidate_scores_divide_by_the_best_score_and_rank_from_zero():
    norm, ranks, gaps = candidate_scores([2.0, 8.0, 4.0])
    assert norm.tolist() == [0.25, 1.0, 0.5]
    assert ranks.tolist() == [2, 0, 1]
    # input order: the 2.0 candidate is ranked last (gap 0); 8 -> 4 is 0.5; 4 -> 2 is 0.25
    assert gaps.tolist() == pytest.approx([0.0, 0.5, 0.25])


def test_candidate_scores_last_ranked_gap_is_zero_and_ties_keep_input_order():
    norm, ranks, gaps = candidate_scores([5.0, 5.0, 1.0])
    assert ranks.tolist() == [0, 1, 2]
    assert gaps[2] == 0.0 and gaps[0] == 0.0
    assert norm.max() == 1.0


def test_candidate_scores_survive_all_zero_and_empty_input():
    norm, ranks, gaps = candidate_scores([0.0, 0.0])
    assert norm.tolist() == [0.0, 0.0] and gaps.tolist() == [0.0, 0.0]
    assert candidate_scores([])[0].size == 0


def test_bm25_retriever_scores_are_on_the_trained_scale():
    results = BM25Retriever(CORPUS).retrieve("capital France Paris", top_k=3)
    assert results[0].bm25_score == pytest.approx(1.0)
    assert all(0.0 <= r.bm25_score <= 1.0 for r in results)
    assert all(0.0 <= r.score_gap <= 1.0 for r in results)
