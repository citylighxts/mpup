"""Candidate retrieval: BM25 and dense retriever."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class RetrievedPassage:
    doc_id: str
    text: str
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rank: int = 0
    score_gap: float = 0.0  # gap to next-ranked passage


def candidate_scores(raw_scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The D1 retrieval features of one query's candidate set, in the caller's order.

    Returns (score divided by the highest score of the query, rank with 0 = best, gap to the
    next-ranked candidate divided by the same maximum; 0 for the last one). Dividing by the
    query's best score keeps D1 comparable across queries and across sources: raw BM25 grows with
    query length and corpus statistics, and the predictor is trained on one scale only.
    """
    raw = np.asarray(raw_scores, dtype=float)
    if raw.size == 0:
        return raw, raw.astype(int), raw
    order = np.argsort(-raw, kind="stable")
    ranks = np.empty(len(raw), dtype=int)
    ranks[order] = np.arange(len(raw))
    top = float(raw.max())
    scale = top if top > 0 else 1.0
    sorted_scores = raw[order]
    gaps_sorted = np.append(sorted_scores[:-1] - sorted_scores[1:], 0.0) / scale
    gaps = np.empty(len(raw))
    gaps[order] = gaps_sorted
    return raw / scale, ranks, gaps


class BM25Retriever:
    def __init__(self, corpus: list[dict]):
        """corpus: list of {"id": str, "text": str}"""
        self.corpus = corpus
        tokenized = [doc["text"].lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievedPassage]:
        """Top-k passages with D1 features on the scale the predictor was trained on
        (`candidate_scores`: BM25 and gap divided by the best score of this query)."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        norm, ranks, gaps = candidate_scores(scores[top_indices])

        passages = []
        for pos, idx in enumerate(top_indices):
            passages.append(RetrievedPassage(
                doc_id=self.corpus[idx]["id"],
                text=self.corpus[idx]["text"],
                bm25_score=float(norm[pos]),
                rank=int(ranks[pos]),
                score_gap=float(gaps[pos]),
            ))
        return passages
