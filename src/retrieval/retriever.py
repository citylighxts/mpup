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


class BM25Retriever:
    def __init__(self, corpus: list[dict]):
        """corpus: list of {"id": str, "text": str}"""
        self.corpus = corpus
        tokenized = [doc["text"].lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievedPassage]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        passages = []
        for rank, idx in enumerate(top_indices):
            score_gap = float(scores[top_indices[rank]] - scores[top_indices[rank + 1]]) \
                if rank + 1 < len(top_indices) else 0.0
            passages.append(RetrievedPassage(
                doc_id=self.corpus[idx]["id"],
                text=self.corpus[idx]["text"],
                bm25_score=float(scores[idx]),
                rank=rank,
                score_gap=score_gap,
            ))
        return passages
