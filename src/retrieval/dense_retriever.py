"""Dense retriever using sentence-transformers cosine similarity.
Used to fill RetrievedPassage.dense_score (D1 feature).
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score_passages(self, query: str, passages: list[str]) -> list[float]:
        """Return cosine similarity scores in the same order as passages."""
        if not passages:
            return []
        q_emb = self.model.encode(query, normalize_embeddings=True)
        p_embs = self.model.encode(passages, normalize_embeddings=True, batch_size=32)
        scores = (p_embs @ q_emb).tolist()
        return scores

    def enrich(self, query: str, passages: list) -> list:
        """Fill dense_score field on a list of RetrievedPassage in-place."""
        texts = [p.text for p in passages]
        scores = self.score_passages(query, texts)
        for p, s in zip(passages, scores):
            p.dense_score = s
        return passages
