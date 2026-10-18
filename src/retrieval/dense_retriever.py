"""Dense similarity between a query and each candidate passage (the D1 `dense_score`).

Uses sentence-transformers cosine similarity. E5 models are trained with a "query: " and a
"passage: " prefix and score noticeably worse without them, so those prefixes are added
automatically for any model whose name contains "e5".
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        batch_size: int = 32,
    ):
        self.model = SentenceTransformer(model_name)
        is_e5 = "e5" in model_name.lower()
        self.query_prefix = ("query: " if is_e5 else "") if query_prefix is None else query_prefix
        self.passage_prefix = ("passage: " if is_e5 else "") if passage_prefix is None else passage_prefix
        self.batch_size = batch_size

    def score_passages(self, query: str, passages: list[str]) -> list[float]:
        """Return cosine similarity scores in the same order as passages."""
        if not passages:
            return []
        q_emb = self.model.encode(self.query_prefix + query, normalize_embeddings=True)
        p_embs = self.model.encode(
            [self.passage_prefix + p for p in passages], normalize_embeddings=True, batch_size=self.batch_size
        )
        return (np.asarray(p_embs) @ np.asarray(q_emb)).tolist()

    def enrich(self, query: str, passages: list) -> list:
        """Fill dense_score field on a list of RetrievedPassage in-place."""
        texts = [p.text for p in passages]
        scores = self.score_passages(query, texts)
        for p, s in zip(passages, scores):
            p.dense_score = s
        return passages
