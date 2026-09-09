"""D1: Retriever-Centric features — BM25 score, dense sim, rank, score gap."""
import numpy as np
from dataclasses import dataclass
from ..retrieval.retriever import RetrievedPassage


@dataclass
class D1Features:
    bm25_score: float
    dense_score: float
    rank: int
    score_gap: float


def extract_d1(passage: RetrievedPassage) -> D1Features:
    return D1Features(
        bm25_score=passage.bm25_score,
        dense_score=passage.dense_score,
        rank=passage.rank,
        score_gap=passage.score_gap,
    )


def d1_to_array(feat: D1Features) -> np.ndarray:
    return np.array([feat.bm25_score, feat.dense_score, feat.rank, feat.score_gap])
