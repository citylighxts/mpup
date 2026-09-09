"""D2: Document-Quality features — length, readability, credibility."""
import numpy as np
from dataclasses import dataclass
import textstat


@dataclass
class D2Features:
    token_length: int
    flesch_kincaid_grade: float
    source_credibility: float  # 0-1 heuristic; extend with domain-specific scorer


def extract_d2(text: str, credibility_score: float = 0.5) -> D2Features:
    return D2Features(
        token_length=len(text.split()),
        flesch_kincaid_grade=textstat.flesch_kincaid_grade(text),
        source_credibility=credibility_score,
    )


def d2_to_array(feat: D2Features) -> np.ndarray:
    return np.array([feat.token_length, feat.flesch_kincaid_grade, feat.source_credibility])
