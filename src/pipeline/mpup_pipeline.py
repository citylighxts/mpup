"""End-to-end MPUP pipeline: retrieval → probing → feature extraction → prediction → RAG."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.retrieval.retriever import RetrievedPassage
from src.probing.logit_probe import LogitProber
from src.features.concatenate import build_feature_vector
from src.predictor.mpup_predictor import MPUPPredictor
from src.predictor.calibration import LinearCalibrator


@dataclass
class RankedPassage:
    passage: RetrievedPassage
    utility_score: float
    feature_vector: np.ndarray = field(default_factory=lambda: np.array([]))


class MPUPPipeline:
    def __init__(
        self,
        predictor: MPUPPredictor,
        prober: Optional[LogitProber] = None,
        calibrator: Optional[LinearCalibrator] = None,
        top_m: int = 5,
        utility_threshold: float = 0.0,
    ):
        self.predictor = predictor
        self.prober = prober
        self.calibrator = calibrator or LinearCalibrator()
        self.top_m = top_m
        self.utility_threshold = utility_threshold

    def rank(self, query: str, passages: list[RetrievedPassage]) -> list[RankedPassage]:
        probe_results = None
        if self.prober is not None:
            probe_results = self.prober.probe_batch(query, [p.text for p in passages])

        X = np.array([
            build_feature_vector(query, passages[i], probe_results[i] if probe_results else None)
            for i in range(len(passages))
        ])
        base_scores = self.predictor.predict(X)
        utility_scores = self.calibrator.transform(base_scores)

        ranked = [
            RankedPassage(passage=passages[i], utility_score=float(utility_scores[i]), feature_vector=X[i])
            for i in range(len(passages))
        ]
        ranked.sort(key=lambda r: r.utility_score, reverse=True)
        return ranked

    def build_rag_prompt(self, query: str, ranked_passages: list[RankedPassage]) -> str:
        top = [r for r in ranked_passages if r.utility_score > self.utility_threshold][:self.top_m]
        context = "\n\n".join(f"[{i+1}] {r.passage.text}" for i, r in enumerate(top))
        return f"Context:\n{context}\n\nQuery: {query}"
