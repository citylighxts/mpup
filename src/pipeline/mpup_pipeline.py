"""End-to-end MPUP pipeline: retrieval → probing → feature extraction → prediction → RAG."""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ..retrieval.retriever import RetrievedPassage
from ..probing.logit_probe import LogitProber
from ..features.d1_retriever_centric import extract_d1, d1_to_array
from ..features.d2_document_quality import extract_d2, d2_to_array
from ..features.d3_semantic_utility import extract_d3, d3_to_array
from ..features.d4_faithfulness import extract_d4, d4_to_array
from ..features.d5_llm_aware import extract_d5, d5_to_array
from ..predictor.mpup_predictor import MPUPPredictor
from ..predictor.calibration import LinearCalibrator


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
    ):
        self.predictor = predictor
        self.prober = prober
        self.calibrator = calibrator or LinearCalibrator()
        self.top_m = top_m

    def _build_feature_vector(
        self,
        query: str,
        passage: RetrievedPassage,
        probe_result=None,
    ) -> np.ndarray:
        d1 = d1_to_array(extract_d1(passage))
        d2 = d2_to_array(extract_d2(passage.text))
        d3 = d3_to_array(extract_d3(query, passage.text))
        d4 = d4_to_array(extract_d4(query, passage.text))
        d5 = d5_to_array(extract_d5(probe_result)) if probe_result else np.zeros(3)
        return np.concatenate([d1, d2, d3, d4, d5])

    def rank(
        self,
        query: str,
        passages: list[RetrievedPassage],
    ) -> list[RankedPassage]:
        probe_results = None
        if self.prober is not None:
            probe_results = self.prober.probe_batch(query, [p.text for p in passages])

        feature_matrix = []
        for i, passage in enumerate(passages):
            pr = probe_results[i] if probe_results else None
            fv = self._build_feature_vector(query, passage, pr)
            feature_matrix.append(fv)

        X = np.array(feature_matrix)
        base_scores = self.predictor.predict(X)
        utility_scores = self.calibrator.transform(base_scores)

        ranked = [
            RankedPassage(passage=passages[i], utility_score=float(utility_scores[i]), feature_vector=X[i])
            for i in range(len(passages))
        ]
        ranked.sort(key=lambda r: r.utility_score, reverse=True)
        return ranked

    def build_rag_prompt(self, query: str, ranked_passages: list[RankedPassage]) -> str:
        top_m = [r for r in ranked_passages if r.utility_score > 0][: self.top_m]
        context = "\n\n".join(f"[{i+1}] {r.passage.text}" for i, r in enumerate(top_m))
        return f"Context:\n{context}\n\nQuery: {query}"
