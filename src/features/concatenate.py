"""Single entry-point for building feature vector x_i = φ(q, p_i, M_target).

Dimensions (14 total):
  D1 [0:4]  — bm25_score, dense_score, rank, score_gap
  D2 [4:7]  — token_length, flesch_kincaid_grade, source_credibility
  D3 [7:9]  — entity_overlap, hyde_answerability
  D4 [9:11] — entailment_prob, contradiction_score
  D5 [11:14]— base_perplexity, ctx_perplexity, delta_h
"""
import numpy as np
from src.retrieval.retriever import RetrievedPassage
from src.probing.logit_probe import LogitProbeResult
from src.features.d1_retriever_centric import extract_d1, d1_to_array
from src.features.d2_document_quality import extract_d2, d2_to_array
from src.features.d3_semantic_utility import extract_d3, d3_to_array
from src.features.d4_faithfulness import extract_d4, d4_to_array
from src.features.d5_llm_aware import extract_d5, d5_to_array

FEATURE_DIM = 14

FEATURE_GROUPS: dict[str, slice] = {
    "d1": slice(0, 4),
    "d2": slice(4, 7),
    "d3": slice(7, 9),
    "d4": slice(9, 11),
    "d5": slice(11, 14),
}


def build_feature_vector(
    query: str,
    passage: RetrievedPassage,
    probe_result: LogitProbeResult | None = None,
) -> np.ndarray:
    d5 = d5_to_array(extract_d5(probe_result)) if probe_result is not None else np.zeros(3)
    vec = np.concatenate([
        d1_to_array(extract_d1(passage)),
        d2_to_array(extract_d2(passage.text)),
        d3_to_array(extract_d3(query, passage.text)),
        d4_to_array(extract_d4(query, passage.text)),
        d5,
    ])
    assert vec.shape == (FEATURE_DIM,), f"Expected ({FEATURE_DIM},) got {vec.shape}"
    return vec
