"""Tests for D3: Semantic Utility features."""
import numpy as np
from src.features.d3_semantic_utility import extract_d3, d3_to_array


def test_d3_shape():
    arr = d3_to_array(extract_d3("What is Paris?", "Paris is the capital of France."))
    assert arr.shape == (2,)


def test_d3_entity_overlap_nonzero_when_query_entities_in_passage():
    feat = extract_d3("Where is Paris", "Paris is the capital.")
    assert feat.entity_overlap > 0.0


def test_d3_overlap_zero_when_no_match():
    feat = extract_d3("quantum physics", "pasta carbonara recipe")
    assert feat.entity_overlap == 0.0


def test_d3_entity_overlap_range():
    feat = extract_d3("test query", "test passage")
    assert 0.0 <= feat.entity_overlap <= 1.0


def test_d3_hyde_answerability_high_when_claim_tokens_in_passage():
    feat = extract_d3(
        "What is the capital of France?",
        "Paris is the capital and largest city of France.",
        hyde_answer="The capital of France is Paris.",
    )
    assert feat.hyde_answerability > 0.5


def test_d3_hyde_answerability_low_when_claim_absent():
    feat = extract_d3(
        "What is the capital of France?",
        "Photosynthesis converts sunlight into chemical energy.",
        hyde_answer="The capital of France is Paris.",
    )
    assert feat.hyde_answerability < 0.3


def test_d3_hyde_default_zero_without_claim():
    feat = extract_d3("query text", "passage text")
    assert feat.hyde_answerability == 0.0


def test_d3_array_dtype():
    arr = d3_to_array(extract_d3("test", "test"))
    assert isinstance(arr, np.ndarray)
