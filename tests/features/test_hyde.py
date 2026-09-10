"""Tests for HyDE pseudo-answer generation."""
from src.features.hyde import query_to_statement, MockHyDEGenerator


def test_query_to_statement_strips_question_mark():
    assert "?" not in query_to_statement("What is the capital of France?")


def test_query_to_statement_handles_what_is():
    out = query_to_statement("What is photosynthesis?")
    assert "photosynthesis" in out.lower()
    assert out != "What is photosynthesis?"


def test_query_to_statement_fallback_for_non_question():
    out = query_to_statement("capital of France")
    assert "capital of france" in out.lower()


def test_query_to_statement_never_empty():
    assert query_to_statement("?").strip()
    assert query_to_statement("").strip()


def test_mock_generator_is_deterministic():
    g = MockHyDEGenerator()
    assert g.generate("What is X?") == g.generate("What is X?")


def test_mock_generator_returns_statement():
    out = MockHyDEGenerator().generate("Who wrote Hamlet?")
    assert "?" not in out
    assert len(out) > 0
