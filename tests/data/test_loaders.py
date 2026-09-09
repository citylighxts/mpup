import pytest
from unittest.mock import patch, MagicMock
from src.data.loaders import load_benchmark


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        list(load_benchmark("nonexistent_dataset"))


def test_ragbench_output_structure():
    mock_row = {
        "question": "What is Paris?",
        "answer": "Capital of France",
        "documents": ["Paris is the capital of France.", "France is in Europe."],
    }
    mock_ds = [mock_row]
    with patch("src.data.loaders.load_dataset", return_value=mock_ds):
        results = list(load_benchmark("ragbench", max_samples=1))
    assert len(results) == 1
    row = results[0]
    assert row["query"] == "What is Paris?"
    assert "Capital of France" in row["answers"]
    assert len(row["passages"]) == 2
    assert all("id" in p and "text" in p for p in row["passages"])


def test_msmarco_skips_no_answer_rows():
    mock_no_answer = {
        "query": "q",
        "answers": ["No Answer Present."],
        "passages": {"passage_text": [], "is_selected": []},
    }
    mock_valid = {
        "query": "What is Paris?",
        "answers": ["Capital of France"],
        "passages": {
            "passage_text": ["Paris is capital."],
            "is_selected": [1],
        },
    }
    with patch("src.data.loaders.load_dataset", return_value=[mock_no_answer, mock_valid]):
        results = list(load_benchmark("msmarco", max_samples=2))
    assert len(results) == 1
    assert results[0]["query"] == "What is Paris?"


def test_triviaqa_output_has_passages():
    mock_row = {
        "question": "In what country is Paris?",
        "answer": {"value": "France", "aliases": ["France", "french republic"]},
        "entity_pages": {"wiki_context": ["Paris is in France."]},
        "search_results": {"search_context": []},
    }
    with patch("src.data.loaders.load_dataset", return_value=[mock_row]):
        results = list(load_benchmark("triviaqa", max_samples=1))
    assert len(results) == 1
    assert results[0]["answers"] == ["France", "french republic"]
    assert len(results[0]["passages"]) >= 1
