import pytest
from unittest.mock import patch, MagicMock
from src.data.loaders import load_benchmark


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        list(load_benchmark("nonexistent_dataset"))


def _ragbench_row() -> dict:
    """A row in the real galileo-ai/ragbench schema: there is no `answer` column, the
    reference text is `response`, and relevance lives in sentence keys."""
    return {
        "question": "What is Paris?",
        "response": "Capital of France",
        "documents": ["Paris is the capital of France.", "France is in Europe."],
        "documents_sentences": [
            [["0a", "Paris is the capital of France."]],
            [["1a", "France is in Europe."]],
        ],
        "all_relevant_sentence_keys": ["0a"],
    }


def test_ragbench_output_structure():
    with patch("src.data.loaders.load_dataset", return_value=[_ragbench_row()]):
        results = list(load_benchmark("ragbench", max_samples=1))
    assert len(results) == 1
    row = results[0]
    assert row["query"] == "What is Paris?"
    assert "Capital of France" in row["answers"]
    assert len(row["passages"]) == 2
    assert all("id" in p and "text" in p for p in row["passages"])


def test_ragbench_passes_config_name_and_marks_relevant_passage():
    # galileo-ai/ragbench is a multi-config dataset; omitting the config raises
    # "Config name is missing" on the real Hub.
    with patch("src.data.loaders.load_dataset", return_value=[_ragbench_row()]) as mock_load:
        results = list(load_benchmark("ragbench", split="test", max_samples=1))
    args, kwargs = mock_load.call_args
    assert args == ("galileo-ai/ragbench", "hotpotqa")
    assert kwargs["split"] == "test"
    flags = [p["is_positive"] for p in results[0]["passages"]]
    assert flags == [True, False]


@pytest.mark.parametrize(
    "name, expected_args",
    [
        ("natural_questions", ("google-research-datasets/natural_questions",)),
        ("triviaqa", ("mandarjoshi/trivia_qa", "unfiltered")),
        ("msmarco", ("microsoft/ms_marco", "v2.1")),
    ],
)
def test_loaders_use_resolvable_hub_ids_without_trust_remote_code(name, expected_args):
    # `google/natural_questions` and the legacy `trivia_qa` id no longer resolve, and
    # datasets>=4 rejects `trust_remote_code` (loading scripts are unsupported).
    with patch("src.data.loaders.load_dataset", return_value=[]) as mock_load:
        list(load_benchmark(name, split="validation", max_samples=1))
    args, kwargs = mock_load.call_args
    assert args == expected_args
    assert "trust_remote_code" not in kwargs


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
