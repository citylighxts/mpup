import io
import json
import zipfile

import pytest

from src.data.benchmarks import (
    ALL_DATASETS,
    build_dataset,
    is_yes_no,
    items_from_2wiki_rows,
    items_from_entityq_zip,
    items_from_flashrag_rows,
    items_from_hotpot_rows,
    items_from_musique_rows,
    passage_text,
    sample_items,
)


def test_passage_text_uses_the_wikipedia_corpus_format():
    assert passage_text("Paris", "Capital of France.") == '"Paris"\nCapital of France.'


def test_yes_no_detection():
    assert is_yes_no(["Yes"]) and is_yes_no(["no"]) and not is_yes_no(["Paris"])


def test_sample_items_is_deterministic_ordered_and_capped():
    items = [{"i": i} for i in range(50)]
    a = sample_items(items, 10, seed=3)
    assert a == sample_items(items, 10, seed=3)
    assert a != sample_items(items, 10, seed=4)
    assert [x["i"] for x in a] == sorted(x["i"] for x in a)
    assert len(sample_items(items[:5], 10, seed=0)) == 5


def test_flashrag_rows_drop_rows_without_gold_and_dedupe_popqa_style():
    rows = [
        {"id": "a", "question": "Who is X?", "golden_answers": ["one"]},
        {"id": "b", "question": "What is Will Smith's occupation?", "golden_answers": ["actor"]},
        {"id": "c", "question": "What is Will Smith's occupation?", "golden_answers": ["producer"]},
        {"id": "d", "question": "No gold", "golden_answers": []},
    ]
    plain = items_from_flashrag_rows(rows, "nq")
    assert [i["query_id"] for i in plain] == ["nq:a", "nq:b", "nq:c"]
    deduped = items_from_flashrag_rows(rows, "popqa", dedupe_questions=True)
    assert [i["query_id"] for i in deduped] == ["popqa:a"]  # b/c share a question; d has no gold
    assert all(i["pool"] == "retrieved" and i["passages"] == [] for i in plain)


def test_hotpot_rows_keep_every_provided_paragraph_and_drop_yes_no():
    rows = [
        {"id": "h1", "question": "Q1?", "answer": "Paris",
         "context": {"title": ["A", "B"], "sentences": [["a1.", "a2."], ["b1."]]}},
        {"id": "h2", "question": "Q2?", "answer": "yes",
         "context": {"title": ["C"], "sentences": [["c1."]]}},
    ]
    items = items_from_hotpot_rows(rows)
    assert len(items) == 1
    item = items[0]
    assert item["pool"] == "provided" and item["gold_answers"] == ["Paris"]
    assert [p["text"] for p in item["passages"]] == ['"A"\na1. a2.', '"B"\nb1.']
    assert len({p["id"] for p in item["passages"]}) == 2


def test_2wiki_rows_use_aliases_and_drop_yes_no():
    rows = [
        {"id": "w1", "question": "Q?", "golden_answers": ["Mask of Fu Manchu", "The Mask of Fu Manchu"],
         "metadata": {"context": {"title": ["T"], "content": [["s1.", "s2."]]}}},
        {"id": "w2", "question": "Q2?", "golden_answers": ["no"],
         "metadata": {"context": {"title": ["T"], "content": [["s."]]}}},
    ]
    items = items_from_2wiki_rows(rows)
    assert len(items) == 1 and len(items[0]["gold_answers"]) == 2
    assert items[0]["passages"][0]["text"] == '"T"\ns1. s2.'


def test_musique_rows_skip_unanswerable_and_merge_aliases():
    rows = [
        {"id": "m1", "question": "Q?", "answer": "Green", "answer_aliases": ["Green (band)"], "answerable": True,
         "paragraphs": [{"idx": 0, "title": "T0", "paragraph_text": "x", "is_supporting": True},
                        {"idx": 1, "title": "T1", "paragraph_text": "y", "is_supporting": False}]},
        {"id": "m2", "question": "Q?", "answer": "", "answer_aliases": [], "answerable": False, "paragraphs": []},
    ]
    items = items_from_musique_rows(rows)
    assert len(items) == 1
    assert items[0]["gold_answers"] == ["Green", "Green (band)"]
    assert len(items[0]["passages"]) == 2


def test_entityquestions_zip_reads_only_the_test_split():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("dataset/test/P19.test.json", json.dumps([{"question": "Where was A born?", "answers": ["Rome"]}]))
        z.writestr("dataset/dev/P19.dev.json", json.dumps([{"question": "dev q", "answers": ["x"]}]))
    items = items_from_entityq_zip(buf.getvalue())
    assert [i["query"] for i in items] == ["Where was A born?"]
    assert items[0]["query_id"] == "entityquestions:P19-0"


def test_ragbench_rows_carry_the_sentence_proxy_and_no_gold_answer():
    from src.data.benchmarks import items_from_ragbench_rows

    rows = [{
        "question": "Q?", "documents": ["d0", "d1"],
        "documents_sentences": [[["0a", "s."], ["0b", "t."]], [["1a", "u."]]],
        "all_utilized_sentence_keys": ["0a"],
    }, {"question": "single doc", "documents": ["only"], "documents_sentences": [[["0a", "s."]]],
        "all_utilized_sentence_keys": []}]
    items = items_from_ragbench_rows(rows)
    assert len(items) == 1  # one-document rows have nothing to rank
    assert items[0]["gold_answers"] == [] and items[0]["pool"] == "provided"
    assert [p["proxy_label"] for p in items[0]["passages"]] == [0.5, 0.0]


def test_build_dataset_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown dataset"):
        build_dataset("not-a-dataset")


def test_all_eleven_evaluation_datasets_are_registered_once():
    assert len(ALL_DATASETS) == len(set(ALL_DATASETS)) == 11
