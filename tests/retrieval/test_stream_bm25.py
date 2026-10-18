import json
import math
import zipfile

import pytest

from src.retrieval.stream_bm25 import (
    CorpusStats,
    bm25_idf,
    collect_stats,
    iter_zip_lines,
    parse_line,
    query_terms,
    retrieve,
    score_text,
    tokenize,
)

DOCS = [
    ("Paris", "Paris is the capital and largest city of France. Paris sits on the river Seine."),
    ("France", "France is a country in Western Europe with Paris as its capital."),
    ("Rome", "Rome is the capital of Italy and sits on the river Tiber."),
    ("Seine", "The Seine is a river in northern France that flows through Paris."),
    ("Cats", "Cats are small carnivorous mammals kept as pets."),
    ("Python", "Python is a programming language."),
] * 5  # repeated so document frequencies are not all 1


def _lines():
    return [json.dumps({"id": str(i), "contents": f'"{t}"\n{b}'}).encode() for i, (t, b) in enumerate(DOCS)]


def _open():
    return iter(_lines())


def test_tokenize_lowercases_splits_punctuation_and_drops_stopwords():
    assert tokenize("The Capital, of France!") == ["capital", "france"]


def test_query_terms_are_unique_and_ordered():
    assert query_terms("capital of the capital of France") == ["capital", "france"]


def test_idf_is_never_negative_even_for_ubiquitous_terms():
    assert bm25_idf(df=100, n_docs=100) > 0
    assert bm25_idf(df=1, n_docs=100) > bm25_idf(df=50, n_docs=100)


def test_parse_line_splits_title_and_body():
    pid, title, body = parse_line(json.dumps({"id": 7, "contents": '"Paris"\nCapital of France.'}))
    assert (pid, title, body) == ("7", "Paris", "Capital of France.")


def test_stats_count_documents_and_document_frequency():
    stats = collect_stats(_open, {"paris", "cats", "zzz"}, workers=1, batch=7)
    assert stats.n_docs == len(DOCS)
    assert stats.df["paris"] == 15  # Paris, France and Seine passages, five copies each
    assert stats.df["cats"] == 5
    assert stats.df.get("zzz", 0) == 0
    assert stats.avgdl > 0


def _brute_force(query, stats, k):
    terms = query_terms(query)
    scored = []
    for i, (t, b) in enumerate(DOCS):
        scored.append((score_text(terms, t + " " + b, stats), i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(s, str(i)) for s, i in scored[:k] if s > 0]


@pytest.mark.parametrize("workers", [1, 2])
def test_streamed_ranking_equals_brute_force(workers):
    queries = {"q1": "What is the capital of France?", "q2": "river Seine Paris", "q3": "carnivorous mammals pets"}
    terms = {t for q in queries.values() for t in query_terms(q)}
    stats = collect_stats(_open, terms, workers=workers, batch=4)
    got = retrieve(_open, queries, stats, k=5, workers=workers, batch=4)
    for qid, q in queries.items():
        expected = _brute_force(q, stats, 5)
        assert [(round(p["bm25"], 9), p["id"]) for p in got[qid]] == [
            (round(s, 9), pid) for s, pid in expected
        ]


def test_a_query_with_no_known_term_gets_an_empty_list():
    stats = collect_stats(_open, {"zzzz"}, workers=1)
    assert retrieve(_open, {"q": "zzzz"}, stats, k=3)["q"] == []


def test_more_matching_terms_rank_higher():
    stats = collect_stats(_open, {"river", "seine", "paris"}, workers=1)
    top = retrieve(_open, {"q": "river Seine Paris"}, stats, k=1)["q"][0]
    assert top["title"] in {"Seine", "Paris"}


def test_iter_zip_lines_streams_a_deflated_member(tmp_path):
    path = tmp_path / "c.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("c.jsonl", b"\n".join(_lines()) + b"\n")
    streamed = list(iter_zip_lines(str(path), chunk=64))
    assert streamed == _lines()


def test_score_is_zero_without_a_shared_term_and_positive_with_one():
    stats = CorpusStats(n_docs=100, avgdl=10.0, df={"paris": 5})
    assert score_text(["paris"], "unrelated words here", stats) == 0.0
    assert score_text(["paris"], "paris is lovely", stats) > 0.0
    assert math.isfinite(score_text(["paris"], "paris", stats))
