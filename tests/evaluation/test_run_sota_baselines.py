import json

import numpy as np
import pytest

from experiments import run_big
from experiments.run_sota_baselines import group_rows, per_query_table, row_texts, score_rows
from src.evaluation.protocol import LABEL_UTILITY


def _toy_dir(tmp_path, n_q=6, k=4):
    rng = np.random.default_rng(0)
    (tmp_path / "features").mkdir()
    (tmp_path / "extra").mkdir()
    items, qids, pids = [], [], []
    for i in range(n_q):
        items.append({"dataset": "nq", "query_id": f"id{i}", "query": f"question {i}",
                      "passages": [{"id": f"id{i}:p{j}", "text": f"text {i}-{j}"} for j in range(k)]})
        qids += [f"id{i}"] * k
        pids += [f"id{i}:p{j}" for j in range(k)]
    n = n_q * k
    np.savez(tmp_path / "features" / "nq.npz", X=rng.normal(size=(n, 14)), y=rng.normal(size=n), query_ids=np.array(qids),
             passage_ids=np.array(pids), dataset=np.array(["nq"] * n), label_source=np.array(LABEL_UTILITY))
    np.savez(tmp_path / "extra" / "nq.npz", passage_ids=np.array(pids), soft=rng.normal(size=n), ce=rng.normal(size=n),
             hidden_21=rng.normal(size=(n, 4)).astype(np.float16))
    (tmp_path / "items.json").write_text(json.dumps(items))
    return tmp_path


def test_row_texts_follow_the_rows_of_the_larger_study(tmp_path):
    d = run_big.load(_toy_dir(tmp_path))
    qtext, ptext = row_texts(tmp_path)
    assert len(qtext) == len(ptext) == len(d["y"])
    for q, qt, pt in zip(d["q"], qtext, ptext):
        i, j = q.split(":id")[1], None
        assert qt == f"question {i}" and pt.startswith(f"text {i}-")
    assert ptext[0] == "text 0-0" and ptext[5] == "text 1-1"


def test_group_rows_gives_the_positions_of_each_query_in_row_order():
    q = np.array(["a", "a", "b", "a", "b"])
    groups = group_rows(q, np.array([0, 1, 2, 3, 4]))
    assert [g.tolist() for g in groups] == [[0, 1, 3], [2, 4]]
    assert [g.tolist() for g in group_rows(q, np.array([1, 2, 4]))] == [[0], [1, 2]]  # positions inside idx


def test_score_rows_calls_the_scorer_once_per_query_and_keeps_the_row_order():
    q = np.array(["a", "b", "a", "b"])
    qtext, ptext = np.array(["Qa", "Qb", "Qa", "Qb"]), np.array(["x", "yy", "zzz", "w"])
    calls = []

    def scorer(query, passages):
        calls.append((query, tuple(passages)))
        return np.array([len(p) for p in passages], float)

    out = score_rows(scorer, qtext, ptext, q, np.arange(4))
    assert out.tolist() == [1, 2, 3, 1] and calls == [("Qa", ("x", "zzz")), ("Qb", ("yy", "w"))]


def test_per_query_table_has_one_r_s_per_usable_query_for_every_method():
    rng = np.random.default_rng(1)
    q = np.repeat([f"q{i}" for i in range(10)], 5)
    y = rng.normal(size=len(q))
    preds = {"good": y + 0.1 * rng.normal(size=len(q)), "noise": rng.normal(size=len(q))}
    table = per_query_table(y, q, preds)
    assert set(table) == {"good", "noise"} and len(table["good"]) == len(table["noise"]) == 10
    assert table["good"].mean() > 0.8 and abs(table["noise"].mean()) < 0.4


def test_the_cd_command_ranks_mpup_and_all_baselines_on_the_same_test_queries(tmp_path):
    from argparse import Namespace

    from experiments.run_sota_baselines import cmd_cd

    (tmp_path / "big").mkdir()
    data = _toy_dir(tmp_path / "big", n_q=80, k=5)
    d = run_big.load(data)
    test = run_big.split(d["q"], set(), 0.3)
    te = np.flatnonzero(test)
    rng = np.random.default_rng(3)
    scores = tmp_path / "scores.npz"
    np.savez(scores, test_rows=te, gain=rng.normal(size=len(te)), judge=rng.normal(size=len(te)))
    out = tmp_path / "cd"
    cmd_cd(Namespace(data=str(data), seen=[], test_share=0.3, scores=str(scores), variant="gbr(100,d2)|u|ce|probe",
                     alpha=0.05, out=str(out)))
    report = json.loads(out.with_suffix(".json").read_text())
    assert set(report["average_rank"]) == {"MPUP", "GainRAG-style", "LLM judgment", "Cross-encoder", "Dense", "BM25"}
    assert report["cd"] > 0 and 0 <= report["friedman_p"] <= 1
    assert out.with_suffix(".tex").read_text().startswith("\\begin{tikzpicture}")
    assert report["n_queries"] == len(np.load(out.with_suffix(".npz"))["MPUP"])
