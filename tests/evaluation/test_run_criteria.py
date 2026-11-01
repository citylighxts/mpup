"""Helpers of the C3-C6 experiments: column views, answer labels, and the parsing of a listwise ranking."""
import json

import numpy as np

from experiments import run_criteria
from experiments.run_probe_baselines import parse_ranking


def test_view_keeps_rows_and_replaces_only_what_is_given():
    d = {"X": np.arange(20.0).reshape(2, 10), "y": np.array([1.0, 2.0]), "H": np.zeros((2, 3)), "q": np.array(["a", "a"])}
    v = run_criteria.view(d, [0, 3], y=np.array([0.0, 1.0]))
    assert v["X"].tolist() == [[0.0, 3.0], [10.0, 13.0]]
    assert v["y"].tolist() == [0.0, 1.0] and v["H"] is d["H"]
    assert d["y"].tolist() == [1.0, 2.0]  # the original is not changed


def test_no_generation_drops_exactly_the_hyde_columns():
    assert run_criteria.NO_GEN == [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13]
    assert run_criteria.LLM_FREE == list(range(8))


def test_golds_parses_the_stored_list_string():
    assert run_criteria._golds("['May 18, 2018']") == ["May 18, 2018"]
    assert run_criteria._golds(["a", "b"]) == ["a", "b"]
    assert run_criteria._golds("plain answer") == ["plain answer"]


def test_answer_labels_follow_the_relaxed_match(tmp_path):
    (tmp_path / "features").mkdir()
    (tmp_path / "extra").mkdir()
    items = [{"query_id": "nq:1", "gold_answers": "['Barack Obama']", "passages": [
                 {"id": "p0", "text": "In 2009 barack obama took office."}, {"id": "p1", "text": "Obama alone."}]},
             {"query_id": "nq:2", "gold_answers": "['yes']", "passages": [{"id": "p2", "text": "Yes, it is."}]}]
    (tmp_path / "items.json").write_text(json.dumps(items))
    np.savez(tmp_path / "features" / "nq.npz", label_source="utility_eq1", query_ids=np.array(["nq:1", "nq:1", "nq:2"]),
             passage_ids=np.array(["p0", "p1", "p2"]))
    np.savez(tmp_path / "extra" / "nq.npz", passage_ids=np.array(["p0", "p1", "p2"]))
    lab, yesno = run_criteria.answer_labels(tmp_path)
    assert lab.tolist() == [1.0, 0.0, 1.0]
    assert yesno.tolist() == [False, False, True]


def test_parse_ranking_scores_by_position_and_puts_missing_last():
    s = parse_ranking("[3] > [1] > [3] > [9]", 4)
    assert s.tolist() == [-1.0, -4.0, 0.0, -4.0]
    assert parse_ranking("no ranking", 2).tolist() == [-2.0, -2.0]


def test_reader_features_are_aligned_and_missing_datasets_give_none(tmp_path):
    for sub in ("features", "extra", "features_r"):
        (tmp_path / sub).mkdir()
    for name, pids in (("a", ["p0", "p1"]), ("b", ["p2"])):
        np.savez(tmp_path / "features" / f"{name}.npz", label_source="utility_eq1", passage_ids=np.array(pids))
        np.savez(tmp_path / "extra" / f"{name}.npz", passage_ids=np.array(pids))
    np.savez(tmp_path / "features_r" / "a.npz", passage_ids=np.array(["p0", "p1"]), X=np.ones((2, 14)))
    assert run_criteria.load_reader_X(tmp_path, "features_r") is None
    np.savez(tmp_path / "features_r" / "b.npz", passage_ids=np.array(["p2"]), X=np.full((1, 14), 2.0))
    X = run_criteria.load_reader_X(tmp_path, "features_r")
    assert X.shape == (3, 14) and X[2, 0] == 2.0
