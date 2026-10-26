import json
from argparse import Namespace

import numpy as np

from experiments.run_mac_sota import cmd_cd, load_rows
from src.evaluation.protocol import LABEL_UTILITY


def _mac_dir(path, n_q=30, k=5, seed=0, signal=True):
    """features/, hidden/ (two layers of 4 dims), loglik/ and items.json, as `extract_multidataset` writes them."""
    rng = np.random.default_rng(seed)
    for sub in ("features", "hidden", "loglik"):
        (path / sub).mkdir(parents=True)
    items, qids, pids = [], [], []
    for i in range(n_q):
        items.append({"dataset": "nq", "query_id": f"s{seed}q{i}", "query": f"question {i}",
                      "passages": [{"id": f"s{seed}q{i}:p{j}", "text": f"text {i}-{j}"} for j in range(k)]})
        qids += [f"s{seed}q{i}"] * k
        pids += [f"s{seed}q{i}:p{j}" for j in range(k)]
    n = n_q * k
    H = rng.normal(size=(n, 2, 4)).astype(np.float16)
    y = (H[:, 0, 0] if signal else rng.normal(size=n)) + 0.1 * rng.normal(size=n)
    np.savez(path / "features" / "nq.npz", X=rng.normal(size=(n, 14)), y=y, query_ids=np.array(qids),
             passage_ids=np.array(pids), dataset=np.array(["nq"] * n), label_source=np.array(LABEL_UTILITY))
    np.savez(path / "hidden" / "nq.npz", H=H, query_ids=np.array(qids), passage_ids=np.array(pids))
    np.savez(path / "loglik" / "nq.npz", u_smooth=rng.normal(size=n), query_ids=np.array(qids), passage_ids=np.array(pids))
    (path / "items.json").write_text(json.dumps(items))
    return path


def test_load_rows_joins_features_hidden_states_and_texts_in_row_order(tmp_path):
    rows = load_rows(_mac_dir(tmp_path / "a", n_q=4, k=3))
    assert rows["X"].shape == (12, 14 + 2 * 4 + 1) and len(rows["y"]) == len(rows["q"]) == 12
    assert rows["q"][0] == "nq:s0q0" and rows["qtext"][4] == "question 1" and rows["ptext"][4] == "text 1-1"


def test_the_cd_command_ranks_mpup_and_the_baselines_on_the_test_sample(tmp_path):
    train = _mac_dir(tmp_path / "train", n_q=60, seed=1)
    test = _mac_dir(tmp_path / "test", n_q=30, seed=2)
    rng = np.random.default_rng(5)
    scores = tmp_path / "scores.npz"
    np.savez(scores, ce=rng.normal(size=150), gain=rng.normal(size=150), judge=rng.normal(size=150))
    out = tmp_path / "cd"
    report = cmd_cd(Namespace(train=str(train), test=str(test), scores=str(scores), config="base+probe",
                              alpha=0.05, out=str(out)))
    assert set(report["average_rank"]) == {"MPUP", "GainRAG-style", "LLM judgment", "Cross-encoder", "Dense", "BM25"}
    assert report["n_queries"] == 30 and report["cd"] > 0
