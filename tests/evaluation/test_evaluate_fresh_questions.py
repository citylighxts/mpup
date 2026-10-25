import numpy as np

from experiments.evaluate_fresh_questions import run, to_markdown
from experiments.nested_selection import Config
from src.evaluation.protocol import LABEL_UTILITY

GRID = [Config("ridge-centered", "ridge", True, (("alpha", 10.0),))]


def _write(dirpath, name, seed, prefix, n_queries=30, per_query=5):
    rng = np.random.default_rng(seed)
    n = n_queries * per_query
    X = rng.normal(size=(n, 14))
    X[:, 1] = rng.normal(size=n)                      # dense cosine: noise
    y = X[:, 13] + rng.normal(0, 0.3, n)              # delta_h carries the utility
    q = np.repeat([f"{prefix}:{name}:{i}" for i in range(n_queries)], per_query)
    (dirpath / "features").mkdir(parents=True, exist_ok=True)
    np.savez(dirpath / "features" / f"{name}.npz", X=X, y=y, query_ids=q, passage_ids=np.arange(n).astype(str),
             dataset=np.array([name] * n), label_source=np.array(LABEL_UTILITY))
    return [{"query_id": str(x)} for x in dict.fromkeys(q.tolist())]


def _dirs(tmp_path):
    import json

    train, fresh = tmp_path / "train", tmp_path / "fresh"
    items = []
    for i, name in enumerate("ab"):
        items += _write(train, name, i + 1, "t")
        _write(fresh, name, i + 10, "f")
    (train / "items.json").write_text(json.dumps(items))
    return train, fresh


def test_criterion_is_met_when_the_pooled_gain_clears_the_threshold(tmp_path):
    train, fresh = _dirs(tmp_path)
    r = run(train, fresh, "ridge-centered", grid=GRID, min_gain=0.08)
    m, lo, hi = r["paired_mpup_vs"]["dense"]["spearman"]
    assert m > 0.08 and lo > 0 and r["criterion"]["met"] is True and r["criterion"]["min_gain"] == 0.08
    assert "criterion met" in to_markdown(r)


def test_criterion_is_not_met_when_the_threshold_is_out_of_reach(tmp_path):
    train, fresh = _dirs(tmp_path)
    r = run(train, fresh, "ridge-centered", grid=GRID, min_gain=5.0)
    assert r["criterion"]["met"] is False and "criterion not met" in to_markdown(r)


def test_no_criterion_without_min_gain(tmp_path):
    train, fresh = _dirs(tmp_path)
    assert "criterion" not in run(train, fresh, "ridge-centered", grid=GRID)
