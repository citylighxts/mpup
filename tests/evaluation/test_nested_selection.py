import numpy as np
import pytest

from experiments.nested_selection import PAPER, Config, nested_lodo, select_config

SMALL_GRID = [
    Config("ridge-raw", "ridge", False, (("alpha", 1.0),)),
    Config("ridge-centered", "ridge", True, (("alpha", 1.0),)),
]


def _simpson(seed, n_queries=30, per_query=5):
    """Across queries the feature rises with the label; within a query it falls (see the predictor tests)."""
    rng = np.random.default_rng(seed)
    q = np.repeat([f"s{seed}:{i}" for i in range(n_queries)], per_query)
    level = np.repeat(rng.normal(0, 5, n_queries), per_query)
    within = rng.normal(0, 1, len(q))
    X = np.c_[level + within, rng.normal(size=(len(q), 13))]
    y = -within + 3.0 * level + rng.normal(0, 0.1, len(q))
    return X, y, q


@pytest.fixture
def pool():
    parts = [_simpson(s) for s in (1, 2, 3)]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    q = np.concatenate([p[2] for p in parts])
    ds = np.concatenate([[f"d{i}"] * len(p[1]) for i, p in enumerate(parts)])
    return X, y, q, ds


def test_the_paper_configuration_is_part_of_the_grid():
    from experiments.nested_selection import GRID

    assert PAPER in GRID and PAPER.algorithm == "xgboost" and not PAPER.center_by_query


def test_selection_picks_the_config_that_ranks_within_queries(pool):
    X, y, q, ds = pool
    best, scores = select_config(X, y, q, ds, SMALL_GRID, n_jobs=1)
    assert best.name == "ridge-centered"
    assert scores["ridge-centered"] > scores["ridge-raw"] + 0.5


def test_nested_report_structure_and_that_it_beats_the_raw_model(pool):
    X, y, q, ds = pool
    rep = nested_lodo(X, y, q, ds, baselines={"noise": lambda M: M[:, 5]}, grid=SMALL_GRID, n_jobs=1)
    assert set(rep["per_dataset"]) == {"d0", "d1", "d2"}
    assert all(d["selected"] == "ridge-centered" for d in rep["per_dataset"].values())
    assert rep["selection_frequency"] == {"ridge-centered": 3}
    assert rep["macro"]["nested"]["spearman"] > 0.8
    assert rep["final_config"]["name"] == "ridge-centered" and rep["final_config"]["center_by_query"] is True
    d = rep["per_dataset"]["d0"]
    assert d["paired_vs"]["noise"]["spearman"][1] > 0  # lower CI bound above zero
    assert rep["optimism_of_oracle_selection"] >= -1e-9  # the oracle can only match or exceed
    assert rep["n_queries"] == sum(x["n_queries"] for x in rep["per_dataset"].values())
    lo = rep["pooled_paired_vs"]["noise"]["spearman"][1]
    assert lo > 0.5  # over all held-out queries together the gap to a noise feature is unmistakable


def test_selection_never_sees_the_held_out_datasets_labels(pool):
    """Scramble the labels of one dataset: the configuration chosen for that outer fold must not change."""
    X, y, q, ds = pool
    base = nested_lodo(X, y, q, ds, grid=SMALL_GRID, n_jobs=1)
    y2 = y.copy()
    m = ds == "d0"
    y2[m] = np.random.default_rng(9).permutation(y2[m])
    scrambled = nested_lodo(X, y2, q, ds, grid=SMALL_GRID, n_jobs=1)
    assert scrambled["per_dataset"]["d0"]["selected"] == base["per_dataset"]["d0"]["selected"]
    # inner scores for outer fold d0 are computed only from d1 and d2, so they must be identical
    assert scrambled["per_dataset"]["d0"]["inner_scores"] == base["per_dataset"]["d0"]["inner_scores"]
