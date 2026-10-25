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


# ------------------------------------------------------------------- grid v2 / extended layout

from experiments.cv_report import make_fit_predict
from experiments.nested_selection import GRID, grid_v2


def _extended(seed, hidden=8, informative_features=True):
    X, y, q = _simpson(seed)
    if not informative_features:
        X = np.random.default_rng(seed).normal(size=X.shape)
    rng = np.random.default_rng(seed + 100)
    Xc = X - np.repeat(np.array([X[q == u].mean(0) for u in dict.fromkeys(q)]), 5, axis=0)
    yc = y - np.repeat(np.array([y[q == u].mean() for u in dict.fromkeys(q)]), 5)
    H = rng.normal(size=(len(y), hidden))
    H[:, 0] += 3.0 * yc  # the hidden states carry the within-query label
    return np.c_[X, H, y + rng.normal(0, 0.1, len(y))], y, q, Xc


def test_grid_v2_names_and_base_equals_the_frozen_config():
    names = [c.name for c in grid_v2()]
    assert names == ["base", "base-noD2D4", "base+probe", "base-rank", "base+smooth"]
    frozen = next(c for c in GRID if c.name == "gbr(100,d2)-centered")
    base = grid_v2()[0]
    assert (base.algorithm, base.center_by_query, base.params) == (frozen.algorithm, frozen.center_by_query, frozen.params)
    combined = grid_v2(combined=("base-noD2D4", "base+probe"))[-1]
    assert combined.name == "combined" and combined.probe and combined.drop_groups == ("d2", "d4") and combined.algorithm == "xgboost"


def test_drop_groups_ignores_those_columns():
    Xe, y, q, _ = _extended(1)
    fp = make_fit_predict("ridge", True, drop_groups=("d2", "d4"), alpha=1.0)
    base = fp(Xe[:100], y[:100], Xe[100:], q[:100], q[100:])
    Xe2 = Xe.copy()
    Xe2[:, 4:7] = 99.0  # D2
    Xe2[:, 9:11] = -7.0  # D4
    np.testing.assert_allclose(fp(Xe2[:100], y[:100], Xe2[100:], q[:100], q[100:]), base)


def test_smooth_label_trains_on_the_last_column():
    Xe, y, q, _ = _extended(2)
    fp = make_fit_predict("ridge", True, smooth_label=True, alpha=1.0)
    plain = make_fit_predict("ridge", True, alpha=1.0)
    Xs = Xe.copy()
    Xs[:, -1] = -Xs[:, -1]  # flip the smooth label: predictions must flip sign, those on the true label must not
    a = fp(Xe[:100], y[:100], Xe[100:], q[:100], q[100:])
    b = fp(Xs[:100], y[:100], Xs[100:], q[:100], q[100:])
    np.testing.assert_allclose(a, -b, atol=1e-8)
    np.testing.assert_allclose(plain(Xe[:100], y[:100], Xe[100:], q[:100], q[100:]), plain(Xs[:100], y[:100], Xs[100:], q[:100], q[100:]))


def test_smooth_label_without_the_column_raises():
    Xe, y, q, _ = _extended(2)
    with pytest.raises(ValueError, match="smooth"):
        make_fit_predict("ridge", True, smooth_label=True)(Xe[:100, :14], y[:100], Xe[100:, :14], q[:100], q[100:])


def test_probe_variant_uses_hidden_states_and_beats_features_alone():
    Xe, y, q, _ = _extended(3, informative_features=False)
    plain = make_fit_predict("ridge", True, alpha=1.0)
    probe = make_fit_predict("ridge", True, probe=True, alpha=1.0)
    from src.evaluation.protocol import per_query_scores
    s = lambda p: per_query_scores(y[100:], p, q[100:])["spearman"].mean()  # noqa: E731
    assert s(probe(Xe[:100], y[:100], Xe[100:], q[:100], q[100:])) > s(plain(Xe[:100], y[:100], Xe[100:], q[:100], q[100:])) + 0.1


def test_nested_lodo_runs_with_grid_v2_on_extended_toy_data():
    parts = [_extended(s) for s in (1, 2, 3)]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    q = np.concatenate([p[2] for p in parts])
    ds = np.concatenate([[f"d{i}"] * len(p[1]) for i, p in enumerate(parts)])
    rep = nested_lodo(X, y, q, ds, grid=grid_v2(), n_jobs=1)
    assert rep["grid"] == [c.name for c in grid_v2()] and set(rep["per_dataset"]) == {"d0", "d1", "d2"}


def test_a_fitted_model_predicts_exactly_what_fit_predict_returns():
    from experiments.cv_report import make_fitter

    Xe, y, q, _ = _extended(4)
    for kw in ({}, {"probe": True}, {"drop_groups": ("d2", "d4")}):
        direct = make_fit_predict("ridge", True, alpha=1.0, **kw)(Xe[:100], y[:100], Xe[100:], q[:100], q[100:])
        predict = make_fitter("ridge", True, alpha=1.0, **kw)(Xe[:100], y[:100], q[:100])
        np.testing.assert_allclose(predict(Xe[100:], q[100:]), direct)
