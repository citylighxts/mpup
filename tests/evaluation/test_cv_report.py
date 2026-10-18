import numpy as np
import pytest

from experiments.cv_report import cv_report, lodo_report


def _linear_fit_predict(X_tr, y_tr, X_te, g_tr=None, g_te=None):
    A = np.c_[X_tr, np.ones(len(X_tr))]
    w, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    return np.c_[X_te, np.ones(len(X_te))] @ w


def _data(n_queries=40, per_query=4, seed=0, signal=1.0):
    rng = np.random.default_rng(seed)
    q = np.repeat(np.arange(n_queries), per_query).astype(str)
    X = rng.normal(size=(n_queries * per_query, 3))
    y = signal * X[:, 0] + rng.normal(0, 0.3, len(X))
    return X, y, q


def test_cv_report_scores_a_real_signal_and_a_useless_baseline():
    X, y, q = _data()
    rep = cv_report(
        X, y, q, _linear_fit_predict,
        baselines={"noise": lambda X: X[:, 2]},  # unrelated column
        n_splits=4, n_repeats=3, seed=0,
    )
    assert rep["n_queries"] == 40
    mean, lo, hi = rep["models"]["mpup"]["spearman"]
    assert lo <= mean <= hi and mean > 0.6
    assert abs(rep["models"]["noise"]["spearman"][0]) < 0.3
    d_mean, d_lo, d_hi = rep["paired_vs"]["noise"]["spearman"]
    assert d_lo > 0  # MPUP beats the noise baseline, and the interval says so


def test_cv_report_is_deterministic_for_a_seed():
    X, y, q = _data()
    a = cv_report(X, y, q, _linear_fit_predict, n_splits=4, n_repeats=2, seed=5)
    b = cv_report(X, y, q, _linear_fit_predict, n_splits=4, n_repeats=2, seed=5)
    assert a["models"]["mpup"]["spearman"] == b["models"]["mpup"]["spearman"]


def test_lodo_report_holds_out_each_dataset():
    Xa, ya, qa = _data(seed=1)
    Xb, yb, qb = _data(seed=2)
    X = np.concatenate([Xa, Xb])
    y = np.concatenate([ya, yb])
    q = np.concatenate([np.char.add("a", qa), np.char.add("b", qb)])
    datasets = np.array(["a"] * len(ya) + ["b"] * len(yb))
    rep = lodo_report(X, y, q, datasets, _linear_fit_predict)
    assert set(rep["per_dataset"]) == {"a", "b"}
    for name in ("a", "b"):
        assert rep["per_dataset"][name]["mpup"]["spearman"][0] > 0.6
        assert rep["per_dataset"][name]["n_queries"] == 40
    assert rep["macro"]["mpup"]["spearman"] > 0.6


def test_lodo_report_needs_two_datasets():
    X, y, q = _data()
    with pytest.raises(ValueError):
        lodo_report(X, y, q, np.array(["a"] * len(y)), _linear_fit_predict)


def test_cv_report_scores_extra_models_on_the_same_folds_with_paired_differences():
    from experiments.cv_report import columns

    X, y, q = _data()
    rep = cv_report(
        X, y, q, _linear_fit_predict,
        extra_models={"without_signal_column": columns(_linear_fit_predict, [1, 2])},
        n_splits=4, n_repeats=3, seed=1,
    )
    full = rep["models"]["mpup"]["spearman"][0]
    dropped = rep["models"]["without_signal_column"]["spearman"][0]
    assert full > dropped + 0.3  # the informative column is what carries the signal
    assert rep["paired_vs"]["without_signal_column"]["spearman"][1] > 0  # lower bound above zero


def test_cv_report_breaks_results_down_by_dataset():
    Xa, ya, qa = _data(seed=1)
    Xb, yb, qb = _data(seed=2, signal=0.0)  # dataset b carries no signal at all
    X, y = np.concatenate([Xa, Xb]), np.concatenate([ya, yb])
    q = np.concatenate([np.char.add("a", qa), np.char.add("b", qb)])
    datasets = np.array(["a"] * len(ya) + ["b"] * len(yb))
    rep = cv_report(X, y, q, _linear_fit_predict, datasets=datasets, n_splits=4, n_repeats=2, seed=0)
    assert set(rep["per_dataset"]) == {"a", "b"}
    assert rep["per_dataset"]["a"]["n_queries"] == 40
    assert rep["per_dataset"]["a"]["models"]["mpup"]["spearman"][0] > rep["per_dataset"]["b"]["models"]["mpup"]["spearman"][0] + 0.3


def test_lodo_report_includes_paired_differences_against_baselines():
    Xa, ya, qa = _data(seed=1)
    Xb, yb, qb = _data(seed=2)
    X, y = np.concatenate([Xa, Xb]), np.concatenate([ya, yb])
    q = np.concatenate([np.char.add("a", qa), np.char.add("b", qb)])
    datasets = np.array(["a"] * len(ya) + ["b"] * len(yb))
    rep = lodo_report(X, y, q, datasets, _linear_fit_predict, baselines={"noise": lambda M: M[:, 2]})
    assert rep["per_dataset"]["a"]["paired_vs"]["noise"]["spearman"][1] > 0
