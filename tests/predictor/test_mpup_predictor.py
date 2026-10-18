import numpy as np
import pytest
from src.predictor.mpup_predictor import MPUPPredictor


@pytest.fixture
def trained_predictor(tiny_feature_matrix):
    X, y = tiny_feature_matrix
    pred = MPUPPredictor(algorithm="xgboost", n_estimators=10, max_depth=3)
    pred.fit(X[:40], y[:40])
    return pred, X[40:], y[40:]


def test_predict_shape(trained_predictor):
    pred, X_val, _ = trained_predictor
    out = pred.predict(X_val)
    assert out.shape == (len(X_val),)


def test_predict_returns_floats(trained_predictor):
    pred, X_val, _ = trained_predictor
    out = pred.predict(X_val)
    assert out.dtype in [np.float32, np.float64]


def test_save_and_load(trained_predictor, tmp_path):
    pred, X_val, _ = trained_predictor
    pred.save(tmp_path / "mpup")
    loaded = MPUPPredictor.load(tmp_path / "mpup", algorithm="xgboost")
    np.testing.assert_array_almost_equal(
        pred.predict(X_val), loaded.predict(X_val)
    )


def test_mlp_algorithm_works(tiny_feature_matrix):
    X, y = tiny_feature_matrix
    pred = MPUPPredictor(algorithm="mlp", hidden_dims=[32, 16], epochs=5)
    pred.fit(X[:40], y[:40])
    out = pred.predict(X[40:])
    assert out.shape == (10,)


# ------------------------------------------------------- query-centred training

from src.predictor.mpup_predictor import center_by_group


def test_center_by_group_removes_each_groups_mean():
    v = np.array([1.0, 3.0, 10.0, 20.0])
    out = center_by_group(v, ["a", "a", "b", "b"])
    assert out.tolist() == [-1.0, 1.0, -5.0, 5.0]
    M = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 0.0]])
    out = center_by_group(M, [1, 1, 2])
    assert out.tolist() == [[-1.0, -10.0], [1.0, 10.0], [0.0, 0.0]]
    assert center_by_group(v).mean() == pytest.approx(0.0)  # no groups: one group


def test_ridge_algorithm_learns_a_linear_signal(tiny_feature_matrix):
    X, y = tiny_feature_matrix
    pred = MPUPPredictor(algorithm="ridge", alpha=1.0)
    pred.fit(X[:40], y[:40])
    assert np.corrcoef(pred.predict(X[40:]), y[40:])[0, 1] > 0.8


def _grouped(n_queries=20, per_query=5, seed=0):
    """Simpson's paradox: across queries a feature rises with the label, within a query it falls."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_queries), per_query)
    level = np.repeat(rng.normal(0, 5, n_queries), per_query)   # per-query level (easy vs hard question)
    within = rng.normal(0, 1, len(g))
    X = np.c_[level + within, rng.normal(size=(len(g), 3))]
    y = -within + 3.0 * level + rng.normal(0, 0.1, len(g))
    return X, y, g


def test_centered_predictor_needs_groups_to_fit():
    X, y, g = _grouped()
    with pytest.raises(ValueError, match="groups"):
        MPUPPredictor(algorithm="ridge", center_by_query=True).fit(X, y)


def test_centered_predictions_ignore_a_per_query_shift_of_the_features():
    X, y, g = _grouped()
    pred = MPUPPredictor(algorithm="ridge", center_by_query=True, alpha=1.0)
    pred.fit(X, y, groups=g)
    shifted = X + np.repeat(np.arange(20), 5)[:, None] * 3.0  # move every query's features around
    np.testing.assert_allclose(pred.predict(X, groups=g), pred.predict(shifted, groups=g), atol=1e-9)


def test_centered_predictor_ranks_within_a_query_better_than_the_raw_one():
    from src.evaluation.protocol import evaluate_ranking

    X, y, g = _grouped(n_queries=40, seed=1)
    tr, te = np.flatnonzero(g < 30), np.flatnonzero(g >= 30)
    raw = MPUPPredictor(algorithm="ridge", alpha=1.0)
    raw.fit(X[tr], y[tr])
    cen = MPUPPredictor(algorithm="ridge", center_by_query=True, alpha=1.0)
    cen.fit(X[tr], y[tr], groups=g[tr])
    r_raw = evaluate_ranking(y[te], raw.predict(X[te]), g[te])["spearman"]
    r_cen = evaluate_ranking(y[te], cen.predict(X[te], groups=g[te]), g[te])["spearman"]
    assert r_raw < 0.0 and r_cen > 0.9  # the raw model learned the between-query trend and inverts the order


def test_save_and_load_keep_the_algorithm_and_the_centering_flag(tmp_path):
    X, y, g = _grouped()
    pred = MPUPPredictor(algorithm="ridge", center_by_query=True, alpha=3.0)
    pred.fit(X, y, groups=g)
    pred.save(tmp_path / "m")
    loaded = MPUPPredictor.load(tmp_path / "m")
    assert loaded.algorithm == "ridge" and loaded.center_by_query is True
    np.testing.assert_allclose(pred.predict(X, groups=g), loaded.predict(X, groups=g))
