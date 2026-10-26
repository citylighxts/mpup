import numpy as np
import pytest

from experiments import run_big
from src.predictor.mpup_plus import MPUP_PLUS_PARAMS, MPUPPlusPredictor


def _toy(n_queries=40, per_query=5, d=12, seed=0):
    rng = np.random.default_rng(seed)
    q = np.repeat([f"ds:{i}" for i in range(n_queries)], per_query)
    X = rng.normal(size=(len(q), 14))
    ce = rng.normal(size=len(q))
    H = rng.normal(size=(len(q), d)).astype(np.float32)
    y = 0.5 * X[:, 13] + 0.3 * ce + 0.4 * H[:, 0] + rng.normal(0, 0.2, len(q))
    return X, ce, H, y, q


def test_the_defaults_are_the_variant_kept_for_testing_in_the_larger_study():
    assert MPUP_PLUS_PARAMS == {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03}
    assert run_big.MODELS["gbr(500,d4)"][1] == MPUP_PLUS_PARAMS


def test_predictions_equal_those_of_the_larger_study_for_the_same_split():
    X, ce, H, y, q = _toy()
    tr, te = np.arange(150), np.arange(150, 200)
    d = {"X": X, "y": y, "q": q, "ds": np.array(["ds"] * len(q)), "soft": y, "ce": ce, "H": H}
    expected = run_big.fit_predict(d, tr, te, "gbr(500,d4)|u|ce|probe")
    model = MPUPPlusPredictor().fit(X[tr], ce[tr], H[tr], y[tr], q[tr])
    np.testing.assert_allclose(model.predict(X[te], ce[te], H[te], q[te]), expected, rtol=1e-6, atol=1e-6)


def test_candidates_of_one_query_can_be_scored_without_groups():
    X, ce, H, y, q = _toy()
    model = MPUPPlusPredictor().fit(X, ce, H, y, q)
    one = slice(0, 5)
    np.testing.assert_allclose(model.predict(X[one], ce[one], H[one]), model.predict(X[one], ce[one], H[one], q[one]))


def test_save_and_load_give_the_same_scores(tmp_path):
    X, ce, H, y, q = _toy()
    model = MPUPPlusPredictor().fit(X, ce, H, y, q)
    model.save(tmp_path / "mpup_plus")
    loaded = MPUPPlusPredictor.load(tmp_path / "mpup_plus")
    np.testing.assert_allclose(loaded.predict(X[:5], ce[:5], H[:5]), model.predict(X[:5], ce[:5], H[:5]))


def test_predict_is_the_regressor_applied_to_the_sixteen_features():
    X, ce, H, y, q = _toy()
    model = MPUPPlusPredictor().fit(X, ce, H, y, q)
    X16 = model.features(X[:5], ce[:5], H[:5])
    assert X16.shape == (5, 16)
    np.testing.assert_allclose(X16[:, :14], X[:5])
    np.testing.assert_allclose(X16[:, 14], ce[:5])
    np.testing.assert_allclose(X16[:, 15], model.probe.predict(H[:5]))
    np.testing.assert_allclose(model.predict_features(X16), model.predict(X[:5], ce[:5], H[:5]))
