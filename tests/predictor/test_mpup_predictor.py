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
