import numpy as np
import pytest
from src.predictor.mpup_predictor import MPUPPredictor
from experiments.evaluate_baselines import run_evaluation


@pytest.fixture
def trained_pred():
    # Labels depend on D1+D2 features, not D5 — so MPUP (all features) outperforms Tian (D5-only)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 14))
    y = X[:, 0] * 0.4 + X[:, 4] * 0.4 + rng.standard_normal(50) * 0.1  # D1+D2 driven
    pred = MPUPPredictor(algorithm="xgboost", n_estimators=20, max_depth=3)
    pred.fit(X[:40], y[:40])
    return pred, X[40:], y[40:]


def test_all_baseline_keys_present(trained_pred):
    pred, X_val, y_val = trained_pred
    results = run_evaluation(X_val, y_val, pred, k_shots_list=[5], ndcg_k=5, seed=42)
    for key in ["no_retrieval", "full_unranked", "tian_et_al_2026", "mpup_zero_shot", "mpup_few_shot_k5"]:
        assert key in results, f"Missing key: {key}"


def test_spearman_rho_in_results(trained_pred):
    pred, X_val, y_val = trained_pred
    results = run_evaluation(X_val, y_val, pred, k_shots_list=[5], ndcg_k=5, seed=42)
    for key, val in results.items():
        assert "spearman_rho" in val, f"{key} missing spearman_rho"
        assert -1.0 <= val["spearman_rho"] <= 1.0


def test_mpup_outperforms_tian(trained_pred):
    pred, X_val, y_val = trained_pred
    results = run_evaluation(X_val, y_val, pred, k_shots_list=[5], ndcg_k=5, seed=42)
    rho_tian = results["tian_et_al_2026"]["spearman_rho"]
    rho_mpup = results["mpup_zero_shot"]["spearman_rho"]
    assert rho_mpup >= rho_tian


def test_deterministic_with_same_seed(trained_pred):
    pred, X_val, y_val = trained_pred
    r1 = run_evaluation(X_val, y_val, pred, k_shots_list=[5], ndcg_k=5, seed=99)
    r2 = run_evaluation(X_val, y_val, pred, k_shots_list=[5], ndcg_k=5, seed=99)
    assert r1["mpup_few_shot_k5"]["spearman_rho"] == r2["mpup_few_shot_k5"]["spearman_rho"]
