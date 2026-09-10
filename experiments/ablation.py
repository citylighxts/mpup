"""Feature ablation study — leave-one-group-out with retraining.

Two views are reported per feature group D1–D5:
  - leave_one_out: retrain on the other 4 groups, measure ρ / NDCG loss
  - group_only:    retrain on that group alone, measure standalone strength

Retraining (not inference-time zeroing) is the honest ablation: a predictor
trained with a group cannot simply have that group masked at test time without
producing misleading numbers.
"""
import numpy as np
from scipy.stats import spearmanr

from src.features.concatenate import FEATURE_GROUPS
from src.evaluation.metrics import spearman_rho, ndcg_at_k
from src.predictor.mpup_predictor import MPUPPredictor


def run_ablation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ndcg_k: int = 10,
    predictor_kwargs: dict | None = None,
) -> dict:
    predictor_kwargs = predictor_kwargs or {"algorithm": "xgboost", "n_estimators": 200, "max_depth": 5}
    all_cols = list(range(X_train.shape[1]))

    def fit_eval(cols: list[int]) -> tuple[float, float]:
        p = MPUPPredictor(**predictor_kwargs)
        p.fit(X_train[:, cols], y_train)
        pred = p.predict(X_test[:, cols])
        return spearman_rho(y_test, pred), ndcg_at_k(y_test, pred, k=ndcg_k)

    base_rho, base_ndcg = fit_eval(all_cols)
    results = {"baseline": {"spearman_rho": base_rho, "ndcg_at_k": base_ndcg}}

    for name, slc in FEATURE_GROUPS.items():
        drop = set(range(slc.start, slc.stop))
        loo_cols = [c for c in all_cols if c not in drop]
        only_cols = list(range(slc.start, slc.stop))

        loo_rho, loo_ndcg = fit_eval(loo_cols)
        only_rho, only_ndcg = fit_eval(only_cols)

        results[f"leave_out_{name}"] = {
            "spearman_rho": loo_rho,
            "ndcg_at_k": loo_ndcg,
            "delta_rho": loo_rho - base_rho,
        }
        results[f"only_{name}"] = {
            "spearman_rho": only_rho,
            "ndcg_at_k": only_ndcg,
        }

    return results
