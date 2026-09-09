"""Feature ablation study: remove D1-D5 one at a time, measure ρ drop."""
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
from src.features.concatenate import FEATURE_GROUPS, FEATURE_DIM


def run_ablation(X: np.ndarray, y_true: np.ndarray, predictor) -> dict:
    """
    For each feature group, zero it out and measure Spearman ρ.
    Returns dict of {ablated_group: rho_drop}.
    """
    baseline_pred = predictor.predict(X)
    baseline_rho, _ = spearmanr(y_true, baseline_pred)
    results = {"baseline": float(baseline_rho)}

    for group_name, slc in FEATURE_GROUPS.items():
        X_ablated = X.copy()
        X_ablated[:, slc] = 0.0
        pred = predictor.predict(X_ablated)
        rho, _ = spearmanr(y_true, pred)
        results[f"ablate_{group_name}"] = float(rho)
        results[f"drop_{group_name}"] = float(baseline_rho - rho)

    return results


if __name__ == "__main__":
    import json, sys
    print("Run ablation after loading your trained predictor and feature matrix.")
    print("Usage: call run_ablation(X, y_true, predictor) in your experiment script.")
