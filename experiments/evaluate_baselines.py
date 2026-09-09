"""Evaluasi MPUP vs semua baseline (Notion doc Section 5 & 6).

Baseline pembanding:
  - no_retrieval          : lower bound — jawab tanpa dokumen sama sekali
  - full_unranked         : naive — pakai semua K kandidat tanpa ranking
  - tian_et_al_2026       : zero-shot logit probing only (D5 tanpa D1-D4), target ρ < 0.3
  - mpup_zero_shot        : MPUP tanpa kalibrasi
  - mpup_few_shot_k{k}    : MPUP dengan linear adapter dari k contoh

Output: results/benchmark_results.json
"""
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from src.evaluation.metrics import evaluate_all, spearman_rho, ndcg_at_k
from src.predictor.mpup_predictor import MPUPPredictor
from src.predictor.calibration import LinearCalibrator


# ── Feature group slices (harus sinkron dengan experiments/ablation.py) ──────
FEATURE_GROUPS = {
    "d1": slice(0, 4),
    "d2": slice(4, 7),
    "d3": slice(7, 9),
    "d4": slice(9, 11),
    "d5": slice(11, 14),
}


def _only_d5(X: np.ndarray) -> np.ndarray:
    """Tian et al. (2026) baseline: gunakan hanya fitur D5 (logit probe)."""
    X_d5 = np.zeros_like(X)
    X_d5[:, FEATURE_GROUPS["d5"]] = X[:, FEATURE_GROUPS["d5"]]
    return X_d5


def run_evaluation(
    X: np.ndarray,
    y_true: np.ndarray,
    predictor: MPUPPredictor,
    k_shots_list: list[int],
    ndcg_k: int = 10,
) -> dict:
    results = {}

    # ── No Retrieval baseline ─────────────────────────────────────────────────
    # Dummy: prediksi semua utility = 0 (model tidak bisa memilih, ambil acak)
    y_zero = np.zeros(len(y_true))
    results["no_retrieval"] = {
        "spearman_rho": spearman_rho(y_true, y_zero),
        "ndcg_at_k": ndcg_at_k(y_true, y_zero, k=ndcg_k),
    }

    # ── Full Unranked baseline ────────────────────────────────────────────────
    # Prediksi uniform score (semua dokumen dianggap sama)
    y_uniform = np.ones(len(y_true))
    results["full_unranked"] = {
        "spearman_rho": spearman_rho(y_true, y_uniform),
        "ndcg_at_k": ndcg_at_k(y_true, y_uniform, k=ndcg_k),
    }

    # ── Tian et al. (2026): D5 only ──────────────────────────────────────────
    X_d5_only = _only_d5(X)
    y_tian = predictor.predict(X_d5_only)
    rho_tian = spearman_rho(y_true, y_tian)
    results["tian_et_al_2026"] = {
        "spearman_rho": rho_tian,
        "ndcg_at_k": ndcg_at_k(y_true, y_tian, k=ndcg_k),
        "note": "D5 logit probe only — expects rho < 0.3",
    }

    # ── MPUP Zero-Shot ────────────────────────────────────────────────────────
    y_mpup_zs = predictor.predict(X)
    results["mpup_zero_shot"] = {
        "spearman_rho": spearman_rho(y_true, y_mpup_zs),
        "ndcg_at_k": ndcg_at_k(y_true, y_mpup_zs, k=ndcg_k),
    }

    # ── MPUP Few-Shot (k varied per Notion 6.3) ───────────────────────────────
    for k in k_shots_list:
        if k >= len(X):
            continue
        calib_idx = np.random.choice(len(X), k, replace=False)
        calibrator = LinearCalibrator()
        calibrator.fit(y_mpup_zs[calib_idx], y_true[calib_idx])
        y_mpup_fs = calibrator.transform(y_mpup_zs)
        results[f"mpup_few_shot_k{k}"] = {
            "spearman_rho": spearman_rho(y_true, y_mpup_fs),
            "ndcg_at_k": ndcg_at_k(y_true, y_mpup_fs, k=ndcg_k),
            "calibration_alpha": round(calibrator.alpha, 4),
            "calibration_beta": round(calibrator.beta, 4),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--features", required=True, help="features.npy")
    parser.add_argument("--labels", required=True, help="labels.npy")
    parser.add_argument("--checkpoint", required=True, help="Path to saved predictor")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    X = np.load(args.features)
    y_true = np.load(args.labels)

    predictor = MPUPPredictor.load(
        args.checkpoint,
        algorithm=cfg["predictor"]["algorithm"],
    )

    k_shots_list = cfg["calibration"].get("k_shots", [5, 10, 20])
    ndcg_k = cfg["evaluation"].get("ndcg_k", 10)

    print(f"Running evaluation on {len(X)} samples ({args.dataset})...")
    results = run_evaluation(X, y_true, predictor, k_shots_list, ndcg_k)

    # Print summary table
    print(f"\n{'Method':<30} {'Spearman ρ':>12} {'NDCG@'+str(ndcg_k):>10}")
    print("-" * 55)
    for method, metrics in results.items():
        rho = metrics.get("spearman_rho", 0.0)
        ndcg = metrics.get("ndcg_at_k", 0.0)
        print(f"{method:<30} {rho:>12.4f} {ndcg:>10.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.dataset}_results.json"
    with open(out_file, "w") as f:
        json.dump({"dataset": args.dataset, "results": results}, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
