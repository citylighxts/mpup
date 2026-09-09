"""Train MPUP predictor on labeled (query, passage, utility_score) data."""
import argparse
import yaml
import numpy as np
from pathlib import Path

from src.predictor.mpup_predictor import MPUPPredictor
from src.evaluation.metrics import evaluate_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--features", required=True, help="Path to .npy feature matrix")
    parser.add_argument("--labels", required=True, help="Path to .npy utility labels")
    parser.add_argument("--output", default="checkpoints/mpup")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    X = np.load(args.features)
    y = np.load(args.labels)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    pred_cfg = cfg["predictor"]
    algo_cfg = pred_cfg.get(pred_cfg["algorithm"], {})
    predictor = MPUPPredictor(algorithm=pred_cfg["algorithm"], **algo_cfg)
    predictor.fit(X_train, y_train)

    y_pred = predictor.predict(X_val)
    metrics = evaluate_all(y_val, y_pred)
    print("Validation metrics:", metrics)

    predictor.save(args.output)
    print(f"Saved predictor to {args.output}")


if __name__ == "__main__":
    main()
