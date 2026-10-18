"""Train MPUP predictor on labeled (query, passage, utility_score) data."""
import argparse
import yaml
import numpy as np
from pathlib import Path

from src.predictor.mpup_predictor import MPUPPredictor
from src.evaluation.metrics import evaluate_all
from src.evaluation.protocol import group_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--features", required=True, help="Path to .npy feature matrix")
    parser.add_argument("--labels", required=True, help="Path to .npy utility labels")
    parser.add_argument("--groups", default=None,
                        help="Path to .npy with one query id per row. Strongly recommended: "
                             "without it the split cuts the file at 80% of rows, which can "
                             "put part of one query's passages on each side.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="checkpoints/mpup")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    X = np.load(args.features)
    y = np.load(args.labels)

    if args.groups:
        groups = np.load(args.groups, allow_pickle=True)
        if len(groups) != len(X):
            raise SystemExit(f"--groups has {len(groups)} entries but there are {len(X)} rows")
        train_idx, val_idx = group_splits(groups, n_splits=5, seed=args.seed)[0]  # 80/20 by query
    else:
        print("WARNING: no --groups given; splitting by row order. Pass --groups to split by query.")
        split = int(0.8 * len(X))
        train_idx, val_idx = np.arange(split), np.arange(split, len(X))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

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
