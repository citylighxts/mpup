"""MPUP Predictor Engine — XGBoost or MLP regressor trained on D1–D5 features."""
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class MPUPPredictor:
    def __init__(self, algorithm: str = "xgboost", **kwargs):
        self.algorithm = algorithm
        self.scaler = StandardScaler()
        if algorithm == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get("n_estimators", 300),
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.05),
                subsample=kwargs.get("subsample", 0.8),
                objective="reg:squarederror",
                tree_method="hist",
            )
        elif algorithm == "mlp":
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(kwargs.get("hidden_dims", [256, 128, 64])),
                dropout=kwargs.get("dropout", 0.2),
                learning_rate_init=kwargs.get("lr", 1e-3),
                max_iter=kwargs.get("epochs", 50),
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        joblib.dump(self.scaler, path / "scaler.joblib")

    @classmethod
    def load(cls, path: str | Path, algorithm: str = "xgboost") -> "MPUPPredictor":
        path = Path(path)
        obj = cls.__new__(cls)
        obj.algorithm = algorithm
        obj.model = joblib.load(path / "model.joblib")
        obj.scaler = joblib.load(path / "scaler.joblib")
        return obj
