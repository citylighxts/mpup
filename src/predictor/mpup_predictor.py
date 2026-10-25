"""MPUP Predictor Engine — gradient boosting, ridge or MLP regressor trained on D1–D5 features.

XGBoost has a numpy buffer protocol incompatibility with Python 3.14 in pytest context;
GradientBoostingRegressor is a drop-in sklearn equivalent used as the 'xgboost' backend.

`center_by_query=True` trains and predicts on *within-query deviations*: each feature, and at
training time the label, has the query's own mean subtracted. Utility is evaluated by how well the
passages of one query are ordered, and a large part of the raw label is a per-query offset (a
question the LLM already answers without help makes every passage useless, a question it cannot
answer lets any good passage help). A model on raw features spends its capacity on that offset,
which no reranker can use. At inference the candidates of the query being ranked are the group,
so the scores are "better or worse than the average candidate", not absolute utilities.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _group_index(groups, n: int) -> np.ndarray:
    return np.zeros(n, dtype=int) if groups is None else np.unique(np.asarray(groups), return_inverse=True)[1]


def center_by_group(values: np.ndarray, groups=None) -> np.ndarray:
    """Subtract each group's mean (column-wise for a matrix). groups=None means a single group."""
    values = np.asarray(values, dtype=float)
    idx = _group_index(groups, len(values))
    counts = np.bincount(idx)
    if values.ndim == 1:
        means = np.bincount(idx, weights=values) / counts
    else:
        means = np.stack([np.bincount(idx, weights=values[:, j]) for j in range(values.shape[1])], axis=1) / counts[:, None]
    return values - means[idx]


class PairwiseLogistic:
    """Linear RankNet: logistic regression on the feature difference of two candidates of one query.

    Stands in for XGBoost's `rank:pairwise`, which segfaults under pytest on Python 3.14 (see the module
    docstring), so a path that cannot be tested would otherwise be the one that gets run.
    """

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, X: np.ndarray, y: np.ndarray, qid: np.ndarray) -> "PairwiseLogistic":
        diffs, signs = [], []
        for g in np.unique(qid):
            idx = np.flatnonzero(qid == g)
            i, j = np.meshgrid(idx, idx, indexing="ij")
            keep = y[i] > y[j]  # every ordered pair whose first member has the higher label
            diffs.append(X[i[keep]] - X[j[keep]])
        D = np.concatenate(diffs) if diffs else np.empty((0, X.shape[1]))
        if len(D) == 0:  # no query has two candidates with different labels: nothing to learn from
            self.coef_ = np.zeros(X.shape[1])
            return self
        D = np.concatenate([D, -D])
        signs = np.r_[np.ones(len(D) // 2), np.zeros(len(D) // 2)]
        self.coef_ = LogisticRegression(C=self.C, fit_intercept=False).fit(D, signs).coef_[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_


class MPUPPredictor:
    def __init__(self, algorithm: str = "xgboost", center_by_query: bool = False, **kwargs):
        self.algorithm = algorithm
        self.center_by_query = center_by_query
        self.scaler = StandardScaler()
        if algorithm == "xgboost":
            self.model = GradientBoostingRegressor(
                n_estimators=kwargs.get("n_estimators", 300),
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.05),
                subsample=kwargs.get("subsample", 0.8),
                random_state=kwargs.get("random_state", 42),
            )
        elif algorithm == "rank":  # pairwise ranking inside each query: only the order of the candidates matters
            self.model = PairwiseLogistic(C=kwargs.get("C", 1.0))
        elif algorithm == "ridge":
            self.model = Ridge(alpha=kwargs.get("alpha", 100.0))
        elif algorithm == "mlp":
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(kwargs.get("hidden_dims", [256, 128, 64])),
                alpha=kwargs.get("dropout", 0.2),  # L2 regularization (no dropout in MLPRegressor)
                learning_rate_init=kwargs.get("lr", 1e-3),
                max_iter=kwargs.get("epochs", 50),
                early_stopping=True,
                random_state=kwargs.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None):
        """`groups` (one query id per row) is required when `center_by_query` is set."""
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if self.center_by_query:
            if groups is None:
                raise ValueError("center_by_query needs `groups`: the query id of every training row")
            X, y = center_by_group(X, groups), center_by_group(y, groups)
        X_scaled = self.scaler.fit_transform(X)
        if self.algorithm == "rank":
            if groups is None:
                raise ValueError("the ranking objective needs `groups`: the query id of every training row")
            self.model.fit(X_scaled, y, _group_index(groups, len(y)))
            return
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray, groups=None) -> np.ndarray:
        """With `center_by_query`, `groups` identifies the queries in X; omitted, X is one query."""
        X = np.asarray(X, dtype=float)
        if self.center_by_query:
            X = center_by_group(X, groups)
        return self.model.predict(self.scaler.transform(X))

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        joblib.dump(self.scaler, path / "scaler.joblib")
        (path / "config.json").write_text(
            json.dumps({"algorithm": self.algorithm, "center_by_query": self.center_by_query})
        )

    @classmethod
    def load(cls, path: str | Path, algorithm: str | None = None) -> "MPUPPredictor":
        path = Path(path)
        config = {}
        if (path / "config.json").exists():
            config = json.loads((path / "config.json").read_text())
        obj = cls.__new__(cls)
        obj.algorithm = algorithm or config.get("algorithm", "xgboost")
        obj.center_by_query = config.get("center_by_query", False)
        obj.model = joblib.load(path / "model.joblib")
        obj.scaler = joblib.load(path / "scaler.joblib")
        return obj
