"""MPUP+: the 14 features, a cross-encoder score and a hidden-state probe score, in one regressor.

Feature order, as in `experiments.run_big`: the 14 features, the cross-encoder score, the probe score. The
regressor is the variant kept for testing in the larger study, `gbr(500,d4)|u|ce|probe`: boosted trees (500 of
depth 4, learning rate 0.03) trained on within-query deviations of the Eq. (1) utility.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.predictor.hidden_state_probe import HiddenStateProbe
from src.predictor.mpup_predictor import MPUPPredictor

MPUP_PLUS_PARAMS = {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03}


class MPUPPlusPredictor:
    def __init__(self, **params):
        self.params = {**MPUP_PLUS_PARAMS, **params}

    def fit(self, X14, ce, H, target, groups) -> "MPUPPlusPredictor":
        """X14 (n, 14), cross-encoder scores (n,), hidden states (n, d), the Eq. (1) utility and the query ids."""
        self.probe = HiddenStateProbe().fit(H, target, groups)
        X = np.c_[np.asarray(X14, dtype=float), np.asarray(ce, dtype=float), self.probe.oof_scores(H, target, groups)]
        self.regressor = _fit_regressor(self.params, X, target, groups)
        return self

    def features(self, X14, ce, H, groups=None) -> np.ndarray:
        """The 16 features: the 14, the cross-encoder score and the probe score (see the module docstring)."""
        return np.c_[np.asarray(X14, dtype=float), np.asarray(ce, dtype=float), self.probe.predict(H, groups)]

    def predict_features(self, X16, groups=None) -> np.ndarray:
        return self.regressor.predict(X16, groups=groups)

    def predict(self, X14, ce, H, groups=None) -> np.ndarray:
        """`groups` identifies the queries; omitted, the rows are the candidates of one query."""
        return self.predict_features(self.features(X14, ce, H, groups), groups)

    def save(self, path) -> None:
        path = Path(path)
        self.regressor.save(path / "regressor")
        self.probe.save(path / "probe")
        (path / "params.json").write_text(json.dumps(self.params))

    @classmethod
    def load(cls, path) -> "MPUPPlusPredictor":
        path = Path(path)
        obj = cls(**json.loads((path / "params.json").read_text()))
        obj.regressor = MPUPPredictor.load(path / "regressor")
        obj.probe = HiddenStateProbe.load(path / "probe")
        return obj


def _fit_regressor(params: dict, X, target, groups) -> MPUPPredictor:
    model = MPUPPredictor(algorithm="xgboost", center_by_query=True, **params)
    model.fit(X, np.asarray(target, dtype=float), groups=groups)
    return model
