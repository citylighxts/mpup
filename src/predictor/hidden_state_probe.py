"""The hidden-state probe of MPUP+: a ridge regression from the hidden state of `M` to the utility.

Hidden states are centred within the query, standardised with the training mean and standard deviation, and a
ridge regression with strength 10 per training row maps them to the within-query deviation of the training
target. The score is one more feature of the regressor. Inside a training set the scores must be out-of-fold
(five folds over queries, `oof_scores`), so the regressor never sees a probe score fitted on the same row. The
arithmetic is that of `experiments.run_big`, which produced the larger study of the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from src.evaluation.protocol import group_splits
from src.predictor.mpup_predictor import center_by_group

ALPHA_PER_ROW = 10.0


def _fit(H, target, groups):
    H, target = np.asarray(H, dtype=np.float32), np.asarray(target, dtype=float)
    Hc = center_by_group(H, groups)
    mu, sd = Hc.mean(0), Hc.std(0) + 1e-6
    ridge = Ridge(alpha=ALPHA_PER_ROW * len(H)).fit((Hc - mu) / sd, center_by_group(target, groups))
    return mu, sd, ridge


def _score(model, H, groups):
    mu, sd, ridge = model
    return ridge.predict((center_by_group(np.asarray(H, dtype=np.float32), groups) - mu) / sd)


class HiddenStateProbe:
    def fit(self, H, target, groups) -> "HiddenStateProbe":
        self._model = _fit(H, target, groups)
        return self

    def predict(self, H, groups=None) -> np.ndarray:
        """`groups` identifies the queries in H; omitted, H holds the candidates of one query."""
        return _score(self._model, H, groups)

    def oof_scores(self, H, target, groups, n_folds: int = 5) -> np.ndarray:
        """Score of every row from a probe fitted on the other folds of queries."""
        H, target, groups = np.asarray(H), np.asarray(target, dtype=float), np.asarray(groups)
        out = np.zeros(len(H))
        for a, b in group_splits(groups, n_splits=n_folds, seed=0):
            out[b] = _score(_fit(H[a], target[a], groups[a]), H[b], groups[b])
        return out

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "probe.joblib")
        (path / "config.json").write_text(json.dumps({"alpha_per_row": ALPHA_PER_ROW}))

    @classmethod
    def load(cls, path) -> "HiddenStateProbe":
        obj = cls.__new__(cls)
        obj._model = joblib.load(Path(path) / "probe.joblib")
        return obj
