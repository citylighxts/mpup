"""A ridge probe on the hidden states of the target LLM.

The hidden states of one (query, passage) prompt have thousands of dimensions and a large per-query offset,
so the probe works on within-query deviations: each column has its query's mean subtracted and is scaled, the
result is projected on its leading principal components (fitted on the training rows only, labels unused), and
a ridge regression on the within-query-centred label is fitted on the projection. `oof_scores` gives the
score every training row would have received from a probe fitted without its query; the predictor stacks
those on its other features, so the stacked model never trains on a score that has seen its own label.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from src.predictor.mpup_predictor import center_by_group


class HiddenProbe:
    def __init__(self, alphas: tuple[float, ...] = (1e1, 1e2, 1e3, 1e4), n_components: int = 256, n_splits: int = 5):
        self.alphas, self.n_components, self.n_splits = tuple(alphas), n_components, n_splits

    def _project(self, H, groups) -> np.ndarray:
        Z = center_by_group(np.asarray(H, dtype=np.float32), groups) / self._scale
        return self._pca.transform(Z)

    def fit(self, H, y, groups) -> "HiddenProbe":
        H, y = np.asarray(H, dtype=np.float32), np.asarray(y, dtype=float)
        Hc = center_by_group(H, groups)
        self._scale = Hc.std(axis=0) + 1e-6
        k = min(self.n_components, len(H) - 1, H.shape[1])
        self._pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(Hc / self._scale)
        T, yc = self._pca.transform(Hc / self._scale), center_by_group(y, groups)
        self.alpha_ = self.alphas[0] if len(self.alphas) == 1 else min(
            self.alphas, key=lambda a: float(np.mean((self._cross_fit(T, yc, groups, a) - yc) ** 2))
        )
        self._ridge = Ridge(alpha=self.alpha_).fit(T, yc)
        return self

    def _cross_fit(self, T, yc, groups, alpha) -> np.ndarray:
        out = np.zeros(len(yc))
        for tr, te in GroupKFold(n_splits=self.n_splits).split(T, yc, np.asarray(groups)):
            out[te] = Ridge(alpha=alpha).fit(T[tr], yc[tr]).predict(T[te])
        return out

    def oof_scores(self, H, y, groups, n_splits: int | None = None) -> np.ndarray:
        """Score of every row from a ridge fitted on the other folds of queries (never its own query)."""
        self.n_splits = n_splits or self.n_splits
        return self._cross_fit(self._project(H, groups), center_by_group(np.asarray(y, dtype=float), groups), groups, self.alpha_)

    def predict(self, H, groups=None) -> np.ndarray:
        """`groups` identifies the queries in H; omitted, H is one query."""
        return self._ridge.predict(self._project(H, groups))
