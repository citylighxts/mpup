"""Linear calibration adapter for few-shot mode.

Fits alpha (scale) and beta (bias) from k calibration examples:
  u_hat_i = alpha * f_source(x_i) + beta
"""
import numpy as np
from sklearn.linear_model import LinearRegression


class LinearCalibrator:
    def __init__(self):
        self.alpha: float = 1.0
        self.beta: float = 0.0
        self._fitted = False

    def fit(self, base_scores: np.ndarray, true_utilities: np.ndarray):
        reg = LinearRegression().fit(base_scores.reshape(-1, 1), true_utilities)
        self.alpha = float(reg.coef_[0])
        self.beta = float(reg.intercept_)
        self._fitted = True

    def transform(self, base_scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return base_scores  # zero-shot: identity
        return self.alpha * base_scores + self.beta
