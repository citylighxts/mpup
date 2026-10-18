"""Few-shot adapters for transferring a predictor to an unseen target LLM.

Two are provided, and the difference decides what they can be used for.

`LinearCalibrator` fits `u = alpha * base + beta`. With alpha > 0 that map preserves the
order of its inputs exactly, and Spearman sees only order while Pearson is invariant to
linear rescaling — so it **provably cannot change rank metrics**. Measured on real data it
returns rho identical to the zero-shot score to six decimal places, and at very small k it
can *hurt*, because a noisy fit can produce alpha < 0 and invert the ranking. It remains the
right tool for calibration error: it cuts MSE by an order of magnitude, which matters when
`u_hat` feeds a threshold for top-m selection.

`FeatureSpaceAdapter` is what can actually improve ranking. It fits a strongly regularized
ridge on the k target examples **in feature space** and blends it with the standardized base
score. Because that blends two different functions of the features rather than monotonically
remapping one, the induced ranking genuinely changes.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


class LinearCalibrator:
    """u' = alpha * u_hat + beta, with alpha kept positive so the ranking can never be inverted.

    A least-squares slope fitted on a handful of examples is negative when those examples happen
    to disagree with the predictor (measured: 12.5% of 5-example draws on one run), and a negative
    slope reverses the whole ranking. The calibrator exists to fix the *scale* of u_hat, not to
    overrule its order, so when the fit is not positive it falls back to a pure shift
    (alpha = 1, beta = mean(true) - mean(base)) and records that in `fell_back`.
    """

    def __init__(self):
        self.alpha: float = 1.0
        self.beta: float = 0.0
        self.fell_back: bool = False
        self._fitted = False

    def fit(self, base_scores: np.ndarray, true_utilities: np.ndarray):
        base_scores = np.asarray(base_scores, dtype=float)
        true_utilities = np.asarray(true_utilities, dtype=float)
        reg = LinearRegression().fit(base_scores.reshape(-1, 1), true_utilities)
        alpha, beta = float(reg.coef_[0]), float(reg.intercept_)
        self.fell_back = alpha <= 0.0
        if self.fell_back:
            alpha, beta = 1.0, float(np.mean(true_utilities) - np.mean(base_scores))
        self.alpha, self.beta = alpha, beta
        self._fitted = True

    def transform(self, base_scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return base_scores  # zero-shot: identity
        return self.alpha * base_scores + self.beta


class FeatureSpaceAdapter:
    """k-shot adaptation that can change the ranking, unlike an affine calibrator.

    `ridge_alpha` is deliberately large: with k in the 8-32 range the design matrix is small
    relative to the feature count, and an unregularized fit just memorises the draw.
    """

    def __init__(self, ridge_alpha: float = 10.0, blend: float = 0.5):
        self.ridge_alpha = ridge_alpha
        self.blend = blend
        self.ridge: Ridge | None = None
        self._base_mean = 0.0
        self._base_std = 1.0
        self._ridge_mean = 0.0
        self._ridge_std = 1.0

    def fit(self, X_k: np.ndarray, base_scores: np.ndarray, true_utilities: np.ndarray):
        self.ridge = Ridge(alpha=self.ridge_alpha).fit(X_k, true_utilities)
        ridge_scores = self.ridge.predict(X_k)
        self._base_mean = float(np.mean(base_scores))
        self._base_std = float(np.std(base_scores)) or 1.0
        self._ridge_mean = float(np.mean(ridge_scores))
        self._ridge_std = float(np.std(ridge_scores)) or 1.0
        return self

    def transform(self, X: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
        """Standardisation uses the calibration set's statistics, never the evaluation set's —
        using evaluation statistics would be transductive and would inflate reported scores."""
        if self.ridge is None:
            return base_scores  # zero-shot: identity
        base_z = (base_scores - self._base_mean) / self._base_std
        ridge_z = (self.ridge.predict(X) - self._ridge_mean) / self._ridge_std
        return (1.0 - self.blend) * base_z + self.blend * ridge_z
