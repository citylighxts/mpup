import numpy as np
import pytest
from scipy.stats import spearmanr
from src.predictor.calibration import LinearCalibrator, FeatureSpaceAdapter


def test_identity_before_fit():
    cal = LinearCalibrator()
    scores = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(cal.transform(scores), scores)


def test_fit_scales_and_biases():
    cal = LinearCalibrator()
    base = np.array([0.0, 1.0, 2.0, 3.0])
    true = base * 2.0 + 0.5
    cal.fit(base, true)
    assert cal.alpha == pytest.approx(2.0, abs=1e-3)
    assert cal.beta == pytest.approx(0.5, abs=1e-3)


def test_transform_after_fit():
    cal = LinearCalibrator()
    base = np.array([1.0, 2.0, 3.0, 4.0])
    true = base * 0.5 - 0.1
    cal.fit(base, true)
    result = cal.transform(np.array([2.0]))
    assert result[0] == pytest.approx(2.0 * 0.5 - 0.1, abs=1e-2)


# ── Why FeatureSpaceAdapter exists — ported from the UtilityTransfer effort ──

def test_affine_calibration_cannot_change_rank_metrics():
    """An affine map with alpha > 0 preserves order, so Spearman cannot move."""
    rng = np.random.default_rng(0)
    truth = rng.choice([-1.0, 0.0, 1.0], size=400)
    base = 4.7 * (0.35 * truth + rng.normal(size=400)) - 2.3

    cal = LinearCalibrator()
    cal.fit(base, truth)
    calibrated = cal.transform(base)

    assert spearmanr(base, truth).statistic == pytest.approx(
        spearmanr(calibrated, truth).statistic, abs=1e-12
    )
    # It is still the right tool for calibration error.
    assert np.mean((calibrated - truth) ** 2) < np.mean((base - truth) ** 2)


def test_feature_space_adapter_changes_the_ranking():
    """Blending two different functions of the features does move the order."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6))
    truth = X @ np.array([1.0, -0.5, 0.3, 0.0, 0.2, -0.1]) + rng.normal(scale=0.3, size=200)
    base = X[:, 0] * 3.0 + 1.0  # a base score that only sees one feature

    adapter = FeatureSpaceAdapter().fit(X, base, truth)
    adapted = adapter.transform(X, base)

    assert spearmanr(adapted, truth).statistic > spearmanr(base, truth).statistic


def test_feature_space_adapter_unfitted_is_identity():
    adapter = FeatureSpaceAdapter()
    base = np.array([0.1, 0.5, 0.9])
    assert np.allclose(adapter.transform(np.zeros((3, 6)), base), base)
