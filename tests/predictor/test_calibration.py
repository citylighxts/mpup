import numpy as np
import pytest
from src.predictor.calibration import LinearCalibrator


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
