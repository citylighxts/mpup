import numpy as np
import pytest

from experiments.significance_tests import describe, holm, sign_flip_pvalue


def test_holm_matches_hand_computation():
    # sorted p: 0.01, 0.02, 0.04 -> 3*0.01, 2*0.02, 1*0.04 (made monotone)
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])


def test_holm_never_lowers_a_p_value_and_caps_at_one():
    p = [0.5, 0.9, 0.001]
    adj = holm(p)
    assert all(a >= b for a, b in zip(adj, p)) and max(adj) <= 1.0


def test_sign_flip_test_rejects_a_clear_shift_and_keeps_a_null():
    rng = np.random.default_rng(0)
    shifted = rng.normal(0.3, 0.2, size=60)
    null = rng.normal(0.0, 0.2, size=60)
    assert sign_flip_pvalue(shifted, n_perm=20_000) < 0.001
    assert sign_flip_pvalue(null, n_perm=20_000) > 0.05


def test_describe_agrees_with_ttest_on_a_large_sample():
    from scipy import stats
    d = np.random.default_rng(1).normal(0.05, 0.3, size=400)
    out = describe(d, n_perm=20_000, seed=0)
    assert out["p_permutation"] == pytest.approx(stats.ttest_1samp(d, 0).pvalue, abs=0.02)
    assert out["cohen_dz"] == pytest.approx(d.mean() / d.std(ddof=1))


def test_p_from_interval_recovers_a_known_normal_case():
    from experiments.significance_tests import p_from_interval
    # estimate exactly 1.96 standard errors from zero -> p = 0.05, and the interval touches zero
    out = p_from_interval(0.0392, 0.0, 0.0784)
    assert out["p"] == pytest.approx(0.05, abs=1e-3)
    assert out["asymmetry"] == pytest.approx(0.0)


def test_p_from_interval_is_large_when_the_interval_straddles_zero():
    from experiments.significance_tests import p_from_interval
    assert p_from_interval(0.01, -0.08, 0.10)["p"] > 0.5
