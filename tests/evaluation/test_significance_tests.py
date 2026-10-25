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


def test_p_from_interval_with_n_uses_the_t_distribution_and_is_more_conservative():
    from experiments.significance_tests import p_from_interval
    z_based = p_from_interval(0.1, 0.02, 0.18)["p"]
    t_based = p_from_interval(0.1, 0.02, 0.18, n=20)["p"]
    assert t_based > z_based
    assert p_from_interval(0.1, 0.02, 0.18, n=100000)["p"] == pytest.approx(z_based, abs=1e-4)


def test_from_results_accepts_a_run_without_the_selected_config_ablation(tmp_path):
    import json

    from experiments.significance_tests import from_results

    ci = lambda m: [m, m - 0.05, m + 0.05]  # noqa: E731
    paired = {f"drop_{g}": {"spearman": ci(0.01)} for g in ("d1", "d2", "d3", "d4", "d5")}
    run = {"nested_lodo": {"n_queries": 100, "pooled_paired_vs": {"dense": {"spearman": ci(0.08)}},
                           "per_dataset": {"a": {"n_queries": 50, "paired_vs": {"dense": {"spearman": ci(0.08)}}}}},
           "ablation_selected": None, "pooled_cv": {"paired_vs": paired}}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(run))
    out = from_results(path)
    assert out["pooled"]["p"] < 0.01 and set(out["ablation"]) == {"reference"}
