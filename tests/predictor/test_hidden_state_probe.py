import numpy as np
import pytest

from experiments import run_big
from src.predictor.hidden_state_probe import HiddenStateProbe


def _data(n_queries=30, per_query=6, d=10, seed=0):
    rng = np.random.default_rng(seed)
    q = np.repeat([f"q{i}" for i in range(n_queries)], per_query)
    y = rng.normal(size=len(q))
    H = rng.normal(size=(len(q), d)) + 4.0 * np.repeat(rng.normal(size=(n_queries, d)), per_query, axis=0)
    H[:, 0] += 3.0 * y
    return H.astype(np.float32), y, q


def test_probe_scores_equal_those_of_the_larger_study():
    H, y, q = _data()
    tr, te = np.arange(120), np.arange(120, 180)
    expected = run_big._probe_fit_predict(H[tr], y[tr], q[tr], H[te], q[te])
    got = HiddenStateProbe().fit(H[tr], y[tr], q[tr]).predict(H[te], q[te])
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_out_of_fold_scores_equal_those_of_the_larger_study():
    H, y, q = _data()
    tr, te = np.arange(120), np.arange(120, 180)
    d = {"H": H, "q": q}
    oof, _ = run_big.probe_scores(d, tr, te, y[tr])
    np.testing.assert_allclose(HiddenStateProbe().oof_scores(H[tr], y[tr], q[tr]), oof, rtol=1e-6, atol=1e-6)


def test_the_probe_recovers_a_planted_signal():
    from src.evaluation.protocol import per_query_scores

    H, y, q = _data()
    oof = HiddenStateProbe().oof_scores(H, y, q)
    assert per_query_scores(y, oof, q)["spearman"].mean() > 0.7


def test_an_out_of_fold_score_never_depends_on_its_own_queries_labels():
    H, y, q = _data()
    base = HiddenStateProbe().oof_scores(H, y, q)
    y2 = y.copy()
    m = q == "q3"
    y2[m] = np.random.default_rng(1).permutation(y2[m]) + 5.0
    changed = HiddenStateProbe().oof_scores(H, y2, q)
    np.testing.assert_allclose(changed[m], base[m], atol=1e-9)


def test_predict_without_groups_treats_the_rows_as_one_query():
    H, y, q = _data()
    probe = HiddenStateProbe().fit(H, y, q)
    one = H[:6]
    np.testing.assert_allclose(probe.predict(one), probe.predict(one, np.zeros(6, dtype=int)))


def test_save_and_load_give_the_same_scores(tmp_path):
    H, y, q = _data()
    probe = HiddenStateProbe().fit(H, y, q)
    probe.save(tmp_path / "probe")
    np.testing.assert_allclose(HiddenStateProbe.load(tmp_path / "probe").predict(H[:6]), probe.predict(H[:6]))
