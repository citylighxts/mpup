import numpy as np
import pytest

from src.evaluation.protocol import per_query_scores
from src.predictor.hidden_probe import HiddenProbe


def _data(seed=0, n_queries=40, per_query=6, d=12, noise=0.3):
    rng = np.random.default_rng(seed)
    q = np.repeat([f"q{i}" for i in range(n_queries)], per_query)
    y = rng.normal(size=len(q))
    H = rng.normal(size=(len(q), d)) + 5.0 * np.repeat(rng.normal(size=(n_queries, d)), per_query, axis=0)  # big per-query offset
    H[:, 0] += y / noise  # the planted signal: one direction carries the label
    return H, y, q


def test_probe_recovers_planted_signal():
    H, y, q = _data()
    probe = HiddenProbe(alphas=(1.0,), n_components=12)
    probe.fit(H, y, q)
    oof = probe.oof_scores(H, y, q)
    assert per_query_scores(y, oof, q)["spearman"].mean() > 0.8


def test_probe_oof_never_sees_own_query():
    H, y, q = _data()
    probe = HiddenProbe(alphas=(1.0,), n_components=12)
    probe.fit(H, y, q)
    base = probe.oof_scores(H, y, q)
    y2 = y.copy()
    m = q == "q3"
    y2[m] = np.random.default_rng(5).permutation(y2[m]) + 10.0  # scramble one query's labels
    changed = probe.oof_scores(H, y2, q)
    np.testing.assert_allclose(changed[m], base[m], atol=1e-9)  # its own score does not depend on its own labels
    assert not np.allclose(changed[~m], base[~m])  # the other queries' models did learn from the scrambled rows


def test_predict_centres_by_the_queries_being_scored():
    H, y, q = _data()
    probe = HiddenProbe(alphas=(1.0,), n_components=12)
    probe.fit(H[:180], y[:180], q[:180])
    pred = probe.predict(H[180:], q[180:])
    assert pred.shape == (60,) and per_query_scores(y[180:], pred, q[180:])["spearman"].mean() > 0.8
