"""The larger experiment's split and probe: no test question leaks into development, no row sees its own probe."""
import numpy as np

from experiments import run_big


def _toy(n_q=60, k=5, seed=0):
    rng = np.random.default_rng(seed)
    n = n_q * k
    q = np.array([f"ds{i % 3}:{i}" for i in range(n_q) for _ in range(k)])
    H = rng.normal(size=(n, 8)).astype(np.float32)
    y = H[:, 0] + 0.1 * rng.normal(size=n)
    X = np.c_[rng.normal(size=(n, 13)), y + rng.normal(size=n)]
    return {"X": X, "y": y, "q": q, "ds": np.array([s.split(":")[0] for s in q]),
            "soft": y + 0.2 * rng.normal(size=n), "ce": rng.normal(size=n), "H": H}


def test_split_keeps_seen_questions_in_development():
    d = _toy()
    seen = set(np.unique(d["q"])[:30])
    test = run_big.split(d["q"], seen, test_share=0.5)
    assert not any(q in seen for q in d["q"][test])
    for qid in np.unique(d["q"]):  # whole queries on one side
        assert len(set(test[d["q"] == qid])) == 1


def test_probe_learns_signal_out_of_fold():
    d = _toy()
    tr, te = np.arange(200), np.arange(200, 300)
    oof, test = run_big.probe_scores(d, tr, te, d["y"][tr])
    assert np.corrcoef(test, d["y"][te])[0, 1] > 0.8
    assert np.corrcoef(oof, d["y"][tr])[0, 1] > 0.8


def test_every_variant_runs():
    d = _toy()
    tr, te = np.arange(200), np.arange(200, 300)
    for v in ("gbr(100,d2)|u", "gbr(100,d2)|mix|ce|probe", "gbr(100,d2)|soft|probe"):
        assert run_big.fit_predict(d, tr, te, v).shape == (100,)
