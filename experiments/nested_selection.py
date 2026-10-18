"""Choosing the predictor configuration without cheating: nested leave-one-dataset-out.

The audit of the first multi-dataset run found that the configuration stated in the paper (200
trees of depth 5 on raw features) loses to a single dense-similarity feature. Changing the
configuration after seeing that is model selection, and the CV score of the winner is then
optimistic. So the selection is wrapped in the same protocol used for everything else:

  outer loop  hold one dataset out (this is what the reported score is measured on)
  inner loop  among the other datasets, score every configuration by leave-one-dataset-out and
              keep the best (the held-out dataset's labels are never touched here)
  then        refit the winner on all the other datasets, score the held-out dataset

The grid is fixed in advance (`GRID`) and small, and the paper's own configuration is part of it.
The score is the within-query Spearman ρ, macro-averaged over datasets for selection.
`oracle` reports what picking the best configuration *on the held-out dataset itself* would have
claimed; the gap to `nested` is the optimism that a non-nested comparison would have added.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from experiments.cv_report import make_fit_predict
from src.evaluation.protocol import bootstrap_ci, leave_one_dataset_out, per_query_scores

METRICS = ("spearman", "ndcg", "top1")


@dataclass(frozen=True)
class Config:
    name: str
    algorithm: str
    center_by_query: bool
    params: tuple = ()

    def fit_predict(self):
        return make_fit_predict(self.algorithm, self.center_by_query, **dict(self.params))


def _grid() -> list[Config]:
    out = []
    for center in (False, True):
        tag = "centered" if center else "raw"
        for alpha in (10.0, 100.0, 1000.0):
            out.append(Config(f"ridge(a={alpha:g})-{tag}", "ridge", center, (("alpha", alpha),)))
        for n, d in ((100, 2), (200, 3), (200, 5)):
            out.append(Config(f"gbr({n},d{d})-{tag}", "xgboost", center,
                              (("n_estimators", n), ("max_depth", d), ("learning_rate", 0.05))))
    return out


GRID = _grid()
PAPER = next(c for c in GRID if c.name == "gbr(200,d5)-raw")  # the configuration stated in the paper


def _mean_rho(y, pred, q) -> float:
    s = per_query_scores(y, pred, q)["spearman"]
    return float(s.mean()) if len(s) else 0.0


def lodo_score(cfg: Config, X, y, q, datasets) -> float:
    """Macro (over held-out datasets) mean per-query ρ of one configuration."""
    fp = cfg.fit_predict()
    scores = []
    for _, train, test in leave_one_dataset_out(datasets):
        scores.append(_mean_rho(y[test], fp(X[train], y[train], X[test], q[train], q[test]), q[test]))
    return float(np.mean(scores))


def select_config(X, y, q, datasets, grid=GRID, n_jobs: int = -1) -> tuple[Config, dict[str, float]]:
    scores = Parallel(n_jobs=n_jobs)(delayed(lodo_score)(c, X, y, q, datasets) for c in grid)
    by_name = {c.name: s for c, s in zip(grid, scores)}
    best = grid[int(np.argmax(scores))]  # ties go to the earlier (simpler) entry
    return best, by_name


def _outer(cfg: Config, X, y, q, train, test):
    return cfg.fit_predict()(X[train], y[train], X[test], q[train], q[test])


def nested_lodo(X, y, query_ids, datasets, baselines=None, grid=GRID, n_jobs: int = -1, seed: int = 0) -> dict:
    X, y, q, ds = np.asarray(X), np.asarray(y, float), np.asarray(query_ids), np.asarray(datasets)
    baselines = baselines or {}
    per_dataset: dict[str, dict] = {}
    diffs = {n: {m: [] for m in METRICS} for n in ["paper", *baselines]}
    for name, train, test in leave_one_dataset_out(ds):
        # inner selection sees only the training datasets
        best, inner = select_config(X[train], y[train], q[train], ds[train], grid, n_jobs)
        outer_preds = {c.name: _outer(c, X, y, q, train, test) for c in grid}
        paper_pred = outer_preds[PAPER.name] if PAPER.name in outer_preds else _outer(PAPER, X, y, q, train, test)
        pq = {"nested": per_query_scores(y[test], outer_preds[best.name], q[test]),
              "paper": per_query_scores(y[test], paper_pred, q[test])}
        for b, f in baselines.items():
            pq[b] = per_query_scores(y[test], np.asarray(f(X[test]), float), q[test])
        for n in diffs:
            for m in METRICS:
                diffs[n][m].append(pq["nested"][m] - pq[n][m])
        outer_rho = {n: _mean_rho(y[test], p, q[test]) for n, p in outer_preds.items()}
        per_dataset[name] = {
            "selected": best.name,
            "inner_scores": inner,
            "outer_rho_all_configs": outer_rho,
            "n_queries": int(len(pq["nested"]["spearman"])),
            "models": {n: {m: bootstrap_ci(v[m], seed=seed) for m in METRICS} for n, v in pq.items()},
            "paired_vs": {
                n: {m: bootstrap_ci(pq["nested"][m] - pq[n][m], seed=seed) for m in METRICS}
                for n in pq if n != "nested"
            },
        }
    names = ["nested", "paper", *baselines]
    # One interval over every held-out query of every dataset: far more power than 11 separate ones.
    pooled_paired = {
        n: {m: bootstrap_ci(np.concatenate(diffs[n][m]), seed=seed) for m in METRICS} for n in names if n != "nested"
    }
    macro = {n: {m: float(np.mean([d["models"][n][m][0] for d in per_dataset.values()])) for m in METRICS} for n in names}
    macro["oracle"] = {"spearman": float(np.mean([max(d["outer_rho_all_configs"].values()) for d in per_dataset.values()]))}
    frequency: dict[str, int] = {}
    for d in per_dataset.values():
        frequency[d["selected"]] = frequency.get(d["selected"], 0) + 1
    final, final_scores = select_config(X, y, q, ds, grid, n_jobs)
    return {
        "grid": [c.name for c in grid],
        "per_dataset": per_dataset,
        "macro": macro,
        "pooled_paired_vs": pooled_paired,
        "n_queries": int(sum(d["n_queries"] for d in per_dataset.values())),
        "optimism_of_oracle_selection": macro["oracle"]["spearman"] - macro["nested"]["spearman"],
        "selection_frequency": frequency,
        "final_config": {"name": final.name, "algorithm": final.algorithm, "center_by_query": final.center_by_query,
                         "params": dict(final.params), "lodo_scores": final_scores},
    }
