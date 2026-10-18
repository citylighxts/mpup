"""Cross-validated ranking report that follows `src.evaluation.protocol`.

`cv_report`   — repeated K-fold over queries. Every query is predicted by models that never
                saw it; per-query metrics are averaged across repeats, then bootstrapped over
                queries, so the interval reflects how many queries there are.
`lodo_report` — leave-one-dataset-out: train on all other datasets, score the held-out one.

Baselines are functions X -> score and `extra_models` are further trained models (ablations),
all evaluated on the same folds and queries. `paired_vs` is the per-query difference
(mpup minus the other method) with its own interval: "does it beat the baseline" is a paired
question, and unpaired overlapping intervals can hide a real gap. Pass `datasets` to also get the
same numbers per dataset (`per_dataset`), each bootstrapped over that dataset's queries.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.evaluation.protocol import (
    bootstrap_ci,
    group_splits,
    leave_one_dataset_out,
    per_query_scores,
)
from src.predictor.mpup_predictor import MPUPPredictor

# fit_predict(X_train, y_train, X_test, groups_train, groups_test) -> scores for X_test.
# The groups are the query ids of the rows, for predictors that work within queries.
FitPredict = Callable[..., np.ndarray]
METRICS = ("spearman", "ndcg", "top1")


def make_fit_predict(algorithm: str = "xgboost", center_by_query: bool = False, **params) -> FitPredict:
    """A fit_predict for one predictor configuration."""
    def fit_predict(X_tr, y_tr, X_te, g_tr=None, g_te=None):
        model = MPUPPredictor(algorithm=algorithm, center_by_query=center_by_query, **params)
        model.fit(X_tr, y_tr, groups=g_tr)
        return model.predict(X_te, groups=g_te)
    return fit_predict


# The configuration stated in the paper: raw features, 200 trees of depth 5. Kept as the reference.
default_fit_predict = make_fit_predict("xgboost", n_estimators=200, max_depth=5)


def columns(fit_predict: FitPredict, cols: list[int]) -> FitPredict:
    """The same predictor restricted to some feature columns (for ablations)."""
    return lambda X_tr, y_tr, X_te, g_tr=None, g_te=None: fit_predict(X_tr[:, cols], y_tr, X_te[:, cols], g_tr, g_te)


def _summarise(per_query: dict[str, np.ndarray], seed: int) -> dict:
    return {m: bootstrap_ci(per_query[m], seed=seed) for m in METRICS}


def cv_report(
    X, y, query_ids,
    fit_predict: FitPredict = default_fit_predict,
    baselines: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    extra_models: dict[str, FitPredict] | None = None,
    datasets=None,
    n_splits: int = 5, n_repeats: int = 4, seed: int = 0, k: int = 10,
) -> dict:
    X, y, query_ids = np.asarray(X), np.asarray(y, float), np.asarray(query_ids)
    baselines = baselines or {}
    trained = {"mpup": fit_predict, **(extra_models or {})}
    names = [*trained, *baselines]
    acc: dict[str, dict[str, list[np.ndarray]]] = {n: {m: [] for m in METRICS} for n in names}
    used_ids = None

    for r in range(n_repeats):
        splits = group_splits(query_ids, n_splits=n_splits, seed=seed + r)
        scored = {}
        for name, fp in trained.items():  # every model sees exactly the same folds
            pred = np.full(len(y), np.nan)
            for train, test in splits:
                pred[test] = fp(X[train], y[train], X[test], query_ids[train], query_ids[test])
            scored[name] = pred
        scored.update({n: np.asarray(f(X), float) for n, f in baselines.items()})
        for n, p in scored.items():
            pq = per_query_scores(y, p, query_ids, k=k)
            used_ids = pq["query_ids"]
            for m in METRICS:
                acc[n][m].append(pq[m])

    per_query = {n: {m: np.mean(acc[n][m], axis=0) for m in METRICS} for n in names}
    others = [n for n in names if n != "mpup"]
    report = {
        "n_queries": int(len(used_ids)),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "models": {n: _summarise(per_query[n], seed) for n in names},
        "paired_vs": {
            n: {m: bootstrap_ci(per_query["mpup"][m] - per_query[n][m], seed=seed) for m in METRICS}
            for n in others
        },
    }
    if datasets is not None:
        datasets = np.asarray(datasets)
        dataset_of = dict(zip(query_ids.tolist(), datasets.tolist()))
        qid_dataset = np.array([dataset_of[q] for q in used_ids.tolist()])
        report["per_dataset"] = {}
        for name in sorted(set(qid_dataset.tolist())):
            sel = qid_dataset == name
            sub = {n: {m: per_query[n][m][sel] for m in METRICS} for n in names}
            report["per_dataset"][name] = {
                "n_queries": int(sel.sum()),
                "models": {n: _summarise(sub[n], seed) for n in names},
                "paired_vs": {
                    n: {m: bootstrap_ci(sub["mpup"][m] - sub[n][m], seed=seed) for m in METRICS}
                    for n in others
                },
            }
    return report


def lodo_report(
    X, y, query_ids, datasets,
    fit_predict: FitPredict = default_fit_predict,
    baselines: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    seed: int = 0, k: int = 10,
) -> dict:
    X, y, query_ids = np.asarray(X), np.asarray(y, float), np.asarray(query_ids)
    baselines = baselines or {}
    names = ["mpup", *baselines]
    per_dataset: dict[str, dict] = {}
    for name, train, test in leave_one_dataset_out(datasets):
        scored = {"mpup": fit_predict(X[train], y[train], X[test], query_ids[train], query_ids[test])}
        scored.update({n: np.asarray(f(X[test]), float) for n, f in baselines.items()})
        entry: dict = {}
        pq_by_name = {n: per_query_scores(y[test], p, query_ids[test], k=k) for n, p in scored.items()}
        for n, pq in pq_by_name.items():
            entry[n] = _summarise(pq, seed)
        entry["n_queries"] = int(len(pq_by_name["mpup"]["spearman"]))
        entry["paired_vs"] = {
            n: {m: bootstrap_ci(pq_by_name["mpup"][m] - pq_by_name[n][m], seed=seed) for m in METRICS}
            for n in baselines
        }
        per_dataset[name] = entry
    # Macro average: every dataset counts once, regardless of how many queries it has.
    macro = {
        n: {m: float(np.mean([per_dataset[d][n][m][0] for d in per_dataset])) for m in METRICS}
        for n in names
    }
    return {"per_dataset": per_dataset, "macro": macro}
