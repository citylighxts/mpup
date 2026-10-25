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
from src.features.concatenate import FEATURE_DIM, FEATURE_GROUPS, split_extended
from src.predictor.hidden_probe import HiddenProbe
from src.predictor.mpup_predictor import MPUPPredictor

# fit_predict(X_train, y_train, X_test, groups_train, groups_test) -> scores for X_test.
# The groups are the query ids of the rows, for predictors that work within queries.
FitPredict = Callable[..., np.ndarray]
METRICS = ("spearman", "ndcg", "top1")


def make_fitter(
    algorithm: str = "xgboost", center_by_query: bool = False,
    drop_groups: tuple[str, ...] = (), probe: bool = False, smooth_label: bool = False, **params,
) -> Callable[..., Callable[..., np.ndarray]]:
    """fit(X_train, y_train, groups_train) -> predict(X_test, groups_test), for one predictor configuration.

    Matrices may be plain (14 columns or fewer) or extended ([X14 | hidden states | u_smooth], see
    `src.features.concatenate.split_extended`). `drop_groups` removes feature groups (e.g. "d2", "d4"),
    `probe` stacks an out-of-fold ridge probe on the hidden states as one more feature, and `smooth_label`
    trains on `u_smooth` instead of the label; scoring is always against the label passed in `y`.
    """
    dropped = {c for g in drop_groups for c in range(FEATURE_GROUPS[g].start, FEATURE_GROUPS[g].stop)}
    keep = [c for c in range(FEATURE_DIM) if c not in dropped]

    def fit(X_tr, y_tr, g_tr=None):
        A_tr, H_tr, s_tr = split_extended(X_tr)
        if dropped:
            A_tr = A_tr[:, keep]
        target = y_tr
        if smooth_label:
            if np.isnan(s_tr).any():
                raise ValueError("smooth_label needs the extended matrix with a u_smooth column for every training row")
            target = s_tr
        hp = None
        if probe:
            if H_tr.shape[1] == 0:
                raise ValueError("probe needs the extended matrix with hidden states")
            hp = HiddenProbe().fit(H_tr, target, g_tr)
            A_tr = np.c_[A_tr, hp.oof_scores(H_tr, target, g_tr)]
        model = MPUPPredictor(algorithm=algorithm, center_by_query=center_by_query, **params)
        model.fit(A_tr, target, groups=g_tr)

        def predict(X_te, g_te=None):
            A_te, H_te, _ = split_extended(X_te)
            if dropped:
                A_te = A_te[:, keep]
            if hp is not None:
                A_te = np.c_[A_te, hp.predict(H_te, g_te)]
            return model.predict(A_te, groups=g_te)
        return predict
    return fit


def make_fit_predict(*args, **kwargs) -> FitPredict:
    """A fit_predict for one predictor configuration (see `make_fitter` for the arguments)."""
    fitter = make_fitter(*args, **kwargs)

    def fit_predict(X_tr, y_tr, X_te, g_tr=None, g_te=None):
        return fitter(X_tr, y_tr, g_tr)(X_te, g_te)
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
