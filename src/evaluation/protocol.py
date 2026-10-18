"""Evaluation protocol for pooled, multi-dataset utility prediction.

Three rules, each fixing a way the single-split benchmark could overstate a result:

1. **The unit of a split is the query.** The passages of one query share features (the HyDE
   claim, the document set, BM25 ranks), so putting some in train and some in test leaks.
   `group_splits` keeps every query wholly on one side, and `repeated_group_splits` repeats
   that with fresh shuffles, because one 80/20 split of ~100 queries moves rho by about
   +-0.09.
2. **Transfer is tested across datasets.** A random split of pooled data only shows that a
   predictor fits the mixture. `leave_one_dataset_out` holds a whole dataset out, which is
   what "one predictor for many datasets" has to survive.
3. **One label definition per pool.** The RAGBench sentence-annotation proxy and the
   utility of Eq. (1) differ in meaning and scale; training on a mixture of them fits
   neither. `assert_single_label_source` and `pool_tables` refuse the mixture.

Metrics are computed within each query and averaged over queries. A pooled Spearman also
rewards telling easy queries from hard ones, which no reranker can use; `evaluate_ranking`
reports it only as `global_spearman`, next to the within-query numbers.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

LABEL_UTILITY = "utility_eq1"
LABEL_RAGBENCH_PROXY = "ragbench_proxy"


# ----------------------------------------------------------------- splitting

def group_splits(groups, n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold over groups (queries): each group is in exactly one test fold, never in both."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < n_splits:
        raise ValueError(f"need at least {n_splits} groups for {n_splits} folds, got {len(unique)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    folds = np.array_split(order, n_splits)
    splits = []
    for fold in folds:
        in_test = np.isin(groups, fold)
        splits.append((np.flatnonzero(~in_test), np.flatnonzero(in_test)))
    return splits


def repeated_group_splits(
    groups, n_splits: int = 5, n_repeats: int = 4, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for r in range(n_repeats):
        splits.extend(group_splits(groups, n_splits=n_splits, seed=seed + r))
    return splits


def leave_one_dataset_out(datasets) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """One (name, train_idx, test_idx) per dataset; the named dataset is entirely held out."""
    datasets = np.asarray(datasets)
    names = sorted(set(datasets.tolist()))
    if len(names) < 2:
        raise ValueError("leave-one-dataset-out needs at least two datasets")
    return [
        (name, np.flatnonzero(datasets != name), np.flatnonzero(datasets == name))
        for name in names
    ]


# -------------------------------------------------------------- label policy

def assert_single_label_source(sources) -> str:
    distinct = sorted(set(sources))
    if not distinct:
        raise ValueError("no label source given")
    if len(distinct) > 1:
        raise ValueError(
            f"refusing to pool different label definitions: {distinct}. Train and evaluate on "
            "one definition; report the other separately."
        )
    return distinct[0]


def pool_tables(tables: list[dict]) -> dict:
    """Concatenate per-dataset tables {X, y, query_ids, dataset, label_source} into one.

    Query ids are prefixed with the dataset name, so "0" in two datasets stays two queries.
    """
    label_source = assert_single_label_source([t["label_source"] for t in tables])
    return {
        "X": np.concatenate([t["X"] for t in tables]),
        "y": np.concatenate([t["y"] for t in tables]),
        "query_ids": np.concatenate(
            [np.array([f"{t['dataset'][0]}:{q}" for q in t["query_ids"]]) for t in tables]
        ),
        "dataset": np.concatenate([np.asarray(t["dataset"]) for t in tables]),
        "label_source": label_source,
    }


# ------------------------------------------------------------------- metrics

def per_query_scores(y_true, y_pred, query_ids, k: int = 10) -> dict[str, np.ndarray]:
    """Within-query Spearman, NDCG@k and top-1 hit, one value per usable query (`query_ids` says which).

    A query is usable if it has at least two passages and its true utility is not constant:
    otherwise there is nothing to rank, and counting it as 0 would dilute the average.
    """
    y_true, y_pred, query_ids = np.asarray(y_true, float), np.asarray(y_pred, float), np.asarray(query_ids)
    spearman, ndcg, top1, used = [], [], [], []
    for qid in np.unique(query_ids):
        mask = query_ids == qid
        t, p = y_true[mask], y_pred[mask]
        if mask.sum() < 2 or np.std(t) < 1e-12:
            continue
        if np.std(p) < 1e-12:
            rho = 0.0  # a constant prediction ranks nothing
        else:
            rho = float(spearmanr(p, t).statistic)
        spearman.append(rho)
        ndcg.append(float(ndcg_score((t - t.min())[None, :], p[None, :], k=min(k, len(t)))))
        top1.append(float(t[int(np.argmax(p))] >= t.max()))
        used.append(qid)
    return {
        "spearman": np.array(spearman), "ndcg": np.array(ndcg), "top1": np.array(top1),
        "query_ids": np.array(used),  # which queries the values belong to, in the same order
    }


def evaluate_ranking(y_true, y_pred, query_ids, k: int = 10) -> dict:
    scores = per_query_scores(y_true, y_pred, query_ids, k=k)
    g = spearmanr(y_pred, y_true).statistic
    return {
        "spearman": float(scores["spearman"].mean()) if len(scores["spearman"]) else float("nan"),
        "ndcg": float(scores["ndcg"].mean()) if len(scores["ndcg"]) else float("nan"),
        "top1": float(scores["top1"].mean()) if len(scores["top1"]) else float("nan"),
        "global_spearman": 0.0 if np.isnan(g) else float(g),
        "n_queries": int(len(scores["spearman"])),
    }


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval, resampling the per-query values."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(values.mean()), float(lo), float(hi)
