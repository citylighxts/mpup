import numpy as np
import pytest

from src.evaluation.protocol import (
    LABEL_RAGBENCH_PROXY,
    LABEL_UTILITY,
    assert_single_label_source,
    bootstrap_ci,
    evaluate_ranking,
    group_splits,
    leave_one_dataset_out,
    per_query_scores,
    pool_tables,
    repeated_group_splits,
)


# ---------------------------------------------------------------- group splits

def _groups(n_queries=20, per_query=4):
    return np.repeat(np.arange(n_queries), per_query)


def test_group_splits_never_split_a_query_across_train_and_test():
    groups = _groups()
    for train, test in group_splits(groups, n_splits=5, seed=0):
        assert not set(groups[train]) & set(groups[test])


def test_group_splits_put_every_query_in_exactly_one_test_fold():
    groups = _groups()
    seen = []
    for _, test in group_splits(groups, n_splits=5, seed=0):
        seen.extend(set(groups[test]))
    assert sorted(seen) == sorted(set(groups))
    assert len(seen) == len(set(seen))


def test_group_splits_are_balanced_and_deterministic():
    groups = _groups(n_queries=23)
    sizes = [len(set(groups[t])) for _, t in group_splits(groups, n_splits=5, seed=3)]
    assert max(sizes) - min(sizes) <= 1
    a = [t.tolist() for _, t in group_splits(groups, n_splits=5, seed=3)]
    b = [t.tolist() for _, t in group_splits(groups, n_splits=5, seed=3)]
    c = [t.tolist() for _, t in group_splits(groups, n_splits=5, seed=4)]
    assert a == b
    assert a != c


def test_group_splits_need_at_least_as_many_queries_as_folds():
    with pytest.raises(ValueError):
        group_splits(_groups(n_queries=3), n_splits=5)


def test_repeated_group_splits_count_and_variety():
    splits = repeated_group_splits(_groups(), n_splits=5, n_repeats=3, seed=0)
    assert len(splits) == 15
    first_tests = {tuple(t.tolist()) for _, t in splits[::5]}
    assert len(first_tests) == 3  # each repeat reshuffles


# --------------------------------------------------- leave-one-dataset-out

def test_leave_one_dataset_out_holds_out_whole_datasets():
    datasets = np.array(["nq"] * 6 + ["tqa"] * 4 + ["hotpot"] * 5)
    out = leave_one_dataset_out(datasets)
    assert [name for name, _, _ in out] == ["hotpot", "nq", "tqa"]
    for name, train, test in out:
        assert set(datasets[test]) == {name}
        assert name not in set(datasets[train])
        assert len(train) + len(test) == len(datasets)


def test_leave_one_dataset_out_needs_two_datasets():
    with pytest.raises(ValueError):
        leave_one_dataset_out(np.array(["nq"] * 5))


# ------------------------------------------------------------- label policy

def test_single_label_source_is_returned():
    assert assert_single_label_source([LABEL_UTILITY, LABEL_UTILITY]) == LABEL_UTILITY


def test_mixed_label_sources_are_refused():
    with pytest.raises(ValueError, match="label"):
        assert_single_label_source([LABEL_UTILITY, LABEL_RAGBENCH_PROXY])


def test_empty_label_sources_are_refused():
    with pytest.raises(ValueError):
        assert_single_label_source([])


def _table(dataset, label_source, n_queries=3, per_query=2):
    n = n_queries * per_query
    return {
        "X": np.ones((n, 14)),
        "y": np.linspace(0, 1, n),
        "query_ids": np.repeat(np.arange(n_queries), per_query).astype(str),
        "dataset": np.array([dataset] * n),
        "label_source": label_source,
    }


def test_pool_tables_keeps_query_ids_unique_across_datasets():
    pooled = pool_tables([_table("nq", LABEL_UTILITY), _table("tqa", LABEL_UTILITY)])
    assert pooled["X"].shape == (12, 14)
    assert len(set(pooled["query_ids"])) == 6  # "0" in nq and "0" in tqa are different queries
    assert pooled["label_source"] == LABEL_UTILITY
    assert set(pooled["dataset"]) == {"nq", "tqa"}


def test_pool_tables_refuses_mixed_label_sources():
    with pytest.raises(ValueError, match="label"):
        pool_tables([_table("nq", LABEL_UTILITY), _table("ragbench", LABEL_RAGBENCH_PROXY)])


# --------------------------------------------------------- per-query metrics

def test_per_query_scores_perfect_and_reversed():
    y = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    q = np.array(["a"] * 3 + ["b"] * 3)
    good = per_query_scores(y, y, q)
    assert np.allclose(good["spearman"], 1.0)
    assert np.allclose(good["top1"], 1.0)
    assert np.allclose(good["ndcg"], 1.0)
    bad = per_query_scores(y, -y, q)
    assert np.allclose(bad["spearman"], -1.0)
    assert np.allclose(bad["top1"], 0.0)


def test_per_query_scores_skip_queries_without_ranking_signal():
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.5, 1.0])  # query "a" has constant utility
    q = np.array(["a"] * 3 + ["b"] * 3)
    scores = per_query_scores(y, np.array([0.1, 0.2, 0.3, 0.0, 0.5, 1.0]), q)
    assert len(scores["spearman"]) == 1


def test_global_rho_can_hide_a_useless_within_query_ranking():
    # Queries differ in difficulty (offset), and the prediction only recovers the offset.
    # A pooled Spearman looks excellent; a within-query ranker would be worthless.
    rng = np.random.default_rng(0)
    q = np.repeat(np.arange(30), 5).astype(str)
    offset = np.repeat(np.linspace(0, 1, 30), 5)
    y = offset + rng.normal(0, 0.05, 150)
    pred = offset
    res = evaluate_ranking(y, pred, q)
    assert res["global_spearman"] > 0.9
    assert abs(res["spearman"]) < 0.3
    assert res["n_queries"] == 30


def test_bootstrap_ci_brackets_the_mean_and_collapses_for_constants():
    vals = np.array([0.1, 0.4, 0.5, 0.6, 0.9, 0.3, 0.2, 0.7])
    mean, lo, hi = bootstrap_ci(vals, n_boot=500, seed=0)
    assert lo <= mean <= hi
    assert mean == pytest.approx(vals.mean())
    mean, lo, hi = bootstrap_ci(np.full(6, 0.25), n_boot=100, seed=0)
    assert lo == hi == mean == pytest.approx(0.25)


def test_per_query_scores_report_which_queries_they_belong_to():
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.0, 1.0])
    q = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 2)
    s = per_query_scores(y, y, q)
    assert s["query_ids"].tolist() == ["b", "c"]  # "a" has constant utility and is skipped
    assert len(s["query_ids"]) == len(s["spearman"]) == len(s["ndcg"]) == len(s["top1"])
