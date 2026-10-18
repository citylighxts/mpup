import numpy as np
import pytest

from experiments.nested_selection import Config
from experiments.run_multidataset import dataset_statistics, load_tables, run, to_markdown

SMALL = [Config('ridge-raw', 'ridge', False, (('alpha', 10.0),)), Config('ridge-centered', 'ridge', True, (('alpha', 10.0),))]
from src.evaluation.protocol import LABEL_RAGBENCH_PROXY, LABEL_UTILITY


def _write(dirpath, name, label_source, seed, n_queries=30, per_query=5, signal=1.0):
    rng = np.random.default_rng(seed)
    n = n_queries * per_query
    X = rng.normal(size=(n, 14))
    X[:, 0] = np.tile(np.linspace(1, 0, per_query), n_queries)  # a rank-like BM25 column
    y = signal * X[:, 13] + rng.normal(0, 0.3, n)               # delta_h carries the utility
    q = np.repeat([f"{name}:{i}" for i in range(n_queries)], per_query)
    (dirpath / "features").mkdir(exist_ok=True)
    np.savez(dirpath / "features" / f"{name}.npz", X=X, y=y, query_ids=q, passage_ids=np.arange(n).astype(str),
             dataset=np.array([name] * n), label_source=np.array(label_source))


@pytest.fixture
def data_dir(tmp_path):
    _write(tmp_path, "a", LABEL_UTILITY, 1)
    _write(tmp_path, "b", LABEL_UTILITY, 2)
    _write(tmp_path, "c", LABEL_UTILITY, 3)
    _write(tmp_path, "rag", LABEL_RAGBENCH_PROXY, 4)
    return tmp_path


def test_tables_keep_their_label_definition(data_dir):
    tables = load_tables(data_dir)
    assert sorted(t["label_source"] for t in tables) == sorted([LABEL_UTILITY] * 3 + [LABEL_RAGBENCH_PROXY])


def test_dataset_statistics_count_queries_and_pairs(data_dir):
    stats = dataset_statistics(load_tables(data_dir))
    assert stats["a"]["n_queries"] == 30 and stats["a"]["n_pairs"] == 150
    assert stats["a"]["passages_per_query"] == 5.0
    assert stats["rag"]["label_source"] == LABEL_RAGBENCH_PROXY


def test_full_run_produces_every_report_and_a_learnable_signal_beats_bm25(data_dir):
    report, pooled = run(data_dir, repeats=2, folds=3, seed=0, nested_grid=SMALL)
    assert set(report) >= {"protocol", "datasets", "pooled_cv", "lodo", "nested_lodo", "proxy_transfer", "feature_names"}
    assert set(report["nested_lodo"]["per_dataset"]) == {"a", "b", "c"}
    cv = report["pooled_cv"]
    assert set(cv["per_dataset"]) == {"a", "b", "c"}  # RAGBench-style data is not in the pool
    assert cv["models"]["mpup"]["spearman"][0] > 0.5
    assert cv["paired_vs"]["bm25"]["spearman"][1] > 0       # beats the BM25 baseline, CI above 0
    assert "drop_d5" in cv["paired_vs"]                      # delta_h lives in D5 here
    assert cv["paired_vs"]["drop_d5"]["spearman"][1] > 0    # so removing D5 must hurt
    assert set(report["lodo"]["per_dataset"]) == {"a", "b", "c"}
    assert set(report["proxy_transfer"]) == {"a", "b", "c"}
    assert pooled["label_source"] == LABEL_UTILITY
    assert report["constant_features"] == []  # the synthetic columns all vary


def test_mixing_label_definitions_in_the_pool_is_impossible(tmp_path):
    # Two datasets, one wrongly stamped with the proxy label, leaves only one utility dataset.
    _write(tmp_path, "a", LABEL_UTILITY, 1)
    _write(tmp_path, "b", LABEL_RAGBENCH_PROXY, 2)
    with pytest.raises(SystemExit, match="at least two"):
        run(tmp_path, repeats=1, folds=3, nested_grid=SMALL)


def test_markdown_report_has_every_section(data_dir):
    report, _ = run(data_dir, repeats=1, folds=3, nested_grid=SMALL)
    md = to_markdown(report)
    for heading in ("## Datasets", "## Headline: nested leave-one-dataset-out", "## Pooled cross-validation",
                    "## Feature-group ablation", "## Proxy transfer"):
        assert heading in md


def test_constant_columns_are_reported(tmp_path):
    # three datasets: nested selection needs two for the inner loop after one is held out
    for i, name in enumerate("abc"):
        _write(tmp_path, name, LABEL_UTILITY, i + 1)
    for name in "abc":  # force source_credibility (column 6) to be constant, as in the real data
        z = dict(np.load(tmp_path / "features" / f"{name}.npz"))
        z["X"][:, 6] = 0.5
        np.savez(tmp_path / "features" / f"{name}.npz", **z)
    report, _ = run(tmp_path, repeats=1, folds=3, nested_grid=SMALL)
    assert report["constant_features"] == ["source_credibility"]
    assert "Constant (uninformative) features" in to_markdown(report)
