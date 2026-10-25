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


# ----------------------------------------------------------- extended layout (hidden states, smooth label)

D_HIDDEN, N_LAYERS = 4, 2


def _write_hidden(dirpath, name, seed, shuffle=False, drop_last=False, loglik=True, n_queries=30, per_query=5):
    """hidden/<name>.npz (and loglik/) for the pairs _write() produced, optionally in another row order."""
    z = np.load(dirpath / "features" / f"{name}.npz")
    rng = np.random.default_rng(seed)
    n = len(z["y"])
    H = rng.normal(size=(n, N_LAYERS, D_HIDDEN)).astype(np.float16)
    u = rng.normal(size=n)
    order = rng.permutation(n) if shuffle else np.arange(n)
    if drop_last:
        order = order[:-1]
    (dirpath / "hidden").mkdir(exist_ok=True)
    np.savez(dirpath / "hidden" / f"{name}.npz", H=H[order], query_ids=z["query_ids"][order], passage_ids=z["passage_ids"][order])
    if loglik:
        (dirpath / "loglik").mkdir(exist_ok=True)
        np.savez(dirpath / "loglik" / f"{name}.npz", u_smooth=u[order], query_ids=z["query_ids"][order], passage_ids=z["passage_ids"][order])
    return H, u


def test_load_tables_extended_joins_hidden_and_smooth_label_by_ids(data_dir):
    H, u = _write_hidden(data_dir, "a", seed=1, shuffle=True)
    for n in "bc":
        _write_hidden(data_dir, n, seed=2)
    _write_hidden(data_dir, "rag", seed=3, loglik=False)
    tables = {str(t["dataset"][0]): t for t in load_tables(data_dir, extended=True)}
    X = tables["a"]["X"]
    assert X.shape == (150, 14 + N_LAYERS * D_HIDDEN + 1)
    np.testing.assert_allclose(X[:, 14:-1], H.reshape(150, -1).astype(np.float32), atol=1e-6)  # right rows despite the shuffle
    np.testing.assert_allclose(X[:, -1], u, atol=1e-6)
    assert np.isnan(tables["rag"]["X"][:, -1]).all()  # a dataset without gold answers has no smooth label


def test_load_tables_plain_is_unchanged_by_the_existence_of_hidden_files(data_dir):
    for n in ("a", "b", "c", "rag"):
        _write_hidden(data_dir, n, seed=1)
    assert all(t["X"].shape[1] == 14 for t in load_tables(data_dir))


def test_load_tables_extended_raises_on_a_missing_pair(data_dir):
    _write_hidden(data_dir, "a", seed=1, drop_last=True)
    with pytest.raises(ValueError, match="hidden"):
        load_tables(data_dir, extended=True)


def test_load_tables_extended_raises_without_a_hidden_cache(data_dir):
    with pytest.raises(ValueError, match="hidden"):
        load_tables(data_dir, extended=True)


def test_run_with_a_probe_variant_on_extended_data(data_dir):
    for i, n in enumerate(("a", "b", "c", "rag")):
        _write_hidden(data_dir, n, seed=i + 1)
    grid = [Config("ridge-centered", "ridge", True, (("alpha", 10.0),)),
            Config("ridge-probe", "ridge", True, (("alpha", 10.0),), probe=True)]
    report, _ = run(data_dir, repeats=1, folds=3, nested_grid=grid, extended=True)
    fc = report["nested_lodo"]["final_config"]
    assert fc["probe"] in (True, False) and "drop_groups" in fc and "smooth_label" in fc
    assert report["ablation_selected"] is None or "paired_vs" in report["ablation_selected"]
    assert "## Headline" in to_markdown(report)
