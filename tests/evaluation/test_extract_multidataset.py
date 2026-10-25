from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from experiments.extract_multidataset import candidate_bm25, item_features, item_hidden, main, process_dataset
from src.features.concatenate import FEATURE_DIM
from src.probing.logit_probe import LogitProbeResult
from src.retrieval.stream_bm25 import CorpusStats


STATS = CorpusStats(n_docs=1000, avgdl=20.0, df={"capital": 50, "france": 40})


def _nli():
    fake = MagicMock()
    fake.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    fake.predict.return_value = np.array([[-2.0, 3.0, -0.5]])
    return fake


class FakeDense:
    def score_passages(self, query, texts):
        return [0.9 - 0.1 * i for i in range(len(texts))]


class FakeProber:
    def probe_batch(self, query, texts):
        return [LogitProbeResult(h_base=3.0, h_ctx=2.0 + 0.1 * i, delta_h=1.0 - 0.1 * i,
                                 base_perplexity=20.0, ctx_perplexity=7.0 + i) for i in range(len(texts))]


def test_retrieved_passages_keep_the_bm25_from_the_corpus_scan():
    item = {"query": "capital of France", "passages": [{"id": "a", "text": "x", "bm25": 7.5}, {"id": "b", "text": "y", "bm25": 3.0}]}
    assert candidate_bm25(item, STATS) == [7.5, 3.0]


def test_provided_passages_are_scored_with_the_same_corpus_statistics():
    item = {"query": "capital of France", "passages": [
        {"id": "a", "text": '"Paris"\nParis is the capital of France.'},
        {"id": "b", "text": '"Cats"\nCats are small mammals.'},
    ]}
    s = candidate_bm25(item, STATS)
    assert s[0] > 0 and s[1] == 0.0


def test_item_features_have_the_trained_scale_and_the_right_shape():
    item = {"query": "capital of France", "passages": [
        {"id": "a", "text": '"Cats"\nCats are small mammals.', "bm25": 2.0},
        {"id": "b", "text": '"Paris"\nParis is the capital of France.', "bm25": 8.0},
    ]}
    with patch("src.features.d4_faithfulness._get_nli", return_value=_nli()):
        X = item_features(item, "The capital of France is Paris.", STATS, FakeDense(), FakeProber())
    assert X.shape == (2, FEATURE_DIM) and np.isfinite(X).all()
    # D1: bm25 divided by the query's best, rank 0 = best, gap to the next-ranked / best
    assert X[1, 0] == pytest.approx(1.0) and X[0, 0] == pytest.approx(0.25)
    assert X[1, 2] == 0 and X[0, 2] == 1
    assert X[1, 3] == pytest.approx(0.75) and X[0, 3] == 0.0
    assert X[0, 1] == pytest.approx(0.9) and X[1, 1] == pytest.approx(0.8)  # dense cosine, passage order
    # D5 comes from the probe: delta_h = h_base - h_ctx
    assert X[0, 13] == pytest.approx(1.0) and X[1, 13] == pytest.approx(0.9)


# ------------------------------------------------------------- hidden-state cache


class HiddenProber(FakeProber):
    D = 5

    def hidden_batch(self, query, texts, layers=(14, -1), max_length=512):
        out = np.zeros((len(texts), len(layers), self.D), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = len(t) + 0.5 * np.arange(len(layers))[:, None] + np.arange(self.D)
        return out


class FakeHyde:
    def generate_batch(self, queries):
        return ["The capital of France is Paris."] * len(queries)


def _proxy_items():
    return [{"query_id": f"q{i}", "query": "capital of France", "gold_answers": [],
             "passages": [{"id": f"q{i}:a", "text": "Cats are small mammals." * (i + 1), "bm25": 2.0, "proxy_label": 0.0},
                          {"id": f"q{i}:b", "text": "Paris is the capital of France.", "bm25": 8.0, "proxy_label": 1.0}]}
            for i in range(3)]


def _run(tmp_path, **kw):
    with patch("src.features.d4_faithfulness._get_nli", return_value=_nli()):
        process_dataset("toy", _proxy_items(), tmp_path, None, FakeHyde(), HiddenProber(), FakeDense(), STATS, **kw)


def test_item_hidden_shape():
    item = _proxy_items()[0]
    assert item_hidden(item, HiddenProber(), (14, -1)).shape == (2, 2, HiddenProber.D)


def test_hidden_file_aligned_with_features(tmp_path):
    _run(tmp_path, hidden_layers=(14, -1))
    f, h = np.load(tmp_path / "features" / "toy.npz"), np.load(tmp_path / "hidden" / "toy.npz")
    assert h["H"].dtype == np.float16 and h["H"].shape == (6, 2, HiddenProber.D)
    assert (h["query_ids"] == f["query_ids"]).all() and (h["passage_ids"] == f["passage_ids"]).all()
    # row j of H belongs to the pair at row j of X: the cats passage of q1 is twice as long as that of q0
    assert h["H"][2, 0, 0] > h["H"][0, 0, 0]


def test_no_hidden_file_without_flag(tmp_path):
    _run(tmp_path)
    assert (tmp_path / "features" / "toy.npz").exists() and not (tmp_path / "hidden").exists()


def test_main_recomputes_dataset_that_lacks_hidden_cache(tmp_path, monkeypatch):
    import json
    import sys

    (tmp_path / "items.json").write_text(json.dumps([{**it, "dataset": "toy", "pool": "provided"} for it in _proxy_items()]))
    (tmp_path / "stats.json").write_text(json.dumps({"n_docs": 1000, "avgdl": 20.0, "df": {}}))
    calls = []
    monkeypatch.setattr("experiments.extract_multidataset.process_dataset", lambda name, *a, **k: calls.append((name, k.get("hidden_layers"))))
    monkeypatch.setattr("experiments.extract_multidataset.DenseRetriever", lambda m: FakeDense())
    fake_mlx = type("M", (), {"MLXLLM": lambda *a, **k: object()})
    fake_probe = type("P", (), {"MLXProber": lambda llm=None: HiddenProber()})
    monkeypatch.setitem(sys.modules, "src.data.mlx_backend", fake_mlx)
    monkeypatch.setitem(sys.modules, "src.probing.logit_probe", fake_probe)
    monkeypatch.setattr("experiments.extract_multidataset.CUDAHyDEGenerator", lambda llm: FakeHyde())
    (tmp_path / "features").mkdir()
    np.savez(tmp_path / "features" / "toy.npz", X=np.zeros((1, 14)))
    monkeypatch.setattr(sys, "argv", ["x", "--data", str(tmp_path), "--hidden-layers", "14", "-1"])
    main()
    assert calls == [("toy", (14, -1))]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--data", str(tmp_path)])
    main()
    assert calls == []  # features exist and no hidden requested: skipped as before
