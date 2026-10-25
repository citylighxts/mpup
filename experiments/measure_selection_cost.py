"""Measured cost of selecting passages for one query on this machine (latency per stage and peak memory).

  python -m experiments.measure_selection_cost --data data/processed/mac_qwen7b \
      --config "<frozen name>" --grid v2 --extended --out results/mac_qwen7b/cost

Runs the stages the frozen pipeline really executes for one query, one query at a time as a user would, on
`--n` questions each of MuSiQue and HotpotQA (the longest contexts): the HyDE claim, dense scoring, the
entropy probe, the hidden-state pass when the configuration has a probe, the NLI check only when D4 is kept,
and the regressor. Training the regressor is not part of selection and is done once, before timing starts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np


def time_stages(stages: dict[str, Callable[[dict], object]], items: list[dict],
                clock: Callable[[], float] = time.perf_counter) -> dict:
    """Run each stage on every item (stages in order, item by item) and summarise seconds per query."""
    per_stage = {name: [] for name in stages}
    for item in items:
        for name, stage in stages.items():
            t0 = clock()
            stage(item)
            per_stage[name].append(clock() - t0)
    total = np.sum([per_stage[n] for n in stages], axis=0)

    def summary(v):
        return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
    return {"n_queries": len(items), "stages": {n: summary(v) for n, v in per_stage.items()}, "total": summary(total)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/mac_qwen7b")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--config", required=True, help="frozen configuration, a name from the grid")
    ap.add_argument("--grid", choices=["v1", "v2"], default="v2")
    ap.add_argument("--combined", default="")
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--n", type=int, default=40, help="questions per dataset (musique, hotpotqa)")
    ap.add_argument("--hidden-layers", type=int, nargs="+", default=[14, -1])
    ap.add_argument("--out", default="results/mac_qwen7b/cost")
    args = ap.parse_args()

    import mlx.core as mx

    from experiments.extract_multidataset import DEFAULT_DENSE, candidate_bm25, load_stats
    from experiments.nested_selection import GRID, grid_v2
    from experiments.run_multidataset import load_tables
    from src.data.mlx_backend import MLXLLM
    from src.evaluation.protocol import LABEL_UTILITY, pool_tables
    from src.features.concatenate import build_feature_vector
    from src.features.hyde import CUDAHyDEGenerator
    from src.features.d4_faithfulness import extract_d4
    from src.probing.logit_probe import MLXProber
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.retriever import RetrievedPassage, candidate_scores

    data = Path(args.data)
    grid = GRID if args.grid == "v1" else grid_v2(tuple(n for n in args.combined.split(",") if n))
    cfg = next(c for c in grid if c.name == args.config)
    pooled = pool_tables([t for t in load_tables(data, extended=args.extended) if t["label_source"] == LABEL_UTILITY])
    predict = cfg.fitter()(pooled["X"], pooled["y"], pooled["query_ids"])  # trained once; not part of selection

    items = [it for d in ("musique", "hotpotqa") for it in [x for x in json.loads((data / "items.json").read_text())
                                                             if x["dataset"] == d][: args.n]]
    stats = load_stats(data / "stats.json")
    llm = MLXLLM(args.model)
    prober, hyde, dense = MLXProber(llm=llm), CUDAHyDEGenerator(llm=llm), DenseRetriever(DEFAULT_DENSE)
    uses_d4 = "d4" not in cfg.drop_groups
    layers = tuple(args.hidden_layers)
    state: dict = {}

    def stage_hyde(it):
        state["claim"] = hyde.generate_batch([it["query"]])[0]
        mx.eval(mx.array(0))

    def stage_dense(it):
        state["cos"] = dense.score_passages(it["query"], [p["text"] for p in it["passages"]])

    def stage_probe(it):
        state["probes"] = prober.probe_batch(it["query"], [p["text"] for p in it["passages"]])

    def stage_hidden(it):
        state["H"] = prober.hidden_batch(it["query"], [p["text"] for p in it["passages"]], layers=layers)

    def stage_nli(it):
        for p in it["passages"]:
            extract_d4(state["claim"], p["text"])

    def stage_regressor(it):
        norm, ranks, gaps = candidate_scores(candidate_bm25(it, stats))
        rows = [build_feature_vector(it["query"], RetrievedPassage(
            doc_id=p["id"], text=p["text"], bm25_score=float(norm[j]), dense_score=float(state["cos"][j]),
            rank=int(ranks[j]), score_gap=float(gaps[j])), state["probes"][j], hyde_answer=state["claim"])
            for j, p in enumerate(it["passages"])]
        X = np.array(rows)
        if args.extended:
            H = state["H"].reshape(len(rows), -1) if cfg.probe else np.zeros((len(rows), pooled["X"].shape[1] - 15))
            X = np.c_[X, H, np.full(len(rows), np.nan)]
        predict(X, np.zeros(len(rows), dtype=int))

    stages = {"hyde": stage_hyde, "dense": stage_dense, "probe": stage_probe}
    if cfg.probe:
        stages["hidden_states"] = stage_hidden
    if uses_d4:
        stages["nli"] = stage_nli
    stages["features_and_regressor"] = stage_regressor
    result = time_stages(stages, items)
    peak = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory()
    result.update({"config": cfg.name, "model": args.model, "peak_gib": round(peak / 2**30, 2), "datasets": ["musique", "hotpotqa"]})
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
