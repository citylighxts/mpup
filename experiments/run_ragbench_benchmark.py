"""End-to-end RAGBench benchmark with the full D1–D5 feature stack.

Pipeline per question:
  1. BM25 over the question's candidate documents            → D1
  2. textstat document-quality features                      → D2
  3. HyDE pseudo-answer (MLX Llama) + entity/lexical overlap  → D3
  4. NLI cross-encoder (passage ⊨ pseudo-answer)             → D4
  5. MLX logit probe ΔH                                       → D5
  label: fraction of the document's sentences that appear in
         RAGBench `all_utilized_sentence_keys`               → u_i_true proxy

Usage:
  python -m experiments.run_ragbench_benchmark --subset hotpotqa --n 100
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252 and crash on 'ρ' or any passage text."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_force_utf8_stdout()

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi

from src.retrieval.retriever import RetrievedPassage
from src.features.concatenate import build_feature_vector, FEATURE_GROUPS
from src.features.hyde import HyDEGenerator, MockHyDEGenerator
from src.probing.logit_probe import MLXProber, MockProber
from src.predictor.mpup_predictor import MPUPPredictor
from experiments.evaluate_baselines import run_evaluation
from experiments.ablation import run_ablation

MLX_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
CUDA_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 4-bit NF4, fits an 8GB card


def _tok(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _build_backend(backend: str):
    """mlx (Apple Silicon), cuda (NVIDIA), or mock (no model).

    The CUDA path shares one quantized model between the prober and HyDE rather than
    loading two copies — on an 8GB card a second 7B model does not fit.
    """
    if backend == "mlx":
        return MLXProber(MLX_MODEL), HyDEGenerator(MLX_MODEL)
    if backend == "cuda":
        from src.data.llm_backend import QuantizedLLM
        from src.features.hyde import CUDAHyDEGenerator
        from src.probing.logit_probe import QuantizedProber

        llm = QuantizedLLM(CUDA_MODEL, batch_size=16)
        return QuantizedProber(llm=llm), CUDAHyDEGenerator(llm=llm)
    return MockProber(seed=0), MockHyDEGenerator()


def build_matrix(subset: str, n_questions: int, backend: str):
    ds = load_dataset("galileo-ai/ragbench", subset, split="test", streaming=True)
    prober, hyde = _build_backend(backend)

    X, y, query_ids = [], [], []
    seen = 0
    for row in ds:
        if seen >= n_questions:
            break
        query = row["question"]
        docs = row["documents"]
        doc_sents = row["documents_sentences"]
        util_keys = set(row["all_utilized_sentence_keys"])
        if len(docs) < 2:
            continue

        bm25 = BM25Okapi([_tok(d) for d in docs])
        scores = bm25.get_scores(_tok(query))
        order = np.argsort(scores)[::-1]
        rank_of = {int(d): r for r, d in enumerate(order)}
        max_s = float(scores.max()) or 1.0
        sorted_s = sorted(scores, reverse=True)

        claim = hyde.generate(query)
        probes = prober.probe_batch(query, docs)

        for d_idx, text in enumerate(docs):
            sents = doc_sents[d_idx]
            doc_keys = {s[0] for s in sents}
            u_i = len(doc_keys & util_keys) / max(len(sents), 1)

            r = rank_of[d_idx]
            gap = (sorted_s[r] - sorted_s[r + 1]) / max_s if r + 1 < len(sorted_s) else 0.0
            passage = RetrievedPassage(
                doc_id=f"{seen}-{d_idx}", text=text,
                bm25_score=float(scores[d_idx]) / max_s, dense_score=0.0,
                rank=r, score_gap=gap,
            )
            vec = build_feature_vector(query, passage, probes[d_idx], hyde_answer=claim)
            X.append(vec)
            y.append(u_i)
            query_ids.append(seen)

        seen += 1
        if seen % 10 == 0:
            print(f"  {seen}/{n_questions} questions")

    return np.array(X), np.array(y), np.array(query_ids), seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mock", action="store_true", help="use MockProber/MockHyDE (no model)")
    ap.add_argument("--backend", choices=["mlx", "cuda", "mock"], default=None,
                    help="mlx (Apple Silicon), cuda (NVIDIA), mock. Overrides --mock.")
    ap.add_argument("--out", default="results/ragbench_hotpotqa_benchmark.json")
    args = ap.parse_args()

    backend = args.backend or ("mock" if args.mock else "mlx")
    print(f"Building feature matrix — subset={args.subset} n={args.n} backend={backend}")
    X, y, query_ids, seen = build_matrix(args.subset, args.n, backend=backend)
    print(f"\n{X.shape[0]} pairs / {seen} questions | u_i mean={y.mean():.3f} nonzero={np.mean(y>0)*100:.1f}%")

    # Split by query, not by row. Passages of one question sit next to each other, so a row
    # split puts some of a question's passages in train and the rest in test — the features
    # they share (BM25 rank, the HyDE claim, document set) then leak across the boundary.
    unique_queries = np.unique(query_ids)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_queries)
    train_queries = set(unique_queries[: int(len(unique_queries) * 0.8)].tolist())
    is_train = np.array([q in train_queries for q in query_ids])
    X_tr, y_tr, X_te, y_te = X[is_train], y[is_train], X[~is_train], y[~is_train]
    print(f"split by query: {is_train.sum()} train / {(~is_train).sum()} test pairs "
          f"({len(train_queries)}/{len(unique_queries)} questions)")

    pred = MPUPPredictor(algorithm="xgboost", n_estimators=200, max_depth=5)
    pred.fit(X_tr, y_tr)

    results = run_evaluation(X_te, y_te, pred, k_shots_list=[5, 10], ndcg_k=10, seed=42,
                             X_train=X_tr, y_train=y_tr)
    abl = run_ablation(X_tr, y_tr, X_te, y_te, ndcg_k=10)

    print(f"\n{'Method':<26} {'ρ':>9} {'NDCG@10':>9}")
    print("-" * 46)
    for k, v in results.items():
        print(f"{k:<26} {v['spearman_rho']:>9.4f} {v['ndcg_at_k']:>9.4f}")
    print("\nAblation — leave-one-group-out (retrained):")
    print(f"  {'ALL D1-D5':<14} ρ={abl['baseline']['spearman_rho']:.4f}")
    for g in ["d1", "d2", "d3", "d4", "d5"]:
        lo = abl[f"leave_out_{g}"]
        on = abl[f"only_{g}"]
        print(f"  drop {g}: ρ={lo['spearman_rho']:.4f} (Δ{lo['delta_rho']:+.4f})   {g} only: ρ={on['spearman_rho']:.4f}")

    payload = {
        "dataset": f"RAGBench {args.subset} (test split)",
        "n_questions": seen,
        "n_pairs": int(X.shape[0]),
        "features": "D1+D2+D3+D4+D5 (14-dim)",
        "backend": backend,
        "split": "by query (80/20)",
        "d5_model": {"mlx": MLX_MODEL, "cuda": CUDA_MODEL}.get(backend, "MockProber"),
        "hyde_model": {"mlx": MLX_MODEL, "cuda": CUDA_MODEL}.get(
            backend, "MockHyDE (rule-based)"
        ),
        "nli_model": "cross-encoder/nli-deberta-v3-small",
        "results": {k: {"spearman_rho": round(v["spearman_rho"], 4),
                        "ndcg_at_10": round(v["ndcg_at_k"], 4)} for k, v in results.items()},
        "ablation": {
            k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in abl.items()
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
