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
from pathlib import Path

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


def _tok(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def build_matrix(subset: str, n_questions: int, use_mlx: bool):
    ds = load_dataset("galileo-ai/ragbench", subset, split="test", streaming=True)
    prober = MLXProber(MLX_MODEL) if use_mlx else MockProber(seed=0)
    hyde = HyDEGenerator(MLX_MODEL) if use_mlx else MockHyDEGenerator()

    X, y = [], []
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

        seen += 1
        if seen % 10 == 0:
            print(f"  {seen}/{n_questions} questions")

    return np.array(X), np.array(y), seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mock", action="store_true", help="use MockProber/MockHyDE (no MLX)")
    ap.add_argument("--out", default="results/ragbench_hotpotqa_benchmark.json")
    args = ap.parse_args()

    print(f"Building feature matrix — subset={args.subset} n={args.n} mlx={not args.mock}")
    X, y, seen = build_matrix(args.subset, args.n, use_mlx=not args.mock)
    print(f"\n{X.shape[0]} pairs / {seen} questions | u_i mean={y.mean():.3f} nonzero={np.mean(y>0)*100:.1f}%")

    split = int(len(X) * 0.8)
    pred = MPUPPredictor(algorithm="xgboost", n_estimators=200, max_depth=5)
    pred.fit(X[:split], y[:split])

    results = run_evaluation(X[split:], y[split:], pred, k_shots_list=[5, 10], ndcg_k=10, seed=42)
    abl = run_ablation(X[:split], y[:split], X[split:], y[split:], ndcg_k=10)

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
        "d5_model": MLX_MODEL if not args.mock else "MockProber",
        "hyde_model": MLX_MODEL if not args.mock else "MockHyDE (rule-based)",
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
