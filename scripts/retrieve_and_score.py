"""Add BM25 retrieval scores and ranks to labeled JSONL.

Setelah label_utility.py, passages belum punya BM25/dense score.
Script ini menjalankan BM25 retrieval pada setiap query dan menambahkan
D1 fields (bm25, dense, rank, gap) ke setiap passage.

Usage:
  python scripts/retrieve_and_score.py \
    --input data/raw/ragbench_labeled.jsonl \
    --output data/raw/ragbench_scored.jsonl
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
from rank_bm25 import BM25Okapi


def add_retrieval_scores(item: dict) -> dict:
    """Re-score passages against query with BM25; add rank and score_gap."""
    query_tokens = item["query"].lower().split()
    passages = item["passages"]

    if not passages:
        return item

    corpus = [p["text"].lower().split() for p in passages]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)

    # rank by BM25 score descending
    ranked_idx = np.argsort(scores)[::-1]

    scored_passages = [None] * len(passages)
    for rank, idx in enumerate(ranked_idx):
        gap = float(scores[ranked_idx[rank]] - scores[ranked_idx[rank + 1]]) \
            if rank + 1 < len(ranked_idx) else 0.0
        scored_passages[idx] = {
            **passages[idx],
            "bm25": float(scores[idx]),
            "dense": 0.0,   # placeholder; replace with dense encoder if available
            "rank": rank,
            "gap": gap,
        }

    item["passages"] = scored_passages
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input) as fin, open(out_path, "w") as fout:
        for line in tqdm(fin, desc="Scoring"):
            item = json.loads(line)
            item = add_retrieval_scores(item)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
