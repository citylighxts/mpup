"""Stage 1 of the multi-dataset experiment: questions, candidate passages, corpus statistics.

  python -m experiments.prepare_data --corpus path/to/wiki18_100w.zip --out data/processed/multidataset

Writes
  items.json  every item of every dataset; "retrieved" items now hold their top-k BM25 passages
  stats.json  N, average passage length and document frequency of every query term, so BM25 of
              the *provided* paragraphs is scored with the same corpus statistics

The corpus zip is the FlashRAG `wiki18_100w` file (5 GB; see the README for where it comes from).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.data.benchmarks import ALL_DATASETS, build_dataset, build_ragbench, passage_text
from src.retrieval.stream_bm25 import collect_stats, iter_zip_lines, query_terms, retrieve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="path to wiki18_100w.zip")
    ap.add_argument("--out", default="data/processed/multidataset")
    ap.add_argument("--n", type=int, default=100, help="questions per dataset")
    ap.add_argument("--k", type=int, default=5, help="passages retrieved per question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="data/raw/cache")
    ap.add_argument("--no-ragbench", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    for name in ALL_DATASETS:
        got = build_dataset(name, n=args.n, seed=args.seed, cache_dir=args.cache)
        print(f"{name:18s} {len(got):4d} questions ({got[0]['pool']})", flush=True)
        items.extend(got)
    if not args.no_ragbench:
        rag = build_ragbench()
        print(f"{'ragbench-hotpotqa':18s} {len(rag):4d} questions (provided, proxy label)", flush=True)
        items.extend(rag)

    terms = {t for it in items for t in query_terms(it["query"])}
    open_lines = lambda: iter_zip_lines(args.corpus)  # noqa: E731
    t0 = time.time()
    stats = collect_stats(open_lines, terms, workers=args.workers)
    print(f"pass 1: {stats.n_docs:,} passages, avg {stats.avgdl:.1f} tokens, "
          f"{len(terms):,} query terms ({time.time() - t0:.0f}s)", flush=True)

    queries = {it["query_id"]: it["query"] for it in items if it["pool"] == "retrieved"}
    t0 = time.time()
    hits = retrieve(open_lines, queries, stats, k=args.k, workers=args.workers)
    print(f"pass 2: retrieved top-{args.k} for {len(queries)} questions ({time.time() - t0:.0f}s)", flush=True)

    for it in items:
        if it["pool"] == "retrieved":
            it["passages"] = [
                {"id": f"{it['query_id']}:w{h['id']}", "text": passage_text(h["title"], h["text"]), "bm25": h["bm25"]}
                for h in hits[it["query_id"]]
            ]
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False))
    (out / "stats.json").write_text(json.dumps(
        {"n_docs": stats.n_docs, "avgdl": stats.avgdl, "df": {t: stats.df.get(t, 0) for t in terms}}
    ))
    short = [it["query_id"] for it in items if it["pool"] == "retrieved" and len(it["passages"]) < args.k]
    print(f"wrote {len(items)} items to {out}; {len(short)} retrieved questions have fewer than {args.k} passages", flush=True)


if __name__ == "__main__":
    main()
