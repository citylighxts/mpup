"""Remove from a freshly prepared sample the questions that earlier samples already contained.

  python -m experiments.drop_seen_questions --data data/processed/mac_qwen7b_s3 \
      --seen-dirs data/processed/mac_qwen7b --seen-seeds 1

`--seen-dirs` are earlier prepared samples (their `items.json`); `--seen-seeds` rebuild the question lists of
samples that are no longer on disk from `build_dataset(name, n, seed)`, which needs no retrieval corpus. The
unfiltered file is kept as `items_all.json`; a second run starts from it, so the result never depends on how
often the command ran. Extraction should run after this, so no time is spent on questions that are dropped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def drop_seen_questions(items: list[dict], seen_ids: set[str]) -> list[dict]:
    return [it for it in items if str(it["query_id"]) not in seen_ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seen-dirs", nargs="*", default=[])
    ap.add_argument("--seen-seeds", type=int, nargs="*", default=[])
    ap.add_argument("--n", type=int, default=100, help="questions per dataset of the seeded samples")
    ap.add_argument("--cache", default="data/raw/cache")
    args = ap.parse_args()
    data = Path(args.data)
    source = data / "items_all.json" if (data / "items_all.json").exists() else data / "items.json"
    items = json.loads(source.read_text())
    if source.name == "items.json":
        (data / "items_all.json").write_text(json.dumps(items, ensure_ascii=False))

    seen: set[str] = set()
    for d in args.seen_dirs:
        seen |= {str(it["query_id"]) for it in json.loads((Path(d) / "items.json").read_text())}
    if args.seen_seeds:
        from src.data.benchmarks import ALL_DATASETS, build_dataset

        for seed in args.seen_seeds:
            for name in ALL_DATASETS:
                seen |= {str(it["query_id"]) for it in build_dataset(name, n=args.n, seed=seed, cache_dir=args.cache)}
    kept = drop_seen_questions(items, seen)
    (data / "items.json").write_text(json.dumps(kept, ensure_ascii=False))
    print(f"kept {len(kept)} of {len(items)} questions; {len(items) - len(kept)} were in earlier samples")


if __name__ == "__main__":
    main()
