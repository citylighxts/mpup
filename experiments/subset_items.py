"""Questions for a second LLM: every test question of the larger study and a hash-chosen share of its development ones.

  python -m experiments.subset_items --out data/processed/big_mistral

Writes `items.json` (and copies `stats.json`) so that `extract_multidataset` and `extract_extra` can be run on
the subset with another `--model`. The test part is the one of `run_big.split`; a development question is kept
when a second, salted hash of its id falls below `--dev-share`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from experiments import run_big


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/big")
    ap.add_argument("--seen", nargs="*", default=["data/processed/multidataset", "data/processed/multidataset_seed1"])
    ap.add_argument("--dev-share", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = json.loads((Path(args.data) / "items.json").read_text())
    q = np.array([f"{it['dataset']}:{it['query_id']}" for it in items])
    test = run_big.split(q, run_big.seen_ids([Path(p) for p in args.seen]))
    h = np.array([int(hashlib.md5(("m2" + x).encode()).hexdigest()[:8], 16) / 16**8 for x in q])
    keep = [it for it, k in zip(items, test | (h < args.dev_share)) if k]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(keep, ensure_ascii=False))
    shutil.copy(Path(args.data) / "stats.json", out / "stats.json")
    print(f"{len(keep)} questions, {sum(len(it['passages']) for it in keep)} pairs")


if __name__ == "__main__":
    main()
