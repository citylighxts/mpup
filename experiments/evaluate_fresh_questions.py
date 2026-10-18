"""Frozen-configuration check on questions the grid was never designed on.

  python -m experiments.evaluate_fresh_questions --train data/processed/multidataset \
      --fresh data/processed/multidataset_seed1 --config "gbr(100,d2)-centered" --out results/fresh_questions

The configuration grid of `nested_selection` was chosen after looking at the first 100 questions of
each dataset, so even the nested score is slightly optimistic. Here the configuration is fixed in
advance, fitted on all Eq. (1) datasets of the original questions, and scored on a second sample of
questions (another sampling seed) from the same datasets. Questions that were already in the first
sample are dropped. Nothing is selected or tuned on the fresh questions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.nested_selection import GRID, PAPER
from experiments.run_multidataset import BASELINES, load_tables
from src.evaluation.protocol import LABEL_UTILITY, bootstrap_ci, per_query_scores, pool_tables

METRICS = ("spearman", "ndcg", "top1")


def _utility_tables(data_dir: Path) -> list[dict]:
    return [t for t in load_tables(data_dir) if t["label_source"] == LABEL_UTILITY]


def run(train_dir: Path, fresh_dir: Path, config_name: str) -> dict:
    cfg = next(c for c in GRID if c.name == config_name)
    pooled = pool_tables(_utility_tables(train_dir))
    fresh_tables = {str(t["dataset"][0]): t for t in _utility_tables(fresh_dir)}
    old_ids = {str(d["query_id"]) for d in json.loads((train_dir / "items.json").read_text())}

    models = {cfg.name: cfg, "paper config": PAPER}
    fitted = {n: c.fit_predict() for n, c in models.items()}
    per_dataset, pooled_scores = {}, {}
    for name, t in sorted(fresh_tables.items()):
        keep = np.array([str(q) not in old_ids for q in t["query_ids"]])
        X, y, q = t["X"][keep], t["y"][keep], t["query_ids"][keep]
        preds = {n: f(pooled["X"], pooled["y"], X, pooled["query_ids"], q) for n, f in fitted.items()}
        preds.update({b: np.asarray(f(X), float) for b, f in BASELINES.items()})
        pq = {n: per_query_scores(y, p, q) for n, p in preds.items()}
        for n in pq:
            for m in METRICS:
                pooled_scores.setdefault(n, {}).setdefault(m, []).append(pq[n][m])
        per_dataset[name] = {
            "n_dropped_as_already_seen": len({str(x) for x in t["query_ids"][~keep]}),
            "n_queries": int(len(pq[cfg.name]["spearman"])),
            "spearman": {n: float(v["spearman"].mean()) for n, v in pq.items()},
            "paired_mpup_vs_dense": bootstrap_ci(pq[cfg.name]["spearman"] - pq["dense"]["spearman"]),
        }
    cat = {n: {m: np.concatenate(v) for m, v in d.items()} for n, d in pooled_scores.items()}
    macro = {n: {m: float(np.mean([np.mean(v) for v in pooled_scores[n][m]])) for m in METRICS} for n in cat}
    paired = {
        b: {m: bootstrap_ci(cat[cfg.name][m] - cat[b][m]) for m in METRICS}
        for b in cat if b != cfg.name
    }
    return {
        "config": cfg.name,
        "trained_on": sorted({str(d) for d in pooled["dataset"]}),
        "n_fresh_queries": int(len(cat[cfg.name]["spearman"])),
        "macro": macro,
        "paired_mpup_vs": paired,
        "per_dataset": per_dataset,
    }


def to_markdown(r: dict) -> str:
    lines = [
        "# Frozen configuration on fresh questions", "",
        f"Configuration `{r['config']}` fitted on the original questions of {len(r['trained_on'])} datasets, "
        f"scored on {r['n_fresh_queries']} rankable questions from a second sample (another seed). "
        "Within-query Spearman, macro-averaged over datasets.", "",
        "| model | ρ | NDCG@10 | top-1 |", "|---|---|---|---|",
    ]
    for n, m in r["macro"].items():
        lines.append(f"| {n} | {m['spearman']:.3f} | {m['ndcg']:.3f} | {m['top1']:.3f} |")
    lines += ["", "Paired difference of the frozen configuration over all fresh queries (95% bootstrap CI):", ""]
    for b, d in r["paired_mpup_vs"].items():
        m, lo, hi = d["spearman"]
        lines.append(f"- vs {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    lines += ["", "| dataset | queries | " + " | ".join(next(iter(r["per_dataset"].values()))["spearman"]) + " |",
              "|---|---|" + "---|" * len(next(iter(r["per_dataset"].values()))["spearman"])]
    for d, v in r["per_dataset"].items():
        lines.append(f"| {d} | {v['n_queries']} | " + " | ".join(f"{x:.3f}" for x in v["spearman"].values()) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/multidataset")
    ap.add_argument("--fresh", default="data/processed/multidataset_seed1")
    ap.add_argument("--config", default="gbr(100,d2)-centered", help="a name from nested_selection.GRID")
    ap.add_argument("--out", default="results/fresh_questions")
    args = ap.parse_args()
    r = run(Path(args.train), Path(args.fresh), args.config)
    Path(args.out).with_suffix(".json").write_text(json.dumps(r, indent=1))
    Path(args.out).with_suffix(".md").write_text(to_markdown(r))
    print(to_markdown(r))


if __name__ == "__main__":
    main()
