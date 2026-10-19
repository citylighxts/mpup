"""Which feature groups are needed together: retrain on subsets of D1-D5 and compare with dense similarity.

  python -m experiments.ablation_subsets --data data/processed/multidataset data/processed/multidataset_seed1 \
      --out results/ablation_subsets

The leave-one-group-out ablation in `run_multidataset` removes one group at a time, which cannot show
whether two weak groups (D2 and D4) can be dropped *together*. Here each subset is retrained under
leave-one-dataset-out with a fixed configuration (no selection, so the numbers are descriptive and a
little optimistic) and scored by the macro mean of within-query rho. The paired difference to dense
similarity is taken over all held-out queries. Each `--data` directory is evaluated on its own.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from experiments.nested_selection import GRID
from experiments.run_multidataset import BASELINES, load_tables
from src.evaluation.protocol import LABEL_UTILITY, bootstrap_ci, leave_one_dataset_out, per_query_scores, pool_tables
from src.features.concatenate import FEATURE_GROUPS

SUBSETS = {
    "all": ("d1", "d2", "d3", "d4", "d5"),
    "d1+d2+d3+d5": ("d1", "d2", "d3", "d5"),
    "d1+d3+d4+d5": ("d1", "d3", "d4", "d5"),
    "d1+d3+d5": ("d1", "d3", "d5"),
    "d1+d5": ("d1", "d5"),
    "d1+d3": ("d1", "d3"),
    "d1": ("d1",),
}
CONFIGS = ("gbr(100,d2)-centered", "ridge(a=1000)-centered")


def _columns(groups) -> list[int]:
    return [c for g in groups for c in range(FEATURE_GROUPS[g].start, FEATURE_GROUPS[g].stop)]


def score_subset(pooled: dict, config: str, subset: str) -> dict:
    X, y, q, ds = pooled["X"], pooled["y"], pooled["query_ids"], pooled["dataset"]
    cols = _columns(SUBSETS[subset])
    fit_predict = next(c for c in GRID if c.name == config).fit_predict()
    per_dataset, mpup, dense = {}, [], []
    for name, train, test in leave_one_dataset_out(ds):
        pred = fit_predict(X[train][:, cols], y[train], X[test][:, cols], q[train], q[test])
        m = per_query_scores(y[test], pred, q[test])["spearman"]
        d = per_query_scores(y[test], BASELINES["dense"](X[test]), q[test])["spearman"]
        per_dataset[name] = float(m.mean())
        mpup.append(m)
        dense.append(d)
    return {
        "config": config, "subset": subset,
        "macro_spearman": float(np.mean(list(per_dataset.values()))),
        "paired_vs_dense": bootstrap_ci(np.concatenate(mpup) - np.concatenate(dense)),
        "per_dataset": per_dataset,
    }


def run(data_dir: Path) -> list[dict]:
    pooled = pool_tables([t for t in load_tables(data_dir) if t["label_source"] == LABEL_UTILITY])
    return Parallel(n_jobs=-1)(delayed(score_subset)(pooled, c, s) for c in CONFIGS for s in SUBSETS)


def to_markdown(report: dict) -> str:
    lines = ["# Feature-group subsets", "",
             "Leave-one-dataset-out, fixed configuration (no selection). Macro within-query Spearman; "
             "paired difference to dense similarity over all held-out queries (95% bootstrap CI).", ""]
    for data_dir, rows in report.items():
        lines += [f"## {data_dir}", "", "| config | features | ρ | − dense |", "|---|---|---|---|"]
        for r in rows:
            m, lo, hi = r["paired_vs_dense"]
            lines.append(f"| {r['config']} | {r['subset'].upper()} | {r['macro_spearman']:.3f} | "
                         f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/processed/multidataset"])
    ap.add_argument("--out", default="results/ablation_subsets")
    args = ap.parse_args()
    report = {d: run(Path(d)) for d in args.data}
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1))
    Path(args.out).with_suffix(".md").write_text(to_markdown(report))
    print(to_markdown(report))


if __name__ == "__main__":
    main()
