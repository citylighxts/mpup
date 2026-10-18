"""Stage 3 of the multi-dataset experiment: evaluate MPUP under the protocol in `src.evaluation.protocol`.

  python -m experiments.run_multidataset --data data/processed/multidataset --out results/multidataset

Reads `features/*.npz` written by `extract_multidataset`. Datasets are split by label definition
and never mixed: those with gold answers carry the utility of Eq. (1) and form the pool; RAGBench
carries the annotation proxy of Eq. (5) and is used only to *train* in the proxy-transfer test.

Reports (all within-query metrics averaged over queries, 95% bootstrap CI over queries):

  pooled_cv      repeated 5-fold over queries of the pooled datasets, with per-dataset breakdown,
                 baselines (BM25 relevance, dense similarity, shortest-first) and the leave-one-
                 feature-group-out ablation as paired differences
  lodo           train on all other datasets, test on the held-out one
  proxy_transfer train on RAGBench (proxy labels), test on every Eq. (1) dataset, no target labels

Hyperparameters are fixed (200 trees, depth 5), so no nested CV is needed; see the README.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cv_report import cv_report, columns, default_fit_predict, lodo_report, make_fit_predict, _summarise
from experiments.nested_selection import GRID, nested_lodo
from src.evaluation.protocol import (
    LABEL_RAGBENCH_PROXY,
    LABEL_UTILITY,
    bootstrap_ci,
    per_query_scores,
    pool_tables,
)
from src.features.concatenate import FEATURE_GROUPS

FEATURE_NAMES = [
    "bm25_norm", "dense_cosine", "rank", "score_gap",
    "token_length", "flesch_kincaid", "source_credibility",
    "entity_overlap", "hyde_answerability",
    "entailment_prob", "contradiction_prob",
    "base_perplexity", "ctx_perplexity", "delta_h",
]
BASELINES = {
    "bm25": lambda X: X[:, 0],            # classic relevance: highest BM25 first
    "dense": lambda X: X[:, 1],           # dense cosine similarity
    "shortest": lambda X: -X[:, 4],       # shortest passage first (length artifact check)
}


def load_tables(data_dir: Path) -> list[dict]:
    tables = []
    for path in sorted((data_dir / "features").glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        tables.append({
            "X": z["X"], "y": z["y"], "query_ids": z["query_ids"],
            "dataset": z["dataset"], "label_source": str(z["label_source"]),
        })
    if not tables:
        raise SystemExit(f"no feature files in {data_dir / 'features'}")
    return tables


def dataset_statistics(tables: list[dict]) -> dict:
    stats = {}
    for t in tables:
        name = str(t["dataset"][0])
        pq = per_query_scores(t["y"], np.zeros(len(t["y"])), t["query_ids"])
        stats[name] = {
            "label_source": t["label_source"],
            "n_queries": int(len(set(t["query_ids"].tolist()))),
            "n_pairs": int(len(t["y"])),
            "passages_per_query": float(len(t["y"]) / len(set(t["query_ids"].tolist()))),
            "label_mean": float(np.mean(t["y"])),
            "label_nonzero_pct": float(100 * np.mean(t["y"] != 0)),
            "label_positive_pct": float(100 * np.mean(t["y"] > 0)),
            "rankable_queries": int(len(pq["query_ids"])),
        }
    return stats


def proxy_transfer_report(train: dict, targets: list[dict], seed: int = 0) -> dict:
    """Train on proxy-labelled data only; score each Eq. (1) dataset without using its labels."""
    out = {}
    for t in targets:
        name = str(t["dataset"][0])
        pred = default_fit_predict(train["X"], train["y"], t["X"], train["query_ids"], t["query_ids"])
        scored = {"mpup_proxy_trained": pred, **{n: f(t["X"]) for n, f in BASELINES.items()}}
        pq = {n: per_query_scores(t["y"], p, t["query_ids"]) for n, p in scored.items()}
        out[name] = {
            "n_queries": int(len(pq["mpup_proxy_trained"]["spearman"])),
            "models": {n: _summarise(v, seed) for n, v in pq.items()},
            "paired_vs": {
                n: {m: bootstrap_ci(pq["mpup_proxy_trained"][m] - pq[n][m], seed=seed)
                    for m in ("spearman", "ndcg", "top1")}
                for n in BASELINES
            },
        }
    return out


def run(data_dir: Path, repeats: int = 4, folds: int = 5, seed: int = 0, nested_grid=GRID) -> tuple[dict, dict]:
    """Returns (report, pooled Eq. (1) table)."""
    tables = load_tables(data_dir)
    utility = [t for t in tables if t["label_source"] == LABEL_UTILITY]
    proxy = [t for t in tables if t["label_source"] == LABEL_RAGBENCH_PROXY]
    if len(utility) < 2:
        raise SystemExit("need at least two datasets with the Eq. (1) utility label")
    pooled = pool_tables(utility)  # raises if label definitions were ever mixed

    ablations = {
        f"drop_{g}": columns(default_fit_predict, [c for c in range(14) if c not in range(sl.start, sl.stop)])
        for g, sl in FEATURE_GROUPS.items()
    }
    report = {
        "protocol": {"folds": folds, "repeats": repeats, "seed": seed,
                     "reference_config": "gbr(200,d5)-raw (as stated in the paper; no selection)",
                     "nested_cv": "leave-one-dataset-out selection over a fixed grid (see nested_lodo)"},
        "feature_names": FEATURE_NAMES,
        # Measured, not assumed: a column with no variance in the pool carries no information.
        "constant_features": [FEATURE_NAMES[i] for i in range(pooled["X"].shape[1]) if np.ptp(pooled["X"][:, i]) == 0],
        "datasets": dataset_statistics(tables),
        "pooled_cv": cv_report(
            pooled["X"], pooled["y"], pooled["query_ids"], baselines=BASELINES, extra_models=ablations,
            datasets=pooled["dataset"], n_splits=folds, n_repeats=repeats, seed=seed,
        ),
        "lodo": lodo_report(
            pooled["X"], pooled["y"], pooled["query_ids"], pooled["dataset"], baselines=BASELINES, seed=seed,
        ),
    }
    report["nested_lodo"] = nested_lodo(
        pooled["X"], pooled["y"], pooled["query_ids"], pooled["dataset"], baselines=BASELINES, grid=nested_grid, seed=seed,
    )
    # Which feature groups matter for the predictor that will actually be deployed. The configuration
    # was selected on these same datasets, so read this as description, not as an unbiased score.
    fc = report["nested_lodo"]["final_config"]
    selected_fp = make_fit_predict(fc["algorithm"], fc["center_by_query"], **fc["params"])
    report["ablation_selected"] = cv_report(
        pooled["X"], pooled["y"], pooled["query_ids"], fit_predict=selected_fp, baselines=BASELINES,
        extra_models={f"drop_{g}": columns(selected_fp, [c for c in range(14) if c not in range(sl.start, sl.stop)])
                      for g, sl in FEATURE_GROUPS.items()},
        n_splits=folds, n_repeats=repeats, seed=seed,
    )
    if proxy:
        report["proxy_transfer"] = proxy_transfer_report(pool_tables(proxy), utility, seed=seed)
    return report, pooled


# ---------------------------------------------------------------- formatting

def _ci(v) -> str:
    return f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"


def to_markdown(report: dict) -> str:
    L: list[str] = []
    prot = report["protocol"]
    L += ["# MPUP multi-dataset results", "",
          "Within-query Spearman ρ (mean over queries, 95% bootstrap CI over queries). "
          f"Pooled CV: {prot['folds']}-fold × {prot['repeats']} repeats over queries. "
          "`paper config` is the predictor as stated in the paper (raw features, 200 trees of depth 5) and involves no selection.", ""]
    if report.get("constant_features"):
        L += [f"Constant (uninformative) features in the pool: {', '.join(report['constant_features'])}.", ""]

    L += ["## Datasets", "", "| dataset | label | queries | pairs | passages/q | mean label | non-zero % | positive % | rankable queries |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, s in report["datasets"].items():
        L.append(f"| {n} | {s['label_source']} | {s['n_queries']} | {s['n_pairs']} | {s['passages_per_query']:.1f} | "
                 f"{s['label_mean']:+.3f} | {s['label_nonzero_pct']:.0f} | {s['label_positive_pct']:.0f} | {s['rankable_queries']} |")

    nl = report["nested_lodo"]
    L += ["", "## Headline: nested leave-one-dataset-out", "",
          "For each held-out dataset the configuration is chosen from the grid using only the other datasets "
          "(inner leave-one-dataset-out), refit on them, and scored on the held-out dataset. "
          f"Grid: {', '.join(nl['grid'])}.", "",
          "| held-out | queries | selected config | MPUP (nested) | paper config | dense | BM25 | MPUP − dense (paired) | MPUP − paper (paired) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, d in nl["per_dataset"].items():
        m = d["models"]
        L.append(f"| {n} | {d['n_queries']} | {d['selected']} | {_ci(m['nested']['spearman'])} | {_ci(m['paper']['spearman'])} | "
                 f"{_ci(m['dense']['spearman'])} | {_ci(m['bm25']['spearman'])} | {_ci(d['paired_vs']['dense']['spearman'])} | "
                 f"{_ci(d['paired_vs']['paper']['spearman'])} |")
    mac = nl["macro"]
    L.append(f"| **macro mean** | | | {mac['nested']['spearman']:+.3f} | {mac['paper']['spearman']:+.3f} | "
             f"{mac['dense']['spearman']:+.3f} | {mac['bm25']['spearman']:+.3f} | | |")
    pp = nl["pooled_paired_vs"]
    L += ["", f"Over all {nl['n_queries']} held-out queries together (paired, per-query ρ): MPUP − dense {_ci(pp['dense']['spearman'])}, "
          f"MPUP − BM25 {_ci(pp['bm25']['spearman'])}, MPUP − paper config {_ci(pp['paper']['spearman'])}.",
          "", f"Top-1 / NDCG@10, macro mean: MPUP {mac['nested']['top1']:+.2f} / {mac['nested']['ndcg']:+.2f}; "
          f"paper config {mac['paper']['top1']:+.2f} / {mac['paper']['ndcg']:+.2f}; dense {mac['dense']['top1']:+.2f} / {mac['dense']['ndcg']:+.2f}; "
          f"BM25 {mac['bm25']['top1']:+.2f} / {mac['bm25']['ndcg']:+.2f}.",
          "", f"Choosing the best configuration on the held-out dataset itself would have reported ρ {mac['oracle']['spearman']:+.3f}, "
          f"an optimism of {nl['optimism_of_oracle_selection']:+.3f} over the nested estimate. "
          f"Selection frequency: {nl['selection_frequency']}.",
          "", f"Configuration selected on all datasets (used for the deployable predictor): **{nl['final_config']['name']}**."]

    cv = report["pooled_cv"]
    L += ["", f"## Pooled cross-validation of the paper config ({cv['n_queries']} rankable queries)", "",
          "| dataset | queries | paper config | BM25 | dense | shortest-first | paper − dense (paired) |", "|---|---|---|---|---|---|---|"]
    m = cv["models"]
    L.append(f"| **all** | {cv['n_queries']} | {_ci(m['mpup']['spearman'])} | {_ci(m['bm25']['spearman'])} | "
             f"{_ci(m['dense']['spearman'])} | {_ci(m['shortest']['spearman'])} | {_ci(cv['paired_vs']['dense']['spearman'])} |")
    for n, d in cv["per_dataset"].items():
        mm = d["models"]
        L.append(f"| {n} | {d['n_queries']} | {_ci(mm['mpup']['spearman'])} | {_ci(mm['bm25']['spearman'])} | "
                 f"{_ci(mm['dense']['spearman'])} | {_ci(mm['shortest']['spearman'])} | {_ci(d['paired_vs']['dense']['spearman'])} |")
    L += ["", "## Feature-group ablation (paired: model − model without the group; positive = the group helps)", "",
          f"Left: the selected configuration ({nl['final_config']['name']}); right: the paper config. Pooled 5-fold CV over queries; "
          "the selected configuration was chosen on these datasets, so its column is descriptive.", "",
          "| group | selected config Δρ | paper config Δρ |", "|---|---|---|"]
    ab = report["ablation_selected"]["paired_vs"]
    for g in FEATURE_GROUPS:
        L.append(f"| {g.upper()} | {_ci(ab[f'drop_{g}']['spearman'])} | {_ci(cv['paired_vs'][f'drop_{g}']['spearman'])} |")
    if "proxy_transfer" in report:
        L += ["", "## Proxy transfer (paper config trained on RAGBench annotation proxy, zero target labels)", "",
              "| target | queries | MPUP (proxy-trained) | BM25 | dense | MPUP − dense (paired) |", "|---|---|---|---|---|---|"]
        for n, d in report["proxy_transfer"].items():
            L.append(f"| {n} | {d['n_queries']} | {_ci(d['models']['mpup_proxy_trained']['spearman'])} | "
                     f"{_ci(d['models']['bm25']['spearman'])} | {_ci(d['models']['dense']['spearman'])} | {_ci(d['paired_vs']['dense']['spearman'])} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/multidataset")
    ap.add_argument("--out", default="results/multidataset", help="writes <out>.json and <out>.md")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-checkpoint", default="checkpoints/mpup_pooled")
    args = ap.parse_args()

    report, pooled = run(Path(args.data), repeats=args.repeats, folds=args.folds, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    out.with_suffix(".md").write_text(to_markdown(report))
    print(to_markdown(report))

    if args.save_checkpoint:  # the deployable predictor: the config selected on all pooled Eq. (1) data
        from src.predictor.mpup_predictor import MPUPPredictor

        cfg = report["nested_lodo"]["final_config"]
        model = MPUPPredictor(algorithm=cfg["algorithm"], center_by_query=cfg["center_by_query"], **cfg["params"])
        model.fit(pooled["X"], pooled["y"], groups=pooled["query_ids"])
        model.save(args.save_checkpoint)
        print(f"saved pooled predictor ({cfg['name']}) to {args.save_checkpoint}")


if __name__ == "__main__":
    main()
