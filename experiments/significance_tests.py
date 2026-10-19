"""Hypothesis tests behind the forest plot (Fig. 2): is MPUP better than dense similarity?

  python -m experiments.significance_tests --data data/processed/multidataset \
      --results results/multidataset.json --out results/significance_mlx

For every held-out dataset of the nested leave-one-dataset-out run, the configuration that the nested
selection chose for it (`per_dataset[name]["selected"]` in --results) is refitted on the other datasets and
scored on the held-out one, which reproduces the outer predictions of `nested_lodo` without repeating the
inner selection. The unit of analysis is the query: d_i = rho_MPUP(i) - rho_dense(i), where rho is the
within-query Spearman correlation with the true utility.

  H0: E[d] = 0   (MPUP and dense similarity order the passages equally well on average)
  H1: E[d] != 0  (two-sided; the one-sided p for "MPUP better" is half of it when mean(d) > 0)

Why three tests. Per-query rho values are bounded and usually not normal (Shapiro-Wilk is reported), so the
primary test is a paired sign-flip permutation test of the mean, which needs no distributional assumption
and matches the bootstrap intervals of the paper. The Wilcoxon signed-rank test (median/ranks) and the paired
t-test (mean, normal theory) are reported as checks: they should agree when the conclusion is robust.

Beyond the p-value, each row also gives the effect size (mean difference, Cohen's d_z, share of queries where
MPUP wins), a Holm correction over the datasets (11 tests are a family; the pooled test is a separate, single
primary test), a dataset-level test with the 11 per-dataset means as units (queries of one dataset are not
independent draws from a population of datasets), and the minimum detectable effect at 80% power, which says
how small a difference a dataset of this size could have shown.

Second mode (`--from-ci`): when the per-query values are not at hand (the Qwen features live on the GPU machine,
only the bootstrap intervals of Table IV are stored in the results JSON), the p-value is recovered from the
interval itself with the normal approximation of Altman and Bland (BMJ 2011): SE = (hi - lo) / (2 * 1.96),
z = estimate / SE, p = 2 * (1 - Phi(|z|)). It is accurate when the interval is roughly symmetric, which is
checked and reported; `validate_against_exact` compares it with the exact permutation p-values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from experiments.nested_selection import GRID
from experiments.run_multidataset import load_tables
from src.evaluation.protocol import LABEL_UTILITY, leave_one_dataset_out, per_query_scores, pool_tables

Z_ALPHA, Z_BETA = 1.959964, 0.841621  # two-sided alpha 0.05, power 0.80


def sign_flip_pvalue(d: np.ndarray, n_perm: int = 200_000, seed: int = 0, chunk: int = 20_000) -> float:
    """Two-sided paired permutation test of mean(d) = 0 by random sign flips (Monte Carlo, with the +1 correction)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    observed = abs(d.mean())
    hits, done = 0, 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        signs = rng.integers(0, 2, size=(b, len(d)), dtype=np.int8) * 2 - 1
        hits += int((np.abs((signs * d).mean(axis=1)) >= observed - 1e-15).sum())
        done += b
    return (hits + 1) / (n_perm + 1)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def describe(d: np.ndarray, n_perm: int, seed: int) -> dict:
    d = np.asarray(d, float)
    n = len(d)
    sd = float(d.std(ddof=1))
    nonzero = d[d != 0]
    out = {
        "n": n,
        "mean_diff": float(d.mean()),
        "sd": sd,
        "cohen_dz": float(d.mean() / sd) if sd > 0 else 0.0,
        "share_mpup_wins": float((d > 0).sum() / max(len(nonzero), 1)),
        "p_permutation": sign_flip_pvalue(d, n_perm, seed),
        "p_ttest": float(stats.ttest_1samp(d, 0.0).pvalue),
        "p_wilcoxon": float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 6 else float("nan"),
        "p_shapiro": float(stats.shapiro(d).pvalue) if 3 <= n <= 5000 else float("nan"),
        "mde_80": float((Z_ALPHA + Z_BETA) * sd / np.sqrt(n)),
    }
    out["n_needed_for_observed_mean"] = int(np.ceil(((Z_ALPHA + Z_BETA) * sd / abs(out["mean_diff"])) ** 2)) if out["mean_diff"] else None
    return out


def per_query_differences(tables: list[dict], selected: dict[str, str]) -> dict[str, np.ndarray]:
    """d_i for the queries of every held-out dataset, from the configuration the nested selection chose for it."""
    pooled = pool_tables(tables)
    X, y, q, ds = pooled["X"], pooled["y"], pooled["query_ids"], pooled["dataset"]
    configs = {c.name: c for c in GRID}
    out = {}
    for name, train, test in leave_one_dataset_out(ds):
        pred = configs[selected[name]].fit_predict()(X[train], y[train], X[test], q[train], q[test])
        mpup = per_query_scores(y[test], pred, q[test])
        dense = per_query_scores(y[test], X[test][:, 1], q[test])  # column 1 = dense cosine similarity
        assert (mpup["query_ids"] == dense["query_ids"]).all()
        out[name] = mpup["spearman"] - dense["spearman"]
    return out


def run(data_dir: Path, results: Path, n_perm: int = 200_000, seed: int = 0) -> dict:
    tables = [t for t in load_tables(data_dir) if t["label_source"] == LABEL_UTILITY]
    nested = json.loads(results.read_text())["nested_lodo"]["per_dataset"]
    selected = {name: d["selected"] for name, d in nested.items()}
    diffs = per_query_differences(tables, selected)
    rows = {name: describe(d, n_perm, seed) for name, d in diffs.items()}
    for key in ("p_permutation", "p_ttest", "p_wilcoxon"):
        adj = holm([rows[n][key] for n in rows])
        for n, a in zip(rows, adj):
            rows[n][key + "_holm"] = a
    pooled = describe(np.concatenate(list(diffs.values())), n_perm, seed)
    means = np.array([r["mean_diff"] for r in rows.values()])
    dataset_level = {
        "n_datasets": len(means), "mean_of_means": float(means.mean()),
        "n_positive": int((means > 0).sum()),
        "p_sign_test": float(stats.binomtest(int((means > 0).sum()), len(means), 0.5).pvalue),
        "p_wilcoxon": float(stats.wilcoxon(means).pvalue),
        "p_ttest": float(stats.ttest_1samp(means, 0.0).pvalue),
    }
    check = {n: float(np.mean(d)) for n, d in diffs.items()}
    return {"per_dataset": rows, "pooled": pooled, "dataset_level": dataset_level, "mean_diff_check": check,
            "selected": selected, "n_perm": n_perm, "seed": seed}


def p_from_interval(estimate: float, lo: float, hi: float) -> dict:
    """Two-sided p-value and standard error recovered from an estimate and its 95% interval (normal approximation)."""
    se = (hi - lo) / (2 * Z_ALPHA)
    z = estimate / se
    asymmetry = abs((hi - estimate) - (estimate - lo)) / (hi - lo)  # 0 = symmetric
    return {"estimate": estimate, "lo": lo, "hi": hi, "se": se, "z": z,
            "p": float(2 * stats.norm.sf(abs(z))), "asymmetry": asymmetry}


def from_results(results: Path, fresh: Path | None = None) -> dict:
    """p-values for the rows of Table IV (every held-out dataset and the pooled difference) and the fresh-question check."""
    nested = json.loads(results.read_text())["nested_lodo"]
    rows = {}
    for name, d in nested["per_dataset"].items():
        est, lo, hi = d["paired_vs"]["dense"]["spearman"]
        rows[name] = {"n": d["n_queries"], **p_from_interval(est, lo, hi)}
        rows[name]["cohen_dz"] = est / (rows[name]["se"] * np.sqrt(d["n_queries"]))
    for key, adj in zip(("p_holm",), (holm([r["p"] for r in rows.values()]),)):
        for r, a in zip(rows.values(), adj):
            r[key] = a
    est, lo, hi = nested["pooled_paired_vs"]["dense"]["spearman"]
    pooled = {"n": nested["n_queries"], **p_from_interval(est, lo, hi)}
    pooled["cohen_dz"] = est / (pooled["se"] * np.sqrt(pooled["n"]))
    out = {"per_dataset": rows, "pooled": pooled, "method": "normal approximation from the 95% bootstrap interval"}
    if fresh is not None:
        f = json.loads(fresh.read_text())
        est, lo, hi = f["paired_mpup_vs"]["dense"]["spearman"]
        out["fresh_questions"] = {"n": f["n_fresh_queries"], **p_from_interval(est, lo, hi)}
    return out


def validate_against_exact(results: Path, exact: dict) -> list[tuple[str, float, float]]:
    """(dataset, p from the interval, exact permutation p) for the same run, to see how good the approximation is."""
    approx = from_results(results)["per_dataset"]
    rows = [(n, approx[n]["p"], exact["per_dataset"][n]["p_permutation"]) for n in approx]
    rows.append(("pooled", from_results(results)["pooled"]["p"], exact["pooled"]["p_permutation"]))
    return rows


def _p(v: float) -> str:
    return "n/a" if v != v else ("<0.001" if v < 0.001 else f"{v:.3f}")


def to_markdown(r: dict) -> str:
    L = ["# MPUP vs dense similarity: hypothesis tests", "",
         "Unit: the query (d = rho_MPUP - rho_dense, within-query Spearman). H0: E[d] = 0, H1: E[d] != 0, alpha = 0.05. "
         f"Primary test: paired sign-flip permutation test ({r['n_perm']:,} flips). Holm correction over the {len(r['per_dataset'])} datasets.", "",
         "| dataset | n | mean d | d_z | MPUP wins | p perm | p Holm | p Wilcoxon | p t-test | Shapiro | MDE (80%) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, x in sorted(r["per_dataset"].items()):
        L.append(f"| {name} | {x['n']} | {x['mean_diff']:+.3f} | {x['cohen_dz']:+.2f} | {x['share_mpup_wins']:.0%} | {_p(x['p_permutation'])} | "
                 f"{_p(x['p_permutation_holm'])} | {_p(x['p_wilcoxon'])} | {_p(x['p_ttest'])} | {_p(x['p_shapiro'])} | {x['mde_80']:.3f} |")
    x = r["pooled"]
    L.append(f"| **pooled** | {x['n']} | {x['mean_diff']:+.3f} | {x['cohen_dz']:+.2f} | {x['share_mpup_wins']:.0%} | {_p(x['p_permutation'])} | "
             f"(single test) | {_p(x['p_wilcoxon'])} | {_p(x['p_ttest'])} | {_p(x['p_shapiro'])} | {x['mde_80']:.3f} |")
    dl = r["dataset_level"]
    L += ["", f"Dataset-level check (units = the {dl['n_datasets']} per-dataset means; mean {dl['mean_of_means']:+.3f}; MPUP ahead on "
          f"{dl['n_positive']} of {dl['n_datasets']}): sign test p = {_p(dl['p_sign_test'])}, Wilcoxon p = {_p(dl['p_wilcoxon'])}, "
          f"t-test p = {_p(dl['p_ttest'])}.", "",
          f"Queries needed for 80% power at the observed pooled difference: {x['n_needed_for_observed_mean']}."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/multidataset")
    ap.add_argument("--results", default="results/multidataset.json", help="run whose nested selection is reproduced")
    ap.add_argument("--out", default="results/significance")
    ap.add_argument("--n-perm", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-ci", action="store_true", help="recover p-values from the stored bootstrap intervals (no features needed)")
    ap.add_argument("--fresh", default=None, help="fresh_questions.json, with --from-ci")
    args = ap.parse_args()
    if args.from_ci:
        r = from_results(Path(args.results), Path(args.fresh) if args.fresh else None)
        Path(args.out).with_suffix(".json").write_text(json.dumps(r, indent=1))
        print(json.dumps({k: v for k, v in r.items() if k != "per_dataset"}, indent=1))
        return
    r = run(Path(args.data), Path(args.results), args.n_perm, args.seed)
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(r, indent=1))
    out.with_suffix(".md").write_text(to_markdown(r))
    print(to_markdown(r))


if __name__ == "__main__":
    main()
