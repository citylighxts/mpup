"""Larger experiment: more questions, more candidates, and three additions to MPUP, tested on held-out questions.

  python -m experiments.run_big dev   --data data/processed/big --out results/big_dev
  python -m experiments.run_big final --data data/processed/big --variant <name> --out results/big_final

Inputs: `features/` (14 features, Eq. (1) labels) and `extra/` (`extract_extra`: soft target, hidden states,
cross-encoder score). Additions, each switched on by a variant:

  ce      the cross-encoder relevance score as one more D1 feature
  probe   a hidden-state probe: ridge regression from M's hidden state at the last prompt token (centred
          within the query) to the training target; its score is one more D5 feature. Inside a training
          set the probe's scores are out-of-fold (5 folds over queries), so the boosting model never sees
          a probe score fitted on the same row
  soft    training target: the change in log-probability of the gold answer (`extract_extra`), alone
          ("soft") or averaged with Eq. (1) after z-scoring within the query ("mix")

Split. Every question is assigned once, by a hash of its id, to the development part (70%) or the test part
(30%). Questions that were already in the samples of the earlier runs (the dev pilot was run on them) always go
to development. `dev` compares the variants by leave-one-dataset-out on the development part only. `final`
fits the chosen variant on the development part and scores the test part once:

  same datasets      trained on the development questions of all datasets, tested on the test questions
  held-out dataset   trained on the development questions of the other datasets only, tested on the test
                     questions of the held-out one (neither the questions nor the dataset were trained on)

Every score is the within-query Spearman correlation with the Eq. (1) utility, as in the paper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

from experiments.cv_report import make_fit_predict
from experiments.significance_tests import describe, holm
from src.evaluation.protocol import LABEL_UTILITY, bootstrap_ci, group_splits, leave_one_dataset_out, per_query_scores
from src.predictor.mpup_predictor import center_by_group

LAYER = 21
OLD = ("xgboost", {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05})  # gbr(100,d2)-centered, the paper's
MODELS = {
    "gbr(100,d2)": OLD,
    "gbr(300,d3)": ("xgboost", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}),
    "gbr(500,d4)": ("xgboost", {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03}),
}


def _variants() -> dict[str, dict]:
    out = {}
    for model in MODELS:
        for target in ("u", "soft", "mix"):  # "soft" lost clearly on the pilot (0.13-0.27); not in the default dev comparison
            for ce in (False, True):
                for probe in (False, True):
                    name = f"{model}|{target}" + ("|ce" if ce else "") + ("|probe" if probe else "")
                    out[name] = {"model": model, "target": target, "ce": ce, "probe": probe}
    return out


VARIANTS = _variants()
OLD_VARIANT = "gbr(100,d2)|u"


# ------------------------------------------------------------------ data

def load(data_dir: Path, layer: int = LAYER) -> dict:
    parts = {k: [] for k in ("X", "y", "q", "ds", "soft", "ce", "H")}
    for path in sorted((data_dir / "features").glob("*.npz")):
        z = np.load(path)
        if str(z["label_source"]) != LABEL_UTILITY or not (data_dir / "extra" / path.name).exists():
            continue
        e = np.load(data_dir / "extra" / path.name)
        assert (e["passage_ids"] == z["passage_ids"]).all(), path.name
        parts["X"].append(z["X"])
        parts["y"].append(z["y"])
        parts["q"].append(np.array([f"{path.stem}:{q}" for q in z["query_ids"]]))
        parts["ds"].append(z["dataset"])
        parts["soft"].append(e["soft"])
        parts["ce"].append(e["ce"])
        parts["H"].append(e[f"hidden_{layer}"])
    d = {k: np.concatenate(v) for k, v in parts.items()}
    d["H"] = d["H"].astype(np.float32)
    return d


def seen_ids(paths: list[Path]) -> set[str]:
    out = set()
    for p in paths:
        if (p / "items.json").exists():
            out |= {f"{it['dataset']}:{it['query_id']}" for it in json.loads((p / "items.json").read_text())}
    return out


def split(q: np.ndarray, seen: set[str], test_share: float = 0.3) -> np.ndarray:
    """True for test rows: a hash of the query id, never a question seen in earlier runs."""
    test = {}
    for qid in np.unique(q):
        h = int(hashlib.md5(qid.encode()).hexdigest()[:8], 16) / 16**8
        test[qid] = h < test_share and qid not in seen
    return np.array([test[x] for x in q])


# ------------------------------------------------------------------ predictor

def target(d: dict, idx: np.ndarray, kind: str) -> np.ndarray:
    u, s, q = d["y"][idx], d["soft"][idx], d["q"][idx]
    if kind == "u":
        return u
    if kind == "soft":
        return s
    z = lambda v: center_by_group(v, q) / (center_by_group(v, q).std() + 1e-12)  # noqa: E731
    return 0.5 * (z(u) + z(s))


def _probe_fit_predict(H_tr, t_tr, q_tr, H_te, q_te, alpha: float | None = None):
    """Ridge from centred hidden states; alpha = 10 per training row (flat optimum 1e5-1e7 on the ~8.7k-row pilot)."""
    alpha = 10.0 * len(H_tr) if alpha is None else alpha
    Hc_tr, Hc_te = center_by_group(H_tr, q_tr), center_by_group(H_te, q_te)
    mu, sd = Hc_tr.mean(0), Hc_tr.std(0) + 1e-6
    r = Ridge(alpha=alpha).fit((Hc_tr - mu) / sd, center_by_group(t_tr, q_tr))
    return r.predict((Hc_te - mu) / sd)


def probe_scores(d: dict, tr: np.ndarray, te: np.ndarray, t_tr: np.ndarray, n_folds: int = 5):
    """Out-of-fold probe scores for the training rows, and scores from a probe on all of them for te."""
    oof = np.zeros(len(tr))
    for a, b in group_splits(d["q"][tr], n_splits=n_folds, seed=0):
        oof[b] = _probe_fit_predict(d["H"][tr][a], t_tr[a], d["q"][tr][a], d["H"][tr][b], d["q"][tr][b])
    test = _probe_fit_predict(d["H"][tr], t_tr, d["q"][tr], d["H"][te], d["q"][te])
    return oof, test


def fit_predict(d: dict, tr: np.ndarray, te: np.ndarray, variant: str, probe=None) -> np.ndarray:
    """`probe`: precomputed (oof, test) probe scores for this split and target, to share them across variants."""
    v = VARIANTS[variant]
    t_tr = target(d, tr, v["target"])
    X_tr, X_te = d["X"][tr], d["X"][te]
    if v["ce"]:
        X_tr, X_te = np.c_[X_tr, d["ce"][tr]], np.c_[X_te, d["ce"][te]]
    if v["probe"]:
        oof, test = probe if probe is not None else probe_scores(d, tr, te, t_tr)
        X_tr, X_te = np.c_[X_tr, oof], np.c_[X_te, test]
    algo, params = MODELS[v["model"]]
    return make_fit_predict(algo, True, **params)(X_tr, t_tr, X_te, d["q"][tr], d["q"][te])


BASELINES = {"dense": lambda d, i: d["X"][i, 1], "bm25": lambda d, i: d["X"][i, 0], "cross-encoder": lambda d, i: d["ce"][i]}


def rho(d, idx, pred) -> np.ndarray:
    return per_query_scores(d["y"][idx], pred, d["q"][idx])


# ------------------------------------------------------------------ dev: compare variants on the development part

def _fold_target(d, tr, te, kind, variants):
    """All variants with one training target on one leave-one-dataset-out fold (the probe is fitted once)."""
    names = [v for v in variants if VARIANTS[v]["target"] == kind]
    probe = probe_scores(d, tr, te, target(d, tr, kind)) if any(VARIANTS[v]["probe"] for v in names) else None
    return {v: float(rho(d, te, fit_predict(d, tr, te, v, probe))["spearman"].mean()) for v in names}


def run_dev(d: dict, test_mask: np.ndarray, variants: list[str]) -> dict:
    dev = np.flatnonzero(~test_mask)
    folds = [(name, dev[tr], dev[te]) for name, tr, te in leave_one_dataset_out(d["ds"][dev])]
    kinds = sorted({VARIANTS[v]["target"] for v in variants})
    jobs = [(name, kind) for name, _, _ in folds for kind in kinds]
    out = Parallel(n_jobs=min(len(jobs), 40))(
        delayed(_fold_target)(d, tr, te, kind, variants) for name, tr, te in folds for kind in kinds)
    res = {v: {"per_dataset": {}} for v in variants}
    for (name, _), scores in zip(jobs, out):
        for v, x in scores.items():
            res[v]["per_dataset"][name] = x
    for v in res:
        res[v]["macro"] = float(np.mean(list(res[v]["per_dataset"].values())))
    base = {}
    for b, f in BASELINES.items():
        per = {name: float(rho(d, te, f(d, te))["spearman"].mean()) for name, _, te in folds}
        base[b] = {"macro": float(np.mean(list(per.values()))), "per_dataset": per}
    best = max(res, key=lambda k: res[k]["macro"])
    return {"n_dev_queries": int(len(np.unique(d["q"][dev]))), "variants": res, "baselines": base, "best": best}


# ------------------------------------------------------------------ final: one look at the test part

def _compare(d, te, preds: dict[str, np.ndarray], ref: str) -> dict:
    pq = {n: rho(d, te, p) for n, p in preds.items()}
    per_ds = {}
    names_ds = [n for n in sorted(set(d["ds"][te].tolist()))  # Bamboogle has 125 questions, nearly all used earlier
                if np.isin(pq[ref]["query_ids"], np.unique(d["q"][te][d["ds"][te] == n])).sum() >= 2]
    for name in names_ds:
        m = np.isin(pq[ref]["query_ids"], np.unique(d["q"][te][d["ds"][te] == name]))
        per_ds[name] = {"n_queries": int(m.sum()), "spearman": {n: float(v["spearman"][m].mean()) for n, v in pq.items()}}
    macro = {n: {"spearman": float(np.mean([per_ds[k]["spearman"][n] for k in names_ds])),
                 "ndcg": float(v["ndcg"].mean()), "top1": float(v["top1"].mean())} for n, v in pq.items()}
    tests = {n: describe(pq[ref]["spearman"] - pq[n]["spearman"], n_perm=200_000, seed=0) for n in pq if n != ref}
    tests_ci = {n: bootstrap_ci(pq[ref]["spearman"] - pq[n]["spearman"]) for n in pq if n != ref}
    per_ds_tests = {}
    for b in ("dense", "cross-encoder"):
        if b not in pq:
            continue
        ps = []
        for name in names_ds:
            m = np.isin(pq[ref]["query_ids"], np.unique(d["q"][te][d["ds"][te] == name]))
            r = describe(pq[ref]["spearman"][m] - pq[b]["spearman"][m], n_perm=50_000, seed=0)
            per_ds_tests.setdefault(b, {})[name] = r
            ps.append(r["p_permutation"])
        for name, ph in zip(names_ds, holm(ps)):
            per_ds_tests[b][name]["p_holm"] = ph
    return {"n_queries": int(len(pq[ref]["spearman"])), "datasets": names_ds, "macro": macro, "per_dataset": per_ds,
            "paired_ci": tests_ci, "tests": tests, "per_dataset_tests": per_ds_tests}


def run_final(d: dict, test_mask: np.ndarray, variant: str) -> dict:
    dev, te = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
    preds = {"mpup+": fit_predict(d, dev, te, variant), "mpup (paper)": fit_predict(d, dev, te, OLD_VARIANT)}
    preds.update({b: f(d, te) for b, f in BASELINES.items()})
    same = _compare(d, te, preds, "mpup+")

    held = {n: np.zeros(len(te)) for n in ("mpup+", "mpup (paper)")}
    for name in sorted(set(d["ds"][te].tolist())):
        tr = dev[d["ds"][dev] != name]
        sub = te[d["ds"][te] == name]
        pos = np.flatnonzero(d["ds"][te] == name)
        held["mpup+"][pos] = fit_predict(d, tr, sub, variant)
        held["mpup (paper)"][pos] = fit_predict(d, tr, sub, OLD_VARIANT)
    held.update({b: f(d, te) for b, f in BASELINES.items()})
    cross = _compare(d, te, held, "mpup+")
    return {"variant": variant, "variant_spec": VARIANTS[variant], "same_datasets": same, "held_out_dataset": cross}


# ------------------------------------------------------------------ report

def md_dev(r: dict) -> str:
    rows = sorted(r["variants"].items(), key=lambda kv: -kv[1]["macro"])
    lines = ["# Variant comparison on the development questions", "",
             f"Leave-one-dataset-out on {r['n_dev_queries']} development questions; macro within-query Spearman with Eq. (1).",
             "Descriptive: the best variant is chosen here, so its score is optimistic. The test questions are not used.", "",
             "| variant | ρ |", "|---|---|"]
    lines += [f"| {k} | {v['macro']:.3f} |" for k, v in rows]
    lines += [f"| baseline: {k} | {v['macro']:.3f} |" for k, v in r["baselines"].items()]
    return "\n".join(lines + ["", f"Chosen: `{r['best']}`", ""])


def md_final(r: dict) -> str:
    lines = ["# Held-out test questions", "", f"Variant `{r['variant']}` (chosen on the development questions only).", ""]
    for key, title in (("same_datasets", "Trained on the development questions of all datasets"),
                       ("held_out_dataset", "Trained on the development questions of the other datasets only")):
        s = r[key]
        lines += [f"## {title}", "", f"{s['n_queries']} rankable test questions. Macro within-query Spearman; NDCG@10 and top-1 over queries.", "",
                  "| model | ρ | NDCG@10 | top-1 |", "|---|---|---|---|"]
        lines += [f"| {n} | {m['spearman']:.3f} | {m['ndcg']:.3f} | {m['top1']:.3f} |" for n, m in s["macro"].items()]
        lines += ["", "Paired difference of mpup+ over all test queries (95% bootstrap CI; sign-flip permutation p; d_z):", ""]
        for n, (m, lo, hi) in s["paired_ci"].items():
            t = s["tests"][n]
            lines.append(f"- vs {n}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}], p = {t['p_permutation']:.2g}, d_z = {t['cohen_dz']:+.2f}")
        names = list(next(iter(s["per_dataset"].values()))["spearman"])
        lines += ["", "| dataset | queries | " + " | ".join(names) + " | p vs dense (Holm) |", "|---|---|" + "---|" * (len(names) + 1)]
        for ds, v in s["per_dataset"].items():
            pt = s["per_dataset_tests"].get("dense", {}).get(ds, {})
            lines.append(f"| {ds} | {v['n_queries']} | " + " | ".join(f"{x:.3f}" for x in v["spearman"].values())
                         + f" | {pt.get('p_permutation', float('nan')):.3f} ({pt.get('p_holm', float('nan')):.3f}) |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "final"])
    ap.add_argument("--data", default="data/processed/big")
    ap.add_argument("--seen", nargs="*", default=["data/processed/multidataset", "data/processed/multidataset_seed1"])
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--test-share", type=float, default=0.3)
    ap.add_argument("--all-dev", action="store_true", help="pilot on earlier data: no test part")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = load(Path(args.data))
    test_mask = np.zeros(len(d["y"]), bool) if args.all_dev else split(d["q"], seen_ids([Path(p) for p in args.seen]), args.test_share)
    if args.mode == "dev":
        r = run_dev(d, test_mask, args.variants or [v for v, s in VARIANTS.items() if s["target"] != "soft"])
        text = md_dev(r)
    else:
        r = run_final(d, test_mask, args.variant)
        text = md_final(r)
    r["n_test_queries"] = int(len(np.unique(d["q"][test_mask])))
    Path(args.out).with_suffix(".json").write_text(json.dumps(r, indent=1, default=float))
    Path(args.out).with_suffix(".md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
