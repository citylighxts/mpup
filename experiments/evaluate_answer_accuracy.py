"""End-to-end check: does feeding the passages MPUP selects raise the accuracy of the answer?

  python -m experiments.evaluate_answer_accuracy --data data/processed/multidataset \
      --results results/multidataset_rtx6000_qwen7b.json --backend cuda --out results/answer_accuracy

The ranking metric (within-query rho) says how well passages are ordered; this script measures what a user
cares about. For every query of the eleven datasets with gold answers, each method ranks the candidates and the
top-m go into the prompt of the answering LLM `M`; the answer is scored with the paper's Acc (average of relaxed
exact match and token F1, Eq. 3).

Methods (each picks its top-m from the same candidates):
  mpup     scores of the configuration the nested selection chose for the held-out dataset (the outer
           predictions of `nested_lodo` are reproduced: nothing is fitted on the dataset being answered)
  dense    E5 cosine similarity        bm25    BM25        random  a seeded shuffle
  oracle   the true utility of Eq. (1) (ceiling: what a perfect ranker would reach with this candidate set)
  none     no passage at all (floor; the LLM answers from memory)

Differences to dense similarity are paired per query, with a bootstrap interval and the permutation /
Wilcoxon / t-test p-values of `significance_tests`, so the claim "MPUP raises accuracy" is tested the same
way as the ranking claim. The answering LLM is the one that produced the labels and the D5 features; swap it
(`--model`) to see whether a predictor trained for one LLM helps another one (criterion C5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.nested_selection import GRID
from experiments.significance_tests import describe, holm
from src.data.utility_labeler import NO_CONTEXT_PROMPT
from src.evaluation.metrics import relaxed_combined_score
from src.evaluation.protocol import LABEL_UTILITY, bootstrap_ci, leave_one_dataset_out

PROMPT_HEAD = "Answer the question as briefly as possible, with just the answer span and no explanation.\n\n"


def build_prompt(query: str, passages: list[str]) -> str:
    """The labelling prompt of Eq. (1), with one `Passage n:` block per selected passage (none -> no-context prompt)."""
    if not passages:
        return NO_CONTEXT_PROMPT.format(query=query)
    if len(passages) == 1:  # identical to the prompt that defined the true utility of a single passage
        return f"{PROMPT_HEAD}Passage: {passages[0]}\n\nQuestion: {query}\nAnswer:"
    blocks = "\n\n".join(f"Passage {i + 1}: {p}" for i, p in enumerate(passages))
    return f"{PROMPT_HEAD}{blocks}\n\nQuestion: {query}\nAnswer:"


def top_m_indices(scores, m: int) -> list[int]:
    """Indices of the m highest scores; ties keep the original candidate order."""
    return np.argsort(-np.asarray(scores, float), kind="stable")[:m].tolist()


def load_utility_features(data_dir: Path) -> dict:
    """Features of the eleven gold-answer datasets, with the passage ids that `load_tables` drops."""
    parts = []
    for path in sorted((data_dir / "features").glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        if str(z["label_source"]) == LABEL_UTILITY:
            parts.append({k: z[k] for k in ("X", "y", "query_ids", "passage_ids", "dataset")})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def outer_predictions(feats: dict, selected: dict[str, str]) -> np.ndarray:
    """MPUP scores of every row from the configuration chosen for its held-out dataset (no inner selection repeated)."""
    X, y, q, ds = feats["X"], feats["y"], feats["query_ids"], feats["dataset"]
    configs = {c.name: c for c in GRID}
    pred = np.zeros(len(y))
    for name, train, test in leave_one_dataset_out(ds):
        pred[test] = configs[selected[name]].fit_predict()(X[train], y[train], X[test], q[train], q[test])
    return pred


def plan_prompts(feats: dict, mpup: np.ndarray, items: dict[str, dict], ms: list[int], seed: int, limit: int | None):
    """For every query, the prompt each (method, m) would send. Returns (rows, unique prompts)."""
    rng = np.random.default_rng(seed)
    text = {(qid, p["id"]): p["text"] for qid, it in items.items() for p in it["passages"]}
    seen: dict[str, int] = {}
    rows, prompts = [], {}
    for qid in dict.fromkeys(feats["query_ids"].tolist()):  # keep file order
        idx = np.flatnonzero(feats["query_ids"] == qid)
        dataset = str(feats["dataset"][idx[0]])
        if limit is not None and seen.get(dataset, 0) >= limit:
            continue
        seen[dataset] = seen.get(dataset, 0) + 1
        pids = feats["passage_ids"][idx].tolist()
        scores = {"mpup": mpup[idx], "dense": feats["X"][idx, 1], "bm25": feats["X"][idx, 0],
                  "random": rng.random(len(idx)), "oracle": feats["y"][idx]}
        y = feats["y"][idx]
        rankable = len(idx) >= 2 and float(np.std(y)) > 1e-12
        it = items[qid]
        choices = {("none", 0): []}
        for method, s in scores.items():
            for m in ms:
                choices[(method, m)] = [pids[i] for i in top_m_indices(s, m)]
        for (method, m), chosen in choices.items():
            prompt = build_prompt(it["query"], [text[(qid, pid)] for pid in chosen])
            prompts[prompt] = None
            rows.append({"query_id": qid, "dataset": dataset, "method": method, "m": m, "prompt": prompt,
                         "gold": it["gold_answers"], "rankable": rankable})
    return rows, list(prompts)


def generate_with_cache(llm, prompts: list[str], cache_path: Path | None) -> dict[str, str]:
    key = lambda p: hashlib.sha1(p.encode()).hexdigest()  # noqa: E731
    cache = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
    todo = [p for p in prompts if key(p) not in cache]
    if todo:
        for k, a in zip([key(p) for p in todo], llm.generate(todo)):
            cache[k] = a
        if cache_path:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False))
    return {p: cache[key(p)] for p in prompts}


def score_rows(rows: list[dict], answers: dict[str, str]) -> None:
    for r in rows:
        r["answer"] = answers[r["prompt"]]
        r["acc"] = relaxed_combined_score(r["answer"], r["gold"])


def summarise(rows: list[dict], ms: list[int], n_perm: int, seed: int, only_rankable: bool = False) -> dict:
    """Accuracy per method and m, and the paired difference of MPUP (and the others) to dense, per query."""
    rows = [r for r in rows if r["rankable"]] if only_rankable else rows
    by = {}
    for r in rows:
        by.setdefault((r["method"], r["m"]), {})[r["query_id"]] = r
    datasets = sorted({r["dataset"] for r in rows})
    out = {"n_queries": len({r["query_id"] for r in rows}), "accuracy": {}, "vs_dense": {}}
    for (method, m), per_q in by.items():
        macro = np.mean([np.mean([r["acc"] for r in per_q.values() if r["dataset"] == d]) for d in datasets])
        out["accuracy"][f"{method}@{m}"] = {"macro": float(macro),
                                            "per_dataset": {d: float(np.mean([r["acc"] for r in per_q.values() if r["dataset"] == d])) for d in datasets}}
    for m in ms:
        dense = by[("dense", m)]
        for method in ("mpup", "bm25", "random", "oracle"):
            qs = [q for q in dense]
            d = np.array([by[(method, m)][q]["acc"] - dense[q]["acc"] for q in qs])
            mean, lo, hi = bootstrap_ci(d, seed=seed)
            entry = {**describe(d, n_perm, seed), "ci": [lo, hi]}
            if method == "mpup":
                per_ds = {}
                for ds in datasets:
                    dd = np.array([by[("mpup", m)][q]["acc"] - dense[q]["acc"] for q in qs if dense[q]["dataset"] == ds])
                    per_ds[ds] = describe(dd, n_perm, seed) if len(dd) > 1 else None
                adj = holm([v["p_permutation"] for v in per_ds.values()])
                for (ds, v), a in zip(per_ds.items(), adj):
                    v["p_permutation_holm"] = a
                entry["per_dataset"] = per_ds
            out["vs_dense"][f"{method}@{m}"] = entry
    return out


def _p(v: float) -> str:
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def to_markdown(summary: dict, ms: list[int], title: str) -> str:
    L = [f"# {title}", "", f"{summary['n_queries']} queries. Acc = 0.5 * relaxed EM + 0.5 * token F1 (Eq. 3); macro mean over datasets.", "",
         "| method | " + " | ".join(f"top-{m}" for m in ms) + " |", "|---|" + "---|" * len(ms)]
    none = summary["accuracy"].get("none@0")
    if none:
        L.append(f"| none (no passage) | {none['macro']:.3f} |" + " |" * (len(ms) - 1))
    for method in ("random", "bm25", "dense", "mpup", "oracle"):
        L.append(f"| {method} | " + " | ".join(f"{summary['accuracy'][f'{method}@{m}']['macro']:.3f}" for m in ms) + " |")
    L += ["", "Paired difference in Acc to dense similarity over all queries (95% bootstrap interval; permutation p; H0: mean difference = 0):", "",
          "| method | m | mean diff | 95% CI | d_z | p perm | p Wilcoxon | p t-test |", "|---|---|---|---|---|---|---|---|"]
    for m in ms:
        for method in ("mpup", "bm25", "random", "oracle"):
            e = summary["vs_dense"][f"{method}@{m}"]
            L.append(f"| {method} | {m} | {e['mean_diff']:+.3f} | [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] | {e['cohen_dz']:+.2f} | "
                     f"{_p(e['p_permutation'])} | {_p(e['p_wilcoxon']) if e['p_wilcoxon'] == e['p_wilcoxon'] else 'n/a'} | {_p(e['p_ttest'])} |")
    return "\n".join(L) + "\n"


def make_llm(args):
    if args.backend == "mlx":
        from src.data.mlx_backend import MLXLLM
        return MLXLLM(args.model or "mlx-community/Llama-3.2-3B-Instruct-4bit", batch_size=args.batch_size or 8)
    from src.data.llm_backend import QuantizedLLM
    bs = args.batch_size or 32
    return QuantizedLLM(args.model or "Qwen/Qwen2.5-7B-Instruct", quantization=args.quantization, batch_size=bs, logprob_batch_size=bs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/multidataset")
    ap.add_argument("--results", default="results/multidataset.json", help="run whose nested selection is reproduced")
    ap.add_argument("--backend", choices=["mlx", "cuda"], default="mlx")
    ap.add_argument("--quantization", choices=["none", "nf4"], default="none")
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--m", type=int, nargs="+", default=[1, 3], help="how many top passages go into the prompt")
    ap.add_argument("--limit", type=int, default=None, help="only the first N queries per dataset (smoke test)")
    ap.add_argument("--n-perm", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/answer_accuracy")
    args = ap.parse_args()

    data = Path(args.data)
    items = {it["query_id"]: it for it in json.loads((data / "items.json").read_text()) if it["gold_answers"]}
    selected = {n: d["selected"] for n, d in json.loads(Path(args.results).read_text())["nested_lodo"]["per_dataset"].items()}
    feats = load_utility_features(data)
    mpup = outer_predictions(feats, selected)
    rows, prompts = plan_prompts(feats, mpup, items, args.m, args.seed, args.limit)
    print(f"{len({r['query_id'] for r in rows})} queries, {len(prompts)} distinct prompts", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    answers = generate_with_cache(make_llm(args), prompts, out.with_name(out.name + "_answers.json"))
    score_rows(rows, answers)
    result = {"all": summarise(rows, args.m, args.n_perm, args.seed), "rankable": summarise(rows, args.m, args.n_perm, args.seed, only_rankable=True),
              "model": args.model, "backend": args.backend, "m": args.m}
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    md = to_markdown(result["all"], args.m, "Answer accuracy with the passages each method selects") \
        + "\n" + to_markdown(result["rankable"], args.m, "Same, only on queries whose candidates differ in utility (rankable)")
    out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
