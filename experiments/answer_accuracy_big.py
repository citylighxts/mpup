"""End-to-end answer accuracy on the held-out test questions of the larger experiment.

  python -m experiments.answer_accuracy_big --data data/processed/big --variant <name> --out results/big_answer_accuracy

Every method ranks the candidates of a test question and the top-m go into the answering prompt of
`evaluate_answer_accuracy` (the Eq. (1) prompt with one block per passage); the answer is scored with Acc
(Eq. 3). The two MPUP predictors are fitted on the development questions only, as in `run_big final`.
Differences are paired per query and tested like the ranking results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import run_big
from experiments.evaluate_answer_accuracy import build_prompt, generate_with_cache, top_m_indices
from experiments.significance_tests import describe, holm
from src.evaluation.metrics import relaxed_combined_score
from src.evaluation.protocol import bootstrap_ci

METHODS = ("mpup+", "mpup (paper)", "cross-encoder", "dense", "bm25", "random", "oracle")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/big")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seen", nargs="*", default=["data/processed/multidataset", "data/processed/multidataset_seed1"])
    ap.add_argument("--m", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-perm", type=int, default=100_000)
    ap.add_argument("--out", default="results/big_answer_accuracy")
    args = ap.parse_args()

    data = Path(args.data)
    d = run_big.load(data)
    test_mask = run_big.split(d["q"], run_big.seen_ids([Path(p) for p in args.seen]))
    dev, te = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
    scores = {"mpup+": run_big.fit_predict(d, dev, te, args.variant),
              "mpup (paper)": run_big.fit_predict(d, dev, te, run_big.OLD_VARIANT),
              "cross-encoder": d["ce"][te], "dense": d["X"][te, 1], "bm25": d["X"][te, 0],
              "random": np.random.default_rng(0).random(len(te)), "oracle": d["y"][te]}

    pids = np.concatenate([np.load(p)["passage_ids"] for p in sorted((data / "features").glob("*.npz"))
                           if (data / "extra" / p.name).exists() and str(np.load(p)["label_source"]) == run_big.LABEL_UTILITY])[te]
    items = {it["query_id"]: it for it in json.loads((data / "items.json").read_text())}
    text = {p["id"]: p["text"] for it in items.values() for p in it["passages"]}
    q_te, ds_te = d["q"][te], d["ds"][te]

    rows, prompts = [], {}
    counts = {n: len(set(q_te[ds_te == n].tolist())) for n in set(ds_te.tolist())}
    for qid in dict.fromkeys(q_te.tolist()):
        idx = np.flatnonzero(q_te == qid)
        if counts[str(ds_te[idx[0]])] < 10:  # Bamboogle: one test question left after the earlier samples
            continue
        it = items[qid.split(":", 1)[1]]
        choices = {("none", 0): []}
        for method in METHODS:
            for m in args.m:
                choices[(method, m)] = [pids[idx][i] for i in top_m_indices(scores[method][idx], m)]
        for (method, m), chosen in choices.items():
            prompt = build_prompt(it["query"], [text[p] for p in chosen])
            prompts[prompt] = None
            rows.append({"query_id": qid, "dataset": str(ds_te[idx[0]]), "method": method, "m": m,
                         "prompt": prompt, "gold": it["gold_answers"]})
    print(f"{len({r['query_id'] for r in rows})} test questions, {len(prompts)} distinct prompts", flush=True)

    from src.data.llm_backend import QuantizedLLM

    llm = QuantizedLLM(args.model, quantization="none", batch_size=args.batch_size)
    out = Path(args.out)
    answers = generate_with_cache(llm, list(prompts), out.with_name(out.name + "_answers.json"))
    by: dict[tuple, dict] = {}
    for r in rows:
        r["acc"] = relaxed_combined_score(answers[r["prompt"]], r["gold"])
        by.setdefault((r["method"], r["m"]), {})[r["query_id"]] = r

    datasets = sorted({r["dataset"] for r in rows})
    acc = {f"{k[0]}@{k[1]}": {d_: float(np.mean([r["acc"] for r in v.values() if r["dataset"] == d_])) for d_ in datasets}
           for k, v in by.items()}
    macro = {k: float(np.mean(list(v.values()))) for k, v in acc.items()}
    tests = {}
    for m in args.m:
        ref = by[("mpup+", m)]
        for method in ("mpup (paper)", "cross-encoder", "dense", "bm25", "random", "none"):
            other = by[(method, 0 if method == "none" else m)]
            diff = np.array([ref[q]["acc"] - other[q]["acc"] for q in ref])
            mean, lo, hi = bootstrap_ci(diff)
            tests[f"mpup+ vs {method} @{m}"] = {**describe(diff, args.n_perm, 0), "ci": [lo, hi]}
        per = {ds: describe(np.array([ref[q]["acc"] - by[("dense", m)][q]["acc"] for q in ref if ref[q]["dataset"] == ds]), 20_000, 0)
               for ds in datasets}
        for (ds, v), a in zip(per.items(), holm([v["p_permutation"] for v in per.values()])):
            v["p_holm"] = a
        tests[f"per-dataset vs dense @{m}"] = per
    result = {"variant": args.variant, "n_queries": len(by[("none", 0)]), "macro": macro, "per_dataset": acc, "tests": tests}
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))

    L = ["# Answer accuracy on the held-out test questions", "",
         f"{result['n_queries']} test questions; Acc = 0.5 relaxed EM + 0.5 token F1, macro mean over datasets; "
         f"answering model {args.model}. mpup+ = `{args.variant}`.", "",
         "| method | " + " | ".join(f"top-{m}" for m in args.m) + " |", "|---|" + "---|" * len(args.m),
         f"| none (no passage) | {macro['none@0']:.3f} |" + " |" * (len(args.m) - 1)]
    for method in METHODS:
        L.append(f"| {method} | " + " | ".join(f"{macro[f'{method}@{m}']:.3f}" for m in args.m) + " |")
    L += ["", "Paired difference of mpup+ (95% bootstrap CI, sign-flip permutation p):", ""]
    for k, v in tests.items():
        if not k.startswith("per-dataset"):
            L.append(f"- {k}: {v['mean_diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], p = {v['p_permutation']:.2g}")
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
