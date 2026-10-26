"""SOTA-style baselines under the protocol of the larger study, and the critical-difference comparison with MPUP.

  # on the GPU machine that holds the larger study (needs features/, extra/ and items.json of --data)
  python -m experiments.run_sota_baselines scores --data data/processed/big --out results/sota_scores \
      --llm Qwen/Qwen2.5-7B-Instruct --selector BAAI/bge-reranker-base
  # on any machine, from the saved scores
  python -m experiments.run_sota_baselines cd --data data/processed/big --scores results/sota_scores.npz \
      --variant "gbr(500,d4)|u|ce|probe" --out results/sota_cd

Methods besides MPUP and the baselines of the paper (dense, BM25, cross-encoder):
  gain   a GainRAG-style selector: a cross-encoder fine-tuned on the Eq. (1) utilities of the development questions by
         listwise knowledge distillation (`src.predictor.gain_selector`); same supervision as MPUP
  judge  zero-shot utility judgment by the same LLM (`src.probing.utility_judge`): "is this passage useful? Yes/No"
Every method is scored on the same test queries with the same within-query Spearman r_s, so the per-query values
can be ranked (Friedman test, Nemenyi critical difference: `experiments.critical_difference`). Numbers published by
other papers cannot enter, since their queries, LLMs and protocols differ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import critical_difference as cd
from experiments import run_big
from src.evaluation.protocol import LABEL_UTILITY, per_query_scores


def row_texts(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Query text and passage text of every row of `run_big.load`, in the same order."""
    items = json.loads((data_dir / "items.json").read_text())
    by_query = {(it["dataset"], str(it["query_id"])): it for it in items}
    qtext, ptext = [], []
    for path in sorted((data_dir / "features").glob("*.npz")):
        z = np.load(path)
        if str(z["label_source"]) != LABEL_UTILITY or not (data_dir / "extra" / path.name).exists():
            continue
        for ds, qid, pid in zip(z["dataset"], z["query_ids"], z["passage_ids"]):
            item = by_query[(str(ds), str(qid))]
            text = {p["id"]: p["text"] for p in item["passages"]}
            qtext.append(item["query"])
            ptext.append(text[str(pid)])
    return np.array(qtext, dtype=object), np.array(ptext, dtype=object)


def group_rows(q: np.ndarray, idx: np.ndarray) -> list[np.ndarray]:
    """Positions (inside `idx`) of the rows of each query, queries in order of first appearance."""
    sub = np.asarray(q)[idx]
    return [np.flatnonzero(sub == u) for u in dict.fromkeys(sub.tolist())]


def score_rows(scorer, qtext: np.ndarray, ptext: np.ndarray, q: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """`scorer(query, passages) -> scores`, called once per query; the result is aligned with `idx`."""
    out = np.zeros(len(idx))
    for pos in group_rows(q, idx):
        rows = idx[pos]
        out[pos] = scorer(qtext[rows[0]], [ptext[r] for r in rows])
    return out


def per_query_table(y: np.ndarray, q: np.ndarray, preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Within-query r_s of every method on the usable queries (the same queries, in the same order, for all)."""
    table = {name: per_query_scores(y, p, q)["spearman"] for name, p in preds.items()}
    assert len({len(v) for v in table.values()}) == 1
    return table


def cmd_scores(args) -> None:
    from src.data.llm_backend import QuantizedLLM
    from src.predictor.gain_selector import GainSelector
    from src.probing.utility_judge import UtilityJudge

    data = Path(args.data)
    d = run_big.load(data)
    qtext, ptext = row_texts(data)
    test = run_big.split(d["q"], run_big.seen_ids([Path(p) for p in args.seen]), args.test_share)
    dev, te = np.flatnonzero(~test), np.flatnonzero(test)
    out = {"test_rows": te}

    selector = GainSelector(args.selector)
    groups = group_rows(d["q"], dev)
    selector.fit([qtext[dev[g[0]]] for g in groups], [[ptext[dev[i]] for i in g] for g in groups],
                 [d["y"][dev[g]] for g in groups], epochs=args.epochs)
    out["gain"] = score_rows(lambda query, ps: selector.score(query, ps), qtext, ptext, d["q"], te)
    del selector

    judge = UtilityJudge(QuantizedLLM(args.llm, quantization="none"), batch_size=args.batch_size)
    out["judge"] = score_rows(lambda query, ps: judge.score(query, ps), qtext, ptext, d["q"], te)
    np.savez(args.out, **out)
    print(f"saved {args.out}.npz: gain and judge scores of {len(te)} test rows")


def cmd_cd(args) -> None:
    d = run_big.load(Path(args.data))
    test = run_big.split(d["q"], run_big.seen_ids([Path(p) for p in args.seen]), args.test_share)
    dev, te = np.flatnonzero(~test), np.flatnonzero(test)
    z = np.load(args.scores)
    assert (z["test_rows"] == te).all(), "the saved scores belong to another split"
    preds = {"MPUP": run_big.fit_predict(d, dev, te, args.variant),
             "GainRAG-style": z["gain"], "LLM judgment": z["judge"]}
    preds.update({"Cross-encoder": d["ce"][te], "Dense": d["X"][te, 1], "BM25": d["X"][te, 0]})
    write_cd_report(per_query_table(d["y"][te], d["q"][te], preds), args.alpha, args.out)


def write_cd_report(table: dict[str, np.ndarray], alpha: float, out: str) -> dict:
    """Friedman test, Nemenyi CD and the diagram for per-query r_s of several methods; writes .json, .tex and .npz."""
    n = len(next(iter(table.values())))
    ranks, crit = cd.average_ranks(table), cd.nemenyi_cd(len(table), n, alpha)
    report = {"n_queries": n, "alpha": alpha, "friedman_p": cd.friedman_pvalue(table), "cd": crit,
              "average_rank": ranks, "mean_r_s": {k: float(v.mean()) for k, v in table.items()},
              "not_significantly_different": cd.cliques(ranks, crit)}
    Path(out).with_suffix(".json").write_text(json.dumps(report, indent=1))
    Path(out).with_suffix(".tex").write_text(cd.to_tikz(ranks, crit))
    np.savez(Path(out).with_suffix(".npz"), **table)
    print(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scores", "cd"])
    ap.add_argument("--data", default="data/processed/big")
    ap.add_argument("--seen", nargs="*", default=["data/processed/multidataset", "data/processed/multidataset_seed1"])
    ap.add_argument("--test-share", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", default=None, help="cd: the .npz written by `scores`")
    ap.add_argument("--variant", default="gbr(500,d4)|u|ce|probe")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--selector", default="BAAI/bge-reranker-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    (cmd_scores if args.mode == "scores" else cmd_cd)(args)


if __name__ == "__main__":
    main()
