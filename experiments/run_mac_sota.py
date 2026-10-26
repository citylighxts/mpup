"""The SOTA-style baselines and the critical-difference comparison on the laptop samples.

  python -m experiments.run_mac_sota scores --train data/processed/mac_qwen7b --test data/processed/mac_qwen7b_s3 \
      --out results/mac_qwen7b/sota_scores
  python -m experiments.run_mac_sota cd --train data/processed/mac_qwen7b --test data/processed/mac_qwen7b_s3 \
      --scores results/mac_qwen7b/sota_scores.npz --out results/mac_qwen7b/sota_cd

Protocol: everything is fitted on the first question sample (`--train`) and scored once on the third sample (`--test`),
whose questions were never used for any choice. `scores` computes, for the rows of the test sample, the scores of
  ce     the cross-encoder bge-reranker-v2-m3,
  gain   a GainRAG-style selector (bge-reranker-base fine-tuned on the Eq. (1) utilities of the training sample),
  judge  the zero-shot yes/no utility judgment of the same LLM (MLX, Qwen2.5-7B-Instruct 4-bit),
and `cd` adds MPUP (the frozen configuration fitted on the training sample), dense similarity and BM25, then ranks all
six methods by their within-query r_s on the same test queries (`experiments.critical_difference`). The samples are
pilot-sized (five BM25 candidates, or the provided paragraphs), so this is supporting evidence next to the larger study.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.nested_selection import grid_v2
from experiments.run_multidataset import extended_matrix
from experiments.run_sota_baselines import group_rows, per_query_table, score_rows, write_cd_report
from src.evaluation.protocol import LABEL_UTILITY

COMBINED = ("base-noD2D4", "base+probe")


def load_rows(data_dir: Path) -> dict:
    """Extended features, labels, query ids and texts of every row with the Eq. (1) utility, in file order."""
    data_dir = Path(data_dir)
    items = {(it["dataset"], str(it["query_id"])): it for it in json.loads((data_dir / "items.json").read_text())}
    parts = {k: [] for k in ("X", "y", "q", "qtext", "ptext")}
    for path in sorted((data_dir / "features").glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        if str(z["label_source"]) != LABEL_UTILITY:
            continue
        parts["X"].append(extended_matrix(data_dir, path.stem, z))
        parts["y"].append(z["y"])
        for ds, qid, pid in zip(z["dataset"], z["query_ids"], z["passage_ids"]):
            item = items[(str(ds), str(qid))]
            parts["q"].append(f"{ds}:{qid}")
            parts["qtext"].append(item["query"])
            parts["ptext"].append({p["id"]: p["text"] for p in item["passages"]}[str(pid)])
    return {"X": np.concatenate(parts["X"]), "y": np.concatenate(parts["y"]), "q": np.array(parts["q"]),
            "qtext": np.array(parts["qtext"], dtype=object), "ptext": np.array(parts["ptext"], dtype=object)}


def cmd_scores(args) -> None:
    from src.data.mlx_backend import MLXLLM
    from src.predictor.gain_selector import GainSelector
    from src.probing.cross_encoder import CrossEncoderScorer
    from src.probing.utility_judge import MLXUtilityJudge

    train, test = load_rows(Path(args.train)), load_rows(Path(args.test))
    te, tr = np.arange(len(test["y"])), np.arange(len(train["y"]))
    out = {"n_test_rows": np.array(len(te))}

    ce = CrossEncoderScorer(args.cross_encoder)
    out["ce"] = score_rows(lambda q, ps: ce.score(q, ps), test["qtext"], test["ptext"], test["q"], te)
    del ce
    print("cross-encoder done", flush=True)

    selector = GainSelector(args.selector, max_length=args.max_length)
    groups = group_rows(train["q"], tr)
    selector.fit([train["qtext"][g[0]] for g in groups], [[train["ptext"][i] for i in g] for g in groups],
                 [train["y"][g] for g in groups], epochs=args.epochs)
    out["gain"] = score_rows(lambda q, ps: selector.score(q, ps), test["qtext"], test["ptext"], test["q"], te)
    del selector
    print("GainRAG-style selector done", flush=True)

    judge = MLXUtilityJudge(MLXLLM(args.llm), batch_size=args.batch_size)
    out["judge"] = score_rows(lambda q, ps: judge.score(q, ps), test["qtext"], test["ptext"], test["q"], te)
    np.savez(args.out, **out)
    print(f"saved {args.out}.npz")


def cmd_cd(args) -> dict:
    train, test = load_rows(Path(args.train)), load_rows(Path(args.test))
    z = np.load(args.scores)
    assert len(z["ce"]) == len(test["y"]), "the saved scores belong to another test sample"
    cfg = next(c for c in grid_v2(COMBINED) if c.name == args.config)
    preds = {"MPUP": cfg.fit_predict()(train["X"], train["y"], test["X"], train["q"], test["q"]),
             "GainRAG-style": z["gain"], "LLM judgment": z["judge"], "Cross-encoder": z["ce"],
             "Dense": test["X"][:, 1], "BM25": test["X"][:, 0]}
    return write_cd_report(per_query_table(test["y"], test["q"], preds), args.alpha, args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scores", "cd"])
    ap.add_argument("--train", default="data/processed/mac_qwen7b")
    ap.add_argument("--test", default="data/processed/mac_qwen7b_s3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", default=None)
    ap.add_argument("--config", default="base+probe", help="the frozen configuration of the laptop run")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--llm", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--cross-encoder", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--selector", default="BAAI/bge-reranker-base")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    (cmd_scores if args.mode == "scores" else cmd_cd)(args)


if __name__ == "__main__":
    main()
