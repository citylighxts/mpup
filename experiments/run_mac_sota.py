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


def load_rows(data_dir: Path, with_features: bool = True) -> dict:
    """Extended features (unless `with_features` is False), labels, query ids and texts of every row with the Eq. (1)
    utility, in file order. The score stages need no features, which saves most of the memory."""
    data_dir = Path(data_dir)
    items = {(it["dataset"], str(it["query_id"])): it for it in json.loads((data_dir / "items.json").read_text())}
    parts = {k: [] for k in ("X", "y", "q", "qtext", "ptext")}
    for path in sorted((data_dir / "features").glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        if str(z["label_source"]) != LABEL_UTILITY:
            continue
        if with_features:
            parts["X"].append(extended_matrix(data_dir, path.stem, z))
        parts["y"].append(z["y"])
        for ds, qid, pid in zip(z["dataset"], z["query_ids"], z["passage_ids"]):
            item = items[(str(ds), str(qid))]
            parts["q"].append(f"{ds}:{qid}")
            parts["qtext"].append(item["query"])
            parts["ptext"].append({p["id"]: p["text"] for p in item["passages"]}[str(pid)])
    rows = {"y": np.concatenate(parts["y"]), "q": np.array(parts["q"]),
            "qtext": np.array(parts["qtext"], dtype=object), "ptext": np.array(parts["ptext"], dtype=object)}
    if with_features:
        rows["X"] = np.concatenate(parts["X"])
    return rows


def done_stages(out: str) -> set[str]:
    path = Path(out).with_suffix(".npz")
    return {k for k in np.load(path).files if k != "n_test_rows"} if path.exists() else set()


def save_stage(out: str, name: str, values: np.ndarray, n_rows: int) -> None:
    """Add one stage to the scores file, keeping the stages that are already there."""
    path = Path(out).with_suffix(".npz")
    data = dict(np.load(path)) if path.exists() else {}
    if data and int(data["n_test_rows"]) != n_rows:
        raise ValueError(f"{path} belongs to another test sample ({int(data['n_test_rows'])} rows, not {n_rows})")
    data[name] = values
    data["n_test_rows"] = np.array(n_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)


def _free_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - freeing memory is best effort
        pass


def cmd_scores(args) -> None:
    """Each stage is computed on its own, saved at once, and skipped when already saved (use --force to redo it)."""
    stages = [x for x in ("ce", "gain", "judge") if x in args.stages]
    test = load_rows(Path(args.test), with_features=False)
    te = np.arange(len(test["y"]))
    todo = [x for x in stages if args.force or x not in done_stages(args.out)]
    if not todo:
        print("all requested stages are already saved")
        return
    train = load_rows(Path(args.train), with_features=False) if "gain" in todo else None
    for name in todo:
        if name == "ce":
            from src.probing.cross_encoder import CrossEncoderScorer

            model = CrossEncoderScorer(args.cross_encoder)
            values = score_rows(lambda q, ps: model.score(q, ps), test["qtext"], test["ptext"], test["q"], te)
        elif name == "gain":
            from src.predictor.gain_selector import GainSelector

            model = GainSelector(args.selector, max_length=args.max_length)
            groups = group_rows(train["q"], np.arange(len(train["y"])))
            model.fit([train["qtext"][g[0]] for g in groups], [[train["ptext"][i] for i in g] for g in groups],
                      [train["y"][g] for g in groups], epochs=args.epochs)
            values = score_rows(lambda q, ps: model.score(q, ps), test["qtext"], test["ptext"], test["q"], te)
        else:
            from src.data.mlx_backend import MLXLLM
            from src.probing.utility_judge import MLXUtilityJudge

            model = MLXUtilityJudge(MLXLLM(args.llm), batch_size=args.batch_size)
            values = score_rows(lambda q, ps: model.score(q, ps), test["qtext"], test["ptext"], test["q"], te)
        save_stage(args.out, name, values, len(te))
        print(f"{name} done and saved", flush=True)
        del model
        _free_memory()


def cmd_cd(args) -> dict:
    train, test = load_rows(Path(args.train)), load_rows(Path(args.test))
    z = np.load(args.scores)
    missing = [k for k in ("ce", "gain", "judge") if k not in z.files]
    if missing:
        raise SystemExit(f"missing stages {missing}: run `scores --stages {' '.join(missing)}` first")
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
    ap.add_argument("--stages", nargs="+", default=["ce", "gain", "judge"], choices=["ce", "gain", "judge"])
    ap.add_argument("--force", action="store_true", help="recompute stages that are already saved")
    args = ap.parse_args()
    (cmd_scores if args.mode == "scores" else cmd_cd)(args)


if __name__ == "__main__":
    main()
