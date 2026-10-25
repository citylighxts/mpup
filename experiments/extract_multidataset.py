"""Stage 2 of the multi-dataset experiment: utility labels (Eq. 1) and the 14 features D1-D5.

  python -m experiments.extract_multidataset --data data/processed/multidataset

For every dataset in `items.json` it writes `features/<dataset>.npz` with

  X            (n_pairs, 14)  D1-D5, built by `build_feature_vector`
  y            (n_pairs,)     the label (see label_source)
  query_ids, passage_ids, dataset, label_source

and, for datasets with gold answers, `labels/<dataset>.json` with both answers and both scores
of every pair, so a label can be traced back to what the model actually said.

Labels. Datasets with gold answers get the true utility of Eq. (1),
u = Acc(answer with passage) - Acc(answer without), computed by the same LLM `M` that writes the
HyDE answer and does the logit probing. RAGBench has no gold answer; its label is the annotation
proxy of Eq. (5). The two are stored under different `label_source` values and never pooled.

One model `M` is loaded once and shared by labelling, HyDE and probing. Finished datasets are
skipped, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data.utility_labeler import BatchedUtilityLabeler
from src.evaluation.protocol import LABEL_RAGBENCH_PROXY, LABEL_UTILITY
from src.features.concatenate import FEATURE_DIM, build_feature_vector
from src.features.hyde import CUDAHyDEGenerator  # generic: works with any `generate(prompts)` LLM
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.retriever import RetrievedPassage, candidate_scores
from src.retrieval.stream_bm25 import CorpusStats, query_terms, score_text

DEFAULT_MODEL = {"mlx": "mlx-community/Llama-3.2-3B-Instruct-4bit", "cuda": "Qwen/Qwen2.5-7B-Instruct"}
DEFAULT_DENSE = "intfloat/multilingual-e5-large"


def load_stats(path: Path) -> CorpusStats:
    raw = json.loads(path.read_text())
    return CorpusStats(n_docs=raw["n_docs"], avgdl=raw["avgdl"], df=raw["df"])


def candidate_bm25(item: dict, stats: CorpusStats) -> list[float]:
    """Raw BM25 of every candidate. Retrieved passages already carry theirs from the corpus
    scan; provided paragraphs are scored here with the same corpus statistics."""
    terms = query_terms(item["query"])
    return [p["bm25"] if "bm25" in p else score_text(terms, p["text"], stats) for p in item["passages"]]


def item_features(item: dict, claim: str, stats: CorpusStats, dense: DenseRetriever, prober) -> np.ndarray:
    texts = [p["text"] for p in item["passages"]]
    norm, ranks, gaps = candidate_scores(candidate_bm25(item, stats))
    cosine = dense.score_passages(item["query"], texts)
    probes = prober.probe_batch(item["query"], texts)
    rows = []
    for j, p in enumerate(item["passages"]):
        passage = RetrievedPassage(
            doc_id=p["id"], text=p["text"], bm25_score=float(norm[j]),
            dense_score=float(cosine[j]), rank=int(ranks[j]), score_gap=float(gaps[j]),
        )
        rows.append(build_feature_vector(item["query"], passage, probes[j], hyde_answer=claim))
    return np.array(rows)


def item_hidden(item: dict, prober, layers: tuple[int, ...]) -> np.ndarray:
    """Last-token hidden states of every candidate, (n_passages, n_layers, d); same prompt as the D5 probe."""
    return prober.hidden_batch(item["query"], [p["text"] for p in item["passages"]], layers=layers)


def process_dataset(name: str, items: list[dict], out: Path, llm, hyde, prober, dense, stats,
                    hidden_layers: tuple[int, ...] | None = None) -> None:
    t0 = time.time()
    has_gold = all(it["gold_answers"] for it in items)

    label_by_pair: dict[tuple[str, str], float] = {}
    if has_gold:
        labeler = BatchedUtilityLabeler(llm)  # relaxed EM + token F1, as in the paper
        rows = labeler.label_corpus(
            [{"query_id": it["query_id"], "query": it["query"], "gold_answers": it["gold_answers"],
              "passages": [{"id": p["id"], "text": p["text"]} for p in it["passages"]]} for it in items]
        )
        label_by_pair = {(r["query_id"], r["doc_id"]): r["u_true"] for r in rows}
        (out / "labels").mkdir(exist_ok=True)
        (out / "labels" / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False))
        print(f"  [{name}] labelled {len(rows)} pairs in {time.time() - t0:.0f}s", flush=True)

    claims = hyde.generate_batch([it["query"] for it in items])
    X, y, qids, pids, hidden = [], [], [], [], []
    for i, (it, claim) in enumerate(zip(items, claims)):
        feats = item_features(it, claim, stats, dense, prober)
        if hidden_layers:
            hidden.append(item_hidden(it, prober, hidden_layers))
        for p, f in zip(it["passages"], feats):
            X.append(f)
            qids.append(it["query_id"])
            pids.append(p["id"])
            y.append(label_by_pair[(it["query_id"], p["id"])] if has_gold else p["proxy_label"])
        if (i + 1) % 25 == 0:
            print(f"  [{name}] features {i + 1}/{len(items)} questions ({time.time() - t0:.0f}s)", flush=True)
    X = np.array(X)
    assert X.shape[1] == FEATURE_DIM and np.isfinite(X).all(), f"{name}: bad feature matrix {X.shape}"
    (out / "features").mkdir(exist_ok=True)
    np.savez(
        out / "features" / f"{name}.npz",
        X=X, y=np.array(y, float), query_ids=np.array(qids), passage_ids=np.array(pids),
        dataset=np.array([name] * len(y)),
        label_source=np.array(LABEL_UTILITY if has_gold else LABEL_RAGBENCH_PROXY),
    )
    if hidden_layers:  # same row order as features/<name>.npz, so rows can be joined by position or by ids
        (out / "hidden").mkdir(exist_ok=True)
        np.savez(out / "hidden" / f"{name}.npz", H=np.concatenate(hidden).astype(np.float16),
                 query_ids=np.array(qids), passage_ids=np.array(pids))
    print(f"{name}: {len(y)} pairs, {len(set(qids))} questions, label mean {np.mean(y):.3f}, "
          f"nonzero {100 * np.mean(np.array(y) != 0):.0f}% ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/multidataset")
    ap.add_argument("--backend", choices=["mlx", "cuda"], default="mlx", help="mlx (Apple Silicon) or cuda (NVIDIA)")
    ap.add_argument("--quantization", choices=["none", "nf4"], default="none", help="cuda only: bf16 weights or 4-bit NF4")
    ap.add_argument("--model", default=None, help="default depends on --backend")
    ap.add_argument("--dense-model", default=DEFAULT_DENSE)
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of dataset names (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions of each dataset (smoke test)")
    ap.add_argument("--batch-size", type=int, default=None, help="default 8 (mlx) or 32 (cuda)")
    ap.add_argument("--hidden-layers", type=int, nargs="+", default=None,
                    help="also cache last-token hidden states of these layers (-1 = final, post-norm); mlx only")
    ap.add_argument("--force", action="store_true", help="recompute datasets that already have features")
    args = ap.parse_args()

    data = Path(args.data)
    items = json.loads((data / "items.json").read_text())
    stats = load_stats(data / "stats.json")
    by_name: dict[str, list[dict]] = {}
    for it in items:
        by_name.setdefault(it["dataset"], []).append(it)
    names = args.datasets or list(by_name)
    out = data if args.limit is None else data / f"smoke{args.limit}"
    out.mkdir(parents=True, exist_ok=True)

    model = args.model or DEFAULT_MODEL[args.backend]
    if args.backend == "mlx":
        from src.data.mlx_backend import MLXLLM
        from src.probing.logit_probe import MLXProber

        llm = MLXLLM(model, batch_size=args.batch_size or 8)
        prober = MLXProber(llm=llm)
    else:
        from src.data.llm_backend import QuantizedLLM
        from src.probing.logit_probe import QuantizedProber

        bs = args.batch_size or 32
        llm = QuantizedLLM(model, quantization=args.quantization, batch_size=bs, logprob_batch_size=bs)
        prober = QuantizedProber(llm=llm, batch_size=bs)
    hyde = CUDAHyDEGenerator(llm=llm)
    dense = DenseRetriever(args.dense_model)
    layers = tuple(args.hidden_layers) if args.hidden_layers else None
    t_start = time.time()
    for name in names:
        done = (out / "features" / f"{name}.npz").exists() and (layers is None or (out / "hidden" / f"{name}.npz").exists())
        if done and not args.force:
            print(f"{name}: features exist, skipping", flush=True)
            continue
        subset = by_name[name][: args.limit] if args.limit else by_name[name]
        process_dataset(name, subset, out, llm, hyde, prober, dense, stats, hidden_layers=layers)
    if args.backend == "mlx":  # what the paper reports about hardware, written next to the features
        import mlx.core as mx

        peak = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory()
        info = {"model": model, "peak_gib": round(peak / 2**30, 2), "batch_size": args.batch_size or 8,
                "datasets": names, "limit": args.limit, "hidden_layers": layers, "seconds": round(time.time() - t_start)}
        (out / "run_info_mlx.json").write_text(json.dumps(info, indent=1))
        print(f"peak memory {info['peak_gib']} GiB (mlx)", flush=True)
    if args.backend == "cuda":  # the numbers the paper reports about hardware, written next to the features
        import torch

        info = {"gpu": torch.cuda.get_device_name(0), "model": model, "quantization": args.quantization,
                "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "batch_size": args.batch_size or 32, "datasets": names, "limit": args.limit}
        (out / f"run_info_{args.quantization}.json").write_text(json.dumps(info, indent=1))
        print(f"peak GPU memory {info['peak_gpu_gib']} GiB ({args.quantization})", flush=True)


if __name__ == "__main__":
    main()
