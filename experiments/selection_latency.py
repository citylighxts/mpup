"""Selection-time cost of the MPUP tiers (`run_criteria light`): latency per query and peak GPU memory.

  python -m experiments.selection_latency --n 100 --out results/selection_latency

For the first `--n` test questions of the larger study (all their candidates, up to twenty), every step that selection
needs is timed one query at a time (batch = the candidates of one query), with the components of the pipeline:

  hyde      one greedy generation of M per query (the claim for D3 answerability and D4)
  dense     the E5 query and passage embeddings (D1; in deployment the retriever usually provides them)
  d2_d3     passage statistics and entity overlap on the CPU (with the answerability overlap)
  d4        NLI of every passage against the claim
  d5        logit probing with M (base prompt once, one forward pass per passage)
  hidden_M  M's hidden state at the last token of the Eq. (1) prompt, one forward pass per passage
  ce        bge-reranker-v2-m3 score of every passage
  hidden_R  the same hidden state from a small reader model (Qwen2.5-1.5B / 0.5B)
  regressor the boosted trees of MPUP on the candidates of one query

The tiers add up their components. Models are loaded in phases so that each phase's peak GPU memory is the peak of
that tier: no-LM (E5, NLI excluded, cross-encoder), reader tiers (+ the small reader), and the full tier (+ M and NLI).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import run_big
from experiments.extract_multidataset import DEFAULT_DENSE, candidate_bm25, load_stats
from experiments.run_criteria import SEEN
from src.data.utility_labeler import WITH_CONTEXT_PROMPT
from src.retrieval.retriever import candidate_scores
from src.features.d2_document_quality import d2_to_array, extract_d2
from src.features.d3_semantic_utility import d3_to_array, extract_d3
from src.features.d4_faithfulness import d4_to_array, extract_d4

TIERS = {"full (MPUP)": ["hyde", "dense", "d2_d3", "d4", "d5", "hidden_M", "ce", "regressor"],
         "no-generation": ["dense", "d2_d3", "d5", "hidden_M", "ce", "regressor"],
         "reader-1.5B": ["dense", "d2_d3", "hidden_R1.5B", "ce", "regressor"],
         "reader-0.5B": ["dense", "d2_d3", "hidden_R0.5B", "ce", "regressor"],
         "no-LM": ["dense", "d2_d3", "ce", "regressor"]}


def timed(fn, *a):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*a)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


@torch.inference_mode()
def hidden_pass(llm, query: str, passages: list[str], layer: int):
    tok = llm.tokenizer
    tok.padding_side = "left"
    enc = tok([llm.chat_format(WITH_CONTEXT_PROMPT.format(query=query, passage=p)) for p in passages],
              return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    out = llm.model.model(**enc, output_hidden_states=True)  # no LM head: only the hidden state is needed
    return out.hidden_states[layer][:, -1].float().cpu().numpy()


@torch.inference_mode()
def ce_scores(model, tok, query: str, passages: list[str]):
    enc = tok([query] * len(passages), passages, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    return model(**enc).logits.view(-1).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/big")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--no-m", action="store_true", help="skip the phase with M (smoke test next to another job)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    d = run_big.load(data)
    test = run_big.split(d["q"], run_big.seen_ids([Path(p) for p in SEEN]))
    items = {str(it["query_id"]): it for it in json.loads((data / "items.json").read_text())}
    qids = list(dict.fromkeys(str(q).split(":", 1)[1] for q in d["q"][test]))
    rng = np.random.default_rng(0)
    qids = [qids[i] for i in sorted(rng.choice(len(qids), args.n, replace=False))]
    sample = [items[q] for q in qids]
    stats = load_stats(data / "stats.json")
    times = {k: [] for k in ("hyde", "dense", "d2_d3", "d4", "d5", "hidden_M", "ce", "hidden_R1.5B", "hidden_R0.5B", "regressor")}
    peak = {}

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.data.llm_backend import QuantizedLLM
    from src.retrieval.dense_retriever import DenseRetriever

    # regressor: trees of MPUP's size (500, depth 4) on 16 columns, fitted once; only prediction is timed
    from sklearn.ensemble import GradientBoostingRegressor
    tr = np.flatnonzero(~test)[:20000]
    Xtr = np.c_[d["X"][tr], d["ce"][tr], d["y"][tr]]
    gbr = GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8, random_state=0)
    gbr.fit(Xtr, d["y"][tr])

    # phase 1: no-LM tier (E5 + cross-encoder) and the CPU features
    torch.cuda.reset_peak_memory_stats()
    dense = DenseRetriever(DEFAULT_DENSE)
    ce_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    ce = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", dtype=torch.float16).cuda().eval()
    for it in sample:
        q, texts = it["query"], [p["text"] for p in it["passages"]]
        _, t = timed(dense.score_passages, q, texts); times["dense"].append(t)
        _, t = timed(lambda: [np.r_[d2_to_array(extract_d2(x)), d3_to_array(extract_d3(q, x, hyde_answer=q))] for x in texts])
        times["d2_d3"].append(t)
        _, t = timed(ce_scores, ce, ce_tok, q, texts); times["ce"].append(t)
        X = np.zeros((len(texts), Xtr.shape[1]))
        _, t = timed(gbr.predict, X); times["regressor"].append(t)
        candidate_scores(candidate_bm25(it, stats))
    peak["no-LM"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    # phase 2/3: small readers on top of phase 1
    for name, path, layer in (("hidden_R0.5B", "Qwen/Qwen2.5-0.5B-Instruct", 18), ("hidden_R1.5B", "Qwen/Qwen2.5-1.5B-Instruct", 21)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        r = QuantizedLLM(path, quantization="none")
        for it in sample:
            _, t = timed(hidden_pass, r, it["query"], [p["text"] for p in it["passages"]], layer); times[name].append(t)
        peak["reader-" + name.split("R")[1]] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        r.unload()

    # phase 4: M and NLI on top of phase 1 (full and no-generation tiers)
    if args.no_m:
        print({k: np.mean(v) if v else None for k, v in times.items()}, peak)
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    from src.features.hyde import CUDAHyDEGenerator
    from src.probing.logit_probe import QuantizedProber

    llm = QuantizedLLM("Qwen/Qwen2.5-7B-Instruct", quantization="none", batch_size=32, logprob_batch_size=32)
    hyde, prober = CUDAHyDEGenerator(llm=llm), QuantizedProber(llm=llm, batch_size=32)
    for it in sample:
        q, texts = it["query"], [p["text"] for p in it["passages"]]
        claim, t = timed(hyde.generate, q); times["hyde"].append(t)
        _, t = timed(lambda: [d4_to_array(extract_d4(claim, x)) for x in texts]); times["d4"].append(t)
        _, t = timed(prober.probe_batch, q, texts); times["d5"].append(t)
        _, t = timed(hidden_pass, llm, q, texts, 21); times["hidden_M"].append(t)
    peak_full = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    peak["full (MPUP)"] = peak["no-generation"] = peak_full

    n_pass = [len(it["passages"]) for it in sample]
    comp = {k: {"mean_ms": 1000 * float(np.mean(v)), "median_ms": 1000 * float(np.median(v))} for k, v in times.items()}
    tiers = {}
    for t, parts in TIERS.items():
        per_q = np.sum([np.array(times[p]) for p in parts], axis=0)
        tiers[t] = {"mean_ms": 1000 * float(per_q.mean()), "median_ms": 1000 * float(np.median(per_q)),
                    "peak_gpu_gib": peak.get(t), "llm_calls": parts}
    res = {"gpu": torch.cuda.get_device_name(0), "n_queries": len(sample), "mean_candidates": float(np.mean(n_pass)),
           "components": comp, "tiers": tiers,
           "note": "bf16, batch = candidates of one query; peak memory is torch.cuda.max_memory_allocated of the phase "
                   "(models of the tier loaded, plus E5 and the cross-encoder)"}
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    L = ["# Selection-time cost per query", "",
         f"{res['n_queries']} test questions, {res['mean_candidates']:.1f} candidates on average, {res['gpu']}, bf16, one query at a time.", "",
         "| tier | mean ms / query | median ms | peak GPU GiB |", "|---|---|---|---|"]
    L += [f"| {t} | {v['mean_ms']:.0f} | {v['median_ms']:.0f} | {v['peak_gpu_gib']} |" for t, v in tiers.items()]
    L += ["", "| component | mean ms / query | median ms |", "|---|---|---|"]
    L += [f"| {k} | {v['mean_ms']:.1f} | {v['median_ms']:.1f} |" for k, v in comp.items()]
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
