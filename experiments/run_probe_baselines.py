"""C4: MPUP against utility estimates that query the target LLM directly, on the test questions of the larger study.

  python -m experiments.run_probe_baselines --data data/processed/big --model Qwen/Qwen2.5-7B-Instruct \
      --out results/probe_baselines
  python -m experiments.run_probe_baselines --data data/processed/big_mistral --model mistralai/Mistral-7B-Instruct-v0.3 \
      --layer 24 --out results/probe_baselines_mistral

The methods are the utility-judgment methods that Zhang et al. (LLM-specific utility, CIKM) evaluate on SpecUBench,
re-implemented with the target LLM M (the one whose Eq. (1) utility is predicted), plus one that reads M's own answers:

  pseudo-answer      one greedy answer of M per query from all candidates of the query (as in SpecUBench), used by
                     the methods marked "w/ A"
  PVS w/o A, w/ A    pointwise verbalized judgment: M is asked whether the passage has utility (the SpecUBench
                     definition: usefulness and novelty) and the score is logit("Yes") - logit("No") of the next token
  likelihood         mean log-probability of the pseudo-answer given the query and the passage (Shi et al.)
  LVR w/ A           listwise verbalized ranking: M reads all candidates and the pseudo-answer and writes a ranking
                     "[3] > [1] > ..."; a passage scores minus its position, unranked passages come last
  answer confidence  mean log-probability of M's own answer with the passage (the answer of the Eq. (1) label run,
                     one generation of M per passage), given the query and the passage

No method is trained. Scores are written to `<out>_scores.npz` (resumable); the comparison with MPUP, fitted on the
development questions as in `run_big final`, goes to `<out>.{md,json}`.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from experiments import run_big
from experiments.run_criteria import SEEN, _files, passage_ids
from src.data.utility_labeler import NO_CONTEXT_PROMPT, WITH_CONTEXT_PROMPT

UTILITY_DEF = ("Determine if the passage has utility based on two criteria: 1. Usefulness: The passage must be relevant "
               "and contribute to generating a correct, reasonable, and complete answer. 2. Novelty: The useful "
               "information should be new, not part of your pre-existing knowledge.")
PVS = "Question: {query}\n\nPassage: {passage}\n\n{ref}" + UTILITY_DEF + "\nAnswer with Yes or No only."
REF = "Reference answer: {answer}\n\n"
LVR = ("The following are {n} passages, each indicated by a number identifier [].\n\n{passages}\n\nQuestion: {query}\n\n"
       "Reference answer: {answer}\n\nRank the passages by their utility for generating a correct, reasonable and "
       "complete answer to the question. List all {n} identifiers in descending order of utility, using the format "
       "[2] > [1] > [3]. Only respond with the ranking.")
PSEUDO = ("Answer the question as briefly as possible, with just the answer span and no explanation.\n\n{passages}\n\n"
          "Question: {query}\nAnswer:")
METHODS = ("PVS w/o A", "PVS w/ A", "likelihood", "LVR w/ A", "answer confidence")


@torch.inference_mode()
def next_token_margin(llm, prompts: list[str], yes: int, no: int, batch_size: int) -> np.ndarray:
    tok = llm.tokenizer
    tok.padding_side = "left"
    texts = [llm.chat_format(p) for p in prompts]
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts), np.float32)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        logits = llm.model(**enc).logits[:, -1].float()
        out[idx] = (logits[:, yes] - logits[:, no]).cpu().numpy()
    return out


def mean_logprob(llm, contexts: list[str], answers: list[str], batch_size: int) -> np.ndarray:
    pairs = [(llm.chat_format(c), " " + (a.strip() or "unknown")) for c, a in zip(contexts, answers)]
    order = np.argsort([len(c) for c, _ in pairs])
    llm.logprob_batch_size = batch_size
    res = llm.sequence_logprobs([pairs[i] for i in order], add_special_tokens=False)
    out = np.zeros(len(pairs), np.float32)
    out[order] = [r.mean_logprob for r in res]
    return out


def parse_ranking(text: str, n: int) -> np.ndarray:
    """Score = minus the position in the ranking; identifiers not ranked share the last position."""
    seen = []
    for m in re.findall(r"\[(\d+)\]", text):
        k = int(m) - 1
        if 0 <= k < n and k not in seen:
            seen.append(k)
    score = np.full(n, -float(n))
    for pos, k in enumerate(seen):
        score[k] = -float(pos)
    return score


def first_token(tok, word: str) -> int:
    ids = tok(word, add_special_tokens=False).input_ids
    return ids[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/big")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, default=run_big.LAYER)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="first N test questions only (smoke test)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    d = run_big.load(data, args.layer)
    test = run_big.split(d["q"], run_big.seen_ids([Path(p) for p in SEEN]))
    dev, te = np.flatnonzero(~test), np.flatnonzero(test)
    if args.limit:
        te = te[np.isin(d["q"][te], list(dict.fromkeys(d["q"][te].tolist()))[:args.limit])]
    pids = passage_ids(data)
    items = {str(it["query_id"]): it for it in json.loads((data / "items.json").read_text())}
    answers = {}
    for p in _files(data):  # the with-passage answers of the Eq. (1) label run
        for r in json.loads((data / "labels" / f"{p.stem}.json").read_text()):
            answers[(r["query_id"], r["doc_id"])] = r["answer_with_ctx"]

    rows_q = [str(q).split(":", 1)[1] for q in d["q"][te]]
    rows_p = [str(p) for p in pids[te]]
    queries = [items[q]["query"] for q in rows_q]
    texts = [{x["id"]: x["text"] for x in items[q]["passages"]}[p] for q, p in zip(rows_q, rows_p)]
    uq = list(dict.fromkeys(rows_q))
    by_q = {q: [i for i, x in enumerate(rows_q) if x == q] for q in uq}

    out = Path(args.out)
    cache = out.with_name(out.name + "_scores.npz")
    scores = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    timing = json.loads(out.with_name(out.name + "_timing.json").read_text()) if out.with_name(out.name + "_timing.json").exists() else {}
    todo = [m for m in METHODS if m not in scores]
    if todo:
        from src.data.llm_backend import QuantizedLLM

        llm = QuantizedLLM(args.model, quantization="none", batch_size=args.batch_size)
        yes, no = first_token(llm.tokenizer, "Yes"), first_token(llm.tokenizer, "No")

        def save(name, value, t0):
            scores[name] = value
            timing[name] = round(time.time() - t0, 1)
            np.savez(cache, **scores)
            out.with_name(out.name + "_timing.json").write_text(json.dumps(timing, indent=1))
            print(f"{name}: {timing[name]:.0f}s", flush=True)

        if "pseudo" not in scores:
            t0 = time.time()
            prompts = [PSEUDO.format(query=items[q]["query"], passages="\n\n".join(
                f"Passage {k + 1}: {texts[i]}" for k, i in enumerate(by_q[q]))) for q in uq]
            llm.batch_size = max(1, args.batch_size // 2)
            pa = llm.generate(prompts, max_new_tokens=32)
            llm.batch_size = args.batch_size
            save("pseudo", np.array(pa, dtype=object), t0)
        pseudo = dict(zip(uq, [str(a) for a in scores["pseudo"]]))
        if "PVS w/o A" in todo:
            t0 = time.time()
            save("PVS w/o A", next_token_margin(llm, [PVS.format(query=q, passage=t, ref="") for q, t in zip(queries, texts)],
                                                yes, no, args.batch_size), t0)
        if "PVS w/ A" in todo:
            t0 = time.time()
            save("PVS w/ A", next_token_margin(llm, [PVS.format(query=q, passage=t, ref=REF.format(answer=pseudo[x]))
                                                     for q, t, x in zip(queries, texts, rows_q)], yes, no, args.batch_size), t0)
        if "likelihood" in todo:
            t0 = time.time()
            save("likelihood", mean_logprob(llm, [WITH_CONTEXT_PROMPT.format(query=q, passage=t) for q, t in zip(queries, texts)],
                                            [pseudo[x] for x in rows_q], args.batch_size), t0)
        if "answer confidence" in todo:
            t0 = time.time()
            own = [answers[(q, p)] for q, p in zip(rows_q, rows_p)]
            save("answer confidence", mean_logprob(llm, [WITH_CONTEXT_PROMPT.format(query=q, passage=t)
                                                         for q, t in zip(queries, texts)], own, args.batch_size), t0)
        if "LVR w/ A" in todo:
            t0 = time.time()
            prompts = [LVR.format(n=len(by_q[q]), query=items[q]["query"], answer=pseudo[q], passages="\n\n".join(
                f"[{k + 1}] {texts[i]}" for k, i in enumerate(by_q[q]))) for q in uq]
            llm.batch_size = max(1, args.batch_size // 2)
            gen = llm.generate(prompts, max_new_tokens=8 * max(len(v) for v in by_q.values()))
            s = np.zeros(len(rows_q), np.float32)
            parsed = []
            for q, g in zip(uq, gen):
                r = parse_ranking(g, len(by_q[q]))
                s[by_q[q]] = r
                parsed.append(int((r > -len(by_q[q])).sum()))
            timing["LVR ranked share"] = float(np.sum(parsed) / sum(len(by_q[q]) for q in uq))
            scores["LVR raw"] = np.array(gen, dtype=object)
            save("LVR w/ A", s, t0)
        timing["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        out.with_name(out.name + "_timing.json").write_text(json.dumps(timing, indent=1))
        llm.unload()

    preds = {"MPUP": run_big.fit_predict(d, dev, te, "gbr(500,d4)|u|ce|probe")}
    preds.update({m: np.asarray(scores[m], dtype=float) for m in METHODS})
    preds["ΔH (D5)"] = d["X"][te, 13]
    preds.update({b: f(d, te) for b, f in run_big.BASELINES.items()})
    c = run_big._compare(d, te, preds, "MPUP")
    res = {"model": args.model, "n_test_queries": c["n_queries"], "timing_s": timing, "comparison": c}
    names = list(preds)
    L = ["# C4: MPUP against utility estimates that query the target LLM", "",
         f"{c['n_queries']} rankable test questions; target LLM {args.model}; within-query Spearman with its Eq. (1) "
         f"utility. Time per method (s, all {len(te)} test pairs): {json.dumps(timing)}", "",
         "| ranking | ρ | NDCG@10 | top-1 |", "|---|---|---|---|"]
    L += [f"| {n} | {c['macro'][n]['spearman']:.3f} | {c['macro'][n]['ndcg']:.3f} | {c['macro'][n]['top1']:.3f} |" for n in names]
    L += ["", "Paired difference of MPUP (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):", ""]
    for n in names[1:]:
        m, lo, hi = c["paired_ci"][n]
        t = c["tests"][n]
        L.append(f"- vs {n}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}], p = {t['p_permutation']:.2g}, p_t = {t['p_ttest']:.2g}, "
                 f"d_z = {t['cohen_dz']:.2f}")
    L += ["", "| dataset | queries | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for ds, x in c["per_dataset"].items():
        L.append(f"| {ds} | {x['n_queries']} | " + " | ".join(f"{x['spearman'][n]:.3f}" for n in names) + " |")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
