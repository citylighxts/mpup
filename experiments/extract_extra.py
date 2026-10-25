"""Extra signals per (query, passage) pair, next to the 14 features of `extract_multidataset`.

  python -m experiments.extract_extra --data data/processed/big

For every dataset with features it writes `extra/<dataset>.npz`, aligned row by row with
`features/<dataset>.npz` (same query and passage order):

  soft        log p(gold | q, d) - log p(gold | q), mean over the gold answer's tokens, scored by the
              same model M and with the same prompts as the Eq. (1) labels. A continuous training
              target: Eq. (1) is a difference of two EM/F1 scores of one greedy answer, so most pairs
              tie at 0, while the log-probability of the gold answer moves with every passage.
              Only used as a *training* target; every reported score is against Eq. (1).
  hidden_L    hidden state of M (layer L) at the last prompt token of the with-passage prompt,
              float16, and hidden_q_L the same for the no-passage prompt (one per query, repeated)
  ce          relevance score of a cross-encoder reranker (bge-reranker-v2-m3) for (q, d)

The first gold answer is scored (the canonical one in every dataset).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.utility_labeler import NO_CONTEXT_PROMPT, WITH_CONTEXT_PROMPT

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CE = "BAAI/bge-reranker-v2-m3"
LAYERS = (14, 21)


@torch.inference_mode()
def score_gold(llm, contexts: list[str], answers: list[str], layers=LAYERS, batch_size: int = 32):
    """Mean log-prob of each answer after its chat-formatted context, and the hidden states of the
    last context token. Batches are formed over length-sorted inputs to keep padding small."""
    tok, model = llm.tokenizer, llm.model
    seqs, plen = [], []
    for c, a in zip(contexts, answers):
        cid = tok(llm.chat_format(c), add_special_tokens=False).input_ids
        aid = tok(" " + a.strip(), add_special_tokens=False).input_ids or [tok.eos_token_id]
        seqs.append(cid + aid)
        plen.append(len(cid))
    order = np.argsort([len(s) for s in seqs])
    lp = np.zeros(len(seqs), np.float32)
    hid = {L: np.zeros((len(seqs), model.config.hidden_size), np.float16) for L in layers}
    for s in range(0, len(seqs), batch_size):
        idx = order[s:s + batch_size]
        m = max(len(seqs[i]) for i in idx)
        ids = torch.full((len(idx), m), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), m), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, :len(seqs[i])] = torch.tensor(seqs[i])
            att[r, :len(seqs[i])] = 1
        out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
        for r, i in enumerate(idx):
            p, n = plen[i], len(seqs[i])
            logits = out.logits[r, p - 1:n - 1].float()
            tgt = ids[r, p:n].cuda()
            lp[i] = float(torch.log_softmax(logits, -1).gather(1, tgt[:, None]).mean())
            for L in layers:
                hid[L][i] = out.hidden_states[L][r, p - 1].float().cpu().numpy().astype(np.float16)
        del out
    return lp, hid


@torch.inference_mode()
def cross_encoder_scores(model, tok, queries: list[str], passages: list[str], batch_size: int = 128) -> np.ndarray:
    out = np.zeros(len(queries), np.float32)
    order = np.argsort([len(p) for p in passages])
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([queries[i] for i in idx], [passages[i] for i in idx], padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to("cuda")
        out[idx] = model(**enc).logits.view(-1).float().cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/big")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ce-model", default=DEFAULT_CE)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    items = json.loads((data / "items.json").read_text())
    by_name: dict[str, dict] = {}
    for it in items:
        by_name.setdefault(it["dataset"], {})[str(it["query_id"])] = it
    (data / "extra").mkdir(exist_ok=True)
    todo = [p for p in sorted((data / "features").glob("*.npz"))
            if (args.force or not (data / "extra" / p.name).exists()) and p.stem in by_name
            and all(it["gold_answers"] for it in by_name[p.stem].values())]
    if not todo:
        print("nothing to do", flush=True)
        return

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.data.llm_backend import QuantizedLLM

    llm = QuantizedLLM(args.model, quantization="none", batch_size=args.batch_size)
    ce_tok = AutoTokenizer.from_pretrained(args.ce_model)
    ce = AutoModelForSequenceClassification.from_pretrained(args.ce_model, dtype=torch.float16).cuda().eval()

    for path in todo:
        t0 = time.time()
        z = np.load(path)
        qids, pids = [str(q) for q in z["query_ids"]], [str(p) for p in z["passage_ids"]]
        items_ds = by_name[path.stem]
        text = {(str(it["query_id"]), p["id"]): p["text"] for it in items_ds.values() for p in it["passages"]}
        queries = [items_ds[q]["query"] for q in qids]
        golds = [str(items_ds[q]["gold_answers"][0]) for q in qids]
        passages = [text[(q, p)] for q, p in zip(qids, pids)]

        uq = list(dict.fromkeys(qids))
        base_lp, base_h = score_gold(llm, [NO_CONTEXT_PROMPT.format(query=items_ds[q]["query"]) for q in uq],
                                     [str(items_ds[q]["gold_answers"][0]) for q in uq], batch_size=args.batch_size)
        pos = {q: i for i, q in enumerate(uq)}
        qi = np.array([pos[q] for q in qids])
        ctx_lp, ctx_h = score_gold(llm, [WITH_CONTEXT_PROMPT.format(query=q, passage=p) for q, p in zip(queries, passages)],
                                   golds, batch_size=args.batch_size)
        ce_s = cross_encoder_scores(ce, ce_tok, queries, passages)
        out = {"query_ids": z["query_ids"], "passage_ids": z["passage_ids"],
               "soft": ctx_lp - base_lp[qi], "ctx_logprob": ctx_lp, "base_logprob": base_lp[qi], "ce": ce_s}
        for L in LAYERS:
            out[f"hidden_{L}"] = ctx_h[L]
            out[f"hidden_q_{L}"] = base_h[L][qi]
        np.savez(data / "extra" / path.name, **out)
        y = z["y"]
        rho = np.corrcoef(out["soft"], y)[0, 1]
        print(f"{path.stem}: {len(y)} pairs, corr(soft, u) {rho:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
