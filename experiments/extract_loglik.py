"""Smooth training label: the change in the log-likelihood of the gold answer when a passage is given.

  python -m experiments.extract_loglik --data data/processed/mac_qwen7b --model mlx-community/Qwen2.5-7B-Instruct-4bit

For every pair of `features/<dataset>.npz` of a dataset with gold answers it writes `loglik/<dataset>.npz`
with `u_smooth` = mean token log-prob of the first gold answer under the Eq. (1) prompt with the passage minus
the same without any passage, clipped to [-5, 5], and the `query_ids` / `passage_ids` of `features/` in the same
row order. It is only a *training* label; every method is still scored against the true utility of Eq. (1).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data.utility_labeler import NO_CONTEXT_PROMPT, WITH_CONTEXT_PROMPT

CLIP = 5.0


def smooth_utility(llm, items: list[dict], pairs: list[tuple[str, str]]) -> np.ndarray:
    """u_smooth of each (query_id, passage_id) in `pairs`, in that order. The no-passage score is computed once per query."""
    by_query = {it["query_id"]: it for it in items}
    text = {(it["query_id"], p["id"]): p["text"] for it in items for p in it["passages"]}
    queries = list(dict.fromkeys(q for q, _ in pairs))
    answer = {q: by_query[q]["gold_answers"][0] for q in queries}
    base = dict(zip(queries, llm.answer_logprobs([NO_CONTEXT_PROMPT.format(query=by_query[q]["query"]) for q in queries],
                                                 [answer[q] for q in queries])))
    prompts = [WITH_CONTEXT_PROMPT.format(query=by_query[q]["query"], passage=text[(q, pid)]) for q, pid in pairs]
    with_passage = llm.answer_logprobs(prompts, [answer[q] for q, _ in pairs])
    return np.clip(np.array([w - base[q] for (q, _), w in zip(pairs, with_passage)]), -CLIP, CLIP)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/mac_qwen7b")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    data = Path(args.data)
    items = json.loads((data / "items.json").read_text())
    from src.data.mlx_backend import MLXLLM

    llm = MLXLLM(args.model)
    (data / "loglik").mkdir(exist_ok=True)
    for path in sorted((data / "features").glob("*.npz")):
        name, out = path.stem, data / "loglik" / path.name
        z = np.load(path, allow_pickle=False)
        subset = [it for it in items if it["dataset"] == name]
        if out.exists() and not args.force or not all(it["gold_answers"] for it in subset):
            print(f"{name}: skipped", flush=True)
            continue
        t0 = time.time()
        pairs = list(zip(z["query_ids"].tolist(), z["passage_ids"].tolist()))
        u = smooth_utility(llm, subset, pairs)
        np.savez(out, u_smooth=u, query_ids=z["query_ids"], passage_ids=z["passage_ids"])
        print(f"{name}: {len(u)} pairs, mean {u.mean():+.3f}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
