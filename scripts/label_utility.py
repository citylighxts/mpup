"""Compute ground-truth utility scores u_i_true for all (query, passage) pairs.

Output: JSONL file dengan format yang siap dipakai extract_features.py:
  {"query": str, "answers": [...], "passages": [{"id", "text", "utility", ...}]}

Usage:
  python scripts/label_utility.py \
    --dataset ragbench \
    --split validation \
    --source-model <model-path-or-hf-id> \
    --output data/raw/ragbench_labeled.jsonl \
    [--max-samples 500]
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.data.loaders import load_benchmark
from src.data.utility_labeler import SourceLLMAnswerer, UtilityLabeler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["ragbench", "natural_questions", "msmarco", "triviaqa"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--source-model", required=True,
                        help="HuggingFace model id or local path for source LLM (cheaper model for labeling)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading source LLM: {args.source_model}")
    answerer = SourceLLMAnswerer(
        args.source_model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    labeler = UtilityLabeler(answerer)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_iter = load_benchmark(args.dataset, split=args.split, max_samples=args.max_samples)
    written = 0

    with open(out_path, "w") as fout:
        for item in tqdm(data_iter, desc=f"Labeling {args.dataset}"):
            labeled = labeler.label_query(item["query"], item["answers"], item["passages"])
            row = {
                "query": item["query"],
                "answers": item["answers"],
                "dataset": item.get("dataset", args.dataset),
                "passages": [
                    {
                        "id": lp.doc_id,
                        "text": lp.text,
                        "utility": lp.u_true,
                        "score_no_ctx": lp.score_no_ctx,
                        "score_with_ctx": lp.score_with_ctx,
                    }
                    for lp in labeled
                ],
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Wrote {written} queries to {out_path}")


if __name__ == "__main__":
    main()
