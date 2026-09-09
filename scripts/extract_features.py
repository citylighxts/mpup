"""Extract D1-D5 feature vectors from a dataset of (query, passage) pairs."""
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.retrieval.retriever import RetrievedPassage
from src.probing.logit_probe import LogitProber
from src.features.d1_retriever_centric import extract_d1, d1_to_array
from src.features.d2_document_quality import extract_d2, d2_to_array
from src.features.d3_semantic_utility import extract_d3, d3_to_array
from src.features.d4_faithfulness import extract_d4, d4_to_array
from src.features.d5_llm_aware import extract_d5, d5_to_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", required=True, help="JSONL: {query, passages:[{id,text,bm25,dense,rank,gap,utility}]}")
    parser.add_argument("--model", required=True, help="Target LLM path/name for logit probing")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prober = LogitProber(args.model, device=cfg["probing"]["device"])
    all_features, all_labels = [], []

    with open(args.data) as f:
        for line in tqdm(f, desc="Extracting features"):
            item = json.loads(line)
            query = item["query"]
            for p in item["passages"]:
                passage = RetrievedPassage(
                    doc_id=p["id"],
                    text=p["text"],
                    bm25_score=p.get("bm25", 0.0),
                    dense_score=p.get("dense", 0.0),
                    rank=p.get("rank", 0),
                    score_gap=p.get("gap", 0.0),
                )
                probe = prober.probe(query, passage.text, max_length=cfg["probing"]["max_length"])
                fv = np.concatenate([
                    d1_to_array(extract_d1(passage)),
                    d2_to_array(extract_d2(passage.text)),
                    d3_to_array(extract_d3(query, passage.text)),
                    d4_to_array(extract_d4(query, passage.text)),
                    d5_to_array(extract_d5(probe)),
                ])
                all_features.append(fv)
                all_labels.append(float(p.get("utility", 0.0)))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "features.npy", np.array(all_features))
    np.save(out / "labels.npy", np.array(all_labels))
    print(f"Saved {len(all_features)} samples to {out}")


if __name__ == "__main__":
    main()
