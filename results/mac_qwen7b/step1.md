# Step-1 gate (Mac, Qwen2.5-7B-Instruct 4-bit, first question sample)

- Extraction (12 datasets, 9,554 pairs, features + hidden states of layers 14 and -1): 10,466 s (2.9 h) at batch size 8, peak MLX memory 6.39 GiB.
- Nested leave-one-dataset-out, v1 grid: macro rho MPUP 0.293, dense 0.281, BM25 0.108; pooled MPUP - dense +0.02 [-0.01, +0.04] over 749 queries (RTX bf16 run: +0.04).
- Answer accuracy, top-1: none 0.246, random 0.246, BM25 0.299, dense 0.360, MPUP 0.367, oracle 0.577. Top-3: dense 0.409, MPUP 0.430, oracle 0.461.
- Gate: oracle - dense at m = 1 is +0.217 (threshold 0.03): **gate passed**. There is room: a perfect ranker would raise top-1 accuracy by 22 points, MPUP gets +0.7 points (p = 0.43) at m = 1 and +2.2 points (p = 0.004) at m = 3.

Decision: continue with the full v2 variant grid (Tasks 5-6 runs), then the third sample.
