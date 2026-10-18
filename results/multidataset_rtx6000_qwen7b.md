# MPUP multi-dataset results

Within-query Spearman ρ (mean over queries, 95% bootstrap CI over queries). Pooled CV: 5-fold × 5 repeats over queries. `paper config` is the predictor as stated in the paper (raw features, 200 trees of depth 5) and involves no selection.

Constant (uninformative) features in the pool: source_credibility.

## Datasets

| dataset | label | queries | pairs | passages/q | mean label | non-zero % | positive % | rankable queries |
|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | utility_eq1 | 100 | 1000 | 10.0 | -0.031 | 43 | 19 | 94 |
| ambigqa | utility_eq1 | 100 | 500 | 5.0 | -0.010 | 44 | 21 | 61 |
| bamboogle | utility_eq1 | 100 | 500 | 5.0 | +0.002 | 31 | 18 | 48 |
| entityquestions | utility_eq1 | 100 | 500 | 5.0 | +0.006 | 41 | 21 | 65 |
| hotpotqa | utility_eq1 | 100 | 1000 | 10.0 | -0.006 | 34 | 13 | 90 |
| musique | utility_eq1 | 100 | 1997 | 20.0 | -0.017 | 24 | 8 | 84 |
| nq | utility_eq1 | 100 | 500 | 5.0 | -0.007 | 38 | 17 | 56 |
| popqa | utility_eq1 | 100 | 500 | 5.0 | +0.043 | 27 | 16 | 44 |
| ragbench-hotpotqa | ragbench_proxy | 390 | 1557 | 4.0 | +0.216 | 45 | 45 | 382 |
| squad | utility_eq1 | 100 | 500 | 5.0 | +0.034 | 40 | 19 | 67 |
| triviaqa | utility_eq1 | 100 | 500 | 5.0 | -0.009 | 43 | 22 | 71 |
| webquestions | utility_eq1 | 100 | 500 | 5.0 | -0.150 | 50 | 12 | 67 |

## Headline: nested leave-one-dataset-out

For each held-out dataset the configuration is chosen from the grid using only the other datasets (inner leave-one-dataset-out), refit on them, and scored on the held-out dataset. Grid: ridge(a=10)-raw, ridge(a=100)-raw, ridge(a=1000)-raw, gbr(100,d2)-raw, gbr(200,d3)-raw, gbr(200,d5)-raw, ridge(a=10)-centered, ridge(a=100)-centered, ridge(a=1000)-centered, gbr(100,d2)-centered, gbr(200,d3)-centered, gbr(200,d5)-centered.

| held-out | queries | selected config | MPUP (nested) | paper config | dense | BM25 | MPUP − dense (paired) | MPUP − paper (paired) |
|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 94 | ridge(a=10)-centered | +0.24 [+0.17, +0.31] | +0.16 [+0.08, +0.23] | +0.21 [+0.14, +0.27] | +0.08 [+0.01, +0.16] | +0.04 [-0.00, +0.08] | +0.09 [+0.01, +0.17] |
| ambigqa | 61 | ridge(a=1000)-centered | +0.43 [+0.32, +0.54] | +0.32 [+0.19, +0.44] | +0.42 [+0.29, +0.54] | +0.08 [-0.05, +0.20] | +0.01 [-0.08, +0.10] | +0.11 [+0.00, +0.22] |
| bamboogle | 48 | gbr(100,d2)-centered | +0.39 [+0.25, +0.52] | +0.20 [+0.05, +0.35] | +0.28 [+0.13, +0.43] | +0.10 [-0.03, +0.23] | +0.11 [-0.00, +0.23] | +0.19 [+0.05, +0.34] |
| entityquestions | 65 | ridge(a=1000)-centered | +0.32 [+0.19, +0.45] | +0.24 [+0.12, +0.36] | +0.27 [+0.13, +0.41] | +0.29 [+0.16, +0.41] | +0.05 [-0.03, +0.15] | +0.08 [-0.04, +0.20] |
| hotpotqa | 90 | gbr(100,d2)-centered | +0.27 [+0.20, +0.35] | +0.14 [+0.07, +0.22] | +0.28 [+0.21, +0.34] | +0.07 [-0.01, +0.15] | -0.00 [-0.05, +0.05] | +0.13 [+0.05, +0.21] |
| musique | 84 | gbr(100,d2)-centered | +0.22 [+0.17, +0.27] | +0.17 [+0.11, +0.22] | +0.17 [+0.12, +0.23] | +0.04 [-0.02, +0.11] | +0.05 [+0.01, +0.09] | +0.06 [+0.01, +0.10] |
| nq | 56 | gbr(100,d2)-centered | +0.44 [+0.33, +0.54] | +0.26 [+0.14, +0.39] | +0.35 [+0.21, +0.47] | +0.03 [-0.09, +0.17] | +0.09 [-0.00, +0.20] | +0.18 [+0.03, +0.34] |
| popqa | 44 | ridge(a=1000)-centered | +0.56 [+0.44, +0.66] | +0.37 [+0.19, +0.52] | +0.51 [+0.37, +0.62] | +0.13 [-0.04, +0.31] | +0.05 [-0.04, +0.13] | +0.19 [+0.04, +0.34] |
| squad | 67 | ridge(a=1000)-centered | +0.37 [+0.26, +0.47] | +0.23 [+0.11, +0.36] | +0.23 [+0.11, +0.34] | +0.25 [+0.12, +0.37] | +0.14 [+0.05, +0.24] | +0.13 [+0.02, +0.24] |
| triviaqa | 71 | gbr(100,d2)-centered | +0.28 [+0.15, +0.40] | +0.33 [+0.21, +0.44] | +0.33 [+0.19, +0.45] | +0.17 [+0.05, +0.30] | -0.05 [-0.13, +0.02] | -0.05 [-0.14, +0.04] |
| webquestions | 67 | gbr(100,d2)-centered | +0.14 [+0.02, +0.27] | +0.07 [-0.05, +0.19] | +0.13 [+0.01, +0.26] | +0.05 [-0.08, +0.18] | +0.01 [-0.11, +0.13] | +0.07 [-0.08, +0.22] |
| **macro mean** | | | +0.333 | +0.227 | +0.289 | +0.117 | | |

Over all 747 held-out queries together (paired, per-query ρ): MPUP − dense +0.04 [+0.02, +0.06], MPUP − BM25 +0.20 [+0.16, +0.24], MPUP − paper config +0.10 [+0.06, +0.13].

Top-1 / NDCG@10, macro mean: MPUP +0.56 / +0.80; paper config +0.50 / +0.77; dense +0.54 / +0.79; BM25 +0.43 / +0.71.

Choosing the best configuration on the held-out dataset itself would have reported ρ +0.357, an optimism of +0.024 over the nested estimate. Selection frequency: {'ridge(a=10)-centered': 1, 'ridge(a=1000)-centered': 4, 'gbr(100,d2)-centered': 6}.

Configuration selected on all datasets (used for the deployable predictor): **gbr(100,d2)-centered**.

## Pooled cross-validation of the paper config (747 rankable queries)

| dataset | queries | paper config | BM25 | dense | shortest-first | paper − dense (paired) |
|---|---|---|---|---|---|---|
| **all** | 747 | +0.21 [+0.18, +0.24] | +0.11 [+0.08, +0.15] | +0.27 [+0.24, +0.31] | -0.03 [-0.07, +0.00] | -0.06 [-0.09, -0.04] |
| 2wikimultihopqa | 94 | +0.21 [+0.15, +0.27] | +0.08 [+0.01, +0.16] | +0.21 [+0.14, +0.27] | -0.21 [-0.28, -0.13] | +0.00 [-0.05, +0.05] |
| ambigqa | 61 | +0.26 [+0.15, +0.37] | +0.08 [-0.05, +0.20] | +0.42 [+0.29, +0.54] | +0.01 [-0.11, +0.13] | -0.16 [-0.27, -0.05] |
| bamboogle | 48 | +0.23 [+0.11, +0.35] | +0.10 [-0.03, +0.23] | +0.28 [+0.13, +0.43] | -0.05 [-0.19, +0.08] | -0.06 [-0.15, +0.04] |
| entityquestions | 65 | +0.27 [+0.17, +0.38] | +0.29 [+0.16, +0.41] | +0.27 [+0.13, +0.41] | +0.03 [-0.09, +0.16] | -0.00 [-0.09, +0.11] |
| hotpotqa | 90 | +0.21 [+0.14, +0.27] | +0.07 [-0.01, +0.15] | +0.28 [+0.21, +0.34] | +0.16 [+0.10, +0.23] | -0.07 [-0.14, -0.01] |
| musique | 84 | +0.15 [+0.11, +0.19] | +0.04 [-0.02, +0.11] | +0.17 [+0.12, +0.23] | -0.08 [-0.13, -0.03] | -0.02 [-0.07, +0.03] |
| nq | 56 | +0.24 [+0.14, +0.35] | +0.03 [-0.09, +0.17] | +0.35 [+0.21, +0.47] | -0.15 [-0.27, -0.02] | -0.11 [-0.23, +0.01] |
| popqa | 44 | +0.40 [+0.27, +0.52] | +0.13 [-0.04, +0.31] | +0.51 [+0.37, +0.62] | -0.08 [-0.23, +0.08] | -0.11 [-0.20, -0.02] |
| squad | 67 | +0.21 [+0.10, +0.31] | +0.25 [+0.12, +0.37] | +0.23 [+0.11, +0.34] | -0.03 [-0.16, +0.09] | -0.02 [-0.13, +0.09] |
| triviaqa | 71 | +0.24 [+0.13, +0.35] | +0.17 [+0.05, +0.30] | +0.33 [+0.19, +0.45] | +0.08 [-0.04, +0.20] | -0.09 [-0.18, -0.00] |
| webquestions | 67 | -0.01 [-0.10, +0.09] | +0.05 [-0.08, +0.18] | +0.13 [+0.01, +0.26] | -0.07 [-0.20, +0.07] | -0.14 [-0.28, -0.01] |

## Feature-group ablation (paired: model − model without the group; positive = the group helps)

Left: the selected configuration (gbr(100,d2)-centered); right: the paper config. Pooled 5-fold CV over queries; the selected configuration was chosen on these datasets, so its column is descriptive.

| group | selected config Δρ | paper config Δρ |
|---|---|---|
| D1 | +0.06 [+0.03, +0.09] | +0.10 [+0.07, +0.13] |
| D2 | +0.00 [-0.00, +0.00] | -0.00 [-0.01, +0.01] |
| D3 | +0.01 [+0.00, +0.03] | -0.01 [-0.02, -0.00] |
| D4 | -0.00 [-0.00, +0.00] | -0.01 [-0.02, +0.01] |
| D5 | +0.02 [+0.00, +0.04] | +0.02 [+0.00, +0.04] |

## Proxy transfer (paper config trained on RAGBench annotation proxy, zero target labels)

| target | queries | MPUP (proxy-trained) | BM25 | dense | MPUP − dense (paired) |
|---|---|---|---|---|---|
| 2wikimultihopqa | 94 | +0.01 [-0.06, +0.08] | +0.08 [+0.01, +0.16] | +0.21 [+0.14, +0.27] | -0.19 [-0.29, -0.09] |
| ambigqa | 61 | +0.32 [+0.20, +0.44] | +0.08 [-0.05, +0.20] | +0.42 [+0.29, +0.54] | -0.10 [-0.21, +0.03] |
| bamboogle | 48 | +0.39 [+0.27, +0.50] | +0.10 [-0.03, +0.23] | +0.28 [+0.13, +0.43] | +0.11 [-0.02, +0.23] |
| entityquestions | 65 | +0.24 [+0.13, +0.35] | +0.29 [+0.16, +0.41] | +0.27 [+0.13, +0.41] | -0.03 [-0.15, +0.09] |
| hotpotqa | 90 | +0.29 [+0.23, +0.34] | +0.07 [-0.01, +0.15] | +0.28 [+0.21, +0.34] | +0.01 [-0.04, +0.06] |
| musique | 84 | +0.09 [+0.04, +0.14] | +0.04 [-0.02, +0.11] | +0.17 [+0.12, +0.23] | -0.08 [-0.14, -0.01] |
| nq | 56 | +0.29 [+0.16, +0.40] | +0.03 [-0.09, +0.17] | +0.35 [+0.21, +0.47] | -0.06 [-0.19, +0.06] |
| popqa | 44 | +0.37 [+0.22, +0.52] | +0.13 [-0.04, +0.31] | +0.51 [+0.37, +0.62] | -0.14 [-0.26, -0.01] |
| squad | 67 | +0.29 [+0.18, +0.40] | +0.25 [+0.12, +0.37] | +0.23 [+0.11, +0.34] | +0.06 [-0.06, +0.18] |
| triviaqa | 71 | +0.24 [+0.12, +0.35] | +0.17 [+0.05, +0.30] | +0.33 [+0.19, +0.45] | -0.09 [-0.19, +0.02] |
| webquestions | 67 | +0.11 [-0.02, +0.23] | +0.05 [-0.08, +0.18] | +0.13 [+0.01, +0.26] | -0.03 [-0.15, +0.09] |
