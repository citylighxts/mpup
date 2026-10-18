# MPUP multi-dataset results

Within-query Spearman ρ (mean over queries, 95% bootstrap CI over queries). Pooled CV: 5-fold × 5 repeats over queries. `paper config` is the predictor as stated in the paper (raw features, 200 trees of depth 5) and involves no selection.

Constant (uninformative) features in the pool: source_credibility.

## Datasets

| dataset | label | queries | pairs | passages/q | mean label | non-zero % | positive % | rankable queries |
|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | utility_eq1 | 100 | 1000 | 10.0 | -0.029 | 37 | 16 | 89 |
| ambigqa | utility_eq1 | 100 | 500 | 5.0 | +0.001 | 45 | 21 | 65 |
| bamboogle | utility_eq1 | 100 | 500 | 5.0 | +0.022 | 22 | 13 | 41 |
| entityquestions | utility_eq1 | 100 | 500 | 5.0 | +0.068 | 33 | 22 | 61 |
| hotpotqa | utility_eq1 | 100 | 1000 | 10.0 | +0.016 | 31 | 16 | 85 |
| musique | utility_eq1 | 100 | 1997 | 20.0 | +0.002 | 22 | 9 | 78 |
| nq | utility_eq1 | 100 | 500 | 5.0 | -0.058 | 38 | 13 | 57 |
| popqa | utility_eq1 | 100 | 500 | 5.0 | +0.048 | 31 | 18 | 43 |
| ragbench-hotpotqa | ragbench_proxy | 390 | 1557 | 4.0 | +0.216 | 45 | 45 | 382 |
| squad | utility_eq1 | 100 | 500 | 5.0 | -0.002 | 42 | 21 | 68 |
| triviaqa | utility_eq1 | 100 | 500 | 5.0 | +0.000 | 41 | 20 | 65 |
| webquestions | utility_eq1 | 100 | 500 | 5.0 | -0.147 | 51 | 14 | 67 |

## Headline: nested leave-one-dataset-out

For each held-out dataset the configuration is chosen from the grid using only the other datasets (inner leave-one-dataset-out), refit on them, and scored on the held-out dataset. Grid: ridge(a=10)-raw, ridge(a=100)-raw, ridge(a=1000)-raw, gbr(100,d2)-raw, gbr(200,d3)-raw, gbr(200,d5)-raw, ridge(a=10)-centered, ridge(a=100)-centered, ridge(a=1000)-centered, gbr(100,d2)-centered, gbr(200,d3)-centered, gbr(200,d5)-centered.

| held-out | queries | selected config | MPUP (nested) | paper config | dense | BM25 | MPUP − dense (paired) | MPUP − paper (paired) |
|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 89 | ridge(a=10)-centered | +0.18 [+0.11, +0.25] | +0.12 [+0.04, +0.20] | +0.19 [+0.12, +0.26] | +0.08 [+0.00, +0.16] | -0.01 [-0.07, +0.03] | +0.06 [-0.02, +0.13] |
| ambigqa | 65 | ridge(a=1000)-centered | +0.35 [+0.21, +0.47] | +0.26 [+0.14, +0.38] | +0.38 [+0.26, +0.51] | +0.11 [-0.01, +0.23] | -0.03 [-0.13, +0.05] | +0.09 [-0.07, +0.24] |
| bamboogle | 41 | gbr(100,d2)-centered | +0.35 [+0.19, +0.50] | +0.26 [+0.10, +0.42] | +0.28 [+0.13, +0.42] | +0.14 [-0.00, +0.27] | +0.07 [-0.04, +0.18] | +0.09 [-0.07, +0.25] |
| entityquestions | 61 | gbr(100,d2)-centered | +0.36 [+0.22, +0.50] | +0.31 [+0.18, +0.44] | +0.30 [+0.15, +0.45] | +0.26 [+0.12, +0.40] | +0.06 [-0.02, +0.13] | +0.05 [-0.06, +0.16] |
| hotpotqa | 85 | ridge(a=1000)-centered | +0.25 [+0.17, +0.32] | +0.24 [+0.18, +0.31] | +0.30 [+0.22, +0.37] | +0.10 [+0.01, +0.19] | -0.05 [-0.12, +0.02] | +0.01 [-0.07, +0.08] |
| musique | 78 | ridge(a=10)-centered | +0.19 [+0.14, +0.24] | +0.13 [+0.08, +0.18] | +0.15 [+0.09, +0.20] | +0.04 [-0.02, +0.09] | +0.04 [+0.00, +0.08] | +0.06 [+0.01, +0.11] |
| nq | 57 | gbr(100,d2)-centered | +0.32 [+0.20, +0.44] | +0.13 [+0.00, +0.27] | +0.23 [+0.10, +0.36] | +0.02 [-0.11, +0.14] | +0.08 [+0.01, +0.16] | +0.19 [+0.07, +0.31] |
| popqa | 43 | gbr(100,d2)-centered | +0.49 [+0.36, +0.61] | +0.39 [+0.27, +0.51] | +0.45 [+0.31, +0.58] | +0.16 [-0.03, +0.33] | +0.04 [-0.04, +0.11] | +0.09 [-0.02, +0.21] |
| squad | 68 | gbr(100,d2)-centered | +0.32 [+0.19, +0.44] | +0.23 [+0.12, +0.33] | +0.27 [+0.15, +0.38] | +0.27 [+0.14, +0.39] | +0.05 [-0.03, +0.13] | +0.09 [-0.03, +0.22] |
| triviaqa | 65 | gbr(100,d2)-centered | +0.29 [+0.16, +0.42] | +0.22 [+0.08, +0.35] | +0.28 [+0.15, +0.41] | +0.05 [-0.09, +0.18] | +0.01 [-0.09, +0.10] | +0.07 [-0.05, +0.20] |
| webquestions | 67 | ridge(a=1000)-centered | +0.07 [-0.06, +0.20] | +0.03 [-0.10, +0.16] | -0.01 [-0.15, +0.13] | -0.00 [-0.13, +0.13] | +0.08 [-0.03, +0.19] | +0.04 [-0.09, +0.17] |
| **macro mean** | | | +0.286 | +0.211 | +0.256 | +0.110 | | |

Over all 719 held-out queries together (paired, per-query ρ): MPUP − dense +0.02 [+0.00, +0.05], MPUP − BM25 +0.16 [+0.12, +0.20], MPUP − paper config +0.07 [+0.04, +0.10].

Top-1 / NDCG@10, macro mean: MPUP +0.54 / +0.79; paper config +0.49 / +0.76; dense +0.52 / +0.78; BM25 +0.41 / +0.72.

Choosing the best configuration on the held-out dataset itself would have reported ρ +0.314, an optimism of +0.028 over the nested estimate. Selection frequency: {'ridge(a=10)-centered': 2, 'ridge(a=1000)-centered': 3, 'gbr(100,d2)-centered': 6}.

Configuration selected on all datasets (used for the deployable predictor): **gbr(100,d2)-centered**.

## Pooled cross-validation of the paper config (719 rankable queries)

| dataset | queries | paper config | BM25 | dense | shortest-first | paper − dense (paired) |
|---|---|---|---|---|---|---|
| **all** | 719 | +0.20 [+0.17, +0.23] | +0.11 [+0.07, +0.14] | +0.25 [+0.21, +0.28] | -0.03 [-0.07, +0.00] | -0.04 [-0.07, -0.01] |
| 2wikimultihopqa | 89 | +0.18 [+0.11, +0.24] | +0.08 [+0.00, +0.16] | +0.19 [+0.12, +0.26] | -0.23 [-0.31, -0.15] | -0.01 [-0.07, +0.05] |
| ambigqa | 65 | +0.21 [+0.11, +0.31] | +0.11 [-0.01, +0.23] | +0.38 [+0.26, +0.51] | -0.02 [-0.13, +0.08] | -0.17 [-0.27, -0.07] |
| bamboogle | 41 | +0.26 [+0.11, +0.40] | +0.14 [-0.00, +0.27] | +0.28 [+0.13, +0.42] | -0.16 [-0.32, +0.00] | -0.02 [-0.14, +0.10] |
| entityquestions | 61 | +0.33 [+0.21, +0.44] | +0.26 [+0.12, +0.40] | +0.30 [+0.15, +0.45] | +0.05 [-0.10, +0.20] | +0.02 [-0.05, +0.11] |
| hotpotqa | 85 | +0.23 [+0.17, +0.29] | +0.10 [+0.01, +0.19] | +0.30 [+0.22, +0.37] | +0.18 [+0.10, +0.25] | -0.07 [-0.14, -0.01] |
| musique | 78 | +0.10 [+0.06, +0.15] | +0.04 [-0.02, +0.09] | +0.15 [+0.09, +0.20] | -0.06 [-0.11, -0.01] | -0.04 [-0.10, +0.01] |
| nq | 57 | +0.15 [+0.03, +0.26] | +0.02 [-0.11, +0.14] | +0.23 [+0.10, +0.36] | -0.07 [-0.19, +0.05] | -0.09 [-0.19, +0.02] |
| popqa | 43 | +0.36 [+0.23, +0.47] | +0.16 [-0.03, +0.33] | +0.45 [+0.31, +0.58] | -0.05 [-0.20, +0.09] | -0.09 [-0.19, +0.01] |
| squad | 68 | +0.26 [+0.16, +0.35] | +0.27 [+0.14, +0.39] | +0.27 [+0.15, +0.38] | +0.05 [-0.06, +0.16] | -0.01 [-0.14, +0.11] |
| triviaqa | 65 | +0.18 [+0.07, +0.29] | +0.05 [-0.09, +0.18] | +0.28 [+0.15, +0.41] | -0.02 [-0.14, +0.11] | -0.10 [-0.20, -0.00] |
| webquestions | 67 | +0.08 [-0.03, +0.19] | -0.00 [-0.13, +0.13] | -0.01 [-0.15, +0.13] | -0.07 [-0.19, +0.05] | +0.09 [-0.05, +0.23] |

## Feature-group ablation (paired: model − model without the group; positive = the group helps)

Left: the selected configuration (gbr(100,d2)-centered); right: the paper config. Pooled 5-fold CV over queries; the selected configuration was chosen on these datasets, so its column is descriptive.

| group | selected config Δρ | paper config Δρ |
|---|---|---|
| D1 | +0.05 [+0.02, +0.07] | +0.08 [+0.05, +0.10] |
| D2 | +0.00 [-0.00, +0.00] | +0.01 [-0.00, +0.02] |
| D3 | +0.02 [+0.00, +0.03] | +0.00 [-0.01, +0.02] |
| D4 | +0.00 [-0.00, +0.01] | -0.02 [-0.03, -0.00] |
| D5 | +0.01 [-0.00, +0.03] | +0.03 [+0.02, +0.05] |

## Proxy transfer (paper config trained on RAGBench annotation proxy, zero target labels)

| target | queries | MPUP (proxy-trained) | BM25 | dense | MPUP − dense (paired) |
|---|---|---|---|---|---|
| 2wikimultihopqa | 89 | -0.02 [-0.09, +0.05] | +0.08 [+0.00, +0.16] | +0.19 [+0.12, +0.26] | -0.21 [-0.30, -0.12] |
| ambigqa | 65 | +0.23 [+0.11, +0.34] | +0.11 [-0.01, +0.23] | +0.38 [+0.26, +0.51] | -0.16 [-0.28, -0.04] |
| bamboogle | 41 | +0.31 [+0.18, +0.44] | +0.14 [-0.00, +0.27] | +0.28 [+0.13, +0.42] | +0.03 [-0.07, +0.13] |
| entityquestions | 61 | +0.24 [+0.10, +0.36] | +0.26 [+0.12, +0.40] | +0.30 [+0.15, +0.45] | -0.06 [-0.20, +0.07] |
| hotpotqa | 85 | +0.35 [+0.28, +0.42] | +0.10 [+0.01, +0.19] | +0.30 [+0.22, +0.37] | +0.05 [-0.01, +0.12] |
| musique | 78 | +0.08 [+0.03, +0.13] | +0.04 [-0.02, +0.09] | +0.15 [+0.09, +0.20] | -0.06 [-0.13, -0.00] |
| nq | 57 | +0.22 [+0.09, +0.35] | +0.02 [-0.11, +0.14] | +0.23 [+0.10, +0.36] | -0.02 [-0.14, +0.12] |
| popqa | 43 | +0.23 [+0.08, +0.39] | +0.16 [-0.03, +0.33] | +0.45 [+0.31, +0.58] | -0.22 [-0.38, -0.07] |
| squad | 68 | +0.27 [+0.14, +0.38] | +0.27 [+0.14, +0.39] | +0.27 [+0.15, +0.38] | -0.00 [-0.09, +0.09] |
| triviaqa | 65 | +0.20 [+0.08, +0.33] | +0.05 [-0.09, +0.18] | +0.28 [+0.15, +0.41] | -0.08 [-0.20, +0.05] |
| webquestions | 67 | +0.03 [-0.09, +0.14] | -0.00 [-0.13, +0.13] | -0.01 [-0.15, +0.13] | +0.04 [-0.09, +0.16] |
