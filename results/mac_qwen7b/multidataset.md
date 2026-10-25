# MPUP multi-dataset results

Within-query Spearman ρ (mean over queries, 95% bootstrap CI over queries). Pooled CV: 5-fold × 4 repeats over queries. `paper config` is the predictor as stated in the paper (raw features, 200 trees of depth 5) and involves no selection.

Constant (uninformative) features in the pool: source_credibility.

## Datasets

| dataset | label | queries | pairs | passages/q | mean label | non-zero % | positive % | rankable queries |
|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | utility_eq1 | 100 | 1000 | 10.0 | -0.080 | 43 | 15 | 95 |
| ambigqa | utility_eq1 | 100 | 500 | 5.0 | -0.031 | 44 | 19 | 65 |
| bamboogle | utility_eq1 | 100 | 500 | 5.0 | +0.007 | 30 | 16 | 44 |
| entityquestions | utility_eq1 | 100 | 500 | 5.0 | +0.000 | 42 | 21 | 65 |
| hotpotqa | utility_eq1 | 100 | 1000 | 10.0 | +0.023 | 34 | 16 | 90 |
| musique | utility_eq1 | 100 | 1997 | 20.0 | -0.013 | 23 | 9 | 81 |
| nq | utility_eq1 | 100 | 500 | 5.0 | +0.012 | 38 | 19 | 58 |
| popqa | utility_eq1 | 100 | 500 | 5.0 | +0.057 | 26 | 17 | 44 |
| ragbench-hotpotqa | ragbench_proxy | 390 | 1557 | 4.0 | +0.216 | 45 | 45 | 382 |
| squad | utility_eq1 | 100 | 500 | 5.0 | +0.026 | 44 | 20 | 67 |
| triviaqa | utility_eq1 | 100 | 500 | 5.0 | +0.048 | 43 | 24 | 68 |
| webquestions | utility_eq1 | 100 | 500 | 5.0 | -0.076 | 47 | 16 | 72 |

## Headline: nested leave-one-dataset-out

For each held-out dataset the configuration is chosen from the grid using only the other datasets (inner leave-one-dataset-out), refit on them, and scored on the held-out dataset. Grid: ridge(a=10)-raw, ridge(a=100)-raw, ridge(a=1000)-raw, gbr(100,d2)-raw, gbr(200,d3)-raw, gbr(200,d5)-raw, ridge(a=10)-centered, ridge(a=100)-centered, ridge(a=1000)-centered, gbr(100,d2)-centered, gbr(200,d3)-centered, gbr(200,d5)-centered.

| held-out | queries | selected config | MPUP (nested) | paper config | dense | BM25 | MPUP − dense (paired) | MPUP − paper (paired) |
|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 95 | gbr(100,d2)-centered | +0.28 [+0.20, +0.34] | +0.25 [+0.18, +0.32] | +0.24 [+0.18, +0.30] | +0.08 [+0.01, +0.15] | +0.04 [+0.00, +0.07] | +0.02 [-0.04, +0.08] |
| ambigqa | 65 | ridge(a=1000)-centered | +0.45 [+0.34, +0.56] | +0.25 [+0.13, +0.36] | +0.41 [+0.29, +0.53] | +0.10 [-0.02, +0.22] | +0.04 [-0.05, +0.13] | +0.20 [+0.07, +0.34] |
| bamboogle | 44 | ridge(a=100)-centered | +0.24 [+0.09, +0.38] | +0.31 [+0.17, +0.44] | +0.33 [+0.18, +0.47] | +0.07 [-0.08, +0.22] | -0.08 [-0.21, +0.05] | -0.07 [-0.18, +0.06] |
| entityquestions | 65 | gbr(100,d2)-centered | +0.34 [+0.21, +0.46] | +0.23 [+0.11, +0.36] | +0.30 [+0.14, +0.44] | +0.27 [+0.13, +0.40] | +0.04 [-0.05, +0.13] | +0.10 [-0.04, +0.25] |
| hotpotqa | 90 | ridge(a=1000)-centered | +0.30 [+0.22, +0.38] | +0.15 [+0.07, +0.22] | +0.30 [+0.24, +0.37] | +0.12 [+0.03, +0.20] | -0.00 [-0.06, +0.05] | +0.15 [+0.07, +0.23] |
| musique | 81 | gbr(100,d2)-centered | +0.19 [+0.14, +0.25] | +0.15 [+0.10, +0.20] | +0.16 [+0.10, +0.21] | +0.04 [-0.01, +0.11] | +0.04 [-0.00, +0.08] | +0.04 [-0.01, +0.09] |
| nq | 58 | gbr(100,d2)-centered | +0.32 [+0.17, +0.44] | +0.22 [+0.07, +0.37] | +0.24 [+0.11, +0.37] | +0.05 [-0.07, +0.16] | +0.07 [-0.00, +0.16] | +0.09 [-0.05, +0.23] |
| popqa | 44 | ridge(a=1000)-centered | +0.48 [+0.32, +0.61] | +0.33 [+0.18, +0.48] | +0.49 [+0.36, +0.61] | +0.14 [-0.03, +0.32] | -0.01 [-0.12, +0.08] | +0.14 [-0.01, +0.29] |
| squad | 67 | ridge(a=1000)-centered | +0.26 [+0.15, +0.37] | +0.13 [-0.00, +0.27] | +0.19 [+0.06, +0.31] | +0.21 [+0.08, +0.33] | +0.08 [-0.01, +0.16] | +0.13 [+0.01, +0.24] |
| triviaqa | 68 | ridge(a=1000)-centered | +0.29 [+0.16, +0.41] | +0.16 [+0.04, +0.29] | +0.34 [+0.21, +0.46] | +0.12 [-0.01, +0.26] | -0.05 [-0.15, +0.04] | +0.13 [-0.00, +0.25] |
| webquestions | 72 | gbr(100,d2)-centered | +0.08 [-0.04, +0.21] | -0.03 [-0.14, +0.09] | +0.10 [-0.02, +0.22] | -0.01 [-0.14, +0.11] | -0.02 [-0.10, +0.06] | +0.11 [-0.06, +0.27] |
| **macro mean** | | | +0.293 | +0.197 | +0.281 | +0.108 | | |

Over all 749 held-out queries together (paired, per-query ρ): MPUP − dense +0.02 [-0.01, +0.04], MPUP − BM25 +0.18 [+0.14, +0.22], MPUP − paper config +0.10 [+0.06, +0.13].

Top-1 / NDCG@10, macro mean: MPUP +0.55 / +0.80; paper config +0.50 / +0.76; dense +0.54 / +0.79; BM25 +0.42 / +0.71.

Choosing the best configuration on the held-out dataset itself would have reported ρ +0.320, an optimism of +0.027 over the nested estimate. Selection frequency: {'gbr(100,d2)-centered': 5, 'ridge(a=1000)-centered': 5, 'ridge(a=100)-centered': 1}.

Configuration selected on all datasets (used for the deployable predictor): **ridge(a=10)-centered**.

## Pooled cross-validation of the paper config (749 rankable queries)

| dataset | queries | paper config | BM25 | dense | shortest-first | paper − dense (paired) |
|---|---|---|---|---|---|---|
| **all** | 749 | +0.21 [+0.18, +0.24] | +0.11 [+0.07, +0.14] | +0.27 [+0.23, +0.30] | -0.04 [-0.07, -0.01] | -0.05 [-0.08, -0.03] |
| 2wikimultihopqa | 95 | +0.27 [+0.21, +0.33] | +0.08 [+0.01, +0.15] | +0.24 [+0.18, +0.30] | -0.27 [-0.33, -0.20] | +0.04 [-0.01, +0.08] |
| ambigqa | 65 | +0.28 [+0.17, +0.39] | +0.10 [-0.02, +0.22] | +0.41 [+0.29, +0.53] | -0.06 [-0.18, +0.06] | -0.13 [-0.22, -0.04] |
| bamboogle | 44 | +0.26 [+0.13, +0.38] | +0.07 [-0.08, +0.22] | +0.33 [+0.18, +0.47] | -0.02 [-0.16, +0.12] | -0.07 [-0.19, +0.05] |
| entityquestions | 65 | +0.24 [+0.12, +0.35] | +0.27 [+0.13, +0.40] | +0.30 [+0.14, +0.44] | +0.06 [-0.07, +0.19] | -0.06 [-0.15, +0.05] |
| hotpotqa | 90 | +0.26 [+0.20, +0.32] | +0.12 [+0.03, +0.20] | +0.30 [+0.24, +0.37] | +0.16 [+0.10, +0.22] | -0.04 [-0.10, +0.01] |
| musique | 81 | +0.17 [+0.12, +0.21] | +0.04 [-0.01, +0.11] | +0.16 [+0.10, +0.21] | -0.06 [-0.11, -0.01] | +0.01 [-0.04, +0.06] |
| nq | 58 | +0.22 [+0.10, +0.34] | +0.05 [-0.07, +0.16] | +0.24 [+0.11, +0.37] | -0.21 [-0.34, -0.07] | -0.02 [-0.13, +0.09] |
| popqa | 44 | +0.33 [+0.19, +0.47] | +0.14 [-0.03, +0.32] | +0.49 [+0.36, +0.61] | -0.08 [-0.25, +0.08] | -0.16 [-0.28, -0.03] |
| squad | 67 | +0.16 [+0.05, +0.27] | +0.21 [+0.08, +0.33] | +0.19 [+0.06, +0.31] | -0.01 [-0.13, +0.11] | -0.02 [-0.12, +0.08] |
| triviaqa | 68 | +0.20 [+0.11, +0.31] | +0.12 [-0.01, +0.26] | +0.34 [+0.21, +0.46] | +0.07 [-0.06, +0.19] | -0.14 [-0.24, -0.03] |
| webquestions | 72 | -0.00 [-0.09, +0.09] | -0.01 [-0.14, +0.11] | +0.10 [-0.02, +0.22] | -0.00 [-0.12, +0.11] | -0.10 [-0.21, +0.01] |

## Feature-group ablation (paired: model − model without the group; positive = the group helps)

Left: the selected configuration (ridge(a=10)-centered); right: the paper config. Pooled 5-fold CV over queries; the selected configuration was chosen on these datasets, so its column is descriptive.

| group | selected config Δρ | paper config Δρ |
|---|---|---|
| D1 | +0.07 [+0.04, +0.10] | +0.10 [+0.08, +0.13] |
| D2 | -0.00 [-0.01, +0.00] | +0.00 [-0.01, +0.01] |
| D3 | +0.00 [-0.01, +0.02] | -0.01 [-0.03, +0.00] |
| D4 | -0.00 [-0.01, +0.01] | +0.01 [-0.01, +0.02] |
| D5 | +0.01 [-0.01, +0.03] | +0.03 [+0.01, +0.04] |

## Proxy transfer (paper config trained on RAGBench annotation proxy, zero target labels)

| target | queries | MPUP (proxy-trained) | BM25 | dense | MPUP − dense (paired) |
|---|---|---|---|---|---|
| 2wikimultihopqa | 95 | -0.09 [-0.16, -0.02] | +0.08 [+0.01, +0.15] | +0.24 [+0.18, +0.30] | -0.33 [-0.43, -0.23] |
| ambigqa | 65 | +0.30 [+0.17, +0.42] | +0.10 [-0.02, +0.22] | +0.41 [+0.29, +0.53] | -0.11 [-0.20, -0.02] |
| bamboogle | 44 | +0.28 [+0.14, +0.42] | +0.07 [-0.08, +0.22] | +0.33 [+0.18, +0.47] | -0.04 [-0.16, +0.07] |
| entityquestions | 65 | +0.20 [+0.07, +0.33] | +0.27 [+0.13, +0.40] | +0.30 [+0.14, +0.44] | -0.09 [-0.22, +0.02] |
| hotpotqa | 90 | +0.35 [+0.28, +0.40] | +0.12 [+0.03, +0.20] | +0.30 [+0.24, +0.37] | +0.04 [-0.01, +0.10] |
| musique | 81 | +0.09 [+0.04, +0.15] | +0.04 [-0.01, +0.11] | +0.16 [+0.10, +0.21] | -0.06 [-0.13, -0.00] |
| nq | 58 | +0.18 [+0.05, +0.31] | +0.05 [-0.07, +0.16] | +0.24 [+0.11, +0.37] | -0.06 [-0.18, +0.06] |
| popqa | 44 | +0.38 [+0.22, +0.52] | +0.14 [-0.03, +0.32] | +0.49 [+0.36, +0.61] | -0.12 [-0.27, +0.03] |
| squad | 67 | +0.15 [+0.03, +0.26] | +0.21 [+0.08, +0.33] | +0.19 [+0.06, +0.31] | -0.04 [-0.15, +0.08] |
| triviaqa | 68 | +0.19 [+0.05, +0.33] | +0.12 [-0.01, +0.26] | +0.34 [+0.21, +0.46] | -0.15 [-0.27, -0.04] |
| webquestions | 72 | +0.09 [-0.04, +0.22] | -0.01 [-0.14, +0.11] | +0.10 [-0.02, +0.22] | -0.01 [-0.12, +0.09] |
