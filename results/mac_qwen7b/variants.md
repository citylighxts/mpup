# MPUP multi-dataset results

Within-query Spearman ρ (mean over queries, 95% bootstrap CI over queries). Pooled CV: 5-fold × 1 repeats over queries. `paper config` is the predictor as stated in the paper (raw features, 200 trees of depth 5) and involves no selection.

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

For each held-out dataset the configuration is chosen from the grid using only the other datasets (inner leave-one-dataset-out), refit on them, and scored on the held-out dataset. Grid: base, base-noD2D4, base+probe, base-rank, base+smooth, combined.

| held-out | queries | selected config | MPUP (nested) | paper config | dense | BM25 | MPUP − dense (paired) | MPUP − paper (paired) |
|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 95 | combined | +0.23 [+0.16, +0.30] | +0.25 [+0.18, +0.32] | +0.24 [+0.18, +0.30] | +0.08 [+0.01, +0.15] | -0.01 [-0.06, +0.05] | -0.02 [-0.09, +0.05] |
| ambigqa | 65 | combined | +0.46 [+0.37, +0.56] | +0.25 [+0.13, +0.37] | +0.41 [+0.29, +0.53] | +0.10 [-0.02, +0.22] | +0.05 [-0.05, +0.16] | +0.21 [+0.08, +0.35] |
| bamboogle | 44 | base+probe | +0.38 [+0.23, +0.52] | +0.31 [+0.17, +0.44] | +0.33 [+0.18, +0.47] | +0.07 [-0.08, +0.22] | +0.05 [-0.09, +0.19] | +0.07 [-0.09, +0.23] |
| entityquestions | 65 | base+probe | +0.45 [+0.34, +0.55] | +0.23 [+0.09, +0.35] | +0.30 [+0.14, +0.44] | +0.27 [+0.13, +0.40] | +0.15 [+0.05, +0.27] | +0.22 [+0.10, +0.34] |
| hotpotqa | 90 | base+probe | +0.42 [+0.35, +0.48] | +0.16 [+0.08, +0.24] | +0.30 [+0.24, +0.37] | +0.12 [+0.03, +0.20] | +0.11 [+0.06, +0.17] | +0.25 [+0.17, +0.34] |
| musique | 81 | base+probe | +0.25 [+0.19, +0.30] | +0.16 [+0.10, +0.21] | +0.16 [+0.10, +0.21] | +0.04 [-0.01, +0.11] | +0.09 [+0.04, +0.14] | +0.09 [+0.03, +0.14] |
| nq | 58 | base+probe | +0.36 [+0.24, +0.48] | +0.22 [+0.07, +0.37] | +0.24 [+0.11, +0.37] | +0.05 [-0.07, +0.16] | +0.12 [+0.03, +0.22] | +0.14 [+0.00, +0.29] |
| popqa | 44 | combined | +0.59 [+0.46, +0.69] | +0.34 [+0.19, +0.48] | +0.49 [+0.36, +0.61] | +0.14 [-0.03, +0.32] | +0.10 [-0.01, +0.21] | +0.24 [+0.11, +0.38] |
| squad | 67 | base+probe | +0.35 [+0.24, +0.46] | +0.14 [-0.00, +0.28] | +0.19 [+0.06, +0.31] | +0.21 [+0.08, +0.33] | +0.17 [+0.08, +0.26] | +0.21 [+0.08, +0.34] |
| triviaqa | 68 | base+probe | +0.33 [+0.20, +0.46] | +0.20 [+0.08, +0.34] | +0.34 [+0.21, +0.46] | +0.12 [-0.01, +0.26] | -0.01 [-0.11, +0.10] | +0.13 [+0.00, +0.25] |
| webquestions | 72 | base+probe | +0.19 [+0.07, +0.31] | -0.03 [-0.14, +0.09] | +0.10 [-0.02, +0.22] | -0.01 [-0.14, +0.11] | +0.09 [-0.03, +0.22] | +0.22 [+0.05, +0.39] |
| **macro mean** | | | +0.365 | +0.204 | +0.281 | +0.108 | | |

Over all 749 held-out queries together (paired, per-query ρ): MPUP − dense +0.08 [+0.05, +0.11], MPUP − BM25 +0.24 [+0.21, +0.28], MPUP − paper config +0.16 [+0.12, +0.19].

Top-1 / NDCG@10, macro mean: MPUP +0.62 / +0.83; paper config +0.50 / +0.76; dense +0.54 / +0.79; BM25 +0.42 / +0.71.

Choosing the best configuration on the held-out dataset itself would have reported ρ +0.373, an optimism of +0.008 over the nested estimate. Selection frequency: {'combined': 3, 'base+probe': 8}.

Configuration selected on all datasets (used for the deployable predictor): **base+probe**.

## Pooled cross-validation of the paper config (749 rankable queries)

| dataset | queries | paper config | BM25 | dense | shortest-first | paper − dense (paired) |
|---|---|---|---|---|---|---|
| **all** | 749 | +0.20 [+0.17, +0.23] | +0.11 [+0.07, +0.14] | +0.27 [+0.23, +0.30] | -0.04 [-0.07, -0.01] | -0.07 [-0.10, -0.04] |
| 2wikimultihopqa | 95 | +0.25 [+0.18, +0.32] | +0.08 [+0.01, +0.15] | +0.24 [+0.18, +0.30] | -0.27 [-0.33, -0.20] | +0.01 [-0.05, +0.07] |
| ambigqa | 65 | +0.27 [+0.15, +0.39] | +0.10 [-0.02, +0.22] | +0.41 [+0.29, +0.53] | -0.06 [-0.18, +0.06] | -0.14 [-0.25, -0.03] |
| bamboogle | 44 | +0.24 [+0.10, +0.37] | +0.07 [-0.08, +0.22] | +0.33 [+0.18, +0.47] | -0.02 [-0.16, +0.12] | -0.09 [-0.25, +0.06] |
| entityquestions | 65 | +0.23 [+0.10, +0.36] | +0.27 [+0.13, +0.40] | +0.30 [+0.14, +0.44] | +0.06 [-0.07, +0.19] | -0.06 [-0.19, +0.08] |
| hotpotqa | 90 | +0.22 [+0.15, +0.29] | +0.12 [+0.03, +0.20] | +0.30 [+0.24, +0.37] | +0.16 [+0.10, +0.22] | -0.08 [-0.15, -0.01] |
| musique | 81 | +0.18 [+0.12, +0.23] | +0.04 [-0.01, +0.11] | +0.16 [+0.10, +0.21] | -0.06 [-0.11, -0.01] | +0.02 [-0.04, +0.08] |
| nq | 58 | +0.18 [+0.04, +0.32] | +0.05 [-0.07, +0.16] | +0.24 [+0.11, +0.37] | -0.21 [-0.34, -0.07] | -0.06 [-0.18, +0.07] |
| popqa | 44 | +0.30 [+0.15, +0.45] | +0.14 [-0.03, +0.32] | +0.49 [+0.36, +0.61] | -0.08 [-0.25, +0.08] | -0.19 [-0.33, -0.05] |
| squad | 67 | +0.20 [+0.07, +0.32] | +0.21 [+0.08, +0.33] | +0.19 [+0.06, +0.31] | -0.01 [-0.13, +0.11] | +0.01 [-0.10, +0.12] |
| triviaqa | 68 | +0.12 [-0.00, +0.25] | +0.12 [-0.01, +0.26] | +0.34 [+0.21, +0.46] | +0.07 [-0.06, +0.19] | -0.22 [-0.34, -0.11] |
| webquestions | 72 | +0.06 [-0.06, +0.18] | -0.01 [-0.14, +0.11] | +0.10 [-0.02, +0.22] | -0.00 [-0.12, +0.11] | -0.04 [-0.18, +0.10] |

## Feature-group ablation (paired: model − model without the group; positive = the group helps)

Left: the selected configuration (base+probe); right: the paper config. Pooled 5-fold CV over queries; the selected configuration was chosen on these datasets, so its column is descriptive.

| group | selected config Δρ | paper config Δρ |
|---|---|---|
| D1 | n/a (variant configuration) | +0.10 [+0.07, +0.14] |
| D2 | n/a (variant configuration) | -0.01 [-0.03, +0.01] |
| D3 | n/a (variant configuration) | -0.03 [-0.05, -0.00] |
| D4 | n/a (variant configuration) | +0.00 [-0.02, +0.03] |
| D5 | n/a (variant configuration) | +0.03 [+0.00, +0.05] |

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
