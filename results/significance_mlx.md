# MPUP vs dense similarity: hypothesis tests

Unit: the query (d = rho_MPUP - rho_dense, within-query Spearman). H0: E[d] = 0, H1: E[d] != 0, alpha = 0.05. Primary test: paired sign-flip permutation test (200,000 flips). Holm correction over the 11 datasets.

| dataset | n | mean d | d_z | MPUP wins | p perm | p Holm | p Wilcoxon | p t-test | Shapiro | MDE (80%) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 89 | -0.015 | -0.06 | 47% | 0.571 | 1.000 | 0.806 | 0.569 | 0.014 | 0.073 |
| ambigqa | 65 | -0.034 | -0.09 | 46% | 0.468 | 1.000 | 0.706 | 0.466 | <0.001 | 0.131 |
| bamboogle | 41 | +0.070 | +0.20 | 67% | 0.215 | 1.000 | 0.197 | 0.216 | 0.002 | 0.155 |
| entityquestions | 61 | +0.056 | +0.19 | 62% | 0.152 | 1.000 | 0.199 | 0.150 | <0.001 | 0.107 |
| hotpotqa | 85 | -0.051 | -0.16 | 46% | 0.136 | 1.000 | 0.080 | 0.136 | 0.013 | 0.095 |
| musique | 78 | +0.044 | +0.24 | 63% | 0.037 | 0.406 | 0.024 | 0.037 | 0.393 | 0.058 |
| nq | 57 | +0.084 | +0.28 | 67% | 0.041 | 0.406 | 0.050 | 0.040 | <0.001 | 0.112 |
| popqa | 43 | +0.039 | +0.15 | 67% | 0.359 | 1.000 | 0.239 | 0.336 | <0.001 | 0.111 |
| squad | 68 | +0.053 | +0.15 | 59% | 0.215 | 1.000 | 0.215 | 0.215 | <0.001 | 0.118 |
| triviaqa | 65 | +0.010 | +0.02 | 57% | 0.847 | 1.000 | 0.348 | 0.843 | <0.001 | 0.137 |
| webquestions | 67 | +0.077 | +0.17 | 60% | 0.178 | 1.000 | 0.168 | 0.177 | 0.017 | 0.158 |
| **pooled** | 719 | +0.025 | +0.07 | 56% | 0.045 | (single test) | 0.019 | 0.045 | <0.001 | 0.034 |

Dataset-level check (units = the 11 per-dataset means; mean +0.030; MPUP ahead on 8 of 11): sign test p = 0.227, Wilcoxon p = 0.054, t-test p = 0.055.

Queries needed for 80% power at the observed pooled difference: 1404.
