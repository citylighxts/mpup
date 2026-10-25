# MPUP vs dense similarity: hypothesis tests

Unit: the query (d = rho_MPUP - rho_dense, within-query Spearman). H0: E[d] = 0, H1: E[d] != 0, alpha = 0.05. Primary test: paired sign-flip permutation test (200,000 flips). Holm correction over the 11 datasets.

| dataset | n | mean d | d_z | MPUP wins | p perm | p Holm | p Wilcoxon | p t-test | Shapiro | MDE (80%) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 94 | +0.037 | +0.19 | 57% | 0.075 | 0.673 | 0.093 | 0.074 | 0.270 | 0.057 |
| ambigqa | 61 | +0.011 | +0.03 | 45% | 0.824 | 1.000 | 0.606 | 0.823 | 0.006 | 0.136 |
| bamboogle | 48 | +0.108 | +0.26 | 60% | 0.078 | 0.673 | 0.128 | 0.077 | <0.001 | 0.168 |
| entityquestions | 65 | +0.048 | +0.13 | 53% | 0.299 | 1.000 | 0.405 | 0.297 | <0.001 | 0.129 |
| hotpotqa | 90 | -0.003 | -0.01 | 53% | 0.902 | 1.000 | 0.908 | 0.900 | 0.001 | 0.070 |
| musique | 84 | +0.051 | +0.27 | 65% | 0.016 | 0.157 | 0.014 | 0.016 | 0.220 | 0.058 |
| nq | 56 | +0.093 | +0.24 | 55% | 0.082 | 0.673 | 0.135 | 0.083 | <0.001 | 0.148 |
| popqa | 44 | +0.048 | +0.16 | 61% | 0.292 | 1.000 | 0.285 | 0.289 | <0.001 | 0.126 |
| squad | 67 | +0.138 | +0.34 | 71% | 0.006 | 0.071 | 0.013 | 0.006 | 0.009 | 0.138 |
| triviaqa | 71 | -0.053 | -0.15 | 47% | 0.200 | 1.000 | 0.179 | 0.199 | <0.001 | 0.115 |
| webquestions | 67 | +0.007 | +0.01 | 55% | 0.905 | 1.000 | 0.799 | 0.905 | 0.064 | 0.165 |
| **pooled** | 747 | +0.040 | +0.12 | 57% | 0.001 | (single test) | 0.002 | 0.001 | <0.001 | 0.035 |

Dataset-level check (units = the 11 per-dataset means; mean +0.044; MPUP ahead on 9 of 11): sign test p = 0.065, Wilcoxon p = 0.032, t-test p = 0.023.

Queries needed for 80% power at the observed pooled difference: 568.
