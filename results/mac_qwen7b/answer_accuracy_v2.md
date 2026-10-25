# Answer accuracy with the passages each method selects

1100 queries. Acc = 0.5 * relaxed EM + 0.5 * token F1 (Eq. 3); macro mean over datasets.

| method | top-1 | top-3 |
|---|---|---|
| none (no passage) | 0.246 | |
| random | 0.246 | 0.331 |
| bm25 | 0.299 | 0.366 |
| dense | 0.360 | 0.409 |
| mpup | 0.414 | 0.438 |
| oracle | 0.577 | 0.461 |

Paired difference in Acc to dense similarity over all queries (95% bootstrap interval; permutation p; H0: mean difference = 0):

| method | m | mean diff | 95% CI | d_z | p perm | p Wilcoxon | p t-test |
|---|---|---|---|---|---|---|---|
| mpup | 1 | +0.054 | [+0.035, +0.075] | +0.16 | <0.001 | <0.001 | <0.001 |
| bm25 | 1 | -0.061 | [-0.081, -0.041] | -0.18 | <0.001 | <0.001 | <0.001 |
| random | 1 | -0.114 | [-0.140, -0.088] | -0.27 | <0.001 | <0.001 | <0.001 |
| oracle | 1 | +0.217 | [+0.195, +0.240] | +0.58 | <0.001 | <0.001 | <0.001 |
| mpup | 3 | +0.030 | [+0.013, +0.048] | +0.10 | <0.001 | 0.002 | 0.001 |
| bm25 | 3 | -0.043 | [-0.061, -0.022] | -0.13 | <0.001 | <0.001 | <0.001 |
| random | 3 | -0.078 | [-0.100, -0.054] | -0.20 | <0.001 | <0.001 | <0.001 |
| oracle | 3 | +0.052 | [+0.034, +0.072] | +0.16 | <0.001 | <0.001 | <0.001 |

# Same, only on queries whose candidates differ in utility (rankable)

749 queries. Acc = 0.5 * relaxed EM + 0.5 * token F1 (Eq. 3); macro mean over datasets.

| method | top-1 | top-3 |
|---|---|---|
| none (no passage) | 0.288 | |
| random | 0.314 | 0.438 |
| bm25 | 0.388 | 0.483 |
| dense | 0.491 | 0.541 |
| mpup | 0.555 | 0.584 |
| oracle | 0.780 | 0.613 |

Paired difference in Acc to dense similarity over all queries (95% bootstrap interval; permutation p; H0: mean difference = 0):

| method | m | mean diff | 95% CI | d_z | p perm | p Wilcoxon | p t-test |
|---|---|---|---|---|---|---|---|
| mpup | 1 | +0.080 | [+0.052, +0.110] | +0.19 | <0.001 | <0.001 | <0.001 |
| bm25 | 1 | -0.090 | [-0.117, -0.060] | -0.22 | <0.001 | <0.001 | <0.001 |
| random | 1 | -0.168 | [-0.203, -0.132] | -0.34 | <0.001 | <0.001 | <0.001 |
| oracle | 1 | +0.318 | [+0.290, +0.348] | +0.76 | <0.001 | <0.001 | <0.001 |
| mpup | 3 | +0.042 | [+0.016, +0.068] | +0.12 | 0.001 | 0.001 | 0.001 |
| bm25 | 3 | -0.064 | [-0.090, -0.038] | -0.17 | <0.001 | <0.001 | <0.001 |
| random | 3 | -0.115 | [-0.147, -0.080] | -0.25 | <0.001 | <0.001 | <0.001 |
| oracle | 3 | +0.078 | [+0.050, +0.105] | +0.20 | <0.001 | <0.001 | <0.001 |
