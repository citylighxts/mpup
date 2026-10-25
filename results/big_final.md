# Held-out test questions

Variant `gbr(500,d4)|u|ce|probe` (chosen on the development questions only).

## Trained on the development questions of all datasets

1641 rankable test questions. Macro within-query Spearman; NDCG@10 and top-1 over queries.

| model | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| mpup+ | 0.356 | 0.780 | 0.618 |
| mpup (paper) | 0.280 | 0.709 | 0.505 |
| dense | 0.226 | 0.659 | 0.459 |
| bm25 | 0.106 | 0.557 | 0.356 |
| cross-encoder | 0.260 | 0.697 | 0.490 |

Paired difference of mpup+ over all test queries (95% bootstrap CI; sign-flip permutation p; d_z):

- vs mpup (paper): +0.075 [+0.064, +0.086], p = 5e-06, d_z = +0.32
- vs dense: +0.130 [+0.116, +0.144], p = 5e-06, d_z = +0.45
- vs bm25: +0.249 [+0.231, +0.266], p = 5e-06, d_z = +0.67
- vs cross-encoder: +0.095 [+0.083, +0.108], p = 5e-06, d_z = +0.37

| dataset | queries | mpup+ | mpup (paper) | dense | bm25 | cross-encoder | p vs dense (Holm) |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 170 | 0.247 | 0.208 | 0.159 | 0.057 | 0.182 | 0.002 (0.002) |
| ambigqa | 147 | 0.405 | 0.331 | 0.309 | 0.151 | 0.317 | 0.000 (0.000) |
| entityquestions | 182 | 0.429 | 0.375 | 0.308 | 0.220 | 0.351 | 0.000 (0.000) |
| hotpotqa | 197 | 0.463 | 0.387 | 0.324 | 0.146 | 0.362 | 0.000 (0.000) |
| musique | 163 | 0.311 | 0.214 | 0.124 | 0.028 | 0.223 | 0.000 (0.000) |
| nq | 151 | 0.340 | 0.248 | 0.226 | 0.037 | 0.217 | 0.000 (0.000) |
| popqa | 134 | 0.390 | 0.312 | 0.252 | 0.146 | 0.300 | 0.000 (0.000) |
| squad | 171 | 0.295 | 0.250 | 0.193 | 0.119 | 0.250 | 0.000 (0.000) |
| triviaqa | 185 | 0.403 | 0.303 | 0.252 | 0.170 | 0.283 | 0.000 (0.000) |
| webquestions | 141 | 0.281 | 0.172 | 0.115 | -0.010 | 0.112 | 0.000 (0.000) |

## Trained on the development questions of the other datasets only

1641 rankable test questions. Macro within-query Spearman; NDCG@10 and top-1 over queries.

| model | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| mpup+ | 0.351 | 0.772 | 0.598 |
| mpup (paper) | 0.276 | 0.703 | 0.499 |
| dense | 0.226 | 0.659 | 0.459 |
| bm25 | 0.106 | 0.557 | 0.356 |
| cross-encoder | 0.260 | 0.697 | 0.490 |

Paired difference of mpup+ over all test queries (95% bootstrap CI; sign-flip permutation p; d_z):

- vs mpup (paper): +0.075 [+0.064, +0.085], p = 5e-06, d_z = +0.32
- vs dense: +0.124 [+0.111, +0.138], p = 5e-06, d_z = +0.44
- vs bm25: +0.243 [+0.226, +0.261], p = 5e-06, d_z = +0.67
- vs cross-encoder: +0.090 [+0.078, +0.102], p = 5e-06, d_z = +0.35

| dataset | queries | mpup+ | mpup (paper) | dense | bm25 | cross-encoder | p vs dense (Holm) |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 170 | 0.243 | 0.201 | 0.159 | 0.057 | 0.182 | 0.002 (0.002) |
| ambigqa | 147 | 0.406 | 0.330 | 0.309 | 0.151 | 0.317 | 0.000 (0.000) |
| entityquestions | 182 | 0.425 | 0.371 | 0.308 | 0.220 | 0.351 | 0.000 (0.000) |
| hotpotqa | 197 | 0.453 | 0.370 | 0.324 | 0.146 | 0.362 | 0.000 (0.000) |
| musique | 163 | 0.296 | 0.210 | 0.124 | 0.028 | 0.223 | 0.000 (0.000) |
| nq | 151 | 0.338 | 0.249 | 0.226 | 0.037 | 0.217 | 0.000 (0.000) |
| popqa | 134 | 0.378 | 0.311 | 0.252 | 0.146 | 0.300 | 0.000 (0.000) |
| squad | 171 | 0.298 | 0.252 | 0.193 | 0.119 | 0.250 | 0.000 (0.000) |
| triviaqa | 185 | 0.403 | 0.297 | 0.252 | 0.170 | 0.283 | 0.000 (0.000) |
| webquestions | 141 | 0.265 | 0.166 | 0.115 | -0.010 | 0.112 | 0.000 (0.000) |
