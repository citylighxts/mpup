# Frozen configuration on fresh questions

Configuration `gbr(100,d2)-centered` fitted on the original questions of 11 datasets, scored on 693 rankable questions from a second sample (another seed). Within-query Spearman, macro-averaged over datasets.

| model | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| gbr(100,d2)-centered | 0.312 | 0.810 | 0.574 |
| paper config | 0.186 | 0.750 | 0.463 |
| bm25 | 0.155 | 0.725 | 0.437 |
| dense | 0.277 | 0.775 | 0.502 |
| shortest | 0.051 | 0.668 | 0.381 |

Paired difference of the frozen configuration over all fresh queries (95% bootstrap CI):

- vs paper config: +0.113 [+0.079, +0.147]
- vs bm25: +0.159 [+0.119, +0.199]
- vs dense: +0.052 [+0.026, +0.079]
- vs shortest: +0.292 [+0.244, +0.337]

| dataset | queries | gbr(100,d2)-centered | paper config | bm25 | dense | shortest |
|---|---|---|---|---|---|---|
| 2wikimultihopqa | 87 | 0.289 | 0.201 | 0.153 | 0.187 | -0.105 |
| ambigqa | 55 | 0.409 | 0.302 | 0.258 | 0.327 | 0.045 |
| bamboogle | 6 | 0.153 | -0.111 | -0.029 | 0.309 | 0.193 |
| entityquestions | 64 | 0.479 | 0.378 | 0.247 | 0.411 | 0.105 |
| hotpotqa | 93 | 0.342 | 0.235 | 0.107 | 0.320 | 0.089 |
| musique | 77 | 0.232 | 0.114 | 0.063 | 0.135 | -0.119 |
| nq | 62 | 0.210 | 0.141 | 0.087 | 0.274 | -0.038 |
| popqa | 50 | 0.376 | 0.243 | 0.291 | 0.280 | 0.149 |
| squad | 68 | 0.437 | 0.281 | 0.356 | 0.404 | 0.089 |
| triviaqa | 68 | 0.306 | 0.138 | 0.127 | 0.215 | 0.003 |
| webquestions | 63 | 0.195 | 0.122 | 0.043 | 0.188 | 0.146 |
