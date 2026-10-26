# Frozen configuration on fresh questions

Configuration `base+probe` fitted on the original questions of 11 datasets, scored on 661 rankable questions from a second sample (another seed). Within-query Spearman, macro-averaged over datasets.

| model | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| base+probe | 0.387 | 0.848 | 0.646 |
| paper config | 0.163 | 0.736 | 0.418 |
| bm25 | 0.126 | 0.708 | 0.395 |
| dense | 0.275 | 0.769 | 0.495 |
| shortest | 0.045 | 0.668 | 0.415 |

Paired difference of the frozen configuration over all fresh queries (95% bootstrap CI):

- vs paper config: +0.183 [+0.146, +0.221]
- vs bm25: +0.245 [+0.203, +0.284]
- vs dense: +0.090 [+0.061, +0.119]
- vs shortest: +0.386 [+0.343, +0.431]

Pre-declared criterion (pooled gain over dense of at least +0.08 with the interval above zero): **criterion met** (gain +0.090, 95% CI [+0.061, +0.119]).

| dataset | queries | base+probe | paper config | bm25 | dense | shortest |
|---|---|---|---|---|---|---|
| 2wikimultihopqa | 85 | 0.290 | 0.171 | 0.065 | 0.220 | -0.195 |
| ambigqa | 52 | 0.441 | 0.227 | 0.089 | 0.393 | -0.028 |
| bamboogle | 2 | 0.498 | -0.144 | 0.000 | 0.144 | 0.661 |
| entityquestions | 68 | 0.507 | 0.363 | 0.304 | 0.459 | 0.010 |
| hotpotqa | 86 | 0.414 | 0.206 | 0.055 | 0.283 | 0.139 |
| musique | 74 | 0.273 | 0.089 | -0.009 | 0.102 | -0.082 |
| nq | 59 | 0.380 | 0.211 | 0.160 | 0.276 | -0.032 |
| popqa | 44 | 0.451 | 0.303 | 0.326 | 0.297 | -0.090 |
| squad | 66 | 0.437 | 0.177 | 0.178 | 0.354 | 0.119 |
| triviaqa | 67 | 0.434 | 0.241 | 0.193 | 0.352 | 0.016 |
| webquestions | 58 | 0.134 | -0.046 | 0.029 | 0.143 | -0.020 |
