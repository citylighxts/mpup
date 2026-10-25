# A second target LLM

Target LLM labels (Eq. 1) on 1714 rankable test questions of the larger study; 2005 development questions for fitting. Variant `gbr(500,d4)|u|ce|probe`, probe layer 24.

Labels: mean u target -0.086 vs Qwen -0.066; nonzero 49% vs 36%; within-query correlation of the two utilities (centred Pearson) 0.596.

| ranking | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| mpup+ (target LLM) | 0.373 | 0.778 | 0.544 |
| mpup (target LLM) | 0.247 | 0.690 | 0.409 |
| mpup+ (Qwen as reader) | 0.322 | 0.757 | 0.536 |
| mpup (Qwen as reader) | 0.258 | 0.696 | 0.412 |
| mpup+ (Qwen probe only) | 0.322 | 0.754 | 0.532 |
| mpup+ (zero-shot from Qwen) | 0.313 | 0.755 | 0.537 |
| probe alone (target LLM) | 0.367 | 0.765 | 0.533 |
| probe, shuffled labels | 0.013 | 0.492 | 0.177 |
| u (Qwen) | 0.512 | 0.787 | 0.613 |
| dense | 0.210 | 0.660 | 0.380 |
| bm25 | 0.095 | 0.571 | 0.285 |
| cross-encoder | 0.243 | 0.688 | 0.397 |

Paired difference of mpup+ (target LLM) (95% bootstrap CI; sign-flip permutation p):

- vs dense: +0.161 [+0.147, +0.176], p = 5e-06
- vs cross-encoder: +0.128 [+0.114, +0.142], p = 5e-06
- vs mpup (target LLM): +0.124 [+0.110, +0.137], p = 5e-06
- vs u (Qwen): -0.140 [-0.159, -0.121], p = 5e-06
- per dataset vs dense: ahead in 10/10, Holm p < 0.05 in 10/10
- per dataset vs cross-encoder: ahead in 10/10, Holm p < 0.05 in 10/10

Paired difference of mpup+ (Qwen as reader) (95% bootstrap CI; sign-flip permutation p):

- vs dense: +0.111 [+0.098, +0.123], p = 5e-06
- vs cross-encoder: +0.078 [+0.066, +0.089], p = 5e-06
- vs mpup (target LLM): +0.073 [+0.061, +0.085], p = 5e-06
- vs mpup+ (target LLM): -0.050 [-0.061, -0.040], p = 5e-06
- vs u (Qwen): -0.190 [-0.208, -0.173], p = 5e-06
- per dataset vs dense: ahead in 10/10, Holm p < 0.05 in 10/10
- per dataset vs cross-encoder: ahead in 10/10, Holm p < 0.05 in 9/10

Paired difference of mpup+ (zero-shot from Qwen) (95% bootstrap CI; sign-flip permutation p):

- vs dense: +0.101 [+0.088, +0.114], p = 5e-06
- vs cross-encoder: +0.068 [+0.056, +0.080], p = 5e-06
- vs mpup (target LLM): +0.063 [+0.052, +0.076], p = 5e-06
- vs mpup+ (target LLM): -0.060 [-0.071, -0.050], p = 5e-06
- vs u (Qwen): -0.200 [-0.217, -0.183], p = 5e-06
- per dataset vs dense: ahead in 10/10, Holm p < 0.05 in 10/10
- per dataset vs cross-encoder: ahead in 10/10, Holm p < 0.05 in 7/10

| dataset | queries | mpup+ (target LLM) | mpup (target LLM) | mpup+ (Qwen as reader) | mpup (Qwen as reader) | mpup+ (Qwen probe only) | mpup+ (zero-shot from Qwen) | probe alone (target LLM) | probe, shuffled labels | u (Qwen) | dense | bm25 | cross-encoder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 182 | 0.344 | 0.262 | 0.329 | 0.286 | 0.319 | 0.290 | 0.347 | 0.030 | 0.471 | 0.236 | 0.132 | 0.252 |
| ambigqa | 155 | 0.456 | 0.302 | 0.373 | 0.304 | 0.378 | 0.363 | 0.453 | -0.005 | 0.495 | 0.273 | 0.133 | 0.282 |
| entityquestions | 186 | 0.449 | 0.347 | 0.420 | 0.374 | 0.415 | 0.425 | 0.441 | -0.053 | 0.662 | 0.302 | 0.215 | 0.354 |
| hotpotqa | 204 | 0.450 | 0.349 | 0.382 | 0.343 | 0.382 | 0.369 | 0.427 | 0.055 | 0.602 | 0.285 | 0.090 | 0.337 |
| musique | 159 | 0.287 | 0.153 | 0.273 | 0.190 | 0.277 | 0.273 | 0.296 | 0.040 | 0.504 | 0.105 | 0.011 | 0.226 |
| nq | 162 | 0.376 | 0.222 | 0.292 | 0.227 | 0.290 | 0.295 | 0.365 | -0.054 | 0.446 | 0.198 | 0.043 | 0.169 |
| popqa | 142 | 0.427 | 0.275 | 0.368 | 0.291 | 0.366 | 0.369 | 0.419 | 0.023 | 0.647 | 0.243 | 0.148 | 0.295 |
| squad | 177 | 0.307 | 0.229 | 0.270 | 0.235 | 0.275 | 0.264 | 0.292 | 0.042 | 0.507 | 0.190 | 0.120 | 0.251 |
| triviaqa | 192 | 0.337 | 0.196 | 0.283 | 0.204 | 0.287 | 0.260 | 0.324 | 0.049 | 0.417 | 0.166 | 0.094 | 0.189 |
| webquestions | 155 | 0.295 | 0.138 | 0.230 | 0.126 | 0.228 | 0.220 | 0.304 | 0.001 | 0.373 | 0.103 | -0.036 | 0.073 |
