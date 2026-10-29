# C4: MPUP against utility estimates that query the target LLM

1641 rankable test questions; target LLM Qwen/Qwen2.5-7B-Instruct; within-query Spearman with its Eq. (1) utility. Time per method (s, all 37710 test pairs): {"pseudo": 1254.8, "PVS w/o A": 1681.3, "PVS w/ A": 1743.9, "likelihood": 1490.3, "answer confidence": 1511.3, "LVR ranked share": 0.8625563511005039, "LVR w/ A": 2323.0, "peak_gpu_gib": 23.94}

| ranking | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| MPUP | 0.356 | 0.780 | 0.618 |
| PVS w/o A | 0.269 | 0.708 | 0.517 |
| PVS w/ A | 0.287 | 0.718 | 0.544 |
| likelihood | 0.253 | 0.682 | 0.520 |
| LVR w/ A | 0.219 | 0.667 | 0.509 |
| answer confidence | -0.060 | 0.383 | 0.081 |
| ΔH (D5) | 0.142 | 0.553 | 0.309 |
| dense | 0.226 | 0.659 | 0.459 |
| bm25 | 0.106 | 0.557 | 0.356 |
| cross-encoder | 0.260 | 0.697 | 0.490 |

Paired difference of MPUP (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.086 [+0.074, +0.097], p = 5e-06, p_t = 3.5e-45, d_z = 0.36
- vs PVS w/ A: +0.067 [+0.054, +0.081], p = 5e-06, p_t = 1e-22, d_z = 0.25
- vs likelihood: +0.102 [+0.082, +0.122], p = 5e-06, p_t = 9e-23, d_z = 0.25
- vs LVR w/ A: +0.134 [+0.119, +0.150], p = 5e-06, p_t = 3.1e-63, d_z = 0.43
- vs answer confidence: +0.422 [+0.394, +0.448], p = 5e-06, p_t = 3.9e-162, d_z = 0.75
- vs ΔH (D5): +0.214 [+0.198, +0.230], p = 5e-06, p_t = 1.7e-120, d_z = 0.63
- vs dense: +0.130 [+0.116, +0.144], p = 5e-06, p_t = 1.6e-67, d_z = 0.45
- vs bm25: +0.249 [+0.231, +0.266], p = 5e-06, p_t = 4e-135, d_z = 0.67
- vs cross-encoder: +0.095 [+0.083, +0.108], p = 5e-06, p_t = 4.4e-47, d_z = 0.37

| dataset | queries | MPUP | PVS w/o A | PVS w/ A | likelihood | LVR w/ A | answer confidence | ΔH (D5) | dense | bm25 | cross-encoder |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 170 | 0.247 | 0.196 | 0.252 | 0.329 | 0.216 | 0.035 | 0.168 | 0.159 | 0.057 | 0.182 |
| ambigqa | 147 | 0.405 | 0.291 | 0.289 | 0.258 | 0.251 | -0.126 | 0.146 | 0.309 | 0.151 | 0.317 |
| entityquestions | 182 | 0.429 | 0.340 | 0.381 | 0.348 | 0.310 | -0.140 | 0.233 | 0.308 | 0.220 | 0.351 |
| hotpotqa | 197 | 0.463 | 0.391 | 0.417 | 0.304 | 0.334 | -0.108 | 0.182 | 0.324 | 0.146 | 0.362 |
| musique | 163 | 0.311 | 0.243 | 0.227 | 0.127 | 0.199 | 0.003 | 0.107 | 0.124 | 0.028 | 0.223 |
| nq | 151 | 0.340 | 0.243 | 0.227 | 0.165 | 0.139 | -0.057 | 0.038 | 0.226 | 0.037 | 0.217 |
| popqa | 134 | 0.390 | 0.310 | 0.339 | 0.325 | 0.247 | -0.066 | 0.204 | 0.252 | 0.146 | 0.300 |
| squad | 171 | 0.295 | 0.262 | 0.283 | 0.205 | 0.223 | -0.016 | 0.147 | 0.193 | 0.119 | 0.250 |
| triviaqa | 185 | 0.403 | 0.289 | 0.304 | 0.313 | 0.219 | -0.138 | 0.156 | 0.252 | 0.170 | 0.283 |
| webquestions | 141 | 0.281 | 0.127 | 0.146 | 0.160 | 0.056 | 0.016 | 0.035 | 0.115 | -0.010 | 0.112 |
