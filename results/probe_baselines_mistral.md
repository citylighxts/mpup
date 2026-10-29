# C4: MPUP against utility estimates that query the target LLM

1714 rankable test questions; target LLM mistralai/Mistral-7B-Instruct-v0.3; within-query Spearman with its Eq. (1) utility. Time per method (s, all 37710 test pairs): {"pseudo": 1850.4, "PVS w/o A": 1736.9, "PVS w/ A": 1855.5, "likelihood": 1563.8, "answer confidence": 1590.7, "LVR ranked share": 0.9759215062317688, "LVR w/ A": 4311.3, "peak_gpu_gib": 27.27}

| ranking | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| MPUP | 0.373 | 0.778 | 0.544 |
| PVS w/o A | 0.238 | 0.683 | 0.417 |
| PVS w/ A | 0.292 | 0.733 | 0.478 |
| likelihood | 0.308 | 0.721 | 0.477 |
| LVR w/ A | 0.184 | 0.641 | 0.363 |
| answer confidence | 0.259 | 0.659 | 0.447 |
| ΔH (D5) | 0.049 | 0.516 | 0.222 |
| dense | 0.210 | 0.660 | 0.380 |
| bm25 | 0.095 | 0.571 | 0.285 |
| cross-encoder | 0.243 | 0.688 | 0.397 |

Paired difference of MPUP (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.134 [+0.120, +0.147], p = 5e-06, p_t = 4.7e-81, d_z = 0.49
- vs PVS w/ A: +0.079 [+0.066, +0.093], p = 5e-06, p_t = 4.7e-30, d_z = 0.28
- vs likelihood: +0.063 [+0.046, +0.082], p = 5e-06, p_t = 8.2e-12, d_z = 0.17
- vs LVR w/ A: +0.187 [+0.171, +0.202], p = 5e-06, p_t = 4.8e-108, d_z = 0.57
- vs answer confidence: +0.112 [+0.096, +0.130], p = 5e-06, p_t = 2.4e-39, d_z = 0.33
- vs ΔH (D5): +0.323 [+0.304, +0.341], p = 5e-06, p_t = 1.4e-212, d_z = 0.87
- vs dense: +0.161 [+0.147, +0.176], p = 5e-06, p_t = 7.1e-93, d_z = 0.53
- vs bm25: +0.277 [+0.259, +0.295], p = 5e-06, p_t = 2.8e-168, d_z = 0.75
- vs cross-encoder: +0.128 [+0.114, +0.142], p = 5e-06, p_t = 2.6e-68, d_z = 0.44

| dataset | queries | MPUP | PVS w/o A | PVS w/ A | likelihood | LVR w/ A | answer confidence | ΔH (D5) | dense | bm25 | cross-encoder |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 182 | 0.344 | 0.264 | 0.301 | 0.281 | 0.260 | 0.244 | 0.027 | 0.236 | 0.132 | 0.252 |
| ambigqa | 155 | 0.456 | 0.264 | 0.332 | 0.341 | 0.233 | 0.329 | 0.060 | 0.273 | 0.133 | 0.282 |
| entityquestions | 186 | 0.449 | 0.342 | 0.391 | 0.361 | 0.252 | 0.304 | 0.090 | 0.302 | 0.215 | 0.354 |
| hotpotqa | 204 | 0.450 | 0.273 | 0.379 | 0.439 | 0.277 | 0.334 | 0.151 | 0.285 | 0.090 | 0.337 |
| musique | 159 | 0.287 | 0.238 | 0.213 | 0.150 | 0.112 | 0.210 | -0.059 | 0.105 | 0.011 | 0.226 |
| nq | 162 | 0.376 | 0.213 | 0.261 | 0.283 | 0.141 | 0.222 | 0.045 | 0.198 | 0.043 | 0.169 |
| popqa | 142 | 0.427 | 0.257 | 0.359 | 0.387 | 0.227 | 0.248 | 0.071 | 0.243 | 0.148 | 0.295 |
| squad | 177 | 0.307 | 0.216 | 0.272 | 0.252 | 0.167 | 0.190 | 0.055 | 0.190 | 0.120 | 0.251 |
| triviaqa | 192 | 0.337 | 0.191 | 0.240 | 0.341 | 0.122 | 0.300 | 0.020 | 0.166 | 0.094 | 0.189 |
| webquestions | 155 | 0.295 | 0.127 | 0.176 | 0.247 | 0.053 | 0.208 | 0.026 | 0.103 | -0.036 | 0.073 |
