# C4: MPUP against utility estimates that query the target LLM

1683 rankable test questions; target LLM allenai/OLMo-2-1124-7B-Instruct; within-query Spearman with its Eq. (1) utility. Time per method (s, all 37710 test pairs): {"pseudo": 2589.3, "PVS w/o A": 1774.2, "PVS w/ A": 1599.6, "likelihood": 1385.1, "answer confidence": 1350.4, "LVR ranked share": 0.532140015910899, "LVR w/ A": 5079.2, "peak_gpu_gib": 27.56}

| ranking | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| MPUP | 0.316 | 0.736 | 0.557 |
| PVS w/o A | 0.195 | 0.609 | 0.386 |
| PVS w/ A | 0.267 | 0.679 | 0.481 |
| likelihood | 0.248 | 0.644 | 0.455 |
| LVR w/ A | 0.087 | 0.509 | 0.301 |
| answer confidence | 0.037 | 0.433 | 0.089 |
| ΔH (D5) | 0.055 | 0.468 | 0.207 |
| dense | 0.211 | 0.623 | 0.401 |
| bm25 | 0.092 | 0.527 | 0.296 |
| cross-encoder | 0.247 | 0.664 | 0.427 |

Paired difference of MPUP (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.122 [+0.108, +0.136], p = 5e-06, p_t = 3.7e-61, d_z = 0.42
- vs PVS w/ A: +0.049 [+0.036, +0.065], p = 5e-06, p_t = 4.8e-11, d_z = 0.16
- vs likelihood: +0.068 [+0.049, +0.089], p = 5e-06, p_t = 1.1e-11, d_z = 0.17
- vs LVR w/ A: +0.231 [+0.215, +0.248], p = 5e-06, p_t = 3.8e-134, d_z = 0.66
- vs answer confidence: +0.282 [+0.259, +0.305], p = 5e-06, p_t = 2.2e-109, d_z = 0.58
- vs ΔH (D5): +0.261 [+0.243, +0.279], p = 5e-06, p_t = 6.1e-143, d_z = 0.69
- vs dense: +0.106 [+0.093, +0.120], p = 5e-06, p_t = 4.2e-51, d_z = 0.38
- vs bm25: +0.224 [+0.207, +0.242], p = 5e-06, p_t = 3.9e-118, d_z = 0.61
- vs cross-encoder: +0.069 [+0.058, +0.081], p = 5e-06, p_t = 6.8e-31, d_z = 0.29

| dataset | queries | MPUP | PVS w/o A | PVS w/ A | likelihood | LVR w/ A | answer confidence | ΔH (D5) | dense | bm25 | cross-encoder |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 164 | 0.282 | 0.158 | 0.281 | 0.301 | 0.023 | 0.116 | 0.097 | 0.152 | 0.028 | 0.166 |
| ambigqa | 152 | 0.369 | 0.242 | 0.307 | 0.272 | 0.111 | 0.026 | 0.016 | 0.304 | 0.117 | 0.313 |
| entityquestions | 185 | 0.383 | 0.241 | 0.331 | 0.383 | 0.210 | -0.024 | 0.021 | 0.270 | 0.213 | 0.334 |
| hotpotqa | 194 | 0.384 | 0.269 | 0.357 | 0.301 | 0.066 | 0.064 | 0.089 | 0.276 | 0.100 | 0.299 |
| musique | 169 | 0.296 | 0.186 | 0.210 | 0.028 | 0.054 | 0.149 | 0.122 | 0.128 | 0.040 | 0.214 |
| nq | 156 | 0.272 | 0.162 | 0.213 | 0.204 | 0.075 | 0.031 | 0.024 | 0.201 | 0.035 | 0.191 |
| popqa | 139 | 0.342 | 0.233 | 0.321 | 0.329 | 0.132 | 0.005 | 0.019 | 0.255 | 0.148 | 0.288 |
| squad | 172 | 0.266 | 0.191 | 0.247 | 0.192 | 0.110 | 0.049 | 0.040 | 0.204 | 0.110 | 0.266 |
| triviaqa | 195 | 0.331 | 0.171 | 0.263 | 0.305 | 0.064 | -0.075 | 0.093 | 0.197 | 0.132 | 0.271 |
| webquestions | 157 | 0.237 | 0.096 | 0.141 | 0.160 | 0.021 | 0.032 | 0.028 | 0.125 | -0.007 | 0.126 |
