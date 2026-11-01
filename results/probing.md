# C4: MPUP tiers against utility estimates that query the target LLM

Probing methods: SpecUBench-style judgments of the target LLM (Zhang et al. 2025c), scores cached by `run_probe_baselines`. MPUP tiers fitted on Qwen2.5-7B development questions (Qwen labels or answer labels); 'native MPUP' is fitted on the target LLM itself.

## Qwen2.5-7B (1641 rankable test questions)

| ranking | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| MPUP (Qwen labels) | 0.356 | 0.780 | 0.618 |
| reader-1.5B (answer labels) | 0.285 | 0.747 | 0.591 |
| no-LM (answer labels) | 0.274 | 0.725 | 0.541 |
| PVS w/o A | 0.269 | 0.708 | 0.517 |
| PVS w/ A | 0.287 | 0.718 | 0.544 |
| likelihood | 0.253 | 0.682 | 0.520 |
| LVR w/ A | 0.219 | 0.667 | 0.509 |
| answer confidence | -0.060 | 0.383 | 0.081 |
| dense | 0.226 | 0.659 | 0.459 |
| bm25 | 0.106 | 0.557 | 0.356 |
| cross-encoder | 0.260 | 0.697 | 0.490 |

Paired difference of MPUP (Qwen labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.086 [+0.074, +0.097], p = 5e-06, p_t = 3.5e-45, d_z = 0.36
- vs PVS w/ A: +0.067 [+0.054, +0.081], p = 5e-06, p_t = 1e-22, d_z = 0.25
- vs likelihood: +0.102 [+0.082, +0.122], p = 5e-06, p_t = 9e-23, d_z = 0.25
- vs LVR w/ A: +0.134 [+0.119, +0.150], p = 5e-06, p_t = 3.1e-63, d_z = 0.43
- vs answer confidence: +0.422 [+0.394, +0.448], p = 5e-06, p_t = 3.9e-162, d_z = 0.75
- vs dense: +0.130 [+0.116, +0.144], p = 5e-06, p_t = 1.6e-67, d_z = 0.45
- vs cross-encoder: +0.095 [+0.083, +0.108], p = 5e-06, p_t = 4.4e-47, d_z = 0.37
- per dataset vs PVS w/o A: ahead in 10/10; Holm p < 0.05 ahead in 8, behind in 0
- per dataset vs PVS w/ A: ahead in 9/10; Holm p < 0.05 ahead in 6, behind in 0
- per dataset vs likelihood: ahead in 9/10; Holm p < 0.05 ahead in 8, behind in 0
- per dataset vs LVR w/ A: ahead in 10/10; Holm p < 0.05 ahead in 9, behind in 0
- per dataset vs answer confidence: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs dense: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs cross-encoder: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0

Paired difference of reader-1.5B (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.016 [+0.006, +0.026], p = 0.0029, p_t = 0.0029, d_z = 0.07
- vs PVS w/ A: -0.002 [-0.016, +0.012], p = 0.76, p_t = 0.76, d_z = -0.01
- vs likelihood: +0.033 [+0.010, +0.054], p = 0.0032, p_t = 0.0031, d_z = 0.07
- vs LVR w/ A: +0.065 [+0.050, +0.079], p = 5e-06, p_t = 5.8e-18, d_z = 0.22
- vs answer confidence: +0.352 [+0.325, +0.379], p = 5e-06, p_t = 6.1e-122, d_z = 0.63
- vs dense: +0.060 [+0.048, +0.072], p = 5e-06, p_t = 5.5e-23, d_z = 0.25
- vs cross-encoder: +0.026 [+0.019, +0.033], p = 5e-06, p_t = 7.2e-14, d_z = 0.19
- per dataset vs PVS w/o A: ahead in 7/10; Holm p < 0.05 ahead in 1, behind in 0
- per dataset vs PVS w/ A: ahead in 4/10; Holm p < 0.05 ahead in 1, behind in 0
- per dataset vs likelihood: ahead in 6/10; Holm p < 0.05 ahead in 2, behind in 0
- per dataset vs LVR w/ A: ahead in 10/10; Holm p < 0.05 ahead in 7, behind in 0
- per dataset vs answer confidence: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs dense: ahead in 10/10; Holm p < 0.05 ahead in 6, behind in 0
- per dataset vs cross-encoder: ahead in 10/10; Holm p < 0.05 ahead in 3, behind in 0

Paired difference of no-LM (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.005 [-0.006, +0.016], p = 0.34, p_t = 0.34, d_z = 0.02
- vs PVS w/ A: -0.013 [-0.027, +0.001], p = 0.075, p_t = 0.076, d_z = -0.04
- vs likelihood: +0.022 [+0.000, +0.043], p = 0.042, p_t = 0.042, d_z = 0.05
- vs LVR w/ A: +0.054 [+0.039, +0.069], p = 5e-06, p_t = 1.7e-13, d_z = 0.18
- vs answer confidence: +0.342 [+0.314, +0.368], p = 5e-06, p_t = 2.9e-117, d_z = 0.62
- vs dense: +0.050 [+0.039, +0.060], p = 5e-06, p_t = 7.6e-19, d_z = 0.22
- vs cross-encoder: +0.015 [+0.010, +0.020], p = 5e-06, p_t = 5.6e-09, d_z = 0.14
- per dataset vs PVS w/o A: ahead in 5/10; Holm p < 0.05 ahead in 0, behind in 0
- per dataset vs PVS w/ A: ahead in 2/10; Holm p < 0.05 ahead in 0, behind in 0
- per dataset vs likelihood: ahead in 6/10; Holm p < 0.05 ahead in 1, behind in 0
- per dataset vs LVR w/ A: ahead in 10/10; Holm p < 0.05 ahead in 4, behind in 0
- per dataset vs answer confidence: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs dense: ahead in 9/10; Holm p < 0.05 ahead in 6, behind in 0
- per dataset vs cross-encoder: ahead in 10/10; Holm p < 0.05 ahead in 3, behind in 0

Timing (s, all test pairs): {"pseudo": 1254.8, "PVS w/o A": 1681.3, "PVS w/ A": 1743.9, "likelihood": 1490.3, "answer confidence": 1511.3, "LVR ranked share": 0.8625563511005039, "LVR w/ A": 2323.0, "peak_gpu_gib": 23.94}

## Mistral-7B (1714 rankable test questions)

| ranking | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| native MPUP | 0.373 | 0.778 | 0.544 |
| MPUP (Qwen labels) | 0.313 | 0.755 | 0.537 |
| reader-1.5B (answer labels) | 0.264 | 0.732 | 0.513 |
| no-LM (answer labels) | 0.255 | 0.715 | 0.464 |
| PVS w/o A | 0.238 | 0.683 | 0.417 |
| PVS w/ A | 0.292 | 0.733 | 0.478 |
| likelihood | 0.308 | 0.721 | 0.477 |
| LVR w/ A | 0.184 | 0.641 | 0.363 |
| answer confidence | 0.259 | 0.659 | 0.447 |
| dense | 0.210 | 0.660 | 0.380 |
| bm25 | 0.095 | 0.571 | 0.285 |
| cross-encoder | 0.243 | 0.688 | 0.397 |

Paired difference of native MPUP (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.134 [+0.120, +0.147], p = 5e-06, p_t = 4.7e-81, d_z = 0.49
- vs PVS w/ A: +0.079 [+0.066, +0.093], p = 5e-06, p_t = 4.7e-30, d_z = 0.28
- vs likelihood: +0.063 [+0.046, +0.082], p = 5e-06, p_t = 8.2e-12, d_z = 0.17
- vs LVR w/ A: +0.187 [+0.171, +0.202], p = 5e-06, p_t = 4.8e-108, d_z = 0.57
- vs answer confidence: +0.112 [+0.096, +0.130], p = 5e-06, p_t = 2.4e-39, d_z = 0.33
- vs dense: +0.161 [+0.147, +0.176], p = 5e-06, p_t = 7.1e-93, d_z = 0.53
- vs cross-encoder: +0.128 [+0.114, +0.142], p = 5e-06, p_t = 2.6e-68, d_z = 0.44
- per dataset vs PVS w/o A: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs PVS w/ A: ahead in 10/10; Holm p < 0.05 ahead in 8, behind in 0
- per dataset vs likelihood: ahead in 9/10; Holm p < 0.05 ahead in 4, behind in 0
- per dataset vs LVR w/ A: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs answer confidence: ahead in 10/10; Holm p < 0.05 ahead in 9, behind in 0
- per dataset vs dense: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs cross-encoder: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0

Paired difference of MPUP (Qwen labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.074 [+0.062, +0.086], p = 5e-06, p_t = 7.1e-31, d_z = 0.28
- vs PVS w/ A: +0.019 [+0.006, +0.032], p = 0.0038, p_t = 0.0037, d_z = 0.07
- vs likelihood: +0.003 [-0.015, +0.022], p = 0.78, p_t = 0.78, d_z = 0.01
- vs LVR w/ A: +0.127 [+0.111, +0.141], p = 5e-06, p_t = 7.7e-56, d_z = 0.39
- vs answer confidence: +0.052 [+0.034, +0.070], p = 5e-06, p_t = 1.2e-08, d_z = 0.14
- vs dense: +0.101 [+0.088, +0.114], p = 5e-06, p_t = 5.9e-47, d_z = 0.36
- vs cross-encoder: +0.068 [+0.056, +0.080], p = 5e-06, p_t = 8.4e-26, d_z = 0.26
- per dataset vs PVS w/o A: ahead in 10/10; Holm p < 0.05 ahead in 8, behind in 0
- per dataset vs PVS w/ A: ahead in 7/10; Holm p < 0.05 ahead in 0, behind in 0
- per dataset vs likelihood: ahead in 6/10; Holm p < 0.05 ahead in 1, behind in 1
- per dataset vs LVR w/ A: ahead in 10/10; Holm p < 0.05 ahead in 9, behind in 0
- per dataset vs answer confidence: ahead in 9/10; Holm p < 0.05 ahead in 3, behind in 0
- per dataset vs dense: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs cross-encoder: ahead in 10/10; Holm p < 0.05 ahead in 7, behind in 0

Paired difference of reader-1.5B (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.027 [+0.015, +0.039], p = 1e-05, p_t = 2e-05, d_z = 0.10
- vs PVS w/ A: -0.028 [-0.042, -0.014], p = 7e-05, p_t = 6.9e-05, d_z = -0.10
- vs likelihood: -0.044 [-0.063, -0.023], p = 1e-05, p_t = 1.1e-05, d_z = -0.11
- vs LVR w/ A: +0.080 [+0.066, +0.095], p = 5e-06, p_t = 1.2e-25, d_z = 0.26
- vs answer confidence: +0.006 [-0.014, +0.025], p = 0.57, p_t = 0.56, d_z = 0.01
- vs dense: +0.054 [+0.043, +0.066], p = 5e-06, p_t = 2.7e-19, d_z = 0.22
- vs cross-encoder: +0.021 [+0.014, +0.028], p = 5e-06, p_t = 2.7e-09, d_z = 0.14
- per dataset vs PVS w/o A: ahead in 7/10; Holm p < 0.05 ahead in 2, behind in 0
- per dataset vs PVS w/ A: ahead in 2/10; Holm p < 0.05 ahead in 0, behind in 3
- per dataset vs likelihood: ahead in 4/10; Holm p < 0.05 ahead in 1, behind in 3
- per dataset vs LVR w/ A: ahead in 10/10; Holm p < 0.05 ahead in 7, behind in 0
- per dataset vs answer confidence: ahead in 6/10; Holm p < 0.05 ahead in 0, behind in 2
- per dataset vs dense: ahead in 8/10; Holm p < 0.05 ahead in 6, behind in 0
- per dataset vs cross-encoder: ahead in 9/10; Holm p < 0.05 ahead in 3, behind in 0

Paired difference of no-LM (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.018 [+0.005, +0.030], p = 0.0056, p_t = 0.0056, d_z = 0.07
- vs PVS w/ A: -0.036 [-0.050, -0.022], p = 5e-06, p_t = 2.6e-07, d_z = -0.12
- vs likelihood: -0.053 [-0.072, -0.033], p = 5e-06, p_t = 8.1e-08, d_z = -0.13
- vs LVR w/ A: +0.071 [+0.057, +0.086], p = 5e-06, p_t = 1.9e-21, d_z = 0.23
- vs answer confidence: -0.003 [-0.023, +0.017], p = 0.74, p_t = 0.74, d_z = -0.01
- vs dense: +0.046 [+0.035, +0.057], p = 5e-06, p_t = 1.6e-15, d_z = 0.19
- vs cross-encoder: +0.013 [+0.007, +0.018], p = 1e-05, p_t = 5e-06, d_z = 0.11
- per dataset vs PVS w/o A: ahead in 7/10; Holm p < 0.05 ahead in 1, behind in 0
- per dataset vs PVS w/ A: ahead in 1/10; Holm p < 0.05 ahead in 0, behind in 3
- per dataset vs likelihood: ahead in 2/10; Holm p < 0.05 ahead in 1, behind in 3
- per dataset vs LVR w/ A: ahead in 10/10; Holm p < 0.05 ahead in 6, behind in 0
- per dataset vs answer confidence: ahead in 6/10; Holm p < 0.05 ahead in 0, behind in 2
- per dataset vs dense: ahead in 8/10; Holm p < 0.05 ahead in 5, behind in 0
- per dataset vs cross-encoder: ahead in 8/10; Holm p < 0.05 ahead in 1, behind in 0

Timing (s, all test pairs): {"pseudo": 1850.4, "PVS w/o A": 1736.9, "PVS w/ A": 1855.5, "likelihood": 1563.8, "answer confidence": 1590.7, "LVR ranked share": 0.9759215062317688, "LVR w/ A": 4311.3, "peak_gpu_gib": 27.27}

## OLMo-2-7B (1683 rankable test questions)

| ranking | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| native MPUP | 0.316 | 0.736 | 0.557 |
| MPUP (Qwen labels) | 0.312 | 0.733 | 0.551 |
| reader-1.5B (answer labels) | 0.277 | 0.719 | 0.537 |
| no-LM (answer labels) | 0.264 | 0.697 | 0.490 |
| PVS w/o A | 0.195 | 0.609 | 0.386 |
| PVS w/ A | 0.267 | 0.679 | 0.481 |
| likelihood | 0.248 | 0.644 | 0.455 |
| LVR w/ A | 0.087 | 0.509 | 0.301 |
| answer confidence | 0.037 | 0.433 | 0.089 |
| dense | 0.211 | 0.623 | 0.401 |
| bm25 | 0.092 | 0.527 | 0.296 |
| cross-encoder | 0.247 | 0.664 | 0.427 |

Paired difference of native MPUP (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.122 [+0.108, +0.136], p = 5e-06, p_t = 3.7e-61, d_z = 0.42
- vs PVS w/ A: +0.049 [+0.036, +0.065], p = 5e-06, p_t = 4.8e-11, d_z = 0.16
- vs likelihood: +0.068 [+0.049, +0.089], p = 5e-06, p_t = 1.1e-11, d_z = 0.17
- vs LVR w/ A: +0.231 [+0.215, +0.248], p = 5e-06, p_t = 3.8e-134, d_z = 0.66
- vs answer confidence: +0.282 [+0.259, +0.305], p = 5e-06, p_t = 2.2e-109, d_z = 0.58
- vs dense: +0.106 [+0.093, +0.120], p = 5e-06, p_t = 4.2e-51, d_z = 0.38
- vs cross-encoder: +0.069 [+0.058, +0.081], p = 5e-06, p_t = 6.8e-31, d_z = 0.29
- per dataset vs PVS w/o A: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs PVS w/ A: ahead in 10/10; Holm p < 0.05 ahead in 3, behind in 0
- per dataset vs likelihood: ahead in 8/10; Holm p < 0.05 ahead in 2, behind in 0
- per dataset vs LVR w/ A: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs answer confidence: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs dense: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs cross-encoder: ahead in 10/10; Holm p < 0.05 ahead in 9, behind in 0

Paired difference of MPUP (Qwen labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.118 [+0.104, +0.132], p = 5e-06, p_t = 7.7e-56, d_z = 0.40
- vs PVS w/ A: +0.046 [+0.031, +0.061], p = 5e-06, p_t = 1.4e-09, d_z = 0.15
- vs likelihood: +0.065 [+0.045, +0.085], p = 5e-06, p_t = 9.2e-11, d_z = 0.16
- vs LVR w/ A: +0.228 [+0.212, +0.245], p = 5e-06, p_t = 1e-128, d_z = 0.64
- vs answer confidence: +0.278 [+0.255, +0.301], p = 5e-06, p_t = 1.2e-105, d_z = 0.57
- vs dense: +0.103 [+0.089, +0.116], p = 5e-06, p_t = 1.5e-48, d_z = 0.37
- vs cross-encoder: +0.065 [+0.053, +0.078], p = 5e-06, p_t = 3.7e-26, d_z = 0.26
- per dataset vs PVS w/o A: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs PVS w/ A: ahead in 9/10; Holm p < 0.05 ahead in 3, behind in 0
- per dataset vs likelihood: ahead in 8/10; Holm p < 0.05 ahead in 2, behind in 0
- per dataset vs LVR w/ A: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs answer confidence: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs dense: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs cross-encoder: ahead in 10/10; Holm p < 0.05 ahead in 8, behind in 0

Paired difference of reader-1.5B (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.083 [+0.070, +0.097], p = 5e-06, p_t = 3.7e-31, d_z = 0.29
- vs PVS w/ A: +0.010 [-0.004, +0.026], p = 0.18, p_t = 0.18, d_z = 0.03
- vs likelihood: +0.029 [+0.009, +0.051], p = 0.0054, p_t = 0.0054, d_z = 0.07
- vs LVR w/ A: +0.192 [+0.176, +0.209], p = 5e-06, p_t = 9e-98, d_z = 0.55
- vs answer confidence: +0.243 [+0.221, +0.266], p = 5e-06, p_t = 4.9e-88, d_z = 0.51
- vs dense: +0.067 [+0.055, +0.079], p = 5e-06, p_t = 9.2e-29, d_z = 0.28
- vs cross-encoder: +0.030 [+0.023, +0.037], p = 5e-06, p_t = 1.5e-17, d_z = 0.21
- per dataset vs PVS w/o A: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs PVS w/ A: ahead in 7/10; Holm p < 0.05 ahead in 1, behind in 0
- per dataset vs likelihood: ahead in 5/10; Holm p < 0.05 ahead in 1, behind in 0
- per dataset vs LVR w/ A: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs answer confidence: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs dense: ahead in 10/10; Holm p < 0.05 ahead in 7, behind in 0
- per dataset vs cross-encoder: ahead in 9/10; Holm p < 0.05 ahead in 5, behind in 0

Paired difference of no-LM (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs PVS w/o A: +0.070 [+0.057, +0.084], p = 5e-06, p_t = 6.8e-23, d_z = 0.24
- vs PVS w/ A: -0.002 [-0.017, +0.013], p = 0.77, p_t = 0.77, d_z = -0.01
- vs likelihood: +0.017 [-0.004, +0.038], p = 0.11, p_t = 0.11, d_z = 0.04
- vs LVR w/ A: +0.180 [+0.164, +0.196], p = 5e-06, p_t = 1.1e-90, d_z = 0.52
- vs answer confidence: +0.230 [+0.208, +0.253], p = 5e-06, p_t = 1.7e-80, d_z = 0.49
- vs dense: +0.055 [+0.043, +0.066], p = 5e-06, p_t = 4.3e-22, d_z = 0.24
- vs cross-encoder: +0.018 [+0.012, +0.023], p = 5e-06, p_t = 6.6e-10, d_z = 0.15
- per dataset vs PVS w/o A: ahead in 10/10; Holm p < 0.05 ahead in 6, behind in 0
- per dataset vs PVS w/ A: ahead in 5/10; Holm p < 0.05 ahead in 0, behind in 0
- per dataset vs likelihood: ahead in 4/10; Holm p < 0.05 ahead in 1, behind in 0
- per dataset vs LVR w/ A: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs answer confidence: ahead in 10/10; Holm p < 0.05 ahead in 10, behind in 0
- per dataset vs dense: ahead in 9/10; Holm p < 0.05 ahead in 6, behind in 0
- per dataset vs cross-encoder: ahead in 8/10; Holm p < 0.05 ahead in 3, behind in 0

Timing (s, all test pairs): {"pseudo": 2589.3, "PVS w/o A": 1774.2, "PVS w/ A": 1599.6, "likelihood": 1385.1, "answer confidence": 1350.4, "LVR ranked share": 0.532140015910899, "LVR w/ A": 5079.2, "peak_gpu_gib": 27.56}

