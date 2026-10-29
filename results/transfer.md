# C5: one predictor for several target LLMs

Every predictor is fitted once on the development questions of Qwen2.5-7B (its utility or the answer labels) and scored unchanged against the Eq. (1) utility of each target LLM on its test questions. 'native' is MPUP fitted on the target LLM's own labels, features and hidden states (an upper reference).

## Qwen2.5-7B (1641 rankable test questions; centred Pearson of its utility with Qwen's 1.000)

| ranking | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| MPUP (Qwen labels) | 0.356 | 0.780 | 0.618 |
| reader-1.5B (Qwen labels) | 0.296 | 0.749 | 0.594 |
| reader-1.5B (answer labels) | 0.285 | 0.747 | 0.591 |
| no-LM (answer labels) | 0.274 | 0.725 | 0.541 |
| dense | 0.226 | 0.659 | 0.459 |
| bm25 | 0.106 | 0.557 | 0.356 |
| cross-encoder | 0.260 | 0.697 | 0.490 |

Paired difference of MPUP (Qwen labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs dense: +0.130 [+0.116, +0.144], p = 5e-06, p_t = 1.6e-67, d_z = 0.45
- vs bm25: +0.249 [+0.231, +0.266], p = 5e-06, p_t = 4e-135, d_z = 0.67
- vs cross-encoder: +0.095 [+0.083, +0.108], p = 5e-06, p_t = 4.4e-47, d_z = 0.37

Paired difference of reader-1.5B (Qwen labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs dense: +0.071 [+0.059, +0.082], p = 5e-06, p_t = 1.9e-32, d_z = 0.30
- vs bm25: +0.190 [+0.172, +0.206], p = 5e-06, p_t = 5.4e-93, d_z = 0.54
- vs cross-encoder: +0.036 [+0.028, +0.044], p = 5e-06, p_t = 2.7e-17, d_z = 0.21

Paired difference of reader-1.5B (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs dense: +0.060 [+0.048, +0.072], p = 5e-06, p_t = 5.5e-23, d_z = 0.25
- vs bm25: +0.179 [+0.162, +0.196], p = 5e-06, p_t = 1.2e-80, d_z = 0.50
- vs cross-encoder: +0.026 [+0.019, +0.033], p = 5e-06, p_t = 7.2e-14, d_z = 0.19

Paired difference of no-LM (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs dense: +0.050 [+0.039, +0.060], p = 5e-06, p_t = 7.6e-19, d_z = 0.22
- vs bm25: +0.169 [+0.151, +0.185], p = 5e-06, p_t = 1.7e-75, d_z = 0.48
- vs cross-encoder: +0.015 [+0.010, +0.020], p = 5e-06, p_t = 5.6e-09, d_z = 0.14

## Mistral-7B (1714 rankable test questions; centred Pearson of its utility with Qwen's 0.596)

| ranking | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| MPUP (Qwen labels) | 0.313 | 0.755 | 0.537 |
| reader-1.5B (Qwen labels) | 0.278 | 0.735 | 0.508 |
| reader-1.5B (answer labels) | 0.264 | 0.732 | 0.513 |
| no-LM (answer labels) | 0.255 | 0.715 | 0.464 |
| native MPUP (own labels) | 0.373 | 0.778 | 0.544 |
| u (Qwen2.5-7B) | 0.512 | 0.787 | 0.613 |
| dense | 0.210 | 0.660 | 0.380 |
| bm25 | 0.095 | 0.571 | 0.285 |
| cross-encoder | 0.243 | 0.688 | 0.397 |

Paired difference of MPUP (Qwen labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs native MPUP (own labels): -0.060 [-0.071, -0.050], p = 5e-06, p_t = 1.6e-27, d_z = -0.27
- vs dense: +0.101 [+0.088, +0.114], p = 5e-06, p_t = 5.9e-47, d_z = 0.36
- vs bm25: +0.217 [+0.199, +0.233], p = 5e-06, p_t = 3.2e-114, d_z = 0.59
- vs cross-encoder: +0.068 [+0.056, +0.080], p = 5e-06, p_t = 8.4e-26, d_z = 0.26

Paired difference of reader-1.5B (Qwen labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs native MPUP (own labels): -0.094 [-0.106, -0.081], p = 5e-06, p_t = 4.9e-49, d_z = -0.37
- vs dense: +0.068 [+0.056, +0.079], p = 5e-06, p_t = 1.5e-30, d_z = 0.28
- vs bm25: +0.183 [+0.167, +0.199], p = 5e-06, p_t = 2.8e-91, d_z = 0.52
- vs cross-encoder: +0.034 [+0.026, +0.043], p = 5e-06, p_t = 2.6e-14, d_z = 0.19

Paired difference of reader-1.5B (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs native MPUP (own labels): -0.107 [-0.121, -0.093], p = 5e-06, p_t = 8.5e-56, d_z = -0.39
- vs dense: +0.054 [+0.043, +0.066], p = 5e-06, p_t = 2.7e-19, d_z = 0.22
- vs bm25: +0.170 [+0.153, +0.187], p = 5e-06, p_t = 1.3e-77, d_z = 0.47
- vs cross-encoder: +0.021 [+0.014, +0.028], p = 5e-06, p_t = 2.7e-09, d_z = 0.14

Paired difference of no-LM (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs native MPUP (own labels): -0.116 [-0.130, -0.102], p = 5e-06, p_t = 5e-58, d_z = -0.40
- vs dense: +0.046 [+0.035, +0.057], p = 5e-06, p_t = 1.6e-15, d_z = 0.19
- vs bm25: +0.161 [+0.145, +0.178], p = 5e-06, p_t = 2.4e-71, d_z = 0.45
- vs cross-encoder: +0.013 [+0.007, +0.018], p = 1e-05, p_t = 5e-06, d_z = 0.11

## OLMo-2-7B (1683 rankable test questions; centred Pearson of its utility with Qwen's 0.611)

| ranking | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| MPUP (Qwen labels) | 0.312 | 0.733 | 0.551 |
| reader-1.5B (Qwen labels) | 0.283 | 0.717 | 0.526 |
| reader-1.5B (answer labels) | 0.277 | 0.719 | 0.537 |
| no-LM (answer labels) | 0.264 | 0.697 | 0.490 |
| native MPUP (own labels) | 0.316 | 0.736 | 0.557 |
| u (Qwen2.5-7B) | 0.554 | 0.782 | 0.656 |
| dense | 0.211 | 0.623 | 0.401 |
| bm25 | 0.092 | 0.527 | 0.296 |
| cross-encoder | 0.247 | 0.664 | 0.427 |

Paired difference of MPUP (Qwen labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs native MPUP (own labels): -0.004 [-0.014, +0.006], p = 0.47, p_t = 0.47, d_z = -0.02
- vs dense: +0.103 [+0.089, +0.116], p = 5e-06, p_t = 1.5e-48, d_z = 0.37
- vs bm25: +0.221 [+0.203, +0.238], p = 5e-06, p_t = 7.5e-113, d_z = 0.59
- vs cross-encoder: +0.065 [+0.053, +0.078], p = 5e-06, p_t = 3.7e-26, d_z = 0.26

Paired difference of reader-1.5B (Qwen labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs native MPUP (own labels): -0.033 [-0.043, -0.023], p = 5e-06, p_t = 4e-10, d_z = -0.15
- vs dense: +0.074 [+0.062, +0.085], p = 5e-06, p_t = 1e-35, d_z = 0.31
- vs bm25: +0.192 [+0.174, +0.209], p = 5e-06, p_t = 1.2e-95, d_z = 0.54
- vs cross-encoder: +0.036 [+0.028, +0.045], p = 5e-06, p_t = 1e-16, d_z = 0.20

Paired difference of reader-1.5B (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs native MPUP (own labels): -0.039 [-0.049, -0.029], p = 5e-06, p_t = 3.9e-13, d_z = -0.18
- vs dense: +0.067 [+0.055, +0.079], p = 5e-06, p_t = 9.2e-29, d_z = 0.28
- vs bm25: +0.185 [+0.168, +0.203], p = 5e-06, p_t = 2e-87, d_z = 0.51
- vs cross-encoder: +0.030 [+0.023, +0.037], p = 5e-06, p_t = 1.5e-17, d_z = 0.21

Paired difference of no-LM (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs native MPUP (own labels): -0.052 [-0.062, -0.041], p = 5e-06, p_t = 4.4e-20, d_z = -0.23
- vs dense: +0.055 [+0.043, +0.066], p = 5e-06, p_t = 4.3e-22, d_z = 0.24
- vs bm25: +0.173 [+0.156, +0.190], p = 5e-06, p_t = 1.2e-79, d_z = 0.49
- vs cross-encoder: +0.018 [+0.012, +0.023], p = 5e-06, p_t = 6.6e-10, d_z = 0.15

