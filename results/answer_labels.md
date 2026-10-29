# C6: training on labels from the dataset's gold answers

1641 rankable test questions, scored against the Eq. (1) utility of Qwen2.5-7B. Answer labels on 90737 development pairs (15.0% positive; 20 yes/no pairs left out); centred Pearson with u 0.612.

| ranking | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| MPUP (Eq. 1 labels) | 0.356 | 0.780 | 0.618 |
| MPUP (answer labels) | 0.310 | 0.770 | 0.620 |
| reader-1.5B (Eq. 1 labels) | 0.296 | 0.749 | 0.594 |
| reader-1.5B (answer labels) | 0.285 | 0.747 | 0.591 |
| no-LM (Eq. 1 labels) | 0.275 | 0.725 | 0.541 |
| no-LM (answer labels) | 0.274 | 0.725 | 0.541 |
| answer containment (uses the gold answer) | 0.527 | 0.785 | 0.707 |
| dense | 0.226 | 0.659 | 0.459 |
| bm25 | 0.106 | 0.557 | 0.356 |
| cross-encoder | 0.260 | 0.697 | 0.490 |

Paired difference of MPUP (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs MPUP (Eq. 1 labels): -0.044 [-0.054, -0.036], p = 5e-06, p_t = 4.1e-22, d_z = -0.24
- vs reader-1.5B (Eq. 1 labels): +0.015 [+0.006, +0.024], p = 0.00086, p_t = 0.00089, d_z = 0.08
- vs reader-1.5B (answer labels): +0.025 [+0.017, +0.033], p = 5e-06, p_t = 3.9e-10, d_z = 0.16
- vs no-LM (Eq. 1 labels): +0.035 [+0.026, +0.045], p = 5e-06, p_t = 3.1e-13, d_z = 0.18
- vs no-LM (answer labels): +0.036 [+0.027, +0.045], p = 5e-06, p_t = 2.9e-15, d_z = 0.20
- vs answer containment (uses the gold answer): -0.219 [-0.236, -0.203], p = 5e-06, p_t = 1.7e-126, d_z = -0.65
- vs dense: +0.085 [+0.072, +0.098], p = 5e-06, p_t = 6.5e-36, d_z = 0.32
- vs bm25: +0.205 [+0.186, +0.223], p = 5e-06, p_t = 2.5e-95, d_z = 0.55
- vs cross-encoder: +0.051 [+0.042, +0.061], p = 5e-06, p_t = 4.6e-25, d_z = 0.26

Paired difference of reader-1.5B (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs MPUP (Eq. 1 labels): -0.070 [-0.081, -0.058], p = 5e-06, p_t = 4.4e-30, d_z = -0.29
- vs MPUP (answer labels): -0.025 [-0.033, -0.017], p = 5e-06, p_t = 3.9e-10, d_z = -0.16
- vs reader-1.5B (Eq. 1 labels): -0.010 [-0.016, -0.005], p = 0.00015, p_t = 0.00016, d_z = -0.09
- vs no-LM (Eq. 1 labels): +0.010 [+0.004, +0.016], p = 0.0018, p_t = 0.002, d_z = 0.08
- vs no-LM (answer labels): +0.011 [+0.005, +0.016], p = 0.00016, p_t = 0.00015, d_z = 0.09
- vs answer containment (uses the gold answer): -0.245 [-0.261, -0.228], p = 5e-06, p_t = 4.2e-153, d_z = -0.73
- vs dense: +0.060 [+0.048, +0.072], p = 5e-06, p_t = 5.5e-23, d_z = 0.25
- vs bm25: +0.179 [+0.162, +0.196], p = 5e-06, p_t = 1.2e-80, d_z = 0.50
- vs cross-encoder: +0.026 [+0.019, +0.033], p = 5e-06, p_t = 7.2e-14, d_z = 0.19

Paired difference of no-LM (answer labels) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs MPUP (Eq. 1 labels): -0.080 [-0.093, -0.068], p = 5e-06, p_t = 9e-37, d_z = -0.32
- vs MPUP (answer labels): -0.036 [-0.045, -0.027], p = 5e-06, p_t = 2.9e-15, d_z = -0.20
- vs reader-1.5B (Eq. 1 labels): -0.021 [-0.028, -0.014], p = 5e-06, p_t = 2.2e-08, d_z = -0.14
- vs reader-1.5B (answer labels): -0.011 [-0.016, -0.005], p = 0.00016, p_t = 0.00015, d_z = -0.09
- vs no-LM (Eq. 1 labels): -0.000 [-0.004, +0.004], p = 0.84, p_t = 0.84, d_z = -0.00
- vs answer containment (uses the gold answer): -0.255 [-0.272, -0.239], p = 5e-06, p_t = 3.3e-159, d_z = -0.74
- vs dense: +0.050 [+0.039, +0.060], p = 5e-06, p_t = 7.6e-19, d_z = 0.22
- vs bm25: +0.169 [+0.151, +0.185], p = 5e-06, p_t = 1.7e-75, d_z = 0.48
- vs cross-encoder: +0.015 [+0.010, +0.020], p = 5e-06, p_t = 5.6e-09, d_z = 0.14
