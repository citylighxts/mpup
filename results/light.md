# C3: cost tiers of MPUP

1641 rankable test questions, target LLM Qwen2.5-7B. Each tier fitted on the development questions with the Eq. (1) utility of Qwen2.5-7B.

| ranking | ρ | NDCG@10 | top-1 |
|---|---|---|---|
| full (MPUP) | 0.356 | 0.780 | 0.618 |
| no-generation | 0.350 | 0.775 | 0.609 |
| reader-1.5B | 0.296 | 0.749 | 0.594 |
| reader-0.5B | 0.280 | 0.732 | 0.553 |
| no-LM | 0.275 | 0.725 | 0.541 |
| dense | 0.226 | 0.659 | 0.459 |
| bm25 | 0.106 | 0.557 | 0.356 |
| cross-encoder | 0.260 | 0.697 | 0.490 |

Paired difference of full (MPUP) (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs no-generation: +0.006 [+0.001, +0.010], p = 0.0087, p_t = 0.0088, d_z = 0.06
- vs reader-1.5B: +0.059 [+0.049, +0.070], p = 5e-06, p_t = 3.8e-26, d_z = 0.27
- vs reader-0.5B: +0.075 [+0.063, +0.087], p = 5e-06, p_t = 1.6e-33, d_z = 0.30
- vs no-LM: +0.080 [+0.068, +0.092], p = 5e-06, p_t = 6.4e-37, d_z = 0.32
- vs dense: +0.130 [+0.116, +0.144], p = 5e-06, p_t = 1.6e-67, d_z = 0.45
- vs bm25: +0.249 [+0.231, +0.266], p = 5e-06, p_t = 4e-135, d_z = 0.67
- vs cross-encoder: +0.095 [+0.083, +0.108], p = 5e-06, p_t = 4.4e-47, d_z = 0.37
- per dataset vs dense: ahead in 10/10, ahead with Holm p < 0.05 in 10
- per dataset vs cross-encoder: ahead in 10/10, ahead with Holm p < 0.05 in 10

Paired difference of no-generation (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs full (MPUP): -0.006 [-0.010, -0.001], p = 0.0087, p_t = 0.0088, d_z = -0.06
- vs reader-1.5B: +0.053 [+0.043, +0.064], p = 5e-06, p_t = 7.9e-23, d_z = 0.25
- vs reader-0.5B: +0.069 [+0.057, +0.081], p = 5e-06, p_t = 3.4e-31, d_z = 0.29
- vs no-LM: +0.074 [+0.063, +0.086], p = 5e-06, p_t = 6.3e-35, d_z = 0.31
- vs dense: +0.124 [+0.111, +0.137], p = 5e-06, p_t = 3.1e-65, d_z = 0.44
- vs bm25: +0.243 [+0.226, +0.260], p = 5e-06, p_t = 2.7e-133, d_z = 0.67
- vs cross-encoder: +0.090 [+0.078, +0.102], p = 5e-06, p_t = 3.1e-44, d_z = 0.35
- per dataset vs dense: ahead in 10/10, ahead with Holm p < 0.05 in 10
- per dataset vs cross-encoder: ahead in 10/10, ahead with Holm p < 0.05 in 10

Paired difference of reader-1.5B (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs full (MPUP): -0.059 [-0.070, -0.049], p = 5e-06, p_t = 3.8e-26, d_z = -0.27
- vs no-generation: -0.053 [-0.064, -0.043], p = 5e-06, p_t = 7.9e-23, d_z = -0.25
- vs reader-0.5B: +0.015 [+0.009, +0.022], p = 5e-06, p_t = 5.9e-06, d_z = 0.11
- vs no-LM: +0.021 [+0.013, +0.027], p = 5e-06, p_t = 2e-08, d_z = 0.14
- vs dense: +0.071 [+0.059, +0.082], p = 5e-06, p_t = 1.9e-32, d_z = 0.30
- vs bm25: +0.190 [+0.172, +0.206], p = 5e-06, p_t = 5.4e-93, d_z = 0.54
- vs cross-encoder: +0.036 [+0.028, +0.044], p = 5e-06, p_t = 2.7e-17, d_z = 0.21
- per dataset vs dense: ahead in 10/10, ahead with Holm p < 0.05 in 7
- per dataset vs cross-encoder: ahead in 10/10, ahead with Holm p < 0.05 in 5

Paired difference of reader-0.5B (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs full (MPUP): -0.075 [-0.087, -0.063], p = 5e-06, p_t = 1.6e-33, d_z = -0.30
- vs no-generation: -0.069 [-0.081, -0.057], p = 5e-06, p_t = 3.4e-31, d_z = -0.29
- vs reader-1.5B: -0.015 [-0.022, -0.009], p = 5e-06, p_t = 5.9e-06, d_z = -0.11
- vs no-LM: +0.005 [+0.000, +0.010], p = 0.029, p_t = 0.029, d_z = 0.05
- vs dense: +0.055 [+0.045, +0.066], p = 5e-06, p_t = 3.1e-24, d_z = 0.25
- vs bm25: +0.174 [+0.158, +0.191], p = 5e-06, p_t = 1.9e-81, d_z = 0.50
- vs cross-encoder: +0.021 [+0.014, +0.028], p = 5e-06, p_t = 4.6e-09, d_z = 0.15
- per dataset vs dense: ahead in 9/10, ahead with Holm p < 0.05 in 6
- per dataset vs cross-encoder: ahead in 9/10, ahead with Holm p < 0.05 in 2

Paired difference of no-LM (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):

- vs full (MPUP): -0.080 [-0.092, -0.068], p = 5e-06, p_t = 6.4e-37, d_z = -0.32
- vs no-generation: -0.074 [-0.086, -0.063], p = 5e-06, p_t = 6.3e-35, d_z = -0.31
- vs reader-1.5B: -0.021 [-0.027, -0.013], p = 5e-06, p_t = 2e-08, d_z = -0.14
- vs reader-0.5B: -0.005 [-0.010, -0.000], p = 0.029, p_t = 0.029, d_z = -0.05
- vs dense: +0.050 [+0.040, +0.060], p = 5e-06, p_t = 5.8e-22, d_z = 0.24
- vs bm25: +0.169 [+0.153, +0.185], p = 5e-06, p_t = 3.3e-79, d_z = 0.49
- vs cross-encoder: +0.016 [+0.009, +0.022], p = 5e-06, p_t = 1.6e-06, d_z = 0.12
- per dataset vs dense: ahead in 9/10, ahead with Holm p < 0.05 in 7
- per dataset vs cross-encoder: ahead in 9/10, ahead with Holm p < 0.05 in 1

Reader runs (hidden states + cross-encoder, all 128,467 pairs):

- reader-1.5B: {'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'ce_model': 'BAAI/bge-reranker-v2-m3', 'layers': [14, 21], 'seconds': 2653, 'pairs': 128467, 'peak_gpu_gib': 7.99}
- reader-0.5B: {'model': 'Qwen/Qwen2.5-0.5B-Instruct', 'ce_model': 'BAAI/bge-reranker-v2-m3', 'layers': [12, 18], 'seconds': 1572, 'pairs': 128467, 'peak_gpu_gib': 5.46}
