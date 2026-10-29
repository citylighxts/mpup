# Transfer matrix: MPUP fitted on one LLM, scored against another LLM's utility

Rows: the LLM whose development labels, features and hidden states MPUP is fitted on (development questions: Qwen2.5-7B 5031, Mistral-7B 2005, OLMo-2-7B 2005). Columns: the LLM whose Eq. (1) utility is predicted, on its test questions. Cell: r_s (difference to the cross-encoder [95% CI], d_z).

| fitted on \ target | Qwen2.5-7B | Mistral-7B | OLMo-2-7B |
|---|---|---|---|
| Qwen2.5-7B | 0.356 (+0.095 [+0.083, +0.108], 0.37) | 0.313 (+0.068 [+0.056, +0.080], 0.26) | 0.312 (+0.065 [+0.053, +0.078], 0.26) |
| Mistral-7B | 0.327 (+0.067 [+0.054, +0.079], 0.25) | 0.373 (+0.128 [+0.114, +0.142], 0.44) | 0.303 (+0.056 [+0.043, +0.068], 0.22) |
| OLMo-2-7B | 0.318 (+0.058 [+0.046, +0.070], 0.24) | 0.293 (+0.049 [+0.037, +0.061], 0.19) | 0.316 (+0.069 [+0.058, +0.081], 0.29) |
| cross-encoder | 0.260 | 0.243 | 0.247 |
| dense | 0.226 | 0.210 | 0.211 |

Datasets where the cell is ahead of the cross-encoder with Holm p < 0.05:

- Qwen2.5-7B -> Qwen2.5-7B: 10/10
- Mistral-7B -> Qwen2.5-7B: 9/10
- OLMo-2-7B -> Qwen2.5-7B: 7/10
- Qwen2.5-7B -> Mistral-7B: 7/10
- Mistral-7B -> Mistral-7B: 10/10
- OLMo-2-7B -> Mistral-7B: 5/10
- Qwen2.5-7B -> OLMo-2-7B: 8/10
- Mistral-7B -> OLMo-2-7B: 5/10
- OLMo-2-7B -> OLMo-2-7B: 9/10
