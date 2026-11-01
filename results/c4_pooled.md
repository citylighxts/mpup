# C4 pooled over three target LLMs

Reader tier (reader 3B, all features, answer labels, fitted once on Qwen's development questions) against the utility judgments of each target LLM. Difference = macro mean over Qwen2.5-7B, Mistral-7B and OLMo-2-7B of the mean per-query difference in r_s; 95% bootstrap CI over questions (10000 draws) and sign-flip p with all LLMs of a question flipped together (20000 draws).

| comparison | Qwen | Mistral | OLMo | macro difference [95% CI] | p |
|---|---|---|---|---|---|
| reader − PVS w/o A | +0.029 | +0.036 | +0.092 | +0.052 [+0.043, +0.061] | 5e-05 |
| reader − PVS w/ A | +0.011 | -0.018 | +0.019 | +0.004 [-0.006, +0.014] | 0.45 |
| reader − likelihood | +0.046 | -0.035 | +0.038 | +0.016 [+0.001, +0.031] | 0.03 |
| reader − LVR w/ A | +0.078 | +0.089 | +0.201 | +0.123 [+0.112, +0.133] | 5e-05 |
| reader − answer confidence | +0.365 | +0.015 | +0.252 | +0.211 [+0.194, +0.227] | 5e-05 |
| reader − best judgment of each LLM (oracle: PVS w/ A, likelihood, PVS w/ A) | +0.011 | -0.035 | +0.019 | -0.002 [-0.013, +0.010] | 0.77 |

1824 distinct test questions, 5038 (LLM, question) pairs.

r_s below is the mean over questions (the paper's r_s averages over datasets first, so it is about 0.005 lower); the differences above are means over questions in both.

| r_s | Qwen2.5-7B | Mistral-7B | OLMo-2-7B | macro |
|---|---|---|---|---|
| reader | 0.303 | 0.276 | 0.288 | 0.289 |
| PVS w/o A | 0.274 | 0.240 | 0.196 | 0.237 |
| PVS w/ A | 0.292 | 0.295 | 0.269 | 0.285 |
| likelihood | 0.257 | 0.311 | 0.250 | 0.273 |
| LVR w/ A | 0.225 | 0.187 | 0.087 | 0.166 |
| answer confidence | -0.063 | 0.261 | 0.036 | 0.078 |
