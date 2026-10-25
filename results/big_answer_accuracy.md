# Answer accuracy on the held-out test questions

2093 test questions; Acc = 0.5 relaxed EM + 0.5 token F1, macro mean over datasets; answering model Qwen/Qwen2.5-7B-Instruct. mpup+ = `gbr(500,d4)|u|ce|probe`.

| method | top-1 | top-3 |
|---|---|---|
| none (no passage) | 0.246 | |
| mpup+ | 0.488 | 0.494 |
| mpup (paper) | 0.409 | 0.480 |
| cross-encoder | 0.400 | 0.475 |
| dense | 0.376 | 0.444 |
| bm25 | 0.297 | 0.370 |
| random | 0.196 | 0.266 |
| oracle | 0.654 | 0.552 |

Paired difference of mpup+ (95% bootstrap CI, sign-flip permutation p):

- mpup+ vs mpup (paper) @1: +0.079 [+0.064, +0.094], p = 1e-05
- mpup+ vs cross-encoder @1: +0.088 [+0.073, +0.103], p = 1e-05
- mpup+ vs dense @1: +0.111 [+0.094, +0.128], p = 1e-05
- mpup+ vs bm25 @1: +0.190 [+0.172, +0.209], p = 1e-05
- mpup+ vs random @1: +0.293 [+0.273, +0.315], p = 1e-05
- mpup+ vs none @1: +0.245 [+0.225, +0.268], p = 1e-05
- mpup+ vs mpup (paper) @3: +0.014 [+0.000, +0.028], p = 0.042
- mpup+ vs cross-encoder @3: +0.019 [+0.006, +0.033], p = 0.004
- mpup+ vs dense @3: +0.050 [+0.035, +0.065], p = 1e-05
- mpup+ vs bm25 @3: +0.123 [+0.107, +0.140], p = 1e-05
- mpup+ vs random @3: +0.229 [+0.210, +0.248], p = 1e-05
- mpup+ vs none @3: +0.251 [+0.231, +0.273], p = 1e-05
