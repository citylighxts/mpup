# Answer accuracy on the held-out test questions

2093 test questions; Acc = 0.5 relaxed EM + 0.5 token F1, macro mean over datasets; answering model mistralai/Mistral-7B-Instruct-v0.3. mpup+ = `gbr(500,d4)|u|ce|probe`.

| method | top-1 | top-3 |
|---|---|---|
| none (no passage) | 0.280 | |
| mpup+ | 0.479 | 0.475 |
| mpup (paper) | 0.399 | 0.453 |
| cross-encoder | 0.391 | 0.450 |
| dense | 0.380 | 0.424 |
| bm25 | 0.309 | 0.349 |
| random | 0.212 | 0.271 |
| oracle | 0.632 | 0.519 |

Paired difference of mpup+ (95% bootstrap CI, sign-flip permutation p):

- mpup+ vs mpup (paper) @1: +0.079 [+0.065, +0.092], p = 1e-05
- mpup+ vs cross-encoder @1: +0.086 [+0.072, +0.100], p = 1e-05
- mpup+ vs dense @1: +0.098 [+0.084, +0.112], p = 1e-05
- mpup+ vs bm25 @1: +0.168 [+0.152, +0.184], p = 1e-05
- mpup+ vs random @1: +0.268 [+0.249, +0.286], p = 1e-05
- mpup+ vs none @1: +0.201 [+0.182, +0.221], p = 1e-05
- mpup+ vs mpup (paper) @3: +0.022 [+0.011, +0.034], p = 0.00036
- mpup+ vs cross-encoder @3: +0.024 [+0.012, +0.036], p = 0.00011
- mpup+ vs dense @3: +0.050 [+0.037, +0.063], p = 1e-05
- mpup+ vs bm25 @3: +0.124 [+0.110, +0.139], p = 1e-05
- mpup+ vs random @3: +0.204 [+0.188, +0.222], p = 1e-05
- mpup+ vs none @3: +0.197 [+0.178, +0.216], p = 1e-05
