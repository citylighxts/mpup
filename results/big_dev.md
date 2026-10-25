# Variant comparison on the development questions

Leave-one-dataset-out on 5031 development questions; macro within-query Spearman with Eq. (1).
Descriptive: the best variant is chosen here, so its score is optimistic. The test questions are not used.

| variant | ρ |
|---|---|
| gbr(500,d4)|u|ce|probe | 0.346 |
| gbr(300,d3)|u|ce|probe | 0.345 |
| gbr(100,d2)|u|ce|probe | 0.344 |
| gbr(300,d3)|u|probe | 0.344 |
| gbr(500,d4)|u|probe | 0.344 |
| gbr(100,d2)|u|probe | 0.344 |
| gbr(100,d2)|mix|ce|probe | 0.320 |
| gbr(300,d3)|mix|ce|probe | 0.319 |
| gbr(500,d4)|mix|ce|probe | 0.317 |
| gbr(300,d3)|mix|probe | 0.315 |
| gbr(100,d2)|mix|probe | 0.314 |
| gbr(500,d4)|mix|probe | 0.312 |
| gbr(500,d4)|u|ce | 0.284 |
| gbr(300,d3)|u|ce | 0.284 |
| gbr(100,d2)|u|ce | 0.282 |
| gbr(100,d2)|mix|ce | 0.273 |
| gbr(300,d3)|mix|ce | 0.271 |
| gbr(500,d4)|u | 0.269 |
| gbr(100,d2)|u | 0.269 |
| gbr(300,d3)|u | 0.268 |
| gbr(500,d4)|mix|ce | 0.268 |
| gbr(100,d2)|mix | 0.262 |
| gbr(300,d3)|mix | 0.256 |
| gbr(500,d4)|mix | 0.255 |
| baseline: dense | 0.227 |
| baseline: bm25 | 0.105 |
| baseline: cross-encoder | 0.257 |

Chosen: `gbr(500,d4)|u|ce|probe`
