# Selection-time cost per query

100 test questions, 17.2 candidates on average, NVIDIA RTX 6000 Ada Generation, bf16, one query at a time.

| tier | mean ms / query | median ms | peak GPU GiB |
|---|---|---|---|
| reader-1.5B | 827 | 925 | 7.69 |
| reader-0.5B | 754 | 837 | 4.92 |
| no-LM | 690 | 781 | 3.47 |
| reader-1.5B, all features | 1785 | 1799 | 9.44 |
| reader-3B, all features | 2133 | 2096 | 12.4 |

| component | mean ms / query | median ms |
|---|---|---|
| dense | 130.0 | 128.0 |
| d2_d3 | 519.7 | 613.5 |
| ce | 38.8 | 35.5 |
| hidden_R1.5B | 137.5 | 138.5 |
| hidden_R0.5B | 63.7 | 57.1 |
| regressor | 1.3 | 1.3 |
| hyde_R1.5B | 512.5 | 486.1 |
| d4_R1.5B | 294.0 | 254.1 |
| d5_R1.5B | 151.6 | 153.3 |
| hidden_R1.5Bf | 137.3 | 137.8 |
| hyde_R3B | 672.7 | 581.0 |
| d4_R3B | 228.0 | 226.0 |
| d5_R3B | 267.8 | 264.2 |
| hidden_R3Bf | 274.9 | 280.0 |
