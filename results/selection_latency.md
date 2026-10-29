# Selection-time cost per query

100 test questions, 17.2 candidates on average, NVIDIA RTX 6000 Ada Generation, bf16, one query at a time.

| tier | mean ms / query | median ms | peak GPU GiB |
|---|---|---|---|
| full (MPUP) | 2516 | 2534 | 21.54 |
| no-generation | 1900 | 1988 | 21.54 |
| reader-1.5B | 832 | 915 | 7.69 |
| reader-0.5B | 757 | 828 | 4.91 |
| no-LM | 695 | 769 | 3.47 |

| component | mean ms / query | median ms |
|---|---|---|
| hyde | 354.2 | 343.3 |
| dense | 129.3 | 126.8 |
| d2_d3 | 524.8 | 601.5 |
| d4 | 262.0 | 219.4 |
| d5 | 568.6 | 567.5 |
| hidden_M | 636.4 | 653.7 |
| ce | 39.3 | 36.0 |
| hidden_R1.5B | 137.3 | 139.3 |
| hidden_R0.5B | 61.9 | 55.5 |
| regressor | 1.3 | 1.3 |
