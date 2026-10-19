# Feature-group subsets

Leave-one-dataset-out, fixed configuration (no selection). Macro within-query Spearman; paired difference to dense similarity over all held-out queries (95% bootstrap CI).

## data/processed/multidataset

| config | features | ρ | − dense |
|---|---|---|---|
| gbr(100,d2)-centered | ALL | 0.337 | +0.045 [+0.020, +0.068] |
| gbr(100,d2)-centered | D1+D2+D3+D5 | 0.335 | +0.043 [+0.018, +0.066] |
| gbr(100,d2)-centered | D1+D3+D4+D5 | 0.336 | +0.045 [+0.020, +0.069] |
| gbr(100,d2)-centered | D1+D3+D5 | 0.336 | +0.045 [+0.021, +0.068] |
| gbr(100,d2)-centered | D1+D5 | 0.322 | +0.032 [+0.010, +0.054] |
| gbr(100,d2)-centered | D1+D3 | 0.298 | +0.012 [-0.007, +0.031] |
| gbr(100,d2)-centered | D1 | 0.288 | +0.002 [-0.008, +0.011] |
| ridge(a=1000)-centered | ALL | 0.334 | +0.042 [+0.018, +0.066] |
| ridge(a=1000)-centered | D1+D2+D3+D5 | 0.336 | +0.044 [+0.020, +0.069] |
| ridge(a=1000)-centered | D1+D3+D4+D5 | 0.337 | +0.044 [+0.021, +0.069] |
| ridge(a=1000)-centered | D1+D3+D5 | 0.340 | +0.048 [+0.023, +0.073] |
| ridge(a=1000)-centered | D1+D5 | 0.324 | +0.031 [+0.008, +0.056] |
| ridge(a=1000)-centered | D1+D3 | 0.301 | +0.014 [-0.005, +0.033] |
| ridge(a=1000)-centered | D1 | 0.283 | -0.003 [-0.011, +0.005] |

## data/processed/multidataset_seed1

| config | features | ρ | − dense |
|---|---|---|---|
| gbr(100,d2)-centered | ALL | 0.336 | +0.061 [+0.035, +0.087] |
| gbr(100,d2)-centered | D1+D2+D3+D5 | 0.334 | +0.059 [+0.031, +0.085] |
| gbr(100,d2)-centered | D1+D3+D4+D5 | 0.337 | +0.061 [+0.034, +0.088] |
| gbr(100,d2)-centered | D1+D3+D5 | 0.335 | +0.061 [+0.033, +0.087] |
| gbr(100,d2)-centered | D1+D5 | 0.297 | +0.024 [+0.003, +0.044] |
| gbr(100,d2)-centered | D1+D3 | 0.314 | +0.042 [+0.018, +0.066] |
| gbr(100,d2)-centered | D1 | 0.273 | +0.002 [-0.010, +0.013] |
| ridge(a=1000)-centered | ALL | 0.332 | +0.057 [+0.029, +0.086] |
| ridge(a=1000)-centered | D1+D2+D3+D5 | 0.330 | +0.055 [+0.027, +0.082] |
| ridge(a=1000)-centered | D1+D3+D4+D5 | 0.335 | +0.060 [+0.033, +0.089] |
| ridge(a=1000)-centered | D1+D3+D5 | 0.332 | +0.058 [+0.030, +0.086] |
| ridge(a=1000)-centered | D1+D5 | 0.302 | +0.027 [+0.003, +0.049] |
| ridge(a=1000)-centered | D1+D3 | 0.309 | +0.036 [+0.010, +0.060] |
| ridge(a=1000)-centered | D1 | 0.278 | +0.005 [-0.003, +0.012] |
