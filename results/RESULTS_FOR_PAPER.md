# RESULTS_FOR_PAPER — multi-seed replication

Seeds: [0, 1, 2, 3, 4]  (N_SEEDS=5). Methods: ['mean', 'locf', 'rf', 'xgb', 'mice', 'saits', 'brits', 'unet', 'ae', 'gnn', 'stgnn'].

Numbers are **pooled R2 across the 19 LOO folds**, reported as **mean±std across seeds**. Aggregate `all` pools the 3 logs; per-log (GR/RHOB/NPHI) is reported because `all` is inflated by between-channel variance (reviewer M2).

## Aggregation formula (exact)

- Per (scenario, log): concatenate masked (pred,true) points over **all 19 folds**, then a **single** R2 = 1 − Σ(pred−true)²/Σ(true−mean)², in **original physical units**, on artificially-masked positions only. (`harness.pooled_agg`). `all` = the three logs pooled together.


## Scenario: single

| method | R2 all | R2 GR | R2 RHOB | R2 NPHI |
|---|---|---|---|---|
| mean | 0.581±0.035 | -0.026±0.002 | -0.089±0.011 | -0.047±0.003 |
| locf | 0.992±0.003 | 0.980±0.007 | 0.975±0.009 | 0.980±0.005 |
| rf | 0.626±0.025 | 0.081±0.032 | 0.453±0.053 | 0.599±0.014 |
| xgb | 0.700±0.035 | 0.265±0.047 | 0.541±0.037 | 0.663±0.012 |
| mice | 0.747±0.043 | 0.382±0.065 | 0.557±0.021 | 0.682±0.022 |
| saits | 0.995±0.004 | 0.987±0.009 | 0.987±0.005 | 0.996±0.000 |
| brits | 0.997±0.001 | 0.993±0.003 | 0.994±0.002 | 0.995±0.001 |
| unet | 0.910±0.016 | 0.779±0.029 | 0.733±0.083 | 0.797±0.019 |
| ae | 0.835±0.007 | 0.594±0.027 | 0.522±0.034 | 0.546±0.036 |
| gnn | 0.731±0.032 | 0.340±0.040 | 0.541±0.034 | 0.478±0.075 |
| stgnn | 0.832±0.027 | 0.588±0.061 | 0.647±0.032 | 0.551±0.114 |

**Zero-skill MEAN floor** (single): all=0.581±0.035, GR=-0.026±0.002, RHOB=-0.089±0.011, NPHI=-0.047±0.003. Note the gap: aggregate `all` is far above 0 even though per-log is ≈0 → confirms the aggregate inflation.

- Top-rank (pooled R2 all) per seed: ['brits', 'brits', 'brits', 'brits', 'saits'] → UNSTABLE top-rank: {'brits': 4, 'saits': 1}.

## Scenario: block

| method | R2 all | R2 GR | R2 RHOB | R2 NPHI |
|---|---|---|---|---|
| mean | 0.563±0.014 | -0.025±0.003 | -0.084±0.002 | -0.049±0.005 |
| locf | 0.847±0.010 | 0.642±0.029 | 0.553±0.087 | 0.435±0.039 |
| rf | 0.610±0.018 | 0.085±0.039 | 0.478±0.033 | 0.585±0.022 |
| xgb | 0.688±0.009 | 0.269±0.013 | 0.560±0.036 | 0.649±0.023 |
| mice | 0.735±0.019 | 0.379±0.047 | 0.571±0.047 | 0.665±0.020 |
| saits | 0.801±0.013 | 0.533±0.018 | 0.447±0.040 | 0.522±0.005 |
| brits | 0.643±0.019 | 0.162±0.020 | 0.162±0.049 | 0.518±0.015 |
| unet | 0.772±0.017 | 0.463±0.053 | 0.609±0.057 | 0.684±0.040 |
| ae | 0.811±0.008 | 0.557±0.028 | 0.482±0.041 | 0.502±0.021 |
| gnn | 0.710±0.030 | 0.320±0.078 | 0.559±0.026 | 0.444±0.060 |
| stgnn | 0.725±0.048 | 0.354±0.119 | 0.536±0.022 | 0.476±0.053 |

**Zero-skill MEAN floor** (block): all=0.563±0.014, GR=-0.025±0.003, RHOB=-0.084±0.002, NPHI=-0.049±0.005. Note the gap: aggregate `all` is far above 0 even though per-log is ≈0 → confirms the aggregate inflation.

- Top-rank (pooled R2 all) per seed: ['locf', 'locf', 'locf', 'locf', 'locf'] → STABLE top-rank = **locf** across all 5 seeds.

## Scenario: profile

| method | R2 all | R2 GR | R2 RHOB | R2 NPHI |
|---|---|---|---|---|
| mean | 0.581±0.022 | -0.026±0.004 | -0.089±0.019 | -0.051±0.005 |
| locf | 0.581±0.022 | -0.026±0.004 | -0.089±0.019 | -0.051±0.005 |
| rf | 0.633±0.021 | 0.100±0.049 | 0.477±0.042 | 0.599±0.023 |
| xgb | 0.704±0.018 | 0.275±0.049 | 0.563±0.033 | 0.660±0.019 |
| mice | 0.752±0.025 | 0.391±0.063 | 0.573±0.026 | 0.656±0.032 |
| saits | 0.603±0.025 | 0.027±0.012 | -0.018±0.021 | 0.091±0.010 |
| brits | 0.557±0.026 | -0.086±0.018 | -0.180±0.024 | 0.296±0.007 |
| unet | 0.755±0.021 | 0.400±0.027 | 0.582±0.031 | 0.665±0.037 |
| ae | 0.623±0.018 | 0.077±0.038 | 0.350±0.030 | 0.409±0.032 |
| gnn | 0.721±0.013 | 0.315±0.045 | 0.566±0.041 | 0.434±0.042 |
| stgnn | 0.738±0.028 | 0.360±0.043 | 0.513±0.066 | 0.483±0.047 |

**Zero-skill MEAN floor** (profile): all=0.581±0.022, GR=-0.026±0.004, RHOB=-0.089±0.019, NPHI=-0.051±0.005. Note the gap: aggregate `all` is far above 0 even though per-log is ≈0 → confirms the aggregate inflation.

- Top-rank (pooled R2 all) per seed: ['stgnn', 'mice', 'mice', 'unet', 'mice'] → UNSTABLE top-rank: {'mice': 3, 'unet': 1, 'stgnn': 1}.

## Scenario: blackout

| method | R2 all | R2 GR | R2 RHOB | R2 NPHI |
|---|---|---|---|---|
| mean | 0.562±0.011 | -0.025±0.001 | -0.083±0.003 | -0.050±0.001 |
| locf | 0.784±0.013 | 0.495±0.018 | 0.348±0.062 | 0.327±0.019 |
| rf | 0.385±0.017 | -0.437±0.012 | -0.163±0.041 | 0.003±0.021 |
| xgb | -0.793±0.071 | -3.192±0.107 | -6.253±0.392 | -2.279±0.201 |
| mice | 0.499±0.014 | -0.172±0.026 | 0.041±0.033 | 0.146±0.024 |
| saits | 0.683±0.011 | 0.260±0.012 | 0.250±0.015 | 0.330±0.020 |
| brits | 0.600±0.013 | 0.065±0.007 | 0.044±0.012 | 0.181±0.017 |
| unet | 0.588±0.023 | 0.036±0.061 | 0.129±0.019 | 0.245±0.046 |
| ae | 0.767±0.013 | 0.456±0.025 | 0.361±0.030 | 0.375±0.024 |
| gnn | 0.636±0.009 | 0.150±0.020 | 0.137±0.039 | -0.148±0.050 |
| stgnn | 0.623±0.023 | 0.117±0.064 | 0.060±0.085 | -0.162±0.072 |

**Zero-skill MEAN floor** (blackout): all=0.562±0.011, GR=-0.025±0.001, RHOB=-0.083±0.003, NPHI=-0.050±0.001. Note the gap: aggregate `all` is far above 0 even though per-log is ≈0 → confirms the aggregate inflation.

- Top-rank (pooled R2 all) per seed: ['locf', 'locf', 'locf', 'locf', 'locf'] → STABLE top-rank = **locf** across all 5 seeds.

## Significance of contested pairs (paired Wilcoxon, 19 folds)

| scenario | A vs B | seeds significant (p<0.05) |
|---|---|---|
| blackout | locf vs ae | 1/5 |
| blackout | locf vs brits | 5/5 |
| blackout | locf vs gnn | 5/5 |
| blackout | locf vs mice | 5/5 |
| blackout | locf vs rf | 5/5 |
| blackout | locf vs saits | 5/5 |
| blackout | locf vs stgnn | 5/5 |
| blackout | locf vs unet | 5/5 |
| blackout | locf vs xgb | 5/5 |
| profile | gnn vs mice | 0/5 |
| profile | stgnn vs gnn | 0/5 |
| profile | stgnn vs mice | 0/5 |
| profile | unet vs gnn | 0/5 |
| profile | unet vs mice | 2/5 |
| profile | unet vs stgnn | 0/5 |
| profile | xgb vs gnn | 0/5 |
| profile | xgb vs mice | 0/5 |
| profile | xgb vs stgnn | 0/5 |

## Friedman (19 folds as blocks, per seed)

| scenario | seed | n_methods | chi2 | p | verdict |
|---|---|---|---|---|---|
| single | 0 | 11 | 156.852 | 1.446e-28 | sig |
| single | 1 | 11 | 143.120 | 9.662e-26 | sig |
| single | 2 | 11 | 144.909 | 4.147e-26 | sig |
| single | 3 | 11 | 152.718 | 1.028e-27 | sig |
| single | 4 | 11 | 147.847 | 1.033e-26 | sig |
| block | 0 | 11 | 70.364 | 3.772e-11 | sig |
| block | 1 | 11 | 59.455 | 4.596e-09 | sig |
| block | 2 | 11 | 58.182 | 7.989e-09 | sig |
| block | 3 | 11 | 66.995 | 1.680e-10 | sig |
| block | 4 | 11 | 64.431 | 5.208e-10 | sig |
| profile | 0 | 11 | 48.428 | 5.183e-07 | sig |
| profile | 1 | 11 | 62.495 | 1.218e-09 | sig |
| profile | 2 | 11 | 45.169 | 2.027e-06 | sig |
| profile | 3 | 11 | 40.440 | 1.417e-05 | sig |
| profile | 4 | 11 | 44.968 | 2.204e-06 | sig |
| blackout | 0 | 11 | 113.100 | 1.263e-19 | sig |
| blackout | 1 | 11 | 107.359 | 1.817e-18 | sig |
| blackout | 2 | 11 | 106.258 | 3.025e-18 | sig |
| blackout | 3 | 11 | 102.660 | 1.597e-17 | sig |
| blackout | 4 | 11 | 113.847 | 8.922e-20 | sig |

_See results/significance/ for Nemenyi matrices + CD diagrams._
