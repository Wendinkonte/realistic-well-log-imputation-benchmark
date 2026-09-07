# Recap par scenario — R2 poole (mean+-std sur les seeds, 11 methodes)

| method | family | single | block | profile | blackout |
|---|---|---|---|---|---|
| MEAN | baseline | 0.581±0.035 | 0.563±0.014 | 0.581±0.022 | 0.561±0.011 |
| LOCF | baseline | 0.992±0.003 | 0.848±0.010 | 0.581±0.022 | 0.784±0.013 |
| RF | tabular | 0.626±0.025 | 0.610±0.018 | 0.633±0.021 | 0.385±0.017 |
| XGBoost | tabular | 0.700±0.035 | 0.688±0.009 | 0.704±0.018 | -0.793±0.071 |
| MICE | tabular | 0.747±0.043 | 0.736±0.019 | 0.752±0.025 | 0.499±0.015 |
| SAITS | sequential | 0.995±0.004 | 0.801±0.013 | 0.603±0.025 | 0.683±0.011 |
| BRITS | sequential | 0.997±0.002 | 0.643±0.019 | 0.557±0.026 | 0.600±0.013 |
| U-Net | sequential | 0.910±0.016 | 0.772±0.017 | 0.755±0.021 | 0.588±0.023 |
| AE | sequential | 0.835±0.007 | 0.811±0.008 | 0.623±0.018 | 0.767±0.013 |
| GNN | spatial | 0.731±0.032 | 0.711±0.030 | 0.721±0.013 | 0.636±0.009 |
| ST-GNN | spatial | 0.832±0.027 | 0.725±0.048 | 0.739±0.028 | 0.623±0.023 |

## single
- **Podium** : 1. BRITS (0.997±0.002, sequential), 2. SAITS (0.995±0.004, sequential), 3. LOCF (0.992±0.003, baseline)
- Famille gagnante : **sequential** | plancher zero-skill MEAN = 0.581
- BRITS (SOTA mono-log) : rang **1/11** | etendue R2 (1er-dernier) = 0.416

## block
- **Podium** : 1. LOCF (0.848±0.010, baseline), 2. AE (0.811±0.008, sequential), 3. SAITS (0.801±0.013, sequential)
- Famille gagnante : **baseline** | plancher zero-skill MEAN = 0.563
- BRITS (SOTA mono-log) : rang **9/11** | etendue R2 (1er-dernier) = 0.284

## profile
- **Podium** : 1. U-Net (0.755±0.021, sequential), 2. MICE (0.752±0.025, tabular), 3. ST-GNN (0.739±0.028, spatial)
- Famille gagnante : **sequential** | plancher zero-skill MEAN = 0.581
- BRITS (SOTA mono-log) : rang **11/11** | etendue R2 (1er-dernier) = 0.198

## blackout
- **Podium** : 1. LOCF (0.784±0.013, baseline), 2. AE (0.767±0.013, sequential), 3. SAITS (0.683±0.011, sequential)
- Famille gagnante : **baseline** | plancher zero-skill MEAN = 0.561
- BRITS (SOTA mono-log) : rang **6/11** | etendue R2 (1er-dernier) = 1.577

