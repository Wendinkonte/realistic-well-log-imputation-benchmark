# Robustesse a l'alea (inter-seed) — coeff. de variation du R2 poole

CV% = ecart-type / |moyenne| du R2 poole sur les seeds (plus petit = plus robuste).
L'alea couvre l'init torch/numpy ET le tirage seed-dependant des positions masquees.

| method | CV% moyen (sur 4 scenarios) |
|---|---|
| AE | 1.60 |
| LOCF | 1.71 |
| SAITS | 1.94 |
| BRITS | 2.44 |
| U-Net | 2.65 |
| GNN | 2.96 |
| MEAN | 3.55 |
| MICE | 3.64 |
| RF | 3.70 |
| ST-GNN | 4.33 |
| XGBoost | 4.43 |

(detail par scenario dans robustness_seed_stability.csv)

