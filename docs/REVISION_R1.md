# Revision R1 — what changed in the code and why

Branch `revision-r1`. Reviewer comments are referenced as R1-n / R2-n.

## Data pipeline (`data/pipeline.py`)
* **Depth-step QC.** Wells whose native depth step differs from 0.1524 m (tolerance 1 mm),
  or with a depth jump > 2 steps or a non-positive step, are excluded and listed in
  `depth_qc.csv`. In the study corpus this excludes one well (0.5 m step, 72.5 m jump),
  which contributed one window but whose gaps, counted at the wrong step, had set the
  published median gap length. Benchmark: 18 wells.
* **Missingness statistics** (`missingness_real.json`, R1-2, R2-2): co-occurrence over
  depth samples (original definition) **and** over maximal intervals; shares of every
  absent-log combination; per-log gap-length quartiles in metres, all gaps and interior
  gaps only; gaps per well; the raw list of intervals (used by the empirical blackout).
* Out-of-range values are set to missing (unchanged behaviour; the article said "clipped").

## Masking and evaluation (`masking/`, `evaluation/`)
* `blackout` draws the number of absent logs from the rebuilt dataset
  (`dataset_cooccur`), not from hard-coded values.
* **Evaluation-mask seeds decoupled from model seeds** (R2-4): `--mask-seeds 0,1,2,3,4`.
  The first is the primary evaluation (same masks for every initialisation seed); the
  others re-evaluate the same trained model (`pools_maskvar/`). `scripts/seed_variance.py`
  separates initialisation and mask variability.
* **Empirical blackout** (`evaluation/empirical.py`, scenario `blackout_emp`, R1-2, R2-2):
  absent-log combination and length drawn jointly from the observed intervals, applied to
  the held-out well before windowing; windowed methods predict from overlapping windows;
  LOCF and interpolation use the whole well; `locf_win` keeps LOCF window-limited.

## Methods (`models/`)
* SAITS/BRITS now receive the master seed (in the original release they always used
  seed 0, so their between-seed spread reflected the evaluation masks only).
* `saits_mix`, `brits_mix`: trained on the curriculum-corrupted windows (R2-3).
* `xgb_miss` (missing predictors seen in training), `xgb_chain_asc/desc` (chained
  equations, imputation-order sensitivity) (R2-3, R1-9).
* Baselines (R1-8): `ridge`, `knn`, `interp_linear`, `interp_pchip`.
* Spatial baselines (R1-7): `idw`, `nnwell`, `okrige` (same depth, training wells only).
* Graph controls (R2-6): `*_noneigh` (identity graph), `*_shuffle` (kNN graph on permuted
  coordinates).

## Statistics (`statistics/wellwise.py`, `scripts/reanalyse_revision.py`)
* Primary metric: per-log pooled R² and their equal-weight mean (R2-1, R1-10).
* Well = statistical unit: seed-averaged per-well nRMSE; Friedman + Kendall's W; exact
  Wilcoxon, rank-biserial r, **Holm within each scenario family**; by-well cluster
  bootstrap of ΔR²_macro (R1-3, R1-4, R1-5, R2-5). Revision families
  (`wilcoxon_revision_holm.csv`) are separate from the pre-registered family.
* `scripts/neighbour_availability.py`: graph-model gain over its no-neighbour control
  against neighbour availability and distance (R2-6).
* `scripts/run_leak_matched.py`: window-split vs well-split on identical windows, masks
  and training-set size, varying same-well overlap and grid-row availability (R2-7).

## Running
`scripts/slurm/revision_r1.sh build | submit | analyse` (edit the paths and partitions first).

## Where each R1 output lives in this repository

| run output (`$OUT` / `$DATA` of `scripts/slurm/revision_r1.sh`) | released as | notes |
|---|---|---|
| `$OUT/revision/table_main_revision.csv`, `table_perlog_revision.csv`, `per_well_nrmse.csv`, `friedman_wellwise.csv`, `wilcoxon_wellwise_holm.csv`, `wilcoxon_allpairs_holm.csv`, `wilcoxon_revision_holm.csv`, `seed_variance.csv` | `results/revision_r1/` | unchanged |
| `$OUT/revision/neighbour_gain_spearman.csv` | `results/revision_r1/` | unchanged |
| `$OUT/revision/neighbour_availability.csv` | not released | per-well nearest-neighbour distance |
| run parameters | `results/revision_r1/run_metadata.json` | rebuilt without paths (commits `1a3fec5` / `cad5dbb`) |
| per-fold dumps (`pools_seeded/`, `pools_maskvar/`) | `results/revision_r1/error_stats/` | as per-well sufficient statistics; the dumps themselves are withheld |
| `$DATA/depth_qc.csv`, `quality_report.csv`, `missingness_real.json` | `results/revision_r1/data_qc/` | anonymised: W01–W18, `EXCLUDED`, non-triplet wells as counts |
| `$DATA/folds_loo.json` | `configs/benchmark/folds_loo_18wells.yaml` | W-ids only |
| `$OUT/ablations/leak_matched.csv` | `results/revision_r1/ablations/leak_matched.csv` | the `.npz` dumps are withheld |
| article figures | `results/revision_r1/figures/` | see its README for what is released and what is not |

Regenerate the R1 tables from the release: `python scripts/recompute_from_export.py
--export results/revision_r1/error_stats --out OUT`; the figures:
`scripts/make_revision_figures.py`.
