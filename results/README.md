# Released results

Every file here is a **derived statistic or aggregate result**. None contains a real well
name, a coordinate, or a measured log value.

Two releases live side by side:

* **Revision R1 (18 wells, the revised article)**, under `revision_r1/`: 18 leave-one-well-out
  folds, five scenarios (including `blackout_emp`), 28 methods (the 11 of the article plus 17
  revision baselines and controls). Wells appear as **W01–W18**, numbered in the order of
  `folds_loo.json`, so that `well = W{fold+1:02d}`. This W-id ↔ fold correspondence **is
  published**, with BRGM approval, in `revision_r1/error_stats/` and
  `configs/benchmark/folds_loo_18wells.yaml`.
* **v1.0 (original submission, 19 wells)**, everything else: 19 folds, four scenarios,
  11 methods. `raw_metrics.csv` and `raw_metrics_full.csv` carry `fold` = 0..18 for every
  (seed, method, scenario, log) cell — 5 seeds x 11 methods x 4 scenarios x 19 folds x 4 log
  channels — and `significance/friedman.csv` blocks on those same 19 folds. In the v1.0
  files a W-identifier (W01–W19) appears only in the two robustness sub-studies, which were
  run on a deliberate subset of wells (two for the multi-window sweep, eight for the
  sensitivity probe). The v1.0 W01–W19 numbering is **not** the R1 W01–W18 numbering, and
  no correspondence between the two is published. The v1.0 fold structure is written out
  in `configs/benchmark/folds_loo_19wells.yaml`.

The v1.0 files are kept unchanged for transparency: the revised article still cites some
of them for the original-submission analysis (see *Status of files* below).

Full inventory, and what is withheld and why: `docs/DATA_AVAILABILITY.md`.

## Layout

```
results/
├── revision_r1/                 REVISION R1, 18 wells (the revised article)
│   ├── table_main_revision.csv, table_perlog_revision.csv   main and per-log tables
│   ├── per_well_nrmse.csv       NRMSE per (method, scenario, well index)
│   ├── friedman_wellwise.csv, wilcoxon_*_holm.csv   well-level Friedman, exact Wilcoxon + Holm
│   ├── friedman_main11.csv      well-level Friedman restricted to the 11 methods of the article
│   ├── seed_variance.csv        initialisation vs evaluation-mask seed variance
│   ├── neighbour_gain_spearman.csv  graph-model gain vs neighbour availability
│   ├── run_metadata.json        commits, seeds, methods, scenarios, test settings
│   ├── error_stats/             per-well sufficient statistics (recompute everything)
│   ├── data_qc/                 depth-step QC, log quality report, observed missingness
│   ├── ablations/               leak_matched.csv (matched leakage experiment)
│   └── figures/                 released figures of the revised article
│
├── results_initial_run.json     v1.0 initial single-seed run (9 methods, 19 folds)
├── aggregated.csv               pooled R² per (method, scenario, log): mean ± std, 95 % CI over 5 seeds
├── aggregated_full.csv          the same for MAE, RMSE, Pearson r
├── raw_metrics.csv              per (seed, method, scenario, fold, log)
├── raw_metrics_full.csv         idem, all metrics
├── pooled_per_seed_full.csv     pooled metrics per seed
├── summary_r2.csv               single-seed summary
├── summary_full.csv             single-seed summary, all metrics and logs
├── table_main.csv               main results table
├── table_perlog.csv             per-log results table
├── bootstrap_ci.csv             by-well cluster-bootstrap intervals
├── bootstrap_pairwise.csv       paired bootstrap comparisons
├── rank_reversal.csv            rank of each method per scenario
├── scenario_recap.{csv,md}      per-scenario podium
├── robustness_seed_stability.*  seed-to-seed variation of each method
├── robustness_severity.csv      degradation with gap severity
├── finding_A_diagnostic.txt     per-fold averaging diagnostic
├── finding_B_wilcoxon.csv       XGBoost vs the profile podium
├── RESULTS_FOR_PAPER.md         narrative synthesis of the campaign
│
├── significance/                Friedman, Nemenyi (20 matrices), CD diagrams (20),
│                                paired Wilcoxon, ranking stability
├── ablations/                   ablation_leak.*  (grid-row leakage ON vs OFF)
│                                leak_randomsplit.*  (window-level split)
├── robustness/                  sensitivity_probe.csv, multiwindow_sweep.csv (wells as
│                                W01–W19), plus their figures
└── figures/                     results-derived figures: R² boxplots, per-log vs
                                 aggregate, ranking stability, four-metric panels,
                                 convergence curves
```

## Status of files

Legend:
**R1** = R1 (18 wells, used in the paper) ·
**v1.0-S** = v1.0 original submission, 19 wells, kept for transparency (cited in Supplementary S-original) ·
**v1.0** = v1.0, not used in the revised paper.

| file or figure | status |
|---|---|
| `revision_r1/` (all tables, `run_metadata.json`) | R1 |
| `revision_r1/error_stats/` | R1 |
| `revision_r1/data_qc/` (`depth_qc.csv`, `quality_report.csv`, `missingness_real.json`, `screened_wells_summary.csv`) | R1 |
| `revision_r1/ablations/leak_matched.csv` | R1 |
| `revision_r1/figures/02_log_distributions_r1`, `03_log_crossplots_r1` | R1 |
| `revision_r1/figures/fig4_aggregation` | v1.0-S (drawn from `results_initial_run.json`; released with the R1 figures because the revised supplement includes it) |
| `revision_r1/figures/missingness_scenarios_synthetic` | illustration on synthetic data, not an article figure |
| `aggregated.csv` | v1.0-S |
| `results_initial_run.json` | v1.0-S: initial single-seed run of the original submission (9 methods, 19 folds), source of `fig4_aggregation`; the five-seed tables are in `aggregated.csv` |
| `raw_metrics_full.csv` | v1.0-S (per-fold values behind the per-fold means quoted in S-original) |
| `figures/08_convergence_curves` | v1.0-S |
| `robustness/fig_sensitivity_probe`, `robustness/sensitivity_probe.csv` | v1.0-S |
| `aggregated_full.csv`, `raw_metrics.csv`, `pooled_per_seed_full.csv`, `summary_r2.csv`, `summary_full.csv`, `table_main.csv`, `table_perlog.csv`, `bootstrap_ci.csv`, `bootstrap_pairwise.csv`, `rank_reversal.csv`, `scenario_recap.*`, `robustness_seed_stability.*`, `robustness_severity.csv`, `finding_A_diagnostic.txt`, `finding_B_wilcoxon.csv`, `RESULTS_FOR_PAPER.md` | v1.0 |
| `figures/09_r2_boxplots_by_scenario`, `10_perlog_vs_aggregate_r2`, `11_ranking_stability`, `metric_{r2,mae,rmse,cc}`, `overview_heatmap` | v1.0 |
| `significance/` (Friedman, Nemenyi, CD diagrams `cd_*`, Wilcoxon pairs, ranking stability) | v1.0 |
| `ablations/` (`ablation_leak.*`, `leak_randomsplit.*`) | v1.0 (superseded by `revision_r1/ablations/leak_matched.csv`) |
| `robustness/fig_multiwindow_sweep`, `robustness/multiwindow_sweep.csv` | v1.0 |

## Reading the numbers

**The pooled `all` channel is inflated.** It pools the three logs, so between-channel
variance inflates it: the zero-skill MEAN scores about 0.76 on it, not 0. Read `all` for
comparing methods *within* a scenario, never as an absolute skill level. The per-log R²
and the MEAN floor are reported alongside for exactly this reason.

*The two paragraphs below describe the v1.0 tables. The R1 tables use per-log R² as the
primary metric and the well as the statistical unit (see `revision_r1/run_metadata.json`).*

**Aggregation is pooled, not averaged per fold.** All 19 folds are concatenated and a
single metric computed. Averaging per fold lets a well contributing one window weigh as
much as a well contributing 237 — the reporting pitfall the paper demonstrates.

**The two ablation tables are single-seed paired experiments**, and are labelled as such
in the article. Everything else comes from the five-seed replication.

## Regenerating

```bash
# R1: every table of the revised article, from the released per-well statistics
python scripts/recompute_from_export.py --export results/revision_r1/error_stats --out OUT
# v1.0
python scripts/reproduce_tables.py     # needs the private per-fold dumps
python scripts/reproduce_figures.py    # runs on the released tables alone
```

`reproduce_figures.py` reproduces the four-metric panels pixel-identically from the
released tables, and the three pooled-R² figures to within rounding.
