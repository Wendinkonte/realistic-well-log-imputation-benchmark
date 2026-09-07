# Released results

Every file here is a **derived statistic or aggregate result**. None contains a real well
name, a coordinate, or a measured log value. Wells appear only as the anonymous
identifiers **W01–W19** used in the publication, and most tables are keyed by fold index
rather than by well at all.

**All 19 leave-one-well-out folds are present.** `raw_metrics.csv` and
`raw_metrics_full.csv` carry `fold` = 0..18 for every (seed, method, scenario, log) cell —
5 seeds x 11 methods x 4 scenarios x 19 folds x 4 log channels — and `significance/friedman.csv`
blocks on those same 19 folds. A W-identifier appears explicitly only in the two
robustness sub-studies, which were run on a deliberate subset of wells (two for the
multi-window sweep, eight for the sensitivity probe); that is a design choice of those
probes, not a truncation of the benchmark. The correspondence between a W-identifier and a
fold index is not published: the W-numbering encodes an ordering of the corpus, so
releasing the mapping would weaken the anonymisation, and no analysis depends on it. The
fold structure itself is written out in `configs/benchmark/folds_loo_19wells.yaml`.

Full inventory, and what is withheld and why: `docs/DATA_AVAILABILITY.md`.

## Layout

```
results/
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

## Reading the numbers

**The pooled `all` channel is inflated.** It pools the three logs, so between-channel
variance inflates it: the zero-skill MEAN scores about 0.76 on it, not 0. Read `all` for
comparing methods *within* a scenario, never as an absolute skill level. The per-log R²
and the MEAN floor are reported alongside for exactly this reason.

**Aggregation is pooled, not averaged per fold.** All 19 folds are concatenated and a
single metric computed. Averaging per fold lets a well contributing one window weigh as
much as a well contributing 237 — the reporting pitfall the paper demonstrates.

**The two ablation tables are single-seed paired experiments**, and are labelled as such
in the article. Everything else comes from the five-seed replication.

## Regenerating

```bash
python scripts/reproduce_tables.py     # needs the private per-fold dumps
python scripts/reproduce_figures.py    # runs on the released tables alone
```

`reproduce_figures.py` reproduces the four-metric panels pixel-identically from the
released tables, and the three pooled-R² figures to within rounding.
