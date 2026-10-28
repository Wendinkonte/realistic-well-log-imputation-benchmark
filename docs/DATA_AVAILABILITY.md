# Data availability

## What is not released

The original and composite well-log data used in the study are owned by BRGM (French
Geological Survey) and are subject to restrictions. They cannot be redistributed.

Concretely, none of the following is present in this repository, and the `.gitignore`
blocks each class:

| Withheld | Why |
|---|---|
| Raw LAS files (measured passes) | BRGM-owned primary data |
| Composite LAS files | BRGM-owned; also blend measured data with model-derived infill |
| `well_logs.npz`, `slices.npz` | the cleaned and windowed measured values |
| `quality_report.csv`, `depth_qc.csv`, `missingness_real.json`, `folds_loo.json` as written by the pipeline | keyed by real well name; anonymised versions of the first three are released in `results/revision_r1/data_qc/` |
| Per-well rows for the screened wells that lack the GR+RHOB+NPHI triplet | not part of the benchmark; released as counts only (`screened_wells_summary.csv`) |
| Well coordinate table | real projected surface positions |
| Any well-name ↔ anonymous-id mapping | it would undo the anonymisation |
| The release-1.0 W-identifier ↔ fold-index correspondence, and any link between the 1.0 (W01–W19) and R1 (W01–W18) numberings | the 1.0 W-numbering encodes an ordering of the corpus (the R1 correspondence is published, see *Anonymisation*) |
| Per-well distance to the nearest neighbour (`neighbour_availability.csv`) | partial location information; only the correlations over the 18 wells are released (`neighbour_gain_spearman.csv`) |
| Pooled prediction dumps (`pool_*.npz`, `pools_seeded/`, `pools_maskvar/`, the `leak_matched_*.npz` of the matched leakage experiment) | they carry the ground-truth log values at every evaluated position, so they are derived proprietary data |
| Per-fold run logs | they print well identifiers |

The last row is worth stating plainly: a file can be "just results" and still be
proprietary. The pool dumps store the true measured value at each scored position, so
publishing them would publish an extract of the logs.

## What is released

Everything needed to inspect the method, check the protocol, and rebuild every
results-derived table and figure:

* **All research code** — the pipeline with depth-step QC, the masking generators
  (including the empirical blackout), the eleven methods and the seventeen baselines and
  controls of the revision, the leave-one-well-out harness, the metrics, the aggregation,
  the well-level significance tests, the bootstrap, and the leakage experiments.
* **Derived statistics and aggregate results**, containing no well name, coordinate or
  individual measured value. `results/README.md` gives, for each file, whether it belongs
  to the revised article (R1, 18 wells) or to release 1.0 (original submission, 19 wells).

**Revision R1 (18 wells, the revised article)**

| File | Content |
|---|---|
| `results/revision_r1/error_stats/` | per-(method, scenario, seed, fold, log) error sufficient statistics, fold index 0–17 and well W01–W18; recomputes every R1 table with `scripts/recompute_from_export.py` |
| `results/revision_r1/table_main_revision.csv`, `table_perlog_revision.csv` | main table (`r2_macro`, ranks, per-well NRMSE quartiles) and per-log R², MAE, RMSE |
| `results/revision_r1/per_well_nrmse.csv` | seed-averaged NRMSE per (method, scenario, well index) |
| `results/revision_r1/friedman_wellwise.csv` | Friedman test and Kendall's W per scenario, wells as blocks |
| `results/revision_r1/wilcoxon_wellwise_holm.csv`, `wilcoxon_allpairs_holm.csv`, `wilcoxon_revision_holm.csv` | exact Wilcoxon, rank-biserial r, Holm within each family; by-well bootstrap of ΔR²_macro |
| `results/revision_r1/seed_variance.csv` | initialisation vs evaluation-mask seed variability |
| `results/revision_r1/neighbour_gain_spearman.csv` | graph-model gain over its no-neighbour control vs neighbour availability and distance (correlations over the 18 wells) |
| `results/revision_r1/run_metadata.json` | commits, seeds, methods, scenarios and test settings of the R1 run |
| `results/revision_r1/data_qc/` | anonymised `depth_qc.csv`, `quality_report.csv`, `missingness_real.json`, and `screened_wells_summary.csv` (counts only) |
| `results/revision_r1/ablations/leak_matched.csv` | matched leakage experiment (`scripts/run_leak_matched.py`) |
| `results/revision_r1/figures/` | released figures of the revised article and a synthetic masking illustration (see its README) |

**Release 1.0 (original submission, 19 wells)**, keyed by fold index:

| File | Content |
|---|---|
| `results/results_initial_run.json` | initial single-seed run of the original submission (9 methods, 19 folds), source of `fig4_aggregation`; the five-seed tables are in `aggregated.csv` |
| `results/aggregated.csv` | pooled R² per (method, scenario, log): mean ± std and 95 % CI over the five seeds |
| `results/aggregated_full.csv` | the same for MAE, RMSE and Pearson r |
| `results/raw_metrics.csv`, `raw_metrics_full.csv` | per (seed, method, scenario, fold, log) metrics |
| `results/pooled_per_seed_full.csv` | pooled metrics per seed |
| `results/summary_r2.csv`, `summary_full.csv` | single-seed summary tables |
| `results/table_main.csv`, `table_perlog.csv` | the result tables reported in the paper |
| `results/bootstrap_ci.csv`, `bootstrap_pairwise.csv` | cluster-bootstrap intervals and paired comparisons |
| `results/rank_reversal.csv`, `scenario_recap.*`, `robustness_*.csv` | ranking and stability summaries |
| `results/significance/friedman.csv` | Friedman per (scenario, seed) |
| `results/significance/wilcoxon_pairs.csv` | paired Wilcoxon on the contested pairs |
| `results/significance/ranking_stability.csv` | rank of every method per seed |
| `results/significance/nemenyi_{scenario}_seed{S}.csv` | 20 Nemenyi post-hoc matrices |
| `results/significance/cd_{scenario}_seed{S}.{pdf,png}` | 20 critical-difference diagrams |
| `results/ablations/ablation_leak.{csv,json}` | grid-row leakage ablation (Section 7.4) |
| `results/ablations/leak_randomsplit.{csv,json}` | window-split leak reproduction (Section 7.4) |
| `results/robustness/sensitivity_probe.csv` | cross-log sensitivity probe, wells as W01–W19 |
| `results/robustness/multiwindow_sweep.csv` | multi-window gap-length sweep, wells as W01–W19 |
| `results/figures/` | the results-derived figures |

* **A fully synthetic example corpus** (`examples/synthetic/`) so the whole pipeline can
  be executed with no data at all. It is **entirely artificial** — sine waves plus white
  noise, on round depth ranges and a regular grid of round coordinates — and is used
  **only to demonstrate the software interface**. It does **not** derive from,
  approximate, reconstruct, or statistically reproduce the BRGM data: no statistic of
  the real corpus enters the generator.

## Anonymisation

**Revision R1.** The revised article refers to its 18 wells as **W01–W18**, numbered in the
order of `folds_loo.json`, so that `well = W{fold+1:02d}`. With BRGM approval this
correspondence is published, in `results/revision_r1/error_stats/` and
`configs/benchmark/folds_loo_18wells.yaml`. The well excluded by the depth-step QC has no
identifier (`EXCLUDED` in `data_qc/depth_qc.csv`). The screened wells that lack the log
triplet are not listed individually.

**Release 1.0.** The original submission referred to its 19 wells as **W01–W19**. The two
robustness tables were anonymised before release: their `well` column, which originally
held the archive identifier, now holds the W-id. Those two probes were run on a subset of
wells by design — two for the multi-window sweep, eight for the sensitivity probe — so not
every W-identifier appears in them. The 1.0 per-fold tables are keyed by **fold index
0-18**, and all 19 folds are present in `raw_metrics.csv` and `raw_metrics_full.csv`. The
1.0 W-identifier ↔ fold-index correspondence is not published; the fold structure is
documented, without it, in `configs/benchmark/folds_loo_19wells.yaml`.

The two numberings are independent, and no correspondence between them is published.

The mapping table itself is a re-identification key and is not released. `.gitignore`
blocks it by name, `utils/figstyle.py` refuses to fall back to a raw identifier when a
map is loaded but a key is missing, and `tests/test_privacy_guards.py` fails the build if
such a table or a well-like identifier appears in the tree.

## Figures that cannot be rebuilt here

The log distributions and cross-plots (`02_log_distributions_r1`, `03_log_crossplots_r1`)
are computed from the well logs themselves; they are released, pooled over the 18 wells,
but rebuilding them needs the corpus (`scripts/make_revision_figures.py --data DIR`).

The following figures of the article are **not released**, because they carry
re-identifying information:

* the well map (`fig5_wellmap_r1`) — relative well positions, which permit
  re-identification against a public borehole database;
* the per-well depth coverage (`04_depth_coverage_r1`);
* the example windows (`05_example_slices_r1`) — measured log traces;
* the masking-scenario illustration on a real window (`06_missingness_scenarios`) — it
  renders measured log values; an equivalent drawn on the synthetic corpus is released as
  `results/revision_r1/figures/missingness_scenarios_synthetic`.

## Access to the underlying data

Requests for access to the original well-log data should be directed to BRGM. The
corresponding author can advise on the procedure but cannot grant redistribution rights.

## Running the code on other data

The pipeline is not specific to this corpus. It reads any LAS collection in the format
described in `docs/DATA_FORMAT.md` and will build the windows, folds, masks and metrics
the same way. Only the spatial models (GNN, ST-GNN) additionally require a coordinate
table, as do the spatial baselines and graph controls of the revision; the other methods
run without one.
