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
| `quality_report.csv`, `folds_loo.json` | keyed by real well identifier |
| Well coordinate table | real projected surface positions |
| Any well-name ↔ anonymous-id mapping | it would undo the anonymisation |
| The W-identifier ↔ fold-index correspondence | the W-numbering encodes an ordering of the corpus |
| Pooled prediction dumps (`pool_*.npz`, `pools_seeded/`) | they carry the ground-truth log values at every evaluated position, so they are derived proprietary data |
| Per-fold run logs | they print well identifiers |

The last row is worth stating plainly: a file can be "just results" and still be
proprietary. The pool dumps store the true measured value at each scored position, so
publishing them would publish an extract of the logs.

## What is released

Everything needed to inspect the method, check the protocol, and rebuild every
results-derived table and figure:

* **All research code** — the pipeline, the four masking generators, the eleven model
  implementations, the leave-one-well-out harness, the metrics, the aggregation, the
  significance tests, the bootstrap, and both leakage experiments.
* **Derived statistics and aggregate results**, keyed by fold index and containing no
  well identifier, coordinate or measured value:

| File | Content |
|---|---|
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

The publication refers to the wells as **W01–W19**. Those anonymous identifiers are the
only well references anywhere in this repository. The two robustness tables above were
anonymised before release: their `well` column, which originally held the archive
identifier, now holds the W-id. Those two probes were run on a subset of wells by design —
two for the multi-window sweep, eight for the sensitivity probe — so not every W-identifier
appears in them.

The benchmark itself covers all 19 wells. Its released per-fold tables are keyed by **fold
index 0-18**, not by W-identifier, and all 19 folds are present in `raw_metrics.csv` and
`raw_metrics_full.csv`. The W-identifier ↔ fold-index correspondence is withheld on
purpose: the W-numbering encodes an ordering of the corpus, so publishing the
correspondence would leak that ordering. No released analysis depends on it. The fold
structure is documented, without that correspondence, in
`configs/benchmark/folds_loo_19wells.yaml`.

The mapping table itself is a re-identification key and is not released. `.gitignore`
blocks it by name, `utils/figstyle.py` refuses to fall back to a raw identifier when a
map is loaded but a key is missing, and `tests/test_privacy_guards.py` fails the build if
such a table or a well-like identifier appears in the tree.

## Figures that cannot be rebuilt here

These are computed from the well logs themselves, so they need the proprietary corpus:
log distributions, log cross-plots, per-well depth coverage, example windows, the masking
illustration drawn on a real window, and the well location map.

The following are **excluded pending explicit clearance**, because they carry
re-identifying information even though they appear in the article:

* the well location map — well positions are plotted on real projected coordinate axes,
  which permits re-identification against a public borehole database;
* the masking-scenario illustration — it renders measured log values from a real window.

## Access to the underlying data

Requests for access to the original well-log data should be directed to BRGM. The
corresponding author can advise on the procedure but cannot grant redistribution rights.

## Running the code on other data

The pipeline is not specific to this corpus. It reads any LAS collection in the format
described in `docs/DATA_FORMAT.md` and will build the windows, folds, masks and metrics
the same way. Only the spatial models (GNN, ST-GNN) additionally require a coordinate
table; the other nine methods run without one.
