# Paper → code mapping

Every claim of the article, traced to the code that produces it. Section numbers follow
the submitted article. This is a code and reproducibility repository: the article sources
are not part of it, and this table is the only link between the two.

> **Note on section numbering.** The article numbers its sections 1 Introduction,
> 2 Related work, 3 Study area and data, 4 Data and diagnoses, 5 Methods,
> 6 Experimental setup, 7 Results, 8 Discussion, 9 Conclusion. The mapping below uses
> that numbering. Where an earlier draft numbered the Methods section 4 and the Results
> section 5, both numbers are given.

---

## 5.1 (draft 4.1) Data, windowing and scaling

| Element | Implementation |
|---|---|
| LAS ingestion, mnemonic selection | `src/wellog_imputation/data/pipeline.py` :: `read_las`, `pick_curve`, `PREFS` |
| Unit harmonisation, physical bounds | `pipeline.py` :: `harmonize`, `BOUNDS` |
| Windowing (256 samples, stride 64) | `pipeline.py` :: `build_dataset`, `SLICE`, `STRIDE` |
| Depth step 0.1524 m (0.5 ft) | `pipeline.py` :: `STEP_M` |
| Real missingness statistics | `pipeline.py` :: `gaps`, `build_dataset` → `missingness_real.json` |
| **Fold-local scaling** | `src/wellog_imputation/evaluation/harness.py` :: `ChannelScaler`, fitted in `run_model` on `X[tr]` only |
| Metrics computed in original units | `harness.run_model` :: `sc_.inverse(pred_z)` before `metrics(...)` |

Design decisions encoded in the pipeline: only MEASURED curves are used (never composite
ones); `NEUT` is excluded from the NPHI candidates because it is a count, not a porosity;
the NPHI percent/fraction decision keys off the observed value range rather than the LAS
header unit; and no normalisation happens in the pipeline — the scaler is fitted per fold,
which is what makes the protocol leakage-free.

## 5.2 (draft 4.2) Leave-one-well-out cross-validation

| Element | Implementation |
|---|---|
| Fold generation | `data/pipeline.py` :: `build_dataset` → `folds_loo.json` (`{"fold_i": {"test": [w], "train": [...]}}`) |
| Fold loop, within-well families | `evaluation/harness.py` :: `run_model` |
| Fold loop, spatial family | `models/spatial.py` :: `run_spatial` |
| Train/test separation | `harness.run_model` :: `tr = wells != test_w`, `te = wells == test_w` |
| Tests | `tests/test_folds_and_scaling.py` |

## 5.3 (draft 4.3) Masking scenarios

| Scenario | Generator |
|---|---|
| `single` | `masking/scenarios.py` :: `mask_mono(mode="single", n_points=5)` |
| `block` | `masking/scenarios.py` :: `mask_mono(mode="block", block_len=(20,100))` |
| `profile` | `masking/scenarios.py` :: `mask_mono(mode="profile")` |
| `blackout` | `masking/scenarios.py` :: `mask_partial_blackout` |
| Co-occurrence calibration | `masking/scenarios.py` :: `COOCCUR_DEFAULT = {1: 0.035, 2: 0.759, 3: 0.206}` |
| Scenario parameters | `harness.EVAL_SCENARIOS` |
| Per-scenario mask seeds | `harness.EVAL_SEED`, combined with the master seed by `harness.eval_seed` |
| Training-time mixed masking | `harness.training_mask` |
| Configuration summary | `configs/masking/scenarios.yaml` |
| Tests | `tests/test_masking.py`, `tests/test_identical_masks.py` |

## 5.4 (draft 4.4) Imputation methods

The eleven methods, by module:

| Method | Class | Module |
|---|---|---|
| MEAN | `MeanImputer` | `models/tabular.py` |
| Random Forest | `TabularImputer("rf")` | `models/tabular.py` |
| XGBoost | `TabularImputer("xgb")` | `models/tabular.py` |
| MICE | `MICEImputer` | `models/tabular.py` |
| LOCF | `LOCFImputer` | `models/sequential.py` |
| SAITS | `SeqImputer("saits")` | `models/sequential.py` |
| BRITS | `SeqImputer("brits")` | `models/sequential.py` |
| U-Net | `UNetImputer` | `models/sequential.py` |
| AE | `AEImputer` | `models/sequential.py` |
| GNN | `SpatialImputer(kind="gnn")` | `models/spatial.py` |
| ST-GNN | `SpatialImputer(kind="stgnn")` | `models/spatial.py` |

Dispatch: `models/factory.py` :: `factory(name)`. Hyperparameters: `configs/models/`.

> **Family labels.** `models/registry.py` defines two mappings. `FAMILY` is what every
> released CSV contains and labels MEAN and LOCF as `"baseline"`. `PAPER_FAMILY` is the
> grouping used in the article's tables, which places MEAN with the tabular family and
> LOCF with the sequential family, on the grounds of the information each one uses.
> `PAPER_FAMILY` is documentation only: it is used by no computation, so the released
> numbers keep the exact labels of the published run.

## 5.5 (draft 4.5) Anti-leakage protocol

| Guarantee | Implementation |
|---|---|
| Well-level split (no window of the held-out well in training) | `harness.run_model` :: `tr = wells != test_w` |
| Scaler fitted on training windows only | `harness.run_model` :: `ChannelScaler().fit(Xtr)` |
| **Graph masking**: held-out well's row blanked in the training grid | `models/spatial.py` :: `SpatialImputer.fit`, `Gz[grid.widx[test_well]] = np.nan` |
| Full grid restored at inference | `SpatialImputer.impute_slice` uses `self.Gz` with the target row replaced by the masked input |
| Held-out loss is monitoring only | `SpatialImputer.fit` :: `_val_mse()` — never enters the gradient, never drives selection |
| Tests | `tests/test_folds_and_scaling.py` |

## 7.3 (draft 5.3) Metric aggregation — the per-fold averaging pitfall

| Element | Implementation |
|---|---|
| Pooled accumulation across folds | `metrics/pooled.py` :: `new_pool`, `add_to_pool` |
| Single metric over the concatenation | `metrics/pooled.py` :: `pooled_agg`, `point_stats` |
| Per-fold metrics (diagnostic only) | `harness.run_model` :: `results[sc][lg][k]`, and `metrics()` |
| Per-seed pooled R² recomputation | `statistics/aggregate.py` :: `load_seed_pool`, `r2` |
| Mean ± std and 95 % CI over seeds | `statistics/aggregate.py` :: `mean_std_ci` → `aggregated.csv` |
| Released table | `results/aggregated.csv` |
| Test demonstrating the pitfall | `tests/test_metrics.py::test_pooling_differs_from_per_fold_averaging` |

## 7.4 (draft 5.4) Spatial leakage experiments

Two distinct experiments, both single-seed paired designs:

| Experiment | Driver | Core code | Released output |
|---|---|---|---|
| Grid-row ablation (leak OFF vs ON, everything else matched) | `scripts/run_leakage_ablation.py` | `models/spatial.py` :: `run_spatial(leak=...)` | `results/ablations/ablation_leak.csv`, `.json` |
| Window-level split instead of leave-one-well-out | `scripts/run_window_split_leak.py` :: `run_leaky` | `SpatialImputer.fit(test_well=None, train_slices=...)` | `results/ablations/leak_randomsplit.csv`, `.json` |

In the second, the windows are partitioned at random 80/20, so windows of the same well
land on both sides; the script prints how many wells are present on both sides, which is
the leak made explicit.

## 7.5 (draft 5.5) Statistical significance

| Test | Implementation | Released output |
|---|---|---|
| Friedman (19 folds as blocks, per scenario and seed) | `statistics/significance.py` :: `friedman_table` | `results/significance/friedman.csv` |
| Nemenyi post-hoc (only when Friedman is significant) | `statistics/significance.py` :: `nemenyi_and_cd` | `results/significance/nemenyi_{scenario}_seed{S}.csv` |
| Critical-difference diagrams | `nemenyi_and_cd` (scikit-posthocs) | `results/significance/cd_{scenario}_seed{S}.{pdf,png}` |
| Paired Wilcoxon on contested pairs | `statistics/significance.py` :: `wilcoxon_table`, `WILCOXON_PAIRS` | `results/significance/wilcoxon_pairs.csv` |
| Ranking stability per seed | `statistics/aggregate.py` :: `aggregate` | `results/significance/ranking_stability.csv` |
| By-well cluster bootstrap, paired | `statistics/bootstrap.py` :: `boot_r2`, `boot_delta` | `results/bootstrap_ci.csv`, `results/bootstrap_pairwise.csv` |

> **On multiple-comparison correction.** Holm is not applied to the Wilcoxon family. The
> pairs are pre-registered — they are exactly the ties the article argues about — and
> each is reported per seed together with the fraction of seeds reaching p < 0.05, which
> is the quantity the text uses. The Nemenyi post-hoc already carries its own
> family-wise control and is what the critical-difference diagrams display. This is a
> reporting choice, stated here so a reader is not left to infer it.

## Supplementary material

| Supplement | Status | Where the content lives |
|---|---|---|
| Supplementary metrics: correlation and per-log error (Tables A1, A2) | Present in the article | `results/aggregated_full.csv` (per-metric means and standard deviations), `results/table_perlog.csv`, figures `results/figures/metric_cc.*`, `metric_mae.*` |
| Implementation, training and compute | Not a supplement; documented here instead | `docs/REPRODUCIBILITY.md` covers the same ground: versions, hardware, seeds, determinism, runtimes |
| Hyperparameter configurations | Not a supplement; documented here instead | `configs/models/tabular.yaml`, `sequential.yaml`, `spatial.yaml` — transcribed from the code, values unchanged |
| Window-level leakage experiment | Not a supplement; the experiment itself is in Section 7.4 | `scripts/run_window_split_leak.py`, `results/ablations/leak_randomsplit.*` |

The submitted article carries exactly one supplementary section (*Supplementary metrics:
correlation and per-log error*). The other rows are not supplements; they name where the
corresponding material lives in this repository.

---

## Figures

| Article figure | Reproducible here? | Source |
|---|---|---|
| Convergence curves | yes | `results/figures/08_convergence_curves.*` (released); regenerate with the private per-fold curves |
| R² boxplots by scenario | **yes** | `scripts/reproduce_figures.py` :: `fig_r2_boxplots` |
| Per-log vs aggregate R² | **yes** | `scripts/reproduce_figures.py` :: `fig_perlog_vs_aggregate` |
| Ranking stability | **yes** | `scripts/reproduce_figures.py` :: `fig_ranking_stability` |
| Critical-difference diagram | **yes** | `scripts/reproduce_tables.py` → `statistics/significance.py` |
| Four-metric panels, overview heatmap | **yes** | `scripts/reproduce_figures.py` :: `metric_figure`, `overview_heatmap` |
| Cross-log sensitivity probe, multi-window sweep | released as PDFs | `results/robustness/` |
| Log distributions, cross-plots, depth coverage, example windows, masking illustration, well map | **no** | computed from the well logs themselves — see `docs/DATA_AVAILABILITY.md` |
