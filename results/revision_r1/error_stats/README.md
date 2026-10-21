# Anonymised per-well error statistics — revision R1 (ACAGS-D-26-00297)

**Status: released with BRGM approval (anonymised: wells identified by fold index and W01–W18 only).**

These files hold, for every evaluated cell, the *sufficient statistics* of the masked
points: counts, sums, sums of squares and error sums. Every aggregate metric and test in
the revised article can be recomputed from them, without the raw logs and without the
per-fold dumps. They contain no well name, no depth, no coordinate and no individual
log value (see *Confidentiality* below).

Produced from the per-fold dumps of the single R1 run (code `revision-r1`, commit
`1a3fec5`, tag `v1.1-r1`; 18 wells, 5 initialisation seeds × 5 evaluation-mask seeds,
28 methods, 5 scenarios). **No model was retrained.**

## Files

### `wellwise_error_stats.csv`: one row per (method, scenario, init_seed, mask_seed, rep, log, well)

| column | meaning |
|---|---|
| `method` | imputation method (28: the 11 of the article, plus 17 revision baselines and controls) |
| `scenario` | `single`, `block`, `profile`, `blackout`, `blackout_emp` (empirical whole-well blackout) |
| `init_seed` | master seed 0–4: training masks and model initialisation |
| `mask_seed` | evaluation-mask seed 0–4. **0 = primary evaluation (main tables)**; 1–4 = the same trained model re-evaluated on other masks |
| `rep` | replicate index. Always 0. `blackout_emp` has 5 replicates; they are **pooled** in this file (see *Confidentiality*) |
| `log` | `GR`, `RHOB`, `NPHI` |
| `fold` | leave-one-well-out fold index, 0–17 (the fold in which that well is held out), in the order of `folds_loo.json` |
| `well` | anonymised well identifier W01–W18, numbered in the same order: `well = W{fold+1:02d}` |
| `n` | number of masked (evaluated) samples |
| `sum_y` | Σ y over these samples, true values in physical units (gAPI, g/cm³, v/v) |
| `sum_y2` | Σ y² |
| `sse` | Σ (ŷ − y)² |
| `sae` | Σ \|ŷ − y\| |

Values are float64, written at full precision (`repr`). They are computed from the
float32 dumps with the same arithmetic as `wellog_imputation.statistics.wellwise`.

### `mask_ids.csv`: one row per (scenario, mask_seed, rep, well)

`effective_seed` is the seed of the evaluation-mask generator:
`eval_seed(scenario, mask_seed) = EVAL_SEED[scenario] + 100000 × mask_seed`, with
`EVAL_SEED = {single: 1000, block: 1001, profile: 1002, blackout: 1003, blackout_emp: 1004}`
(`wellog_imputation.evaluation.harness`). A fresh generator is created with this seed
for each held-out well. Given the dataset, it regenerates the masked positions exactly
(`make_eval_set`, or `empirical.draw_masks` for `blackout_emp`). For `blackout_emp`,
the 5 replicates are successive draws `rep = 0..4` from that one generator.
`n_GR`, `n_RHOB` and `n_NPHI` are the number of masked samples per log. The masks do
not depend on the method or the init seed (checked: identical counts in all 25 × 28
cells). The positions themselves are not released.

### `fold_train_sigma.csv`: one row per (well, log)

`sigma` is the standard deviation of the log over the **training** windows of the fold in
which the well of index `fold` is held out (`ChannelScaler.sd`, i.e. `nanstd + 1e-8`). This file is
needed for nRMSE, which is standardised by the training σ. It is an aggregate of the
measured logs (see *Confidentiality*).

## Reconstruction formulas

Notation: `c` = method, `ℓ` = log, `w` = well, `s` = init seed. The main tables use
`mask_seed = 0`, and every `blackout_emp` replicate is included.

* Pooled R² of log ℓ for one (c, s, scenario), all wells pooled:

  $$R^2_{c,\ell} = 1 - \frac{\sum_w \mathrm{SSE}_{w\ell}}{\sum_w \Sigma y^2_{w\ell} - \big(\sum_w \Sigma y_{w\ell}\big)^2 / \sum_w n_{w\ell}}$$

* Primary metric: $\bar R^2 = \tfrac13 \sum_{\ell\in\{GR,RHOB,NPHI\}} R^2_{c,\ell}$. The
  tables report its mean ± sd (ddof = 1) over the 5 init seeds, and the rank of that mean
  (1 = best) within each scenario.
* MAE and RMSE per log: $\sum_w \mathrm{SAE}/\sum_w n$ and $\sqrt{\sum_w \mathrm{SSE}/\sum_w n}$.
* Per-well standardised error:

  $$\mathrm{nRMSE}_w = \sqrt{\frac{1}{|L_w|}\sum_{\ell\in L_w} \frac{\mathrm{SSE}_{w\ell}}{n_{w\ell}\,\sigma_{w\ell}^2}}$$

  Here $\sigma_{w\ell}$ comes from `fold_train_sigma.csv` and $L_w$ is the set of logs with
  $n_{w\ell} > 0$. nRMSE is averaged over the 5 init seeds for each (method, well). The
  main table reports its median, Q1, Q3 and mean over wells.
* **Friedman**: blocks = wells, treatments = methods, values = seed-averaged nRMSE_w.
  Wells with a missing value are dropped (`scipy.stats.friedmanchisquare`). Kendall's
  W = χ²_F / (N (k − 1)).
* **Wilcoxon**: paired across wells on the seed-averaged nRMSE_w, two-sided,
  `zero_method="wilcox"`, `method="auto"` (exact for 18 wells without ties). The effect
  size is the matched-pairs rank-biserial r. Holm step-down at α = 0.05 is applied within
  each (family, scenario). The families are the pre-registered pairs of the article and
  the revision pairs.
* **By-well cluster bootstrap** of ΔR̄² (A − B): B = 10 000. Wells are resampled with
  replacement, and each statistic row is weighted by its multiplicity. The same draw is
  used for A, B and all seeds, and Δ is averaged over seeds. The CI is percentile
  2.5/97.5, and p_boot = 2 min(P(Δ ≤ 0), P(Δ ≥ 0)). There is ONE generator
  `numpy.random.default_rng(20261016)`, consumed in this order: pre-registered pairs
  (`profile`, then `blackout`), then revision pairs (`profile`, `blackout`, `blackout_emp`).
* Initialisation vs mask variability (`seed_variance.csv`): R̄² is computed on each of the
  25 (init, mask) cells. sd_init = sd over init seeds of the mean over masks; sd_mask =
  sd over masks of the mean over init seeds.

## Recompute everything

```bash
python scripts/recompute_from_export.py --export results/revision_r1/error_stats --out recomputed [--compare <results>/revision]
```

The script needs only `numpy` and `scipy`. It reads only `wellwise_error_stats.csv`
and `fold_train_sigma.csv` (`mask_ids.csv` documents the masks and is not needed) and
writes `table_main_revision.csv`, `table_perlog_revision.csv`, `per_well_nrmse.csv`,
`friedman_wellwise.csv`, `wilcoxon_wellwise_holm.csv`, `wilcoxon_revision_holm.csv` and
`seed_variance.csv`, with the published rounding (`--full` gives full precision).
`--compare` prints the largest absolute difference per file. The same script is in the
code repository as `scripts/recompute_from_export.py`; `scripts/make_error_stats_export.py` built this export from the per-fold dumps (not released).

## Confidentiality

* Wells are identified only by the fold index 0–17 and the label W01–W18
  (`well = W{fold+1:02d}`, order of the 18-well `folds_loo.json` of R1). There is no
  name, depth, coordinate or individual sample.
* These W01–W18 labels are those of revision R1 (18 wells). They are not the W01–W19
  of release 1.0: `configs/benchmark/folds_loo_19wells.yaml` documents the 19-well
  protocol of the original article, where the W ↔ fold correspondence was deliberately
  left unpublished. It is not used by this export.
* `sum_y` and `sum_y2` are aggregates of the measured logs. With `n`, they give the
  mean and variance of the true values on the masked samples of each (scenario, mask
  seed, well, log). `fold_train_sigma.csv` gives the training σ of each fold. The
  smallest released cell has `n = 5` (`single`). Per replicate, `blackout_emp` has cells
  with `n = 1` or `2`, where the values themselves would be recoverable. This is why
  its replicates are pooled here (smallest pooled cell: `n = 107`).
