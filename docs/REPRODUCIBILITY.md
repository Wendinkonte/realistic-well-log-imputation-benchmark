# Reproducibility

Everything below is taken from the code and the run records of this project. Where a
value could not be determined from them, the entry says so explicitly rather than
guessing.

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.12.6 |
| Platform | Linux x86-64 |
| GPU | NVIDIA H100 NVL (one per SLURM task; a 12 GB MIG slice for the lighter jobs) |
| Scheduler | SLURM, `gpu` partition |

Exact package versions of the published run (also in `requirements-lock.txt`):

| Package | Version | | Package | Version |
|---|---|---|---|---|
| numpy | 2.2.6 | | torch | 2.8.0 |
| scipy | 1.16.2 | | pypots | 1.0 |
| pandas | 2.3.3 | | benchpots | 0.4 |
| scikit-learn | 1.7.2 | | pygrinder | 0.7 |
| xgboost | 3.2.0 | | tsdb | 0.7.1 |
| matplotlib | 3.10.7 | | lasio | 0.32 |
| seaborn | 0.13.2 | | networkx | 3.5 |
| statsmodels | 0.14.6 | | h5py | 3.15.1 |
| scikit-posthocs | 0.14.0 | | tqdm | 4.67.1 |

`requirements.txt` uses compatible ranges instead, because pinning hard makes a
repository un-installable within a year. Use `requirements-lock.txt` for a
bit-comparable environment.

---

## 2. Random seeds and what each one controls

The master seed of a run drives every source of randomness. `scripts/run_replication.py`
is invoked once per master seed; the published campaign used seeds **0, 1, 2, 3, 4**.

| Source | Where | How it is seeded |
|---|---|---|
| Evaluation masks | `masking/scenarios.make_eval_set` | `np.random.default_rng(eval_seed(scenario, master_seed))` |
| Per-scenario base seed | `evaluation/harness.EVAL_SEED` | fixed table `{single: 1000, block: 1001, profile: 1002, blackout: 1003}` |
| Seed derivation | `evaluation/harness.eval_seed` | `EVAL_SEED[scenario] + master_seed * 100000` |
| Training masks | `harness.training_mask` | `np.random.default_rng(seed)` in `run_model`; `default_rng(self.seed)` in `SpatialImputer.fit` |
| Neural weight init, dropout, shuffling | torch | `utils/determinism.set_determinism(seed)` |
| Minibatch shuffle (AE, U-Net, spatial) | `np.random.default_rng(self.seed)` | the model's own seed |
| RF / XGBoost / MICE | `models/tabular.py` | `random_state=0`, hard-coded |
| `PYTHONHASHSEED` | process environment | set to 0 by `set_determinism` and by the SLURM templates |

**Why the seed table is explicit.** Deriving the mask seed from `hash(scenario)` would be
wrong: Python salts string hashing per process, so the masked positions would silently
change on every run. `EVAL_SEED` is a literal table for that reason, and
`tests/test_identical_masks.py` asserts it.

### Identical evaluation masks across methods

The fairness guarantee of the benchmark is that every method is scored at exactly the
same masked positions. It follows from the derivation above: within one `(scenario,
master_seed)`, every method calls `make_eval_set` with a generator seeded identically, so
the masks coincide. This is what makes the paired Wilcoxon and the paired bootstrap
valid. It is verified by `tests/test_identical_masks.py`.

Masks DO vary across master seeds. That is deliberate: inter-seed variability is a
reported result, not noise to be suppressed.

---

## 3. Determinism, and its documented limits

### Deterministic
- mask generation, given a seed (evaluation and training);
- fold construction (sorted well list, one fold per well);
- fold-local scaling;
- RF, XGBoost, MICE (fixed `random_state`; MICE uses `sample_posterior=False`);
- MEAN and LOCF (no randomness at all);
- all aggregation, significance testing and bootstrap resampling (seeded).

### Not fully deterministic — CUDA/cuDNN
`utils/determinism.set_determinism` deliberately sets:

```python
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
```

Forcing `cudnn.deterministic = True` pushes the bidirectional RNN of BRITS onto an
unoptimised cuDNN kernel and makes it roughly **80× slower — 33 minutes per fold instead
of 24 seconds** — which put the five-seed campaign out of budget. Seeding numpy and torch
is enough to reproduce results at the precision the paper reports; the residual cuDNN
nondeterminism on convolution and RNN kernels stays **below 1e-3 on the pooled R²**.

This is a real limitation and is stated as such: a GPU re-run of the neural models will
reproduce the reported figures to three decimals, not bit-for-bit. The non-neural methods
(MEAN, LOCF, RF, XGBoost, MICE) are bit-reproducible.

---

## 4. Two documented departures from the originally published single-seed run

Both are recorded here because they are the only ways in which re-running this code does
not reproduce the earlier single-seed numbers bit-for-bit.

**4.1 — Evaluation-seed semantics.** The original single-seed code used the bare
`EVAL_SEED` table. Multi-seed replication requires the master seed to drive mask
generation too, so the derivation became `EVAL_SEED[scenario] + master_seed * 100000`.
With `master_seed = 0` this returns the bare table value, so the masks match; but the
training masks and initialisations do differ, and `master_seed = 0` is therefore **not
bit-identical** to the original run. It reproduces it within the reported precision.

**4.2 — Blackout co-occurrence constants.** The originally published run used
`{1: 0.035, 2: 0.761, 3: 0.204}`; the current constant is
`{1: 0.035, 2: 0.759, 3: 0.206}`, recomputed from the cleaned corpus. The ~0.2
percentage-point shift is below the precision at which results are reported and changes
no R² at three decimals. See `masking/scenarios.py :: COOCCUR_DEFAULT`.

All headline numbers in the paper come from the **five-seed replication**, not from the
earlier single-seed run, so neither departure affects a reported result.

---

## 5. Protocol constants

| Constant | Value | Defined in |
|---|---|---|
| Window length | 256 samples | `data/pipeline.py :: SLICE` |
| Window stride | 64 samples | `data/pipeline.py :: STRIDE` |
| Depth step | 0.1524 m (0.5 ft) | `data/pipeline.py :: STEP_M` |
| Logs | GR, RHOB, NPHI | `metrics/pooled.py :: LOGS` |
| Physical bounds | GR [0, 400]; RHOB [1.2, 3.1]; NPHI [-0.05, 1.0] | `data/pipeline.py :: BOUNDS` |
| Wells (study corpus) | 19 with the complete triplet | derived from the corpus |
| Windows (study corpus) | 802 intact | derived from the corpus |
| Folds | 19, leave-one-well-out | `folds_loo.json` |
| Seeds | 5 (0–4) | `scripts/run_replication.py :: N_SEEDS` |
| Neural epochs | 100 | campaign setting |
| Bootstrap replicates | 5000 | `scripts/run_bootstrap.py` default |
| Significance level | 0.05 | `statistics/significance.py` |

---

## 6. Leave-one-well-out construction

`data/pipeline.py :: build_dataset` keeps the wells in which GR, RHOB and NPHI co-exist
over more than `SLICE` samples, sorts them, and emits one fold per well:
`{"fold_i": {"test": [w_i], "train": [every other well]}}`.

Windows are cut only where all three logs are present over the full 256 samples, so the
ground truth is never itself imputed. Train/test selection is by well identifier
(`wells != test_w`), never by window index — that distinction is the whole point of
Section 7.4, and `tests/test_folds_and_scaling.py` asserts it.

---

## 7. Fold-local scaling

`ChannelScaler` standardises each of the three channels. It is fitted **inside** the fold
loop on `X[tr]` — the training windows only — then applied to both splits. Metrics are
computed after `inverse()`, in original physical units. Fitting on the full dataset would
leak the held-out well's distribution into the scaler;
`tests/test_folds_and_scaling.py::test_scaler_is_fitted_on_training_windows_only`
verifies the two are not equivalent.

---

## 8. Metric aggregation

The primary metric is the **pooled R²**: the masked positions of all 19 folds are
concatenated and a single R² is computed over them, in original units, at artificially
masked positions only (`metrics/pooled.py :: pooled_agg`).

Averaging R² per fold instead lets a well contributing one window weigh as much as a well
contributing 237, which fabricates an apparent collapse of healthy models. That is the
reporting pitfall of Section 7.3, and
`tests/test_metrics.py::test_pooling_differs_from_per_fold_averaging` demonstrates it on a
minimal case.

The pooled `all` channel (three logs together) is inflated by between-channel variance —
the zero-skill MEAN scores about 0.76 on it, not 0 — so the per-log R² and the MEAN floor
are always reported alongside it. Read `all` for cross-method comparison within a
scenario, never as an absolute skill level.

---

## 9. Statistical tests

* **Friedman**, per `(scenario, seed)`, with the 19 folds as blocks and only the methods
  complete over every fold.
* **Nemenyi** post-hoc plus a critical-difference diagram, computed only where Friedman
  gives p < 0.05.
* **Paired Wilcoxon** over the 19 folds on pre-registered contested pairs
  (`statistics/significance.py :: WILCOXON_PAIRS`), reported per seed together with the
  fraction of seeds reaching p < 0.05.
* **By-well cluster bootstrap** (B = 5000) for confidence intervals, and a *paired*
  variant applying the same well draw to both methods for differences. Resampling wells
  rather than points is essential: points within a well are strongly autocorrelated, so
  an i.i.d. point bootstrap would produce intervals that are far too tight.

Holm correction is not applied to the Wilcoxon family; the reasoning is stated in
`docs/METHODS_MAPPING.md`.

---

## 10. Runtime

Measured on one NVIDIA H100 NVL, from the campaign's SLURM records.

| Job | Budget requested | Notes |
|---|---|---|
| Replication, one master seed, non-BRITS methods | up to 23 h | one array task per seed |
| BRITS, one master seed | ~10 h serially | decomposed to one task per (seed, fold), ≤ 4 h each |
| BRITS, one fold | ~24 s with `cudnn.benchmark`; ~33 min if cuDNN determinism is forced | the reason determinism is off |
| Leakage ablation (both models, leak OFF and ON) | up to 12 h | single-seed paired experiment |
| Window-split leak reproduction | within the same 12 h budget | single-seed |
| Aggregation and significance | minutes, CPU only | `scripts/reproduce_tables.py` |
| Figures from released tables | seconds, CPU only | `scripts/reproduce_figures.py` |

**What these numbers are.** They are the *requested* SLURM wall-clock budgets, which is
what the run records preserve. Precise per-method elapsed times were not retained in a
machine-readable form, and none is reconstructed here.

Full campaign scale: 5 seeds × 11 methods × 19 folds × 4 scenarios.

---

## 11. Reproducing, in order

```bash
pip install -e ".[neural,stats,dev]"
cp configs/paths.example.yaml configs/paths.yaml     # point at your LAS corpus

python scripts/build_dataset.py                      # LAS -> windows + folds
for s in 0 1 2 3 4; do python scripts/run_replication.py --seed $s --epochs 100; done
python scripts/run_leakage_ablation.py --models gnn,stgnn --epochs 100
python scripts/run_window_split_leak.py --models gnn,stgnn --epochs 100
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
```

Steps 2–5 need the well-log data. Step 6 and the results-derived part of step 7 run on
the released aggregate tables alone, with no data at all — that is the part a reviewer
can execute directly from a fresh clone.

`run_replication.py` is append-only and resumable: re-running the same command skips
every `(seed, method, fold)` already on disk, so a preempted campaign is restarted by
resubmitting it.

---

## 12. Known limitations

1. **CUDA nondeterminism** on the neural models — Section 3 above. Reproducible to ~1e-3
   on pooled R², not bit-for-bit.
2. **The data cannot be redistributed**, so the exact numbers of the paper cannot be
   reproduced from this repository alone. The aggregate outputs they were computed from
   are released instead; see `docs/DATA_AVAILABILITY.md`.
3. **Well coordinates are required by the spatial family** and are proprietary for the
   study corpus. GNN and ST-GNN need a coordinate table in the format of
   `docs/DATA_FORMAT.md`; the other nine methods do not.
4. **Per-method elapsed times** were not retained — Section 10.
