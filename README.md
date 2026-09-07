# Realistic Missingness Changes the Ranking of Well-Log Imputation Models

Reference implementation and reproducibility package for a benchmark of **eleven well-log
imputation methods** across **four missingness regimes**, under a strict, leakage-
controlled, leave-one-well-out protocol on 19 Paris Basin geothermal wells.

[![CI](https://github.com/Wendinkonte/realistic-well-log-imputation-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Wendinkonte/realistic-well-log-imputation-benchmark/actions/workflows/ci.yml)

---

## 1. Overview

Machine-learning imputation of well logs is now organised around benchmarks whose method
rankings guide practice. This repository contains the code and the derived results
showing that those rankings **do not survive realistic conditions**.

It provides the complete pipeline — LAS ingestion, windowing, the four masking
generators, eleven imputers in three families, the leave-one-well-out harness, the pooled
metrics, the significance tests, the cluster bootstrap and both leakage experiments —
plus every aggregate result table the paper reports.

> **Scope.** This repository provides the implementation and reproducibility framework
> used in the study. The proprietary BRGM well-log data cannot be redistributed.
> Synthetic data are provided to validate and demonstrate the software pipeline.

## 2. Scientific motivation

Existing benchmarks mask within a single log at a time: a few points, a short gap, or at
most one whole log. On the study corpus, **96.5 % of real gaps instead involve several
logs absent together over the same contiguous interval** — entire tool passes lost, a
structured missing-not-at-random *partial blackout* that no benchmark reproduces.

A second hazard compounds it. The "composite" curves often used as ground truth are
depth-merged products that blend measured wireline data with model-derived infill, so
evaluating an imputer against them is partly circular. This benchmark uses **measured
curves only**.

## 3. Main result

**The ranking inverts with the missingness regime, and the inversion is robust across a
five-seed replication with formal significance testing.**

* On easy mono-log gaps, sequential and carry-forward methods are near-perfect
  (R² ≈ 0.99) — but collapse to the mean predictor when a whole log is missing
  (R² ≈ 0.58). **BRITS, the best method on the standard benchmark, ranks last of eleven
  there.**
* In the whole-log (`profile`) regime the lead passes not to one method but to a **broad
  statistical tie** (R² ≈ 0.70–0.76, Wilcoxon n.s.) spanning U-Net, MICE, the two graph
  models and even XGBoost, with the top rank flipping between them across seeds. The deep
  failure is specific to the imputation-specialised recurrent and attention models, not to
  deep learning as such.
* On the field-dominant multi-log **blackout**, none of the learned methods wins: simple
  within-well carry-forward (LOCF) prevails.

Two evaluation pitfalls are isolated, each of which alone dictates the apparent winner:

* **per-fold R² averaging**, which fabricates an apparent collapse of healthy models under
  leave-one-well-out;
* **spatial data leakage**, where a window-level rather than well-level split alone
  inflates accuracy from R² = 0.75 to 0.91.

## 4. Methods compared

| Family | Methods |
|---|---|
| **Tabular** | MEAN, Random Forest, XGBoost, MICE |
| **Sequential** | LOCF, SAITS, BRITS, U-Net, AE |
| **Spatial** | GNN, ST-GNN |

Tabular methods predict a missing entry from the other logs at the same depth. Sequential
methods read the target well's window along depth. Spatial methods additionally borrow
from neighbouring wells through a kNN graph built on surface coordinates.

Implementations: `src/wellog_imputation/models/`. Hyperparameters: `configs/models/`.

## 5. Missingness scenarios

| Scenario | What is masked | Regime |
|---|---|---|
| `single` | 5 scattered samples in one log | mono-log, easy |
| `block` | one contiguous block in one log | mono-log |
| `profile` | an entire log of the window | mono-log, hardest in prior benchmarks |
| `blackout` | **k logs over the same contiguous interval**, k drawn from the measured co-occurrence `{1: 3.5 %, 2: 75.9 %, 3: 20.6 %}` | **the realistic, field-dominant regime** |

Implementation: `src/wellog_imputation/masking/scenarios.py`. Parameters:
`configs/masking/scenarios.yaml`.

## 6. Evaluation protocol

* **19 wells**, each carrying the complete GR / RHOB / NPHI triplet; **802 intact windows**
  of 256 samples (stride 64) at a 0.1524 m depth step.
* **Leave-one-well-out cross-validation**, 19 folds — one per well. No window of the
  held-out well ever reaches training. The fold structure is written out in
  `configs/benchmark/folds_loo_19wells.yaml`; the released per-fold tables
  (`results/raw_metrics.csv`, `raw_metrics_full.csv`) carry all 19 of them, keyed by fold
  index 0-18.
* **Fold-local scaling**: the channel scaler is fitted on the training windows of the fold
  only.
* **Identical evaluation masks across methods**: masks derive from a fixed per-scenario
  seed table combined with the run's master seed, so every method is scored at exactly the
  same positions. This is what makes the paired statistics valid.
* **Mixed training missingness**: a quarter of the training windows under each scenario, so
  no model is tuned to one regime.
* **Five seeds**, with the master seed driving every source of randomness.
* **Pooled R²** as the primary metric, in original physical units, at artificially masked
  positions only. Per-log R² and the zero-skill MEAN floor are always reported alongside,
  because the pooled `all` channel is inflated by between-channel variance.
* **Significance**: Friedman over the 19 folds as blocks, Nemenyi post-hoc with
  critical-difference diagrams, paired Wilcoxon on pre-registered contested pairs, and a
  by-well cluster bootstrap.
* **Anti-leakage**: well-level splits, fold-local scaling, and — for the graph models —
  the held-out well's row blanked in the training grid so it is not even available as a
  neighbour.

## 7. Repository structure

```
├── configs/                    protocol and hyperparameters, transcribed from the code
│   ├── benchmark/default.yaml
│   ├── benchmark/folds_loo_19wells.yaml   the 19 leave-one-well-out folds, W01-W19
│   ├── masking/scenarios.yaml
│   ├── models/{tabular,sequential,spatial}.yaml
│   └── paths.example.yaml      copy to paths.yaml and point at your own data
├── src/wellog_imputation/
│   ├── config.py               path resolution; no absolute path lives anywhere else
│   ├── data/pipeline.py        LAS -> cleaned logs, windows, folds
│   ├── masking/scenarios.py    the four mask generators
│   ├── models/
│   │   ├── tabular.py          MEAN, RF, XGBoost, MICE
│   │   ├── sequential.py       LOCF, SAITS, BRITS, AE, U-Net
│   │   ├── spatial.py          GNN, ST-GNN, grid, adjacency, leakage toggle
│   │   ├── factory.py          name -> constructor
│   │   └── registry.py         method order, family labels, display names
│   ├── evaluation/harness.py   LOO loop, fold-local scaler, seed tables
│   ├── metrics/pooled.py       metrics and pooled aggregation
│   ├── statistics/             aggregation, significance, cluster bootstrap
│   └── utils/                  determinism, figure style, anonymisation
├── scripts/                    command-line entry points (+ SLURM templates)
├── examples/synthetic/         a fully synthetic corpus, runnable with no data
├── results/                    released aggregate results and figures
├── docs/                       data format, availability, reproducibility, paper mapping
└── tests/                      masking, folds, scaling, metrics, privacy and release guards
```

## 8. Installation

Python 3.10 or newer.

```bash
git clone https://github.com/Wendinkonte/realistic-well-log-imputation-benchmark.git
cd realistic-well-log-imputation-benchmark

python -m venv .venv && source .venv/bin/activate
pip install -e ".[stats]"          # tabular methods, metrics, statistics, figures
```

Add the neural imputers (SAITS, BRITS, AE, U-Net, GNN, ST-GNN) when you need them —
they pull in PyTorch:

```bash
pip install -e ".[neural,stats,dev]"
```

For a bit-comparable environment, `requirements-lock.txt` pins the exact versions of the
published run.

## 9. Quick start

No data required — this runs on the synthetic example corpus:

```bash
python examples/synthetic/make_synthetic.py

export WELLOG_DATA_DIR=examples/synthetic/data/processed
export WELLOG_COORDS_CSV=examples/synthetic/data/processed/well_coordinates.csv
export WELLOG_RESULTS_DIR=/tmp/wellog_demo

python scripts/run_single_experiment.py --model mean --folds 2
python scripts/run_single_experiment.py --model mice --folds 2
```

> **The synthetic example data are provided solely to demonstrate the software interface
> and pipeline. They do not reproduce, approximate, or derive from the proprietary BRGM
> well-log data used in the study.**
>
> They are **entirely artificial**: each curve is a sum of a few sine waves plus white
> noise on a round depth range, and the wells sit on a regular grid of round coordinates.
> No statistic of the real corpus — no mean, variance, correlation, gap-length
> distribution, well count, depth coverage or position — enters the generator. The
> synthetic data therefore do **not derive from**, **approximate**, **reconstruct**, or
> **statistically reproduce** the BRGM data in any respect. Numbers computed on them are
> meaningless as science; they exist only to prove the code paths execute.

To point the pipeline at your own LAS corpus, see `docs/DATA_FORMAT.md`:

```bash
cp configs/paths.example.yaml configs/paths.yaml    # edit; git-ignored
python scripts/build_dataset.py
```

## 10. Running individual experiments

```bash
# one method, all four scenarios
python scripts/run_single_experiment.py --model stgnn --epochs 100

# one method, one scenario, a subset of folds
python scripts/run_single_experiment.py --model brits --scenario profile --folds 3

# machine-readable output
python scripts/run_single_experiment.py --model mice --json
```

## 11. Running the benchmark

```bash
# single seed, all methods
python scripts/run_benchmark.py --models mean,locf,rf,xgb,mice,saits,brits,unet,ae,gnn,stgnn \
                               --epochs 100 --dump

# the five-seed replication (append-only and resumable)
for s in 0 1 2 3 4; do
  python scripts/run_replication.py --seed $s --epochs 100
done
```

On a cluster, use the templates in `scripts/slurm/`. `run_replication.py` skips any
`(seed, method, fold)` already on disk, so a preempted campaign is restarted simply by
resubmitting it.

## 12. Reproducing the tables

```bash
python scripts/reproduce_tables.py
```

Consolidates the per-cell metrics, recomputes the pooled R² per seed, and writes
`aggregated.csv`, `raw_metrics.csv`, everything under `results/significance/`, and
`RESULTS_FOR_PAPER.md`. Needs the per-fold dumps produced by the replication; the
already-aggregated outputs are released in `results/`.

## 13. Reproducing the figures

```bash
python scripts/reproduce_figures.py
```

**This one runs from a fresh clone with no data at all.** When the private per-fold dumps
are absent it falls back to the released aggregate tables, which reproduces the R²
boxplots, the per-log-versus-aggregate comparison, the ranking-stability heatmap and the
four-metric panels. The figures that depend on the well logs themselves cannot be rebuilt
without the corpus — they are listed in `docs/DATA_AVAILABILITY.md`.

## 14. Reproducing the significance tests

`scripts/reproduce_tables.py` produces all of them:

| Output | Test |
|---|---|
| `results/significance/friedman.csv` | Friedman, 19 folds as blocks, per (scenario, seed) |
| `results/significance/nemenyi_{scenario}_seed{S}.csv` | Nemenyi post-hoc, where Friedman is significant |
| `results/significance/cd_{scenario}_seed{S}.{pdf,png}` | critical-difference diagrams |
| `results/significance/wilcoxon_pairs.csv` | paired Wilcoxon on the contested pairs |
| `results/significance/ranking_stability.csv` | rank of each method per seed |

Confidence intervals and paired method comparisons use a **by-well cluster bootstrap**:

```bash
python scripts/run_bootstrap.py --B 5000
```

Resampling wells rather than points matters: points within a well are strongly
autocorrelated, and an i.i.d. point bootstrap would give intervals that are far too tight.

## 15. Reproducing the leakage experiments

```bash
# grid-row ablation: anti-leakage ON vs OFF, everything else matched
python scripts/run_leakage_ablation.py --models gnn,stgnn --epochs 100

# window-level split instead of leave-one-well-out
python scripts/run_window_split_leak.py --models gnn,stgnn --epochs 100
```

Released outputs: `results/ablations/ablation_leak.csv` and `leak_randomsplit.csv`.

## 16. Using the pipeline on another dataset

Nothing in the pipeline is specific to the study corpus. Point it at any LAS collection
that matches `docs/DATA_FORMAT.md` and the same windowing, folds, masks, scaling and
metrics are applied unchanged:

```bash
cp configs/paths.example.yaml configs/paths.yaml     # edit las_dir / data_dir; git-ignored
python scripts/build_dataset.py                      # -> well_logs.npz, slices.npz, folds_loo.json
python scripts/run_benchmark.py --models mean,locf,rf,xgb,mice --epochs 100 --dump
```

Requirements and caveats:

* the three logs **GR, RHOB, NPHI** must be present; a well missing one is read, reported
  and then excluded from the folds;
* the number of folds follows your corpus — the harness derives leave-one-well-out folds
  from `folds_loo.json`, it does not assume 19;
* only the two spatial models (GNN, ST-GNN) need a coordinate table
  (`coords_csv`); the other nine methods run without one;
* if your log files and your coordinate table spell some wells differently, copy
  `configs/well_aliases.example.yaml` to `configs/well_aliases.yaml` (git-ignored) and add
  the substitutions;
* `WELLOG_LAS_DIR`, `WELLOG_DATA_DIR`, `WELLOG_COORDS_CSV`, `WELLOG_RESULTS_DIR` and
  `WELLOG_FIGURES_DIR` override the config file, so no path is ever hard-coded.

## 17. Data availability

**The original and composite well-log data used in the study are owned by BRGM (French
Geological Survey) and are subject to restrictions. They cannot be redistributed.**

This repository therefore provides the implementation and the reproducibility framework,
not the study corpus. A user without the BRGM data **cannot** reproduce the exact
numerical results of the paper from scratch: the released aggregate tables let those
numbers be re-derived and re-analysed, and the synthetic corpus lets the whole pipeline be
executed and validated end to end, but the two are different things.

**The research code, masking logic, evaluation pipeline, model implementations, derived
statistics, aggregate results, and reproducibility documentation are released publicly.**

The synthetic example data shipped in `examples/synthetic/` are **entirely artificial**
and are used **only to demonstrate the software interface**. They do **not** derive from,
approximate, reconstruct, or statistically reproduce the BRGM data.

The publication refers to the wells as **W01–W19**; those anonymous identifiers are the
only well references anywhere in this repository. Most released tables are keyed by fold
index rather than by well, so a W-identifier appears explicitly only where a table is
genuinely per-well: `configs/benchmark/folds_loo_19wells.yaml` (all 19) and the two
robustness sub-studies, which were run on a deliberate subset of wells. The
W-identifier ↔ fold-index correspondence is not published, because the W-numbering
encodes an ordering of the corpus.

Note that the pooled prediction dumps are *also* withheld: they carry the ground-truth log
value at every scored position and are therefore derived proprietary data, even though
they look like results.

Full inventory of what is and is not released: `docs/DATA_AVAILABILITY.md`.

## 18. Reproducibility notes

`docs/REPRODUCIBILITY.md` documents the environment, every random seed and what it
controls, which components are deterministic and which are not, the protocol constants,
and the measured runtimes. Two points deserve highlighting here:

* **cuDNN determinism is deliberately off.** Forcing it makes BRITS about 80× slower
  (33 min per fold instead of 24 s), which puts the five-seed campaign out of budget.
  Seeding numpy and torch reproduces the reported numbers to the precision the paper
  states; the residual nondeterminism stays below 1e-3 on the pooled R². The non-neural
  methods are bit-reproducible.
* **Two documented departures** from the earlier single-seed run — the evaluation-seed
  derivation and a ~0.2 percentage-point revision of the blackout co-occurrence constants
  — are described in `docs/REPRODUCIBILITY.md` §4. Neither affects a reported number:
  all headline results come from the five-seed replication.

Paper-to-code traceability, section by section: `docs/METHODS_MAPPING.md`.

## 19. Citation

See `CITATION.cff`.

```bibtex
@article{sawadogo_realistic_missingness,
  title   = {Realistic Missingness Changes the Ranking of Well-Log Imputation Models},
  author  = {Sawadogo, Wendinkont{\'e} and Chassagne, Romain and Joshua, Pwavodi
             and Atteia, Olivier},
  journal = {Computers \& Geosciences},
  note    = {Under review}
}
```

## 20. Contact

Wendinkonté Sawadogo — ED Sciences et Environnements, Université de Bordeaux, and BRGM
(French Geological Survey), Orléans, France. Please open an issue for questions about the
code, and contact the corresponding author for questions about the data.

---

## AI-assisted development

Parts of the code in this repository were written with the assistance of Claude
(Anthropic's AI coding assistant), which was used for selected code generation,
refactoring, debugging and implementation tasks. The scientific authors retain full
responsibility for the methodology, the implementation, the validation, the analyses and
all reported results.
