# Evaluating Well-Log Imputation Under Structured Missingness: Simple Baselines, Protocol Effects and Inter-Well Information in a Paris Basin Geothermal Case Study

Reference implementation and reproducibility package for a benchmark of **eleven well-log
imputation methods**, plus **seventeen baselines and controls**, across **five missingness
regimes**, under a leakage-controlled, leave-one-well-out protocol on **18 Paris Basin
geothermal wells**.

> **Release 1.1.0 (revision R1, tag `v1.1-r1`).** The results of the revised article are in
> `results/revision_r1/`. The files elsewhere in `results/` are those of release 1.0
> (original submission, 19 wells); they are kept unchanged for transparency, and
> `results/README.md` gives the status of each one. What changed in R1 and why:
> `docs/REVISION_R1.md`.

[![CI](https://github.com/Wendinkonte/realistic-well-log-imputation-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Wendinkonte/realistic-well-log-imputation-benchmark/actions/workflows/ci.yml)

---

## 1. Overview

Machine-learning imputation of well logs is usually assessed on masks that remove one log
at a time. This repository contains the code and the derived results of a benchmark that
evaluates imputers under the **structured missingness observed in the field** — several
logs absent together over the same interval — and asks how much of the apparent skill of
learned methods survives simple baselines, careful protocol choices and well-level
statistics.

It provides the complete pipeline — LAS ingestion with depth-step QC, windowing, the five
masking generators, the imputers and controls, the leave-one-well-out harness, per-log
metrics, well-level significance tests, the by-well bootstrap and the leakage experiments
— plus every derived table the revised article reports.

> **Scope.** This repository provides the implementation and reproducibility framework
> used in the study. The proprietary BRGM well-log data cannot be redistributed.
> Synthetic data are provided to validate and demonstrate the software pipeline.

## 2. Scientific motivation

Existing benchmarks mask within a single log at a time: a few points, a short gap, or at
most one whole log. On the study corpus, missing samples are dominated by **several logs
absent together over the same contiguous interval** — entire tool passes lost, a
structured, missing-not-at-random *partial blackout*
(`results/revision_r1/data_qc/missingness_real.json`).

A second hazard compounds it. The "composite" curves often used as ground truth are
depth-merged products that may blend measured wireline data with model-derived infill, so
evaluating an imputer against them can be partly circular. The revision audits the
composite logs, and the benchmark scores **measured curves only**.

## 3. Main result

All statements below are read from `results/revision_r1/` (18 wells, five initialisation
seeds, primary evaluation masks).

* **Simple baselines match or beat the learned methods in four of the five regimes.**
  Linear (or PCHIP) interpolation along depth leads on `single`, `block` and `blackout`;
  on `profile`, linear regression on the co-located logs (`ridge`) ties the best learned
  method. Only on the empirical blackout (`blackout_emp`) does no simple baseline lead.
* **Inter-well information does not help the graph models.** GNN and ST-GNN score at
  least as well with the neighbours removed (`*_noneigh`, identity graph) as with them, in
  every regime; shuffling the coordinates (`*_shuffle`) does not hurt either.
* **No method recovers RHOB or NPHI on observed gaps.** Under `blackout_emp`, where the
  absent-log combination and gap length are drawn from the gaps actually observed in the
  corpus, per-log R² stays near zero for density and neutron porosity for every method;
  only GR is partly recovered.
* **The spatial-leakage effect comes from overlapping training windows.** In the matched
  experiment (`ablations/leak_matched.csv`: identical windows, masks and training-set size),
  the window split with overlapping windows scores higher than the non-overlapping split,
  the split with blanked borders and the well split, which agree with one another.
* **Protocol choices change the ranking.** Per-log R² rather than a pooled multi-log score,
  the well as the statistical unit, and the empirical rather than the synthetic blackout
  each change which method appears to win.

## 4. Methods compared

The eleven methods of the article:

| Family | Methods |
|---|---|
| **Tabular** | MEAN, Random Forest, XGBoost, MICE |
| **Sequential** | LOCF, SAITS, BRITS, U-Net, AE |
| **Spatial** | GNN, ST-GNN |

The seventeen baselines and controls added in the revision:

| Purpose | Methods |
|---|---|
| Simple baselines | `ridge` (linear regression), `knn`, `interp_linear`, `interp_pchip`, `locf_win` (window-limited LOCF) |
| Spatial baselines (same depth, training wells only) | `idw`, `nnwell`, `okrige` |
| Training on corrupted windows | `saits_mix`, `brits_mix` |
| XGBoost variants | `xgb_miss` (missing predictors seen in training), `xgb_chain_asc`, `xgb_chain_desc` (chained equations, both orders) |
| Graph controls | `gnn_noneigh`, `stgnn_noneigh` (identity graph), `gnn_shuffle`, `stgnn_shuffle` (kNN graph on permuted coordinates) |

Tabular methods predict a missing entry from the other logs at the same depth. Sequential
methods read the target well's window along depth. Spatial methods additionally borrow
from neighbouring wells through a kNN graph built on surface coordinates.

Implementations: `src/wellog_imputation/models/`. Hyperparameters: `configs/models/`.

## 5. Missingness scenarios

| Scenario | What is masked | Regime |
|---|---|---|
| `single` | 5 scattered samples in one log | mono-log, easy |
| `block` | one contiguous block in one log | mono-log |
| `profile` | an entire log of the window | mono-log |
| `blackout` | **k logs over the same contiguous interval**, k drawn from the co-occurrence measured on the corpus (`missingness_real.json`) | synthetic multi-log |
| `blackout_emp` | absent-log combination and length drawn **jointly from the observed gaps**, applied to the held-out well before windowing | **empirical, field-like** |

Implementation: `src/wellog_imputation/masking/scenarios.py` and
`src/wellog_imputation/evaluation/empirical.py`. Parameters:
`configs/masking/scenarios.yaml`.

## 6. Evaluation protocol

* **18 wells** carrying the complete GR / RHOB / NPHI triplet and passing the depth-step QC
  (0.1524 m step; one well excluded, see `results/revision_r1/data_qc/`); windows of 256
  samples, stride 64.
* **Leave-one-well-out cross-validation**, 18 folds — one per well. No window of the
  held-out well ever reaches training. Fold structure:
  `configs/benchmark/folds_loo_18wells.yaml` (W01–W18, `well = W{fold+1:02d}`).
* **Fold-local scaling**: the channel scaler is fitted on the training windows of the fold
  only.
* **Identical evaluation masks across methods**, from a fixed per-scenario seed table, so
  every method is scored at exactly the same positions.
* **Seeds**: five initialisation seeds, and five evaluation-mask seeds decoupled from them
  (mask seed 0 is the primary evaluation; `seed_variance.csv` separates the two sources).
* **Primary metric: per-log R²** (GR, RHOB, NPHI) at artificially masked positions, in
  physical units, and their equal-weight mean (`r2_macro`). A pooled multi-log score is
  not used, because between-log variance inflates it.
* **The well is the statistical unit**: seed-averaged per-well NRMSE; Friedman test with
  Kendall's W; exact Wilcoxon signed-rank tests with Holm correction within each scenario
  family; by-well cluster bootstrap of ΔR²_macro.
* **Anti-leakage**: well-level splits, fold-local scaling, and — for the graph models —
  the held-out well's row blanked in the training grid so it is not even available as a
  neighbour.

Every table of the revised article can be recomputed from the released per-well
sufficient statistics, without the raw logs:

```bash
python scripts/recompute_from_export.py --export results/revision_r1/error_stats --out OUT
```

## 7. Repository structure

```
├── configs/                    protocol and hyperparameters, transcribed from the code
│   ├── benchmark/default.yaml
│   ├── benchmark/folds_loo_18wells.yaml   the 18 leave-one-well-out folds of R1, W01-W18
│   ├── benchmark/folds_loo_19wells.yaml   the 19 folds of release 1.0 (legacy)
│   ├── masking/scenarios.yaml
│   ├── models/{tabular,sequential,spatial}.yaml
│   └── paths.example.yaml      copy to paths.yaml and point at your own data
├── src/wellog_imputation/
│   ├── config.py               path resolution; no absolute path lives anywhere else
│   ├── data/pipeline.py        LAS -> depth-step QC, cleaned logs, windows, folds
│   ├── masking/scenarios.py    the mask generators
│   ├── models/
│   │   ├── tabular.py          MEAN, RF, XGBoost, MICE, ridge, kNN, XGBoost variants
│   │   ├── sequential.py       LOCF, interpolation, SAITS, BRITS, AE, U-Net
│   │   ├── spatial.py          GNN, ST-GNN, spatial baselines, grid, adjacency, leakage toggle
│   │   ├── factory.py          name -> constructor
│   │   └── registry.py         method order, family labels, display names
│   ├── evaluation/harness.py   LOO loop, fold-local scaler, seed tables
│   ├── evaluation/empirical.py the empirical blackout (blackout_emp)
│   ├── metrics/pooled.py       metrics and pooled aggregation
│   ├── statistics/             aggregation, well-wise tests, cluster bootstrap
│   └── utils/                  determinism, figure style, provenance
├── scripts/                    command-line entry points (+ SLURM templates)
├── examples/synthetic/         a fully synthetic corpus, runnable with no data
├── results/
│   ├── revision_r1/            R1 results of the revised article (tables, error_stats,
│   │                           data_qc, ablations, figures)
│   └── ...                     release 1.0 results, kept for transparency
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

The single R1 campaign of the revised article (28 methods, five scenarios, five
initialisation and five evaluation-mask seeds) is driven by
`scripts/slurm/revision_r1.sh build | submit | analyse` (edit the paths and partitions
first).

## 12. Reproducing the tables

Revised article (R1), from the released per-well sufficient statistics, no data needed:

```bash
python scripts/recompute_from_export.py --export results/revision_r1/error_stats --out OUT
```

or, from your own per-fold dumps, `scripts/reanalyse_revision.py`.

Release 1.0:

```bash
python scripts/reproduce_tables.py
```

Consolidates the per-cell metrics, recomputes the pooled R² per seed, and writes
`aggregated.csv`, `raw_metrics.csv`, everything under `results/significance/`, and
`RESULTS_FOR_PAPER.md`. Needs the per-fold dumps produced by the replication; the
already-aggregated outputs are released in `results/`.

## 13. Reproducing the figures

Revised article (R1): `scripts/make_revision_figures.py`; see
`results/revision_r1/figures/README.md` for what each figure needs.

Release 1.0:

```bash
python scripts/reproduce_figures.py
```

**This one runs from a fresh clone with no data at all.** When the private per-fold dumps
are absent it falls back to the released aggregate tables, which reproduces the R²
boxplots, the per-log-versus-aggregate comparison, the ranking-stability heatmap and the
four-metric panels. The figures that depend on the well logs themselves cannot be rebuilt
without the corpus — they are listed in `docs/DATA_AVAILABILITY.md`.

## 14. Reproducing the significance tests

**Revised article (R1).** The well-level tests — Friedman with Kendall's W
(`friedman_wellwise.csv` over the 28 methods, `friedman_main11.csv` over the 11 methods
of the original article), exact Wilcoxon with Holm correction within each scenario family
(`wilcoxon_wellwise_holm.csv`, `wilcoxon_allpairs_holm.csv`, `wilcoxon_revision_holm.csv`)
and the by-well bootstrap of ΔR²_macro — are written to `results/revision_r1/` by
`scripts/reanalyse_revision.py`, or recomputed from the released export by
`scripts/recompute_from_export.py`.

**Release 1.0.** `scripts/reproduce_tables.py` produces all of them:

| Output | Test |
|---|---|
| `results/significance/friedman.csv` | Friedman, 19 folds as blocks, per (scenario, seed) — release 1.0 |
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

Released outputs: `results/ablations/ablation_leak.csv` and `leak_randomsplit.csv`
(release 1.0).

The matched experiment of the revision (identical windows, masks and training-set size;
overlapping vs non-overlapping windows, blanked borders, well split):

```bash
python scripts/run_leak_matched.py
```

Released output: `results/revision_r1/ablations/leak_matched.csv`.

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

The revised article refers to its 18 wells as **W01–W18**, numbered in the order of the
leave-one-well-out folds (`well = W{fold+1:02d}`); with BRGM approval this correspondence
is published in `results/revision_r1/error_stats/` and
`configs/benchmark/folds_loo_18wells.yaml`. The well excluded by the depth-step QC has no
identifier. The release-1.0 files use a separate W01–W19 numbering (in the two robustness
sub-studies only); no correspondence between the two numberings is published. No real
well name, file name or coordinate appears anywhere in this repository.

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
@article{sawadogo_structured_missingness,
  title   = {Evaluating Well-Log Imputation Under Structured Missingness: Simple Baselines,
             Protocol Effects and Inter-Well Information in a {Paris Basin} Geothermal
             Case Study},
  author  = {Sawadogo, Wendinkont{\'e} and Chassagne, Romain and Joshua, Pwavodi
             and Atteia, Olivier},
  journal = {Applied Computing and Geosciences},
  note    = {Under review (manuscript ACAGS-D-26-00297)}
}

@software{sawadogo_wellog_imputation_software,
  title   = {wellog\_imputation: reproducibility package for {Evaluating Well-Log Imputation
             Under Structured Missingness}},
  author  = {Sawadogo, Wendinkont{\'e} and Chassagne, Romain and Joshua, Pwavodi
             and Atteia, Olivier},
  version = {1.1.0},
  note    = {Tag v1.1-r1},
  url     = {https://github.com/Wendinkonte/realistic-well-log-imputation-benchmark}
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
