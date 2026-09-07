# Synthetic example data

The synthetic example data are provided solely to demonstrate the software interface and
pipeline. They do not reproduce, approximate, or derive from the proprietary BRGM
well-log data used in the study.

They are **entirely artificial**, and they do **not** derive from, approximate,
**reconstruct**, or **statistically reproduce** the BRGM data in any respect.

Every curve here is a sum of a few sine waves plus white noise, on a round depth range,
with values clipped into obviously artificial ranges. No statistic of the real corpus —
no mean, variance, correlation, gap-length distribution, well count, depth coverage or
coordinate — is used anywhere in the generator. The six wells are named `SYN01`–`SYN06`
and sit on a regular grid of round coordinates.

## Generate

```bash
python examples/synthetic/make_synthetic.py
```

This writes, under `examples/synthetic/data/`:

| Path | Purpose |
|---|---|
| `las/SYN*.las` | minimal LAS 2.0 files, to exercise the reader and `scripts/build_dataset.py` |
| `processed/well_logs.npz` | per-well `(L,4)` arrays: depth, GR, RHOB, NPHI |
| `processed/slices.npz` | the windowed tensor `X` plus `wells` and `starts` |
| `processed/folds_loo.json` | leave-one-well-out folds |
| `processed/well_coordinates.csv` | the coordinate table the spatial models need |

## Run the pipeline on it

```bash
export WELLOG_LAS_DIR=examples/synthetic/data/las
export WELLOG_DATA_DIR=examples/synthetic/data/processed
export WELLOG_COORDS_CSV=examples/synthetic/data/processed/well_coordinates.csv
export WELLOG_RESULTS_DIR=/tmp/wellog_demo

# rebuild the processed arrays from the LAS files (optional: they already exist)
python scripts/build_dataset.py

# a couple of cheap methods over two folds
python scripts/run_single_experiment.py --model mean --folds 2
python scripts/run_single_experiment.py --model mice --folds 2

# the spatial family (needs PyTorch)
python scripts/run_single_experiment.py --model stgnn --folds 2 --epochs 2
```

The numbers produced on this corpus are meaningless as science. They exist to prove the
code paths run: loading, windowing, masking, fold-local scaling, model execution and
evaluation.
