# SLURM templates

Templates, not turnkey scripts. They carry no absolute path: set `WELLOG_*` (or write
`configs/paths.yaml`) and point `PROJECT` and `VENV` at your own installation.

The published campaign ran on one NVIDIA H100 NVL per task. Wall-clock budgets in the
`--time` directives are the ones actually used; see `docs/REPRODUCIBILITY.md`.

| Template | What it runs |
|---|---|
| `run_replication.sbatch` | one array task per master seed — the five-seed campaign |
| `run_replication_perfold.sbatch` | one array task per (seed, fold) — used for BRITS, which dominates the cost |
| `run_ablations.sbatch` | both leakage experiments, then the aggregation |
