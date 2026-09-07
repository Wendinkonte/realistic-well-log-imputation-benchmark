#!/bin/bash
# The two controlled leakage experiments, then the multi-seed aggregation.
# Both ablations are single-seed paired experiments, as labelled in the paper.
# Template -- set PROJECT and VENV.
#SBATCH --job-name=wellog_ablations
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
VENV="${VENV:-$PROJECT/.venv}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJECT"

export PYTHONHASHSEED=0
export WELLOG_DATA_DIR="${WELLOG_DATA_DIR:-$PROJECT/data/processed}"
export WELLOG_RESULTS_DIR="${WELLOG_RESULTS_DIR:-$PROJECT/results}"

# Section 4.3 -- grid-row ablation: anti-leakage ON vs OFF, everything else matched.
python scripts/run_leakage_ablation.py --models gnn,stgnn --epochs 100

# Section 5.4 -- window-level split instead of leave-one-well-out.
python scripts/run_window_split_leak.py --models gnn,stgnn --epochs 100

# Aggregate whatever replication output is present, and rebuild the figures.
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
