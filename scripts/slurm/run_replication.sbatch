#!/bin/bash
# One array task per master seed: the five-seed replication campaign.
# Template -- set PROJECT and VENV, then: sbatch scripts/slurm/run_replication.sbatch
#SBATCH --job-name=wellog_repl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
VENV="${VENV:-$PROJECT/.venv}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJECT"

# Fixed so mask seeding never drifts with Python's salted string hash.
export PYTHONHASHSEED=0
export WELLOG_DATA_DIR="${WELLOG_DATA_DIR:-$PROJECT/data/processed}"
export WELLOG_RESULTS_DIR="${WELLOG_RESULTS_DIR:-$PROJECT/results}"

# Append-only and resumable: re-submitting skips every fold already on disk.
python scripts/run_replication.py --seed "$SLURM_ARRAY_TASK_ID" --epochs 100
