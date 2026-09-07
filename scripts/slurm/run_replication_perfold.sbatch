#!/bin/bash
# One array task per (seed, fold) for the expensive methods -- BRITS is ~10 h per seed
# when run serially, so it is decomposed across 5 seeds x 19 folds = 95 tasks.
# Template -- set PROJECT and VENV.
#SBATCH --job-name=wellog_repl_perfold
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-94%8
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
VENV="${VENV:-$PROJECT/.venv}"
METHODS="${METHODS:-brits}"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJECT"

export PYTHONHASHSEED=0
export WELLOG_DATA_DIR="${WELLOG_DATA_DIR:-$PROJECT/data/processed}"
export WELLOG_RESULTS_DIR="${WELLOG_RESULTS_DIR:-$PROJECT/results}"

# task id -> (seed, fold) over 5 seeds x 19 folds
SEED=$(( SLURM_ARRAY_TASK_ID / 19 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 19 ))

python scripts/run_replication.py --seed "$SEED" --only-folds "$FOLD" \
                                  --methods "$METHODS" --epochs 100
