#!/bin/bash
# Revision R1 -- single complete re-run on the corrected dataset (W17 excluded).
#
# Edit the five variables below, then run the steps IN ORDER from the repository root:
#   bash scripts/slurm/revision_r1.sh build        # 1. rebuild the dataset (login node, minutes)
#   bash scripts/slurm/revision_r1.sh submit       # 2. submit the arrays (GPU / CPU)
#   bash scripts/slurm/revision_r1.sh analyse      # 3. after all jobs: reanalysis + tables
#
# Design
#   * every model is trained with init seed S = 0..4 and evaluated on the SAME evaluation
#     masks (mask seed 0 = main tables), then re-evaluated on mask seeds 1..4
#     (pools_maskvar/) -> initialisation vs mask variability (Reviewer 2, comment 4);
#   * scenario blackout_emp (empirical whole-well blackout, 5 replicates) for every method;
#   * BRITS / BRITS-mix are split per (seed, fold) as in the original run.
set -euo pipefail

LAS_DIR=/path/to/las                                   # raw LAS files (BRGM)
COORDS=/path/to/well_coordinates.csv                   # Lambert-93 coordinates
DATA=/path/to/data_clean_r1                    # NEW dataset dir (old one untouched)
OUT=/path/to/results_r1                             # NEW results dir (old one untouched)
GPU_OPTS="--partition=CHANGE_ME_GPU --gres=gpu:1 --cpus-per-task=4 --mem=32G"
CPU_OPTS="--partition=CHANGE_ME_CPU --cpus-per-task=8 --mem=32G"
PY=python

export WELLOG_LAS_DIR=$LAS_DIR WELLOG_DATA_DIR=$DATA WELLOG_RESULTS_DIR=$OUT WELLOG_COORDS_CSV=$COORDS
COMMON="--epochs 100 --mask-seeds 0,1,2,3,4 --empirical-reps 5 --data-dir $DATA --out $OUT"
mkdir -p "$OUT/logs"

FAST_CPU="mean,locf,locf_win,interp_linear,interp_pchip,rf,xgb,mice,ridge,knn,xgb_miss,xgb_chain_asc,xgb_chain_desc,idw,nnwell,okrige"
FAST_GPU="saits,saits_mix,unet,ae,gnn,stgnn,gnn_noneigh,stgnn_noneigh,gnn_shuffle,stgnn_shuffle"
SLOW_GPU="brits brits_mix"

case "${1:-}" in
build)
  $PY scripts/build_dataset.py --las-dir "$LAS_DIR" --out "$DATA"
  echo "--- depth QC (W17 must be EXCLUDED) ---"; cat "$DATA/depth_qc.csv"
  $PY - <<PYEOF
import json; d = json.load(open("$DATA/missingness_real.json"))
for k in ["n_wells_triplet", "cooccur_pct_samples", "cooccur_pct_intervals",
          "gap_length_m_all", "gap_length_m_interior"]:
    print(k, d[k])
print("combinations:", {k: v["share_samples_pct"] for k, v in d["combinations"].items()})
PYEOF
  ;;
submit)
  for S in 0 1 2 3 4; do
    sbatch $CPU_OPTS --time=12:00:00 -J r1cpu_s$S -o "$OUT/logs/%x_%j.out" \
      --wrap "$PY scripts/run_replication.py --seed $S --methods $FAST_CPU $COMMON"
    sbatch $GPU_OPTS --time=12:00:00 -J r1gpu_s$S -o "$OUT/logs/%x_%j.out" \
      --wrap "$PY scripts/run_replication.py --seed $S --methods $FAST_GPU $COMMON"
    NF=$($PY -c "import json;print(json.load(open('$DATA/folds_loo.json'))['n_folds'])")
    for M in $SLOW_GPU; do
      sbatch $GPU_OPTS --time=04:00:00 --array=0-$((NF-1)) -J r1${M}_s$S -o "$OUT/logs/%x_%A_%a.out" \
        --wrap "$PY scripts/run_replication.py --seed $S --methods $M --only-folds \$SLURM_ARRAY_TASK_ID $COMMON"
    done
  done
  sbatch $GPU_OPTS --time=08:00:00 -J r1leak -o "$OUT/logs/%x_%j.out" \
    --wrap "$PY scripts/run_leak_matched.py --models gnn,stgnn --epochs 100 --seed 0 --out $OUT --data-dir $DATA"
  squeue -u "$USER"
  ;;
analyse)
  NW=$($PY -c "import json;print(json.load(open('$DATA/folds_loo.json'))['n_folds'])")
  $PY scripts/reanalyse_revision.py --results "$OUT" --data-dir "$DATA" --n-wells "$NW" --boot 10000
  $PY scripts/seed_variance.py --results "$OUT" --n-wells "$NW"
  $PY scripts/neighbour_availability.py --results "$OUT" --data-dir "$DATA"
  $PY scripts/make_revision_tables.py "$OUT/revision"
  ls -la "$OUT/revision" "$OUT/revision/latex" "$OUT/ablations"
  ;;
*) echo "usage: $0 {build|submit|analyse}"; exit 1;;
esac
