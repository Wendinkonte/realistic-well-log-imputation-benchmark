#!/usr/bin/env python
"""Multi-seed replication of the full benchmark (paper Section 5.5).

For ONE master seed, runs the whole matrix ``methods x scenarios x 19 LOO folds``
WITHOUT changing anything in the experimental design (architectures, hyperparameters,
CV scheme, scenarios).  The master seed drives every source of randomness -- evaluation
masks (by derivation), training masks, torch/numpy initialisation, shuffles -- giving
reproducibility WITHIN a (seed, fold) and honest variability BETWEEN seeds.

Outputs (append-only, robust to SLURM preemption):

``<results>/raw/raw_seed{S}_{method}_fold{WI}.csv``
    Columns ``seed, method, scenario, fold, log, r2, mae``.  One file per
    ``(seed, method, fold)``, so nothing races even when 19 processes of the same seed
    write at once.
``<results>/pools_seeded/seed{S}/{method}/fold{WI}.npz``
    Points ``(pred, true, well)`` of the fold.  Written LAST and renamed atomically, so
    its presence is the RESUME MARKER: re-running the same command skips finished folds.
``<results>/curves/curve_{method}_seed{S}_fold{WI}.npz``
    Per-epoch train/val loss of the neural models.

Re-running the same command never recomputes an expensive fold (BRITS is ~10 h/seed).

Usage:
    python scripts/run_replication.py --seed S [--epochs 100] [--methods a,b,c]
                                      [--folds N] [--only-folds 0,3,5]
"""
import argparse
import csv
import os
import time

import _bootstrap_path  # noqa: F401
from wellog_imputation.config import get_path
from wellog_imputation.evaluation.harness import run_model
from wellog_imputation.metrics.pooled import LOGS, save_pool
from wellog_imputation.models.factory import factory
from wellog_imputation.models.registry import FAMILY, SCENARIOS

#: Floor for the significance tests (Friedman/Wilcoxon).  Raise if the budget allows.
N_SEEDS = 5
#: BRITS last: it dominates the cost.
METHODS = ["mean", "locf", "rf", "xgb", "mice", "gnn", "stgnn", "saits", "brits"]
SCEN = SCENARIOS
RAW_HEADER = ["seed", "method", "scenario", "fold", "log", "r2", "mae"]


class RawWriter:
    """Write the per-cell metrics and dump the fold's points (the resume marker).

    Race-free by construction: ONE CSV per ``(seed, method, fold)`` at a unique path,
    written atomically -- never an append into a shared file.  This matters because
    BRITS is decomposed per fold, so up to 19 processes of the same seed write in
    parallel.  The aggregation step concatenates every ``raw_seed*.csv`` shard.
    """

    def __init__(self, seed, method, raw_dir, pool_dir):
        self.seed = seed
        self.method = method
        self.raw_dir = raw_dir
        self.pool_dir = pool_dir
        os.makedirs(raw_dir, exist_ok=True)

    def fold_npz_path(self, wi):
        return os.path.join(self.pool_dir, f"seed{self.seed}", self.method, f"fold{wi:02d}.npz")

    def on_fold(self, wi, fold_key, test_well, fold_metrics, fold_pool):
        # 1) per-(seed, method, fold) CSV shard at a unique path -> no race.
        rows = []
        for sc in SCEN:
            mm = fold_metrics[sc]
            for lg in (["all"] + LOGS):
                r2 = mm[lg]["r2"]; mae = mm[lg]["mae"]
                rows.append([self.seed, self.method, sc, wi, lg,
                             "" if r2 != r2 else f"{r2:.6f}",
                             "" if mae != mae else f"{mae:.6f}"])
        csv_path = os.path.join(self.raw_dir,
                                f"raw_seed{self.seed}_{self.method}_fold{wi:02d}.csv")
        tmp_csv = csv_path + ".tmp"
        with open(tmp_csv, "w", newline="") as f:
            cw = csv.writer(f); cw.writerow(RAW_HEADER); cw.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp_csv, csv_path)
        # 2) dump the fold's points = RESUME MARKER (written last, renamed atomically).
        path = self.fold_npz_path(wi)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # the temporary name MUST end in .npz, otherwise np.savez appends the extension
        tmp = path[:-4] + ".tmp.npz"
        save_pool(fold_pool, SCEN, tmp)
        os.replace(tmp, path)   # atomic: the fold counts as done only once renamed


def run_one_method(method, seed, epochs, folds_subset, raw_dir, pool_dir, curve_dir,
                   fold_indices=None, data_dir=None):
    w = RawWriter(seed, method, raw_dir, pool_dir)

    def skip_fold(wi, fk):
        return os.path.exists(w.fold_npz_path(wi))

    if method in ("gnn", "stgnn"):
        from wellog_imputation.models.spatial import run_spatial
        run_spatial(kind=method, folds_subset=folds_subset, epochs=epochs, seed=seed,
                    on_fold=w.on_fold, skip_fold=skip_fold, curve_dir=curve_dir,
                    fold_indices=fold_indices, data_dir=data_dir)
    elif method in ("unet", "ae"):
        # Deep sequential family: the master seed is PROPAGATED to the torch/numpy
        # initialisation of the model (the spatial family's convention), giving
        # determinism within a seed and honest variance between seeds.
        run_model(factory(method, epochs=epochs, seed=seed), method.upper(),
                  folds_subset=folds_subset, seed=seed, on_fold=w.on_fold,
                  skip_fold=skip_fold, curve_dir=curve_dir, fold_indices=fold_indices,
                  data_dir=data_dir)
    elif method in ("saits", "brits"):
        run_model(factory(method, epochs=epochs), method.upper(), folds_subset=folds_subset,
                  seed=seed, on_fold=w.on_fold, skip_fold=skip_fold, curve_dir=curve_dir,
                  fold_indices=fold_indices, data_dir=data_dir)
    else:
        run_model(factory(method), method.upper(), folds_subset=folds_subset,
                  seed=seed, on_fold=w.on_fold, skip_fold=skip_fold,
                  fold_indices=fold_indices, data_dir=data_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True,
                    help="master seed of the run (one SLURM array task)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    ap.add_argument("--folds", type=int, default=None, help="fold subset (smoke test)")
    ap.add_argument("--only-folds", type=str, default=None,
                    help="fold indices, e.g. '7' or '0,3,5' -> per-fold parallelism")
    ap.add_argument("--out", type=str, default=None,
                    help="results directory (default: configured results_dir)")
    ap.add_argument("--data-dir", type=str, default=None)
    args = ap.parse_args()

    OUT = str(get_path("results_dir", args.out))
    raw_dir = os.path.join(OUT, "raw")
    pool_dir = os.path.join(OUT, "pools_seeded")
    curve_dir = os.path.join(OUT, "curves")

    methods = args.methods.split(",")
    fold_indices = ([int(x) for x in args.only_folds.split(",")]
                    if args.only_folds else None)
    print(f"=== REPLICATION seed={args.seed} | epochs={args.epochs} | methods={methods}"
          f" | folds={fold_indices if fold_indices is not None else 'all'} ===", flush=True)
    for m in methods:
        sel = fold_indices if fold_indices is not None else range(19)
        n_done = sum(os.path.exists(os.path.join(pool_dir, f"seed{args.seed}", m,
                                                 f"fold{wi:02d}.npz")) for wi in sel)
        t0 = time.time()
        print(f"\n##### seed{args.seed} {m.upper()} ({FAMILY[m]}) | "
              f"{n_done}/{len(list(sel))} target folds already done #####", flush=True)
        run_one_method(m, args.seed, args.epochs, args.folds, raw_dir, pool_dir, curve_dir,
                       fold_indices=fold_indices, data_dir=args.data_dir)
        print(f"  -> seed{args.seed} {m} finished in {int(time.time()-t0)}s", flush=True)
    print(f"\n=== seed{args.seed} DONE ===", flush=True)


if __name__ == "__main__":
    main()
