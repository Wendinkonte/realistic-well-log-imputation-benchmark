#!/usr/bin/env python
"""Revision reanalysis from the existing per-fold dumps -- NO retraining.

Reads ``<results>/pools_seeded/seed{S}/{method}/fold*.npz`` (written by
``run_replication.py``) and writes, under ``<out>``:

* ``table_perlog_revision.csv``   pooled R^2 / MAE / RMSE per log, mean +/- sd over seeds
* ``table_main_revision.csv``     R2_macro (equal-weight mean of the 3 logs) + rank, and the
                                  per-well nRMSE median / IQR (equal weight per well)
* ``per_well_nrmse.csv``          seed-averaged nRMSE per (method, scenario, well)
* ``friedman_wellwise.csv``       Friedman across the 19 wells + Kendall's W
* ``wilcoxon_wellwise_holm.csv``  pre-registered pairs: exact p, Holm, rank-biserial r,
                                  by-well bootstrap 95 % CI of Delta R2_macro
* ``wilcoxon_allpairs_holm.csv``  all pairs per scenario (supplementary family)
* ``sufficient_stats_per_well.csv`` only with ``--release-suff`` (check licensing first)

Usage:
    python scripts/reanalyse_revision.py [--results DIR] [--out DIR] [--data-dir DIR]
                                         [--boot 10000] [--release-suff]
"""
import argparse
import os

import _bootstrap_path  # noqa: F401
from wellog_imputation.config import get_path
from wellog_imputation.statistics.wellwise import run_revision


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None, help="results dir holding pools_seeded/")
    ap.add_argument("--out", default=None, help="output dir (default <results>/revision)")
    ap.add_argument("--data-dir", default=None,
                    help="dataset dir (slices.npz + folds_loo.json) for exact training sigmas")
    ap.add_argument("--n-wells", type=int, default=19)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--release-suff", action="store_true")
    a = ap.parse_args()
    R = str(get_path("results_dir", a.results))
    out = a.out or os.path.join(R, "revision")
    info = run_revision(os.path.join(R, "pools_seeded"), out, n_wells=a.n_wells,
                        data_dir=a.data_dir, B=a.boot, release_suff=a.release_suff)
    print(f"seeds={info['seeds']} methods={info['methods']} sigma={info['sigma_source']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
