#!/usr/bin/env python
"""Cluster-bootstrap confidence intervals and paired method comparisons.

Resamples WELLS (not points), matching the leave-one-well-out protocol, and writes
``bootstrap_ci.csv`` and ``bootstrap_pairwise.csv``.

Needs the pool dumps of ``scripts/run_benchmark.py --dump``; those carry ground-truth log
values and are not redistributed.

Usage:
    python scripts/run_bootstrap.py [--dump-dir D] [--out O] [--B 5000] [--seed 0]
"""
import argparse
import os

import _bootstrap_path  # noqa: F401
from wellog_imputation.config import get_path
from wellog_imputation.statistics.bootstrap import run_bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", default=None,
                    help="directory of pool_*.npz (default: <results_dir>/pools)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--B", type=int, default=5000, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    results = str(get_path("results_dir", args.out))
    dump_dir = args.dump_dir or os.path.join(results, "pools")
    run_bootstrap(dump_dir, results, B=args.B, seed=args.seed)


if __name__ == "__main__":
    main()
