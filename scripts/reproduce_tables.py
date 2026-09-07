#!/usr/bin/env python
"""Aggregate the multi-seed replication and regenerate the result tables.

Runs the whole statistical layer of the paper:

* consolidates ``results/raw/raw_seed*.csv`` into ``raw_metrics.csv``;
* recomputes the pooled R^2 per seed from ``results/pools_seeded/`` and writes
  ``aggregated.csv`` (mean +/- std and 95 % CI over seeds);
* writes ``results/significance/``: ``friedman.csv``, the Nemenyi matrices, the
  critical-difference diagrams, ``wilcoxon_pairs.csv`` and ``ranking_stability.csv``;
* writes ``RESULTS_FOR_PAPER.md``.

This needs the per-fold dumps produced by ``scripts/run_replication.py``, which are
derived from the proprietary corpus and are therefore NOT part of this repository.  The
already-aggregated outputs it would produce are released under ``results/`` -- see
docs/DATA_AVAILABILITY.md.

Usage:
    python scripts/reproduce_tables.py [--results DIR] [--out DIR]
"""
import argparse

import _bootstrap_path  # noqa: F401
from wellog_imputation.config import get_path
from wellog_imputation.statistics.aggregate import aggregate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None,
                    help="results directory (default: configured results_dir)")
    ap.add_argument("--out", default=None, help="output directory (default: --results)")
    args = ap.parse_args()
    results = str(get_path("results_dir", args.results))
    aggregate(results, args.out or results)


if __name__ == "__main__":
    main()
