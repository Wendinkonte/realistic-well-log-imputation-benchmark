#!/usr/bin/env python
"""Build the clean windowed dataset from a LAS corpus (paper Section 4.1).

The BRGM corpus used in the study is not redistributed; point ``--las-dir`` at your own
LAS files.  See docs/DATA_FORMAT.md for the expected mnemonics and depth step.

Usage:
    python scripts/build_dataset.py [--las-dir DIR] [--out DIR]
"""
import argparse

import _bootstrap_path  # noqa: F401
from wellog_imputation.data.pipeline import build_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--las-dir", default=None,
                    help="directory of *.las files (default: configured las_dir)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: configured data_dir)")
    args = ap.parse_args()
    build_dataset(las_dir=args.las_dir, out_dir=args.out)


if __name__ == "__main__":
    main()
