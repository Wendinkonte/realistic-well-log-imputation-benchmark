#!/usr/bin/env python
"""Separate initialisation and evaluation-mask variability (revision R1, Reviewer 2 c. 4).

Design produced by ``run_replication.py --seed S --mask-seeds 0,1,2,3,4``: every model
trained with initialisation seed S is evaluated on the same five evaluation-mask seeds.
Mask seed 0 is the primary evaluation (``pools_seeded/seed{S}``); the others are in
``pools_maskvar/seed{S}_mask{M}``.  For each method and scenario this script builds the
5 x 5 matrix R2_macro[S, M] and reports

* sd_init  -- standard deviation over S of the mean over M (initialisation variability);
* sd_mask  -- standard deviation over M of the mean over S (mask variability);
* sd_init_at_mask0 -- standard deviation over S with masks fixed (the main tables).

Usage: python scripts/seed_variance.py [--results DIR] [--n-wells 18]
Output: <results>/revision/seed_variance.csv
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

import _bootstrap_path  # noqa: F401
from wellog_imputation.config import get_path
from wellog_imputation.statistics.wellwise import ALL_SCENARIOS, LOGS3, _suff, pooled_r2


def macro_from_dir(d, n_wells):
    """{(method, scenario): R2_macro} from one directory of per-fold dumps."""
    out = {}
    for mdir in sorted(glob.glob(os.path.join(d, "*"))):
        m = os.path.basename(mdir)
        acc = {}
        for fp in sorted(glob.glob(os.path.join(mdir, "fold*.npz"))):
            z = np.load(fp)
            for sc in ALL_SCENARIOS:
                for lg in LOGS3:
                    k = f"{sc}__{lg}"
                    if f"{k}__t" in z.files:
                        a = _suff(z[f"{k}__p"], z[f"{k}__t"], z[f"{k}__w"], n_wells)
                        acc[(sc, lg)] = acc.get((sc, lg), 0) + a
        for sc in ALL_SCENARIOS:
            vals = [pooled_r2(acc[(sc, lg)]) for lg in LOGS3 if (sc, lg) in acc]
            if len(vals) == 3:
                out[(m, sc)] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None)
    ap.add_argument("--n-wells", type=int, default=18)
    a = ap.parse_args()
    R = str(get_path("results_dir", a.results))
    cells = {}                                   # (S, M) -> {(m, sc): r2}
    for d in glob.glob(os.path.join(R, "pools_seeded", "seed*")):
        S = int(re.fullmatch(r"seed(\d+)", os.path.basename(d)).group(1))
        cells[(S, 0)] = macro_from_dir(d, a.n_wells)
    for d in glob.glob(os.path.join(R, "pools_maskvar", "seed*_mask*")):
        S, M = map(int, re.fullmatch(r"seed(\d+)_mask(\d+)", os.path.basename(d)).groups())
        cells[(S, M)] = macro_from_dir(d, a.n_wells)
    Ss = sorted({s for s, _ in cells}); Ms = sorted({m for _, m in cells})
    keys = sorted({k for v in cells.values() for k in v})
    rows = []
    for m, sc in keys:
        Rm = np.full((len(Ss), len(Ms)), np.nan)
        for i, s in enumerate(Ss):
            for j, mm in enumerate(Ms):
                Rm[i, j] = cells.get((s, mm), {}).get((m, sc), np.nan)
        with np.errstate(invalid="ignore"):
            sd_init = float(np.nanstd(np.nanmean(Rm, 1), ddof=1)) if len(Ss) > 1 else np.nan
            sd_mask = float(np.nanstd(np.nanmean(Rm, 0), ddof=1)) if len(Ms) > 1 else np.nan
            sd0 = float(np.nanstd(Rm[:, 0], ddof=1)) if len(Ss) > 1 else np.nan
        rows.append([m, sc, len(Ss), len(Ms), f"{np.nanmean(Rm):.4f}", f"{sd_init:.4f}",
                     f"{sd_mask:.4f}", f"{sd0:.4f}", int(np.isfinite(Rm).sum())])
    out = os.path.join(R, "revision"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "seed_variance.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["method", "scenario", "n_init_seeds", "n_mask_seeds", "r2_macro_mean",
                     "sd_init", "sd_mask", "sd_init_at_mask0", "n_cells"])
        cw.writerows(rows)
    print(f"{len(rows)} rows -> {out}/seed_variance.csv  (init seeds {Ss}, mask seeds {Ms})")


if __name__ == "__main__":
    main()
