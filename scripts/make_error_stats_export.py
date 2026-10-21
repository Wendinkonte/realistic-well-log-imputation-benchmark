#!/usr/bin/env python
"""Build the anonymised error-statistics export of the R1 run -- NO retraining.

Reads the per-fold dumps already on disk

    <results>/pools_seeded/seed{S}/{method}/fold{WI}.npz         (mask seed 0)
    <results>/pools_maskvar/seed{S}_mask{M}/{method}/fold{WI}.npz (mask seeds 1..4)

and reduces each (method, init seed, mask seed, scenario, log, well) to the sufficient
statistics (n, sum y, sum y^2, SSE, SAE) with the SAME arithmetic as
``wellog_imputation.statistics.wellwise._suff`` (float32 dumps cast to float64, bincount).

``blackout_emp`` pools the 5 replicates of a fold in one vector.  The replicates are
split here by re-drawing their masks (``empirical.draw_masks``, same seeded generator
``eval_seed('blackout_emp', M)``) to get the number of masked samples of each replicate
and log; the split is checked against the pooled count.

CONFIDENTIALITY.  Per replicate, some (well, log) cells hold 1 or 2 masked samples, for
which (n, sum y, sum y^2) give the sample values exactly.  The release file therefore
POOLS the replicates of ``blackout_emp`` (rep = 0, as for the other scenarios; every table
pools them anyway); the per-replicate breakdown is written only with ``--per-rep-out``,
to a file that must not be released.  ``mask_ids.csv`` keeps per-replicate COUNTS.

Writes (into --out):
    wellwise_error_stats.csv  method, scenario, init_seed, mask_seed, rep, log, fold,
                              n, sum_y, sum_y2, sse, sae   (full float64 precision)
                              blackout_emp replicates pooled (see CONFIDENTIALITY)
    mask_ids.csv              scenario, mask_seed, rep, fold, effective_seed, n_GR, n_RHOB, n_NPHI
    fold_train_sigma.csv      fold, log, sigma  (training std of the fold, ChannelScaler.sd)

Usage (from the repository root):
    python <export>/make_error_stats_export.py --results RESULTS --data-dir DATA --out EXPORT
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=".", help="repository root (for src/)")
    ap.add_argument("--per-rep-out", default=None,
                    help="also write the per-replicate file here (NOT for release)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.repo, "src"))
    from wellog_imputation.evaluation.empirical import draw_masks, intact_segments, load_intervals
    from wellog_imputation.evaluation.harness import EVAL_SEED, eval_seed
    from wellog_imputation.statistics.wellwise import (ALL_SCENARIOS, LOGS3, _suff,
                                                       training_sigmas)
    from wellog_imputation.models.registry import ORDER

    import json
    F = json.load(open(os.path.join(a.data_dir, "folds_loo.json")))
    fold_keys = list(F["folds"])
    NW = len(fold_keys)
    WID = list(range(NW))   # fold index; no W label (the W numbering would encode a corpus order)
    T = int(np.load(os.path.join(a.data_dir, "slices.npz"), allow_pickle=True)["X"].shape[1])

    # ---- per-replicate counts of blackout_emp, re-drawn from the seeded generator ----
    wl = np.load(os.path.join(a.data_dir, "well_logs.npz"), allow_pickle=True)
    intervals = load_intervals(a.data_dir)
    N_REP = 5
    emp_counts = {}                     # (mask_seed, wi) -> (N_REP, 3) int
    for ms in range(5):
        for wi, fk in enumerate(fold_keys):
            arr = wl[F["folds"][fk]["test"][0]][:, 1:4]
            segs = intact_segments(arr, T)
            c = np.zeros((N_REP, 3), np.int64)
            if segs:
                rng = np.random.default_rng(eval_seed("blackout_emp", ms))
                for r, rep in enumerate(draw_masks(segs, intervals, rng, N_REP)):
                    for m in rep:
                        c[r] += m.sum(0)
            emp_counts[(ms, wi)] = c

    # ---- the pools ----
    cells = []                          # (init_seed, mask_seed, directory)
    for d in glob.glob(os.path.join(a.results, "pools_seeded", "seed*")):
        cells.append((int(re.fullmatch(r"seed(\d+)", os.path.basename(d)).group(1)), 0, d))
    for d in glob.glob(os.path.join(a.results, "pools_maskvar", "seed*_mask*")):
        s, m = map(int, re.fullmatch(r"seed(\d+)_mask(\d+)", os.path.basename(d)).groups())
        cells.append((s, m, d))
    cells.sort()
    found = sorted({os.path.basename(p) for _, _, d in cells for p in glob.glob(os.path.join(d, "*"))})
    methods = [m for m in ORDER if m in found] + sorted(set(found) - set(ORDER))

    rows, rows_rep = [], []
    nref = {}                           # (sc, ms, rep, wi, lg) -> n, must agree across all
    for m in methods:
        for s, ms, d in cells:
            files = sorted(glob.glob(os.path.join(d, m, "fold*.npz")))
            if len(files) != NW:
                raise SystemExit(f"{d}/{m}: {len(files)} folds, expected {NW}")
            for fp in files:
                wi = int(re.search(r"fold(\d+)\.npz$", fp).group(1))
                z = np.load(fp)
                for sc in ALL_SCENARIOS:
                    for li, lg in enumerate(LOGS3):
                        k = f"{sc}__{lg}"
                        if f"{k}__t" not in z.files:
                            continue
                        p, t, w = z[f"{k}__p"], z[f"{k}__t"], z[f"{k}__w"]
                        if w.size and not np.all(w == wi):
                            raise SystemExit(f"{fp}: well label != fold index")
                        st = _suff(p, t, w, NW)[wi]      # whole fold vector: exact original sums
                        if sc == "blackout_emp":
                            rows.append([m, sc, s, ms, 0, lg, WID[wi], int(st[0])]
                                        + [repr(float(x)) for x in st[1:]])
                            cnt = emp_counts[(ms, wi)][:, li]
                            if cnt.sum() != t.size:
                                raise SystemExit(f"{fp} {k}: replicate split {cnt.sum()} != {t.size}")
                            bounds = np.concatenate(([0], np.cumsum(cnt)))
                            parts = [(r, slice(bounds[r], bounds[r + 1])) for r in range(N_REP)]
                        else:
                            parts = [(0, slice(0, t.size))]
                        for r, sl in parts:
                            if sc == "blackout_emp":
                                st = _suff(p[sl], t[sl], np.full(sl.stop - sl.start, wi), NW)[wi]
                            key = (sc, ms, r, wi, lg)
                            if nref.setdefault(key, int(st[0])) != int(st[0]):
                                raise SystemExit(f"mask count differs across methods/seeds at {key}")
                            row = [m, sc, s, ms, r, lg, WID[wi], int(st[0])] + [repr(float(x)) for x in st[1:]]
                            (rows_rep if sc == "blackout_emp" else rows).append(row)
        print(f"{m}: done ({len(rows)} rows)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    hdr = ["method", "scenario", "init_seed", "mask_seed", "rep", "log", "fold",
           "n", "sum_y", "sum_y2", "sse", "sae"]
    with open(os.path.join(a.out, "wellwise_error_stats.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(hdr); cw.writerows(rows)
    if a.per_rep_out:
        with open(a.per_rep_out, "w", newline="") as f:
            cw = csv.writer(f); cw.writerow(hdr)
            cw.writerows([r for r in rows if r[1] != "blackout_emp"] + rows_rep)

    mrows = []
    for (sc, ms, r, wi) in sorted({k[:4] for k in nref}, key=lambda k: (ALL_SCENARIOS.index(k[0]), k[1], k[2], k[3])):
        eff = eval_seed(sc, ms)
        mrows.append([sc, ms, r, WID[wi], eff] + [nref[(sc, ms, r, wi, lg)] for lg in LOGS3])
    with open(os.path.join(a.out, "mask_ids.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["scenario", "mask_seed", "rep", "fold", "effective_seed",
                     "n_GR", "n_RHOB", "n_NPHI"])
        cw.writerows(mrows)

    sig = training_sigmas(a.data_dir)
    with open(os.path.join(a.out, "fold_train_sigma.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["fold", "log", "sigma"])
        for wi in range(NW):
            for lg in LOGS3:
                cw.writerow([WID[wi], lg, repr(float(sig[lg][wi]))])
    print(f"{len(rows)} stat rows, {len(mrows)} mask rows, methods={len(methods)}, "
          f"cells={len(cells)}, EVAL_SEED={EVAL_SEED}")


if __name__ == "__main__":
    main()
