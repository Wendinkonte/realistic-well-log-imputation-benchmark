#!/usr/bin/env python
"""Matched leakage experiment (revision R1, Reviewer 2 comment 7).

The original window-split experiment compared a random 80/20 window split with the
leave-one-well-out run, which were evaluated on different windows and differed in
training-grid availability.  Here every configuration is evaluated on EXACTLY the same
held-out windows and masked positions, with the same number of training windows, and the
factors are varied one at a time:

  A  window split; training windows of the same well, including those overlapping the
     test windows (stride 64 => up to 192/256 shared samples); full training grid
     -- the original leaky configuration;
  B  as A, minus the training windows that overlap a test window (same-well exposure
     without shared samples);
  C  as B, and the test windows' depth ranges blanked in the training grid
     (no graph-row availability of the test positions);
  D  leave-one-well-out for each well that has test windows (no same-well exposure),
     the held-out well's row blanked -- the protocol of the paper.

Training sets are subsampled (seeded) to the size of the smallest configuration.
Output: <results>/ablations/leak_matched.csv (pooled per-log R2 and R2_macro per cell and
scenario) and leak_matched_points.npz (pooled points with well labels, for the by-well
statistics).

Usage: python scripts/run_leak_matched.py [--models gnn,stgnn] [--epochs 100] [--seed 0]
"""
import argparse
import csv
import json
import os

import numpy as np

import _bootstrap_path  # noqa: F401
from wellog_imputation.config import get_path
from wellog_imputation.evaluation.harness import EVAL_SCENARIOS, EVAL_SEED, ChannelScaler
from wellog_imputation.masking.scenarios import make_eval_set
from wellog_imputation.metrics.pooled import LOGS, add_to_pool, new_pool, pooled_agg, save_pool
from wellog_imputation.models.registry import SCENARIOS


def _subsample(idx, n, rng):
    idx = np.asarray(sorted(idx))
    return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


def run(kind, epochs, seed, test_frac, data_dir, coords_csv, out_dir):
    from wellog_imputation.models.spatial import SpatialGrid, SpatialImputer
    data_dir = get_path("data_dir", data_dir)
    s = np.load(os.path.join(data_dir, "slices.npz"), allow_pickle=True)
    X = s["X"].astype(np.float32); wells = s["wells"].astype(str); starts = s["starts"].astype(int)
    with open(os.path.join(data_dir, "folds_loo.json")) as fh:
        well_list = json.load(fh)["wells"]
    widx = {w: i for i, w in enumerate(well_list)}
    grid = SpatialGrid(data_dir=data_dir, coords_csv=coords_csv)
    N, T = len(X), X.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    test = np.sort(perm[: int(round(test_frac * N))])
    tset = set(test.tolist())
    test_wells = sorted(set(wells[test]))

    def overlaps_test(i):
        return any(wells[j] == wells[i] and abs(int(starts[j]) - int(starts[i])) < T for j in test)

    trA = [i for i in range(N) if i not in tset]
    trB = [i for i in trA if not overlaps_test(i)]
    trD = {w: [i for i in range(N) if wells[i] != w] for w in test_wells}
    n_match = min(len(trA), len(trB), min(len(v) for v in trD.values()))
    sub = np.random.default_rng(seed + 1)
    cfg = {
        "A_window_overlap": dict(train=_subsample(trA, n_match, sub), blank=None, loo=False),
        "B_window_no_overlap": dict(train=_subsample(trB, n_match, sub), blank=None, loo=False),
        "C_window_no_overlap_blanked": dict(train=_subsample(trB, n_match, sub),
                                            blank=[(wells[j], int(starts[j])) for j in test],
                                            loo=False),
        "D_well_split": dict(train={w: _subsample(v, n_match, sub) for w, v in trD.items()},
                             blank=None, loo=True),
    }
    # identical evaluation masks for every configuration (masks depend on shape + rng only)
    masks = {sc: make_eval_set(np.zeros((len(test), T, 3)), rng=np.random.default_rng(EVAL_SEED[sc]),
                               **EVAL_SCENARIOS[sc])[2] > 0.5 for sc in SCENARIOS}
    rows, dump = [], {}
    for name, c in cfg.items():
        pool = new_pool(SCENARIOS)
        groups = ([(w, c["train"][w], [k for k, j in enumerate(test) if wells[j] == w])
                   for w in test_wells] if c["loo"] else [(None, c["train"], list(range(len(test))))])
        for test_well, tr, kk in groups:
            if not kk:
                continue
            sc_ = ChannelScaler().fit(X[tr])
            imp = SpatialImputer(kind=kind, epochs=epochs, seed=seed)
            imp.fit(grid, set(wells), list(range(N)), X, wells, starts, sc_,
                    test_well=test_well, train_slices=list(tr), blank_ranges=c["blank"])
            idx = test[kk]
            Z = sc_.transform(X[idx])
            for sc in SCENARIOS:
                M = masks[sc][kk]
                Zin = Z.copy(); Zin[M] = np.nan
                P = np.stack([imp.impute_slice(wells[j], int(starts[j]), z) for j, z in zip(idx, Zin)])
                wl = np.array([widx[w] for w in wells[idx]])
                for wv in np.unique(wl):
                    sel = wl == wv
                    add_to_pool(pool, sc, sc_.inverse(P[sel]), X[idx][sel],
                                M[sel].astype(np.float32), well_idx=int(wv))
        agg = pooled_agg(pool, SCENARIOS)
        for sc in SCENARIOS:
            r2s = [agg[sc][lg]["r2"] for lg in LOGS]
            rows.append([kind, name, sc, len(test), n_match] + [f"{v:.4f}" for v in r2s]
                        + [f"{np.nanmean(r2s):.4f}"])
        save_pool(pool, SCENARIOS, os.path.join(out_dir, f"leak_matched_{kind}_{name}.npz"))
        print(f"  [{kind}] {name}: " + " ".join(f"{r[2]}={r[-1]}" for r in rows[-4:]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gnn,stgnn")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    out = os.path.join(str(get_path("results_dir", a.out)), "ablations")
    os.makedirs(out, exist_ok=True)
    rows = []
    for k in a.models.split(","):
        rows += run(k, a.epochs, a.seed, a.test_frac, a.data_dir, None, out)
    with open(os.path.join(out, "leak_matched.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["model", "config", "scenario", "n_test_windows", "n_train_windows",
                     "r2_GR", "r2_RHOB", "r2_NPHI", "r2_macro"])
        cw.writerows(rows)
    print(f"-> {out}/leak_matched.csv")


if __name__ == "__main__":
    main()
