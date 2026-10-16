#!/usr/bin/env python
"""Spatial performance against neighbour availability and distance (revision R1, R2 c. 6).

For every benchmark well (held out in turn), over the depths of its windows:
  * n_avail   -- mean number of its 8 graph neighbours with a measured value at the same
                 grid depth (averaged over the three logs);
  * d_nearest -- mean distance (km) to the nearest neighbour with a measured value;
  * w_avail   -- mean total graph weight carried by available neighbours.
These are joined with the seed-averaged per-well nRMSE of the revision analysis
(results/revision/per_well_nrmse.csv) and, for each scenario, the gain of each graph model
over its no-neighbour control (nRMSE_control - nRMSE_model, > 0 = the neighbours help)
is correlated with availability and distance (Spearman, 19 or 18 wells).

Usage: python scripts/neighbour_availability.py [--results DIR] [--data-dir DIR]
Outputs: <results>/revision/neighbour_availability.csv, neighbour_gain_spearman.csv
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

import _bootstrap_path  # noqa: F401
from wellog_imputation.config import get_path
from wellog_imputation.models.spatial import SpatialGrid

PAIRS = [("gnn", "gnn_noneigh"), ("stgnn", "stgnn_noneigh"),
         ("gnn", "gnn_shuffle"), ("stgnn", "stgnn_shuffle")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    R = str(get_path("results_dir", a.results)); D = str(get_path("data_dir", a.data_dir))
    grid = SpatialGrid(data_dir=D)
    s = np.load(os.path.join(D, "slices.npz"), allow_pickle=True)
    wells = s["wells"].astype(str); starts = s["starts"].astype(int); T = s["X"].shape[1]
    with open(os.path.join(D, "folds_loo.json")) as fh:
        order = json.load(fh)["wells"]
    Dkm = np.sqrt(((grid.P[:, None] - grid.P[None]) ** 2).sum(-1)) / 1000.0
    rows = []
    for wi, w in enumerate(order):
        i = grid.widx[w]
        nb = np.flatnonzero((grid.A[i] > 0) & (np.arange(len(grid.wells)) != i))
        na, dn, wa = [], [], []
        for st in starts[wells == w]:
            g0 = grid.gstart(w, int(st))
            V = np.isfinite(grid.G[nb, g0:g0 + T, :])           # (k, T, C)
            na.append(V.sum(0).mean())
            wa.append((grid.A[i, nb][:, None, None] * V).sum(0).mean())
            d = np.where(V, Dkm[i, nb][:, None, None], np.inf).min(0)
            dn.append(np.mean(d[np.isfinite(d)]) if np.isfinite(d).any() else np.nan)
        rows.append([wi, f"W{wi + 1:02d}", len(nb), f"{np.mean(na):.3f}", f"{np.nanmean(dn):.2f}",
                     f"{np.mean(wa):.3f}"])
    out = os.path.join(R, "revision"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "neighbour_availability.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["well_index", "well", "k_neighbours", "n_avail", "d_nearest_km", "w_avail"])
        cw.writerows(rows)

    from scipy.stats import spearmanr
    E = defaultdict(dict)
    with open(os.path.join(out, "per_well_nrmse.csv")) as f:
        for r in csv.DictReader(f):
            if r["nrmse"] != "":
                E[(r["method"], r["scenario"])][int(r["well_index"])] = float(r["nrmse"])
    avail = {int(r[0]): (float(r[3]), float(r[4])) for r in rows}
    res = []
    for model, ctrl in PAIRS:
        for sc in sorted({k[1] for k in E}):
            if (model, sc) not in E or (ctrl, sc) not in E:
                continue
            ws = sorted(set(E[(model, sc)]) & set(E[(ctrl, sc)]) & set(avail))
            gain = np.array([E[(ctrl, sc)][x] - E[(model, sc)][x] for x in ws])
            na = np.array([avail[x][0] for x in ws]); dn = np.array([avail[x][1] for x in ws])
            ok = np.isfinite(gain) & np.isfinite(dn)
            r1 = spearmanr(gain[ok], na[ok]); r2 = spearmanr(gain[ok], dn[ok])
            res.append([model, ctrl, sc, int(ok.sum()), f"{np.median(gain[ok]):.4f}",
                        f"{r1.statistic:.3f}", f"{r1.pvalue:.3g}", f"{r2.statistic:.3f}", f"{r2.pvalue:.3g}"])
    with open(os.path.join(out, "neighbour_gain_spearman.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["model", "control", "scenario", "n_wells", "median_gain_nrmse",
                     "rho_gain_vs_n_avail", "p_n_avail", "rho_gain_vs_d_nearest", "p_d_nearest"])
        cw.writerows(res)
    print(f"-> {out}/neighbour_availability.csv, neighbour_gain_spearman.csv")


if __name__ == "__main__":
    main()
