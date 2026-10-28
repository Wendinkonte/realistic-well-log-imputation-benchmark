#!/usr/bin/env python
"""Regenerate the figures of the revised article (R1) that are released in
results/revision_r1/figures/.

What each figure needs
----------------------
* ``02_log_distributions_r1`` and ``03_log_crossplots_r1``: histograms and cross-plots of
  GR, RHOB and NPHI pooled over the 18 benchmark wells. They are computed from the well
  logs themselves (``well_logs.npz`` and ``folds_loo.json`` written by
  ``scripts/build_dataset.py``), so they CANNOT be rebuilt without the proprietary corpus.
  Point ``--data`` at your own dataset directory (the synthetic corpus of
  ``examples/synthetic/data/processed`` works too).
* ``fig4_aggregation``: per-fold vs pooled R^2 for three cases of the original
  (single-seed, 19-well) run of the first submission. Its input is that run's
  ``results.json`` (per-fold and pooled metrics, keyed by fold index), released as
  ``results/results_initial_run.json``; pass it with ``--results-json``.
* ``missingness_scenarios_synthetic``: the four masking scenarios drawn on one window of
  the SYNTHETIC example corpus (``--slices examples/synthetic/data/processed/slices.npz``).
  It illustrates the masks only; the article's version of this figure shows a window of a
  real well and is not released.
* ``fig1_missingness_r1`` and ``fig2_rank_reversal_r1`` were drawn outside this pipeline;
  they are released as produced (see results/revision_r1/figures/README.md).

The well map, the per-well depth coverage and the example windows of the article reveal
locations or individual log traces and are neither released nor produced here.

Usage:
    python scripts/make_revision_figures.py [--data DIR] [--results-json FILE]
                                            [--slices FILE] --out DIR
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _bootstrap_path  # noqa: F401,E402
from wellog_imputation.evaluation.harness import EVAL_SCENARIOS, EVAL_SEED  # noqa: E402
from wellog_imputation.masking.scenarios import make_eval_set  # noqa: E402
from wellog_imputation.utils import figstyle as fs  # noqa: E402

LOGS = ["GR", "RHOB", "NPHI"]
UNITS = {"GR": "API", "RHOB": "g/cc", "NPHI": "frac"}
LOG_COLOR = {"GR": "#1b7837", "RHOB": "#b2182b", "NPHI": "#2166ac"}


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}", flush=True)


def _pooled_logs(data):
    """Values of each log over the wells of folds_loo.json, concatenated."""
    wl = np.load(os.path.join(data, "well_logs.npz"), allow_pickle=True)
    wells = json.load(open(os.path.join(data, "folds_loo.json")))["wells"]
    # each well is an (L, 4) array: depth, GR, RHOB, NPHI
    return {lg: np.concatenate([np.asarray(wl[w])[:, c + 1] for w in wells])
            for c, lg in enumerate(LOGS)}


def fig_log_distributions(allv, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, lg in zip(axes, LOGS):
        v = allv[lg][np.isfinite(allv[lg])]
        ax.hist(v, bins=80, color=LOG_COLOR[lg], alpha=0.85)
        ax.set_title(f"{lg} ({UNITS[lg]})  n={v.size}")
        ax.set_xlabel(lg)
        ax.axvline(np.median(v), color="black", ls="--", lw=1, label=f"med={np.median(v):.2f}")
        ax.legend()
    _save(fig, out, "02_log_distributions_r1")


def fig_log_crossplots(allv, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    for ax, (a, b) in zip(axes, [("GR", "RHOB"), ("GR", "NPHI"), ("RHOB", "NPHI")]):
        x, y = allv[a], allv[b]
        ok = np.isfinite(x) & np.isfinite(y)
        ax.hexbin(x[ok], y[ok], gridsize=60, cmap="viridis", bins="log", rasterized=True)
        ax.set_xlabel(f"{a} ({UNITS[a]})", fontsize=10)
        ax.set_ylabel(f"{b} ({UNITS[b]})", fontsize=10)
        ax.set_title(f"{a} vs {b}  (r={np.corrcoef(x[ok], y[ok])[0, 1]:+.2f})")
    _save(fig, out, "03_log_crossplots_r1")


def fig_aggregation(results_json, out):
    r = json.load(open(results_json))
    cases = [("mean", "single", "MEAN · single"),
             ("gnn", "blackout", "GNN · blackout"),
             ("stgnn", "blackout", "ST-GNN · blackout")]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    rng = np.random.default_rng(0)                  # jitter only
    for j, (nm, sc, lab) in enumerate(cases):
        pf = np.array(r[nm]["per_fold"][sc]["all"]["r2"], float)
        pf = pf[np.isfinite(pf)]
        pooled = r[nm]["agg"][sc]["all"]["r2"]
        mean_, med_ = np.nanmean(pf), np.median(pf)
        x = j + rng.uniform(-0.13, 0.13, len(pf))
        ax.scatter(x, np.clip(pf, -6, 1.05), s=26, color="#9e9e9e", edgecolor="black",
                   linewidth=0.3, alpha=0.8, zorder=3, label="per-fold $R^2$" if j == 0 else None)
        ax.plot([j - 0.30, j + 0.30], [mean_] * 2, color="#e41a1c", lw=3, zorder=4,
                label="per-fold mean (flawed)" if j == 0 else None)
        ax.plot([j - 0.30, j + 0.30], [med_] * 2, color="#377eb8", lw=3, zorder=4,
                label="per-fold median" if j == 0 else None)
        ax.plot([j - 0.30, j + 0.30], [pooled] * 2, color="#4daf4a", lw=3, zorder=4,
                label="pooled (ours)" if j == 0 else None)
        ax.annotate(f"{mean_:.2f}", (j + 0.33, mean_), fontsize=8, color="#e41a1c", va="center")
        ax.annotate(f"pooled {pooled:.2f}", (j + 0.33, pooled + 0.02), fontsize=8,
                    color="#2e7d32", va="bottom")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c[2] for c in cases])
    ax.set_xlim(-0.5, len(cases) - 0.1)
    ax.set_ylabel("$R^2$ (clipped at $-6$ for display)")
    ax.set_ylim(-6.3, 1.2)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    _save(fig, out, "fig4_aggregation")


def fig_missingness_scenarios(slices, out, window=0):
    X = np.load(slices, allow_pickle=True)["X"].astype(float)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.4), sharey=True)
    depth = np.arange(X.shape[1])
    for ax, sc in zip(axes, ["single", "block", "profile", "blackout"]):
        rng = np.random.default_rng(EVAL_SEED[sc])
        _, _, mask = make_eval_set(X[window:window + 1], rng=rng, **EVAL_SCENARIOS[sc])
        for c, lg in enumerate(LOGS):
            v = X[window, :, c]
            v = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-8)
            ax.plot(v, depth, color=LOG_COLOR[lg], lw=1, label=lg)
            mk = mask[0, :, c] > 0.5
            ax.scatter(v[mk], depth[mk], s=14, color=LOG_COLOR[lg], marker="s", zorder=5)
        ax.invert_yaxis()
        ax.set_title(f"{sc}\n({100 * mask.mean():.1f}% masked)", fontsize=10)
        ax.set_xlabel("z-score")
    axes[0].legend()
    axes[0].set_ylabel("depth (sample)")
    _save(fig, out, "missingness_scenarios_synthetic")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="dataset directory (well_logs.npz, folds_loo.json)")
    ap.add_argument("--results-json", help="results/results_initial_run.json (original single-seed run)")
    ap.add_argument("--slices", help="slices.npz of the synthetic corpus, for the masking illustration")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not (a.data or a.results_json or a.slices):
        ap.error("nothing to do: give --data, --results-json and/or --slices")
    fs.apply()
    os.makedirs(a.out, exist_ok=True)
    if a.data:
        allv = _pooled_logs(a.data)
        fig_log_distributions(allv, a.out)
        fig_log_crossplots(allv, a.out)
    if a.results_json:
        fig_aggregation(a.results_json, a.out)
    if a.slices:
        fig_missingness_scenarios(a.slices, a.out)


if __name__ == "__main__":
    main()
