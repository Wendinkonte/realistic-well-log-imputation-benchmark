#!/usr/bin/env python
"""Regenerate the results figures of the paper from the released aggregate tables.

Which figures can be rebuilt here
---------------------------------
Every figure whose input is a DERIVED table is reproducible from this repository alone:

* ``09_r2_boxplots_by_scenario`` -- pooled R^2 per method and regime, one point per seed,
  with the zero-skill MEAN floor;
* ``10_perlog_vs_aggregate_r2``  -- aggregate vs per-log R^2, showing the between-channel
  inflation of the pooled ``all`` score;
* ``11_ranking_stability``       -- rank of each method per seed;
* ``metric_{r2,mae,rmse,cc}`` and ``overview_heatmap`` -- the four-metric panels.

The critical-difference diagrams are written by ``scripts/reproduce_tables.py``, next to
the Nemenyi matrices.

The remaining article figures (log distributions, cross-plots, depth coverage, example
windows, the masking illustration, the well map, the convergence curves) are computed
from the well logs themselves and CANNOT be rebuilt without the proprietary corpus.  They
are listed in docs/DATA_AVAILABILITY.md.

Data source
-----------
When the per-fold dumps of ``results/pools_seeded/`` are present -- i.e. you have run the
replication on your own data -- the pooled R^2 is recomputed from them, exactly as in the
original figure code.  Otherwise the script falls back to the released tables
(``results/significance/ranking_stability.csv`` for the per-seed pooled ``all`` values and
``results/aggregated.csv`` for the per-log means), which reproduces the same figures from
public inputs.

Usage:
    python scripts/reproduce_figures.py [--results DIR] [--out DIR]
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import _bootstrap_path  # noqa: F401,E402
from wellog_imputation.config import get_path  # noqa: E402
from wellog_imputation.metrics.pooled import LOGS  # noqa: E402
from wellog_imputation.models.registry import ORDER, SCENARIOS  # noqa: E402
from wellog_imputation.utils import figstyle as fs  # noqa: E402

SCEN = SCENARIOS
#: Colours of the four log categories (all + the three logs).
CAT_COLORS = ["#444444", "#1b7837", "#b2182b", "#2166ac"]


def _save(fig, figs, name):
    p = os.path.join(figs, name)
    # 300 dpi so rasterised elements stay crisp once placed in the article
    fig.savefig(p, dpi=300, bbox_inches="tight")
    if name.lower().endswith(".png"):
        fig.savefig(p[:-4] + ".pdf", bbox_inches="tight")   # vector copy
    plt.close(fig)
    print(f"  [fig] {name}", flush=True)


# ---------------------------------------------------------------- data loading
def _load_pooled_r2_from_dumps(R):
    """Pooled R^2 per ``(method, seed, scenario, log)`` recomputed from pools_seeded/."""
    POOL = os.path.join(R, "pools_seeded")
    out = {}
    for sd in sorted(glob.glob(os.path.join(POOL, "seed*"))):
        seed = int(os.path.basename(sd)[4:])
        for md in sorted(glob.glob(os.path.join(sd, "*"))):
            m = os.path.basename(md)
            acc = defaultdict(lambda: {"p": [], "t": []})
            for fp in sorted(glob.glob(os.path.join(md, "fold*.npz"))):
                z = np.load(fp)
                for sc in SCEN:
                    for lg in ["all"] + LOGS:
                        k = f"{sc}__{lg}"
                        if f"{k}__t" in z.files:
                            acc[(sc, lg)]["p"].append(z[f"{k}__p"])
                            acc[(sc, lg)]["t"].append(z[f"{k}__t"])
            for (sc, lg), d in acc.items():
                if d["t"]:
                    p = np.concatenate(d["p"]); t = np.concatenate(d["t"])
                    ss = np.sum((t - t.mean()) ** 2)
                    out[(m, seed, sc, lg)] = float(1 - np.sum((p - t) ** 2) / ss) if ss > 0 else np.nan
    return out


def _load_pooled_r2_from_tables(R):
    """Same mapping, rebuilt from the released aggregate tables.

    ``ranking_stability.csv`` carries the per-seed pooled ``all`` R^2 exactly; the per-log
    entries are only available as the mean over seeds (``aggregated.csv``), which is what
    the per-log figure plots anyway.
    """
    out = {}
    rs = os.path.join(R, "significance", "ranking_stability.csv")
    if os.path.isfile(rs):
        with open(rs) as f:
            for r in csv.DictReader(f):
                try:
                    out[(r["method"], int(r["seed"]), r["scenario"], "all")] = float(r["pooledR2_all"])
                except (ValueError, KeyError):
                    continue
    agg = os.path.join(R, "aggregated.csv")
    if os.path.isfile(agg):
        seeds = sorted({k[1] for k in out}) or [0]
        with open(agg) as f:
            for r in csv.DictReader(f):
                if r["log"] == "all":
                    continue          # per-seed values already came from ranking_stability
                try:
                    v = float(r["pooledR2_mean"])
                except ValueError:
                    continue
                for s in seeds:       # the mean over seeds, replicated per seed
                    out[(r["method"], s, r["scenario"], r["log"])] = v
    return out


def load_pooled_r2(R):
    """Prefer the per-fold dumps; fall back to the released tables."""
    pr = _load_pooled_r2_from_dumps(R)
    if pr:
        print("[figures] pooled R2 recomputed from results/pools_seeded/", flush=True)
        return pr
    pr = _load_pooled_r2_from_tables(R)
    if pr:
        print("[figures] pooled R2 read from the released aggregate tables", flush=True)
    return pr


# ---------------------------------------------------------------- figures
def fig_r2_boxplots(figs, pr):
    methods = [m for m in ORDER if any(k[0] == m for k in pr)]
    labels = [fs.DISP.get(m, m) for m in methods]
    seeds = sorted({k[1] for k in pr})
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.0))
    for ax, sc in zip(axes.ravel(), SCEN):
        data = [[pr.get((m, s, sc, "all"), np.nan) for s in seeds] for m in methods]
        data = [[v for v in col if v == v] for col in data]
        bp = ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
        for patch, m in zip(bp["boxes"], methods):        # ONE colour = ONE method
            patch.set_facecolor(fs.METHOD_COLOR[m]); patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        for j, col in enumerate(data, 1):
            ax.scatter([j] * len(col), col, s=12, color="k", zorder=5, alpha=0.7)
        floor = [pr.get(("mean", s, sc, "all"), np.nan) for s in seeds]
        floor = np.nanmean([v for v in floor if v == v]) if any(v == v for v in floor) else np.nan
        if floor == floor:
            ax.axhline(floor, color="#b30000", ls="--", lw=1, label=f"MEAN floor={floor:.2f}")
        ax.set_title(f"{sc}   (points = {len(seeds)} seeds)")
        ax.set_ylabel("pooled $R^2$")
        ax.tick_params(axis="x", rotation=45, labelsize=7.5)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, figs, "09_r2_boxplots_by_scenario.png")


def fig_perlog_vs_aggregate(figs, pr):
    methods = [m for m in ORDER if any(k[0] == m for k in pr)]
    labels = [fs.DISP.get(m, m) for m in methods]
    seeds = sorted({k[1] for k in pr})
    cats = ["all"] + LOGS
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
    x = np.arange(len(methods)); width = 0.2
    for ax, sc in zip(axes.ravel(), SCEN):
        for li, lg in enumerate(cats):
            means = [np.nanmean([pr.get((m, s, sc, lg), np.nan) for s in seeds]) for m in methods]
            ax.bar(x + (li - 1.5) * width, means, width, label=lg, color=CAT_COLORS[li])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, fontsize=7.5)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
        ax.set_title(f"{sc}"); ax.set_ylabel("$R^2$")
        ax.grid(axis="y", alpha=0.3)
    handles, labs = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labs, ncol=4, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.02), title="log")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, figs, "10_perlog_vs_aggregate_r2.png")


def fig_ranking_stability(figs, pr):
    methods = [m for m in ORDER if any(k[0] == m for k in pr)]
    labels = [fs.DISP.get(m, m) for m in methods]
    seeds = sorted({k[1] for k in pr})
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4))
    im = None
    for ax, sc in zip(axes.ravel(), SCEN):
        ranks = np.full((len(methods), len(seeds)), np.nan)
        for sj, s in enumerate(seeds):
            scored = [(m, pr.get((m, s, sc, "all"), np.nan)) for m in methods]
            scored = [(m, v) for m, v in scored if v == v]
            scored.sort(key=lambda z: -z[1])
            rk = {m: i + 1 for i, (m, _) in enumerate(scored)}
            for mi, m in enumerate(methods):
                ranks[mi, sj] = rk.get(m, np.nan)
        im = ax.imshow(ranks, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=len(methods))
        ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"s{s}" for s in seeds], fontsize=8)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(labels, fontsize=8)
        for mi in range(len(methods)):
            for sj in range(len(seeds)):
                if ranks[mi, sj] == ranks[mi, sj]:
                    ax.text(sj, mi, int(ranks[mi, sj]), ha="center", va="center", fontsize=8)
        ax.set_title(sc)
    fig.tight_layout(rect=(0, 0, 0.90, 1))
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(im, cax=cax); cb.set_label("rank (1 = best)")
    cb.ax.text(0.5, 1.04, "green = better", transform=cb.ax.transAxes, ha="center",
               va="bottom", fontsize=7.5)
    _save(fig, figs, "11_ranking_stability.png")


# ---- four-metric panels, driven by aggregated_full.csv -----------------------
SCEN_LABEL = {"single": "Single-gap", "block": "Block",
              "profile": "Profile", "blackout": "Blackout"}
# This figure family uses its own family split and palette (AE/U-Net shown as 'deep'),
# kept exactly as published.
MF_FAMILY = {"mean": "baseline", "locf": "baseline",
             "rf": "tabular", "xgb": "tabular", "mice": "tabular",
             "saits": "sequential", "brits": "sequential",
             "gnn": "spatial", "stgnn": "spatial",
             "ae": "deep", "unet": "deep"}
FAM_COLOR = {"baseline": "#9aa0a6", "tabular": "#1f77b4", "sequential": "#ff7f0e",
             "spatial": "#2ca02c", "deep": "#9467bd"}
PRETTY = {"mean": "Mean", "locf": "LOCF", "rf": "RF", "xgb": "XGB", "mice": "MICE",
          "saits": "SAITS", "brits": "BRITS", "gnn": "GNN", "stgnn": "ST-GNN",
          "ae": "AE", "unet": "U-Net"}
METRICS = {
    "mae":  dict(title="MAE",  better="lower",  fmt="{:.3g}"),
    "rmse": dict(title="RMSE", better="lower",  fmt="{:.3g}"),
    "r2":   dict(title="R²",   better="higher", fmt="{:.3f}"),
    "cc":   dict(title="CC",   better="higher", fmt="{:.3f}"),
}


def _metric_style():
    # Reset first: the metric panels have their own look (sans-serif, no in-facing ticks)
    # and must not inherit the serif publication style applied to the figures above.
    matplotlib.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
        "axes.titleweight": "bold", "figure.dpi": 120,
    })


def metric_figure(metric, AGG, out):
    info = METRICS[metric]
    d = AGG[AGG["log"] == "all"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    for ax, sc in zip(axes, SCEN):
        sub = d[d["scenario"] == sc].copy()
        asc = (info["better"] == "lower")
        sub = sub.sort_values(f"{metric}_mean", ascending=not asc)
        y = np.arange(len(sub))
        vals = sub[f"{metric}_mean"].to_numpy()
        err = sub[f"{metric}_std"].to_numpy()
        colors = [FAM_COLOR[MF_FAMILY[m]] for m in sub["method"]]
        ax.barh(y, vals, xerr=err, color=colors, edgecolor="white", linewidth=0.6,
                height=0.74, error_kw=dict(ecolor="#333333", lw=1.0, capsize=2.5))
        ax.set_yticks(y)
        ax.set_yticklabels([PRETTY[m] for m in sub["method"]], fontsize=9.5)
        ax.set_title(SCEN_LABEL[sc], fontsize=12, pad=6)
        ax.grid(axis="x", color="#dddddd", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        if info["better"] == "higher":
            ax.axvline(0, color="#888888", lw=0.8, ls="--")
        span = (np.nanmax(vals) - min(0, np.nanmin(vals))) or 1
        for yi, v in zip(y, vals):
            off = 0.012 * span * (1 if v >= 0 else -1)
            ax.text(v + off, yi, info["fmt"].format(v), va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.2, color="#222222")
        ax.margins(x=0.16)
    arrow = "↓ lower = better" if info["better"] == "lower" else "↑ higher = better"
    fig.suptitle(f"{info['title']}  —  imputation by method and missingness regime  "
                 f"(log = all, mean over 5 seeds, {arrow})",
                 fontsize=14, fontweight="bold", y=0.99)
    handles = [Patch(facecolor=FAM_COLOR[f], label=f) for f in
               ["baseline", "tabular", "sequential", "spatial", "deep"]]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.005), fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"metric_{metric}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  [fig] metric_{metric}.png/.pdf", flush=True)


def overview_heatmap(AGG, out):
    fig, axes = plt.subplots(1, 4, figsize=(18, 6.2))
    methods = ["mean", "locf", "rf", "xgb", "mice", "saits", "brits",
               "gnn", "stgnn", "ae", "unet"]
    d = AGG[AGG["log"] == "all"]
    for ax, (metric, info) in zip(axes, METRICS.items()):
        M = np.full((len(methods), len(SCEN)), np.nan)
        for i, m in enumerate(methods):
            for j, sc in enumerate(SCEN):
                r = d[(d["method"] == m) & (d["scenario"] == sc)]
                if len(r):
                    M[i, j] = r[f"{metric}_mean"].iloc[0]
        cmap = "RdYlGn" if info["better"] == "higher" else "RdYlGn_r"
        if info["better"] == "higher":
            vmin, vmax = max(-1, np.nanmin(M)), 1
        else:
            vmin, vmax = np.nanmin(M), np.nanpercentile(M, 90)
        im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(SCEN)))
        ax.set_xticklabels([SCEN_LABEL[s] for s in SCEN], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([PRETTY[m] for m in methods], fontsize=9)
        ax.set_title(info["title"], fontsize=13)
        for i in range(len(methods)):
            for j in range(len(SCEN)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, info["fmt"].format(M[i, j]), ha="center", va="center",
                            fontsize=7.3, color="#000000")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Overview — metrics by method (rows) and regime (columns), log = all",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out, "overview_heatmap.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  [fig] overview_heatmap.png", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None,
                    help="results directory (default: configured results_dir)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: configured figures_dir)")
    args = ap.parse_args()
    R = str(get_path("results_dir", args.results))
    figs = str(get_path("figures_dir", args.out))
    os.makedirs(figs, exist_ok=True)
    print(f"=== figures -> {figs} ===", flush=True)

    fs.apply()
    pr = load_pooled_r2(R)
    if pr:
        for name, fn in [("r2_boxplots", fig_r2_boxplots),
                         ("perlog_vs_aggregate", fig_perlog_vs_aggregate),
                         ("ranking_stability", fig_ranking_stability)]:
            try:
                fn(figs, pr)
            except Exception as e:      # one failure must not stop the others
                print(f"  [skip] {name}: {type(e).__name__}: {e}", flush=True)
    else:
        print("  [skip] no pooled R2 available (need results/pools_seeded/ or "
              "results/significance/ranking_stability.csv)", flush=True)

    agg_full = os.path.join(R, "aggregated_full.csv")
    if os.path.isfile(agg_full):
        try:
            import pandas as pd
            AGG = pd.read_csv(agg_full)
            _metric_style()
            for m in METRICS:
                metric_figure(m, AGG, figs)
            overview_heatmap(AGG, figs)
        except Exception as e:
            print(f"  [skip] metric panels: {type(e).__name__}: {e}", flush=True)
    else:
        print(f"  [skip] metric panels: {agg_full} not found", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
