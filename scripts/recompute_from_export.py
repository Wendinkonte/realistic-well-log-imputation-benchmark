#!/usr/bin/env python
"""Recompute the revision R1 tables from the anonymised error-statistics export ONLY.

Inputs (``--export DIR``; no raw log, no per-fold dump, no package import needed):

    wellwise_error_stats.csv   n, sum_y, sum_y2, sse, sae per
                               (method, scenario, init_seed, mask_seed, rep, log, fold);
                               ``well`` (W01-W18) is the label of ``fold`` (W{fold+1:02d})
    fold_train_sigma.csv       training standard deviation of each log in each fold
                               (the fold of well w is the one where w is held out)

Outputs (``--out DIR``), same layout and rounding as ``scripts/reanalyse_revision.py``:

    table_perlog_revision.csv, table_main_revision.csv, per_well_nrmse.csv,
    friedman_wellwise.csv, wilcoxon_wellwise_holm.csv, wilcoxon_revision_holm.csv,
    seed_variance.csv

The main tables use mask seed 0 (the primary evaluation) and sum the replicates of
``blackout_emp``; ``seed_variance.csv`` uses every (init seed, mask seed) cell.
``--full`` writes every number at full float64 precision instead of the published
rounding.  ``--compare DIR`` then reports, file by file, the largest absolute difference
with the CSVs of the same name in DIR.

Usage:
    python recompute_from_export.py --export EXPORT --out OUT [--boot 10000] [--full]
                                    [--compare REFERENCE_DIR]
"""
import argparse
import csv
import itertools
import os
from collections import defaultdict

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

# ---- design constants (identical to wellog_imputation, commit 1a3fec5) -----------------
LOGS3 = ["GR", "RHOB", "NPHI"]
SCENARIOS = ["single", "block", "profile", "blackout", "blackout_emp"]
#: the 11 methods of the original article come first, the revision methods follow sorted
ORDER = ["mean", "locf", "rf", "xgb", "mice", "saits", "brits", "unet", "ae", "gnn", "stgnn"]
LEARNED = ["rf", "xgb", "mice", "saits", "brits", "unet", "ae", "gnn", "stgnn"]
#: pre-registered family of the original article (significance.WILCOXON_PAIRS)
WILCOXON_PAIRS = {
    "profile": [("stgnn", "gnn"), ("stgnn", "mice"), ("gnn", "mice"),
                ("xgb", "mice"), ("xgb", "stgnn"), ("xgb", "gnn"),
                ("unet", "mice"), ("unet", "stgnn"), ("unet", "gnn")],
    "blackout": [("locf", m) for m in LEARNED],
}
#: revision R1 families (wellwise.REVISION_PAIRS)
REVISION_PAIRS = {
    "profile": [("saits_mix", "saits"), ("brits_mix", "brits"),
                ("gnn", "gnn_noneigh"), ("stgnn", "stgnn_noneigh"),
                ("gnn", "gnn_shuffle"), ("stgnn", "stgnn_shuffle"),
                ("stgnn", "idw"), ("stgnn", "okrige"), ("mice", "okrige"),
                ("unet", "ridge"), ("mice", "ridge")],
    "blackout": [("xgb_miss", "xgb"), ("xgb_chain_asc", "xgb_chain_desc"),
                 ("saits_mix", "saits"), ("brits_mix", "brits"),
                 ("locf", "interp_linear")],
    "blackout_emp": [("locf", "locf_win"), ("locf", "interp_linear")]
                    + [("locf", m) for m in LEARNED],
}
BOOT_SEED = 20261016
ALPHA = 0.05
N, SY, SYY, SSE, SAE = range(5)


# ---- reading -----------------------------------------------------------------------------
def read_export(d):
    rows = list(csv.DictReader(open(os.path.join(d, "wellwise_error_stats.csv"))))
    NW = 1 + max(int(r["fold"]) for r in rows)                    # fold index 0..NW-1
    bad = [r for r in rows if "well" in r and r["well"] != f"W{int(r['fold']) + 1:02d}"]
    if bad:
        raise SystemExit(f"well label != W{{fold+1}} in {len(bad)} rows")
    cells = defaultdict(lambda: np.zeros((NW, 5)))   # (m, init, mask, sc, lg) -> (W, 5)
    for r in rows:
        a = cells[(r["method"], int(r["init_seed"]), int(r["mask_seed"]), r["scenario"], r["log"])]
        a[int(r["fold"])] += [float(r["n"]), float(r["sum_y"]), float(r["sum_y2"]),
                              float(r["sse"]), float(r["sae"])]    # += sums the replicates
    sig = {lg: np.full(NW, np.nan) for lg in LOGS3}
    for r in csv.DictReader(open(os.path.join(d, "fold_train_sigma.csv"))):
        sig[r["log"]][int(r["fold"])] = float(r["sigma"])
    found = {k[0] for k in cells}
    methods = [m for m in ORDER if m in found] + sorted(found - set(ORDER))
    return cells, sig, methods, NW


# ---- metrics -----------------------------------------------------------------------------
def r2(a, wts=None):
    if wts is not None:
        a = a * wts[:, None]
    n, sy, syy, sse = a[:, N].sum(), a[:, SY].sum(), a[:, SYY].sum(), a[:, SSE].sum()
    if n <= 1:
        return np.nan
    sst = syy - sy * sy / n
    return float(1.0 - sse / sst) if sst > 0 else np.nan


def r2_macro(S, m, s, sc, wts=None):
    v = [r2(S[(m, s, sc, lg)], wts) for lg in LOGS3 if (m, s, sc, lg) in S]
    return float(np.mean(v)) if len(v) == 3 else np.nan


def nrmse_wells(S, m, s, sc, sig):
    T = []
    for lg in LOGS3:
        a = S[(m, s, sc, lg)]
        with np.errstate(invalid="ignore", divide="ignore"):
            T.append(np.where(a[:, N] > 0, a[:, SSE] / (a[:, N] * sig[lg] ** 2), np.nan))
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(np.vstack(T), axis=0))


def holm(p):
    p = np.asarray(p, float)
    adj = np.full_like(p, np.nan); rej = np.zeros(p.shape, bool)
    ok = np.where(np.isfinite(p))[0]
    if ok.size == 0:
        return adj, rej
    order = ok[np.argsort(p[ok])]
    run, stop = 0.0, False
    for i, j in enumerate(order):
        run = max(run, min(1.0, (ok.size - i) * p[j])); adj[j] = run
    for i, j in enumerate(order):
        if not stop and p[j] <= ALPHA / (ok.size - i):
            rej[j] = True
        else:
            stop = True
    return adj, rej


def wx(ea, eb):
    keep = np.isfinite(ea) & np.isfinite(eb)
    d = ea[keep] - eb[keep]
    if d.size < 5 or np.all(d == 0):
        return dict(n=int(d.size), stat=np.nan, p=np.nan, r_rb=np.nan,
                    med=float(np.median(d)) if d.size else np.nan)
    res = wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="auto")
    dd = d[d != 0]; rk = rankdata(np.abs(dd))
    tp, tm = rk[dd > 0].sum(), rk[dd < 0].sum()
    return dict(n=int(d.size), stat=float(res.statistic), p=float(res.pvalue),
                r_rb=float((tp - tm) / (tp + tm)), med=float(np.median(d)))


def boot(S, a, b, sc, seeds, NW, B, rng):
    """By-well cluster bootstrap of Delta R2_macro; same draw for both methods, all seeds."""
    seeds = [s for s in seeds if all((m, s, sc, lg) in S for m in (a, b) for lg in LOGS3)]
    if not seeds:
        return (np.nan,) * 4
    point = np.mean([r2_macro(S, a, s, sc) - r2_macro(S, b, s, sc) for s in seeds])
    present = np.where(sum(S[(a, seeds[0], sc, lg)][:, N] for lg in LOGS3) > 0)[0]
    dl = np.empty(B)
    for i in range(B):
        w = np.bincount(rng.choice(present, size=present.size, replace=True),
                        minlength=NW).astype(float)
        dl[i] = np.mean([r2_macro(S, a, s, sc, w) - r2_macro(S, b, s, sc, w) for s in seeds])
    lo, hi = np.nanpercentile(dl, [2.5, 97.5])
    pb = 2 * min(np.mean(dl <= 0), np.mean(dl >= 0))
    return float(point), float(lo), float(hi), float(min(1.0, pb))


# ---- driver ------------------------------------------------------------------------------
def run(export, out, B=10000, full=False):
    cells, sig, methods, NW = read_export(export)

    def f(x, nd=4):                                   # published rounding, or full
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return ""
        return repr(float(x)) if full else round(float(x), nd)

    def g(x, spec):                                   # p-value formats
        return repr(float(x)) if full else format(x, spec)

    def w(name, header, rows):
        with open(os.path.join(out, name), "w", newline="") as fh:
            cw = csv.writer(fh, lineterminator="\n"); cw.writerow(header); cw.writerows(rows)

    os.makedirs(out, exist_ok=True)
    S = {(m, s, sc, lg): a for (m, s, ms, sc, lg), a in cells.items() if ms == 0}
    seeds = sorted({k[1] for k in S})
    SCN = [sc for sc in SCENARIOS if any(k[2] == sc for k in S)]

    rows_log, macro = [], {}
    for m in methods:
        for sc in SCN:
            for lg in LOGS3:
                v = [(r2(S[k]), S[k][:, SAE].sum() / S[k][:, N].sum(),
                      np.sqrt(S[k][:, SSE].sum() / S[k][:, N].sum()))
                     for k in [(m, s, sc, lg) for s in seeds] if k in S]
                if v:
                    r2s, maes, rmses = map(list, zip(*v))
                    rows_log.append([m, sc, lg, len(r2s), f(np.mean(r2s)),
                                     f(np.std(r2s, ddof=1) if len(r2s) > 1 else 0.0),
                                     f(np.mean(maes)), f(np.mean(rmses))])
            macro[(m, sc)] = [r2_macro(S, m, s, sc) for s in seeds if (m, s, sc, "GR") in S]
    w("table_perlog_revision.csv", ["method", "scenario", "log", "n_seeds", "r2_mean", "r2_sd",
                                    "mae_mean", "rmse_mean"], rows_log)

    E, rows_well = {}, []
    for m in methods:
        for sc in SCN:
            per = [nrmse_wells(S, m, s, sc, sig) for s in seeds
                   if all((m, s, sc, lg) in S for lg in LOGS3)]
            if per:
                with np.errstate(invalid="ignore"):
                    E[(m, sc)] = np.nanmean(np.vstack(per), axis=0)
                rows_well += [[m, sc, i, f(v)] for i, v in enumerate(E[(m, sc)])]
    w("per_well_nrmse.csv", ["method", "scenario", "well_index", "nrmse"], rows_well)

    rows_main = []
    for sc in SCN:
        ms = [m for m in methods if macro.get((m, sc))]
        mean = {m: np.mean(macro[(m, sc)]) for m in ms}
        rank = {m: i + 1 for i, m in enumerate(sorted(ms, key=lambda x: -mean[x]))}
        for m in ms:
            v = macro[(m, sc)]; e = E.get((m, sc), np.array([np.nan]))
            q1, med, q3 = np.nanpercentile(e, [25, 50, 75])
            rows_main.append([m, sc, len(v), f(np.mean(v)),
                              f(np.std(v, ddof=1) if len(v) > 1 else 0.0), rank[m],
                              f(med), f(q1), f(q3), f(np.nanmean(e))])
    w("table_main_revision.csv", ["method", "scenario", "n_seeds", "r2_macro_mean", "r2_macro_sd",
                                  "rank_r2_macro", "nrmse_well_median", "nrmse_well_q1",
                                  "nrmse_well_q3", "nrmse_well_mean"], rows_main)

    rows_f = []
    for sc in SCN:
        ms = [m for m in methods if (m, sc) in E]
        if len(ms) < 3:
            continue
        X = np.column_stack([E[(m, sc)] for m in ms]); X = X[np.all(np.isfinite(X), axis=1)]
        chi2, p = friedmanchisquare(*X.T)
        rows_f.append([sc, len(ms), X.shape[0], f(chi2), g(p, ".3e"),
                       f(chi2 / (X.shape[0] * (len(ms) - 1)))])
    w("friedman_wellwise.csv", ["scenario", "k_methods", "n_wells", "chi2_F", "p", "kendall_W"],
      rows_f)

    hdr = ["scenario", "family_size_m", "A", "B", "n_wells", "median_diff_nrmse_A_minus_B",
           "W_stat", "p_exact", "p_holm", "reject_holm_0.05", "rank_biserial",
           "delta_r2_macro_A_minus_B", "boot_ci_lo", "boot_ci_hi", "p_boot"]
    rng = np.random.default_rng(BOOT_SEED)            # ONE stream, both families, in order
    for name, fam in [("wilcoxon_wellwise_holm.csv", WILCOXON_PAIRS),
                      ("wilcoxon_revision_holm.csv", REVISION_PAIRS)]:
        rows = []
        for sc, pl in fam.items():
            pl = [(a, b) for a, b in pl if (a, sc) in E and (b, sc) in E]
            if not pl:
                continue
            res = [wx(E[(a, sc)], E[(b, sc)]) for a, b in pl]
            adj, rej = holm([r["p"] for r in res])
            for (a, b), r, pa, rj in zip(pl, res, adj, rej):
                d, lo, hi, pb = boot(S, a, b, sc, seeds, NW, B, rng)
                rows.append([sc, len(pl), a, b, r["n"], f(r["med"]), f(r["stat"]),
                             g(r["p"], ".4g") if np.isfinite(r["p"]) else "",
                             g(pa, ".4g") if np.isfinite(pa) else "", bool(rj),
                             f(r["r_rb"], 3), f(d), f(lo), f(hi), f(pb, 4)])
        w(name, hdr, rows)

    # seed_variance: R2_macro on every (init seed, mask seed) cell
    allc = defaultdict(dict)
    for (m, s, ms, sc, lg), a in cells.items():
        allc[(s, ms)][(m, sc, lg)] = a
    Ss = sorted({k[0] for k in allc}); Ms = sorted({k[1] for k in allc})
    rows_sv = []
    for m, sc in sorted({(k[0], k[1]) for c in allc.values() for k in c}):
        R = np.full((len(Ss), len(Ms)), np.nan)
        for i, s in enumerate(Ss):
            for j, mm in enumerate(Ms):
                c = allc.get((s, mm), {})
                if all((m, sc, lg) in c for lg in LOGS3):
                    R[i, j] = np.mean([r2(c[(m, sc, lg)]) for lg in LOGS3])
        with np.errstate(invalid="ignore"):
            sd_i = float(np.nanstd(np.nanmean(R, 1), ddof=1))
            sd_m = float(np.nanstd(np.nanmean(R, 0), ddof=1))
            sd0 = float(np.nanstd(R[:, 0], ddof=1))
        rows_sv.append([m, sc, len(Ss), len(Ms)]
                       + [g(x, ".4f") for x in (np.nanmean(R), sd_i, sd_m, sd0)]
                       + [int(np.isfinite(R).sum())])
    w("seed_variance.csv", ["method", "scenario", "n_init_seeds", "n_mask_seeds", "r2_macro_mean",
                            "sd_init", "sd_mask", "sd_init_at_mask0", "n_cells"], rows_sv)


# ---- comparison --------------------------------------------------------------------------
EXACT_COLS = {"rank_r2_macro", "p_exact", "p_holm", "reject_holm_0.05", "n_seeds", "n_wells",
              "k_methods", "family_size_m", "W_stat", "n_cells"}


def compare(a_dir, b_dir, names):
    """{file: (n_cells, max |a-b| on numeric cells, n differing exact-column cells,
    n differing text cells)}."""
    out = {}
    for nm in names:
        A = list(csv.reader(open(os.path.join(a_dir, nm))))
        Bf = os.path.join(b_dir, nm)
        if not os.path.exists(Bf):
            continue
        B_ = list(csv.reader(open(Bf)))
        if len(A) != len(B_) or A[0] != B_[0]:
            out[nm] = ("shape/header mismatch", len(A), len(B_)); continue
        hdr = A[0]; mx = 0.0; nex = ntx = ncell = 0
        for ra, rb in zip(A[1:], B_[1:]):
            for col, x, y in zip(hdr, ra, rb):
                ncell += 1
                try:
                    fx, fy = float(x), float(y)
                except ValueError:
                    ntx += x != y; continue
                if np.isnan(fx) and np.isnan(fy):
                    continue
                dxy = abs(fx - fy)
                if col in EXACT_COLS:
                    nex += dxy != 0
                mx = max(mx, dxy)
        out[nm] = (ncell, mx, nex, ntx)
    return out


FILES = ["table_main_revision.csv", "friedman_wellwise.csv", "wilcoxon_wellwise_holm.csv",
         "wilcoxon_revision_holm.csv", "table_perlog_revision.csv", "per_well_nrmse.csv",
         "seed_variance.csv"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    run(a.export, a.out, B=a.boot, full=a.full)
    print(f"-> {a.out}")
    if a.compare:
        print(f"{'file':32s} {'cells':>7s} {'max|diff|':>11s} {'exact≠':>7s} {'text≠':>6s}")
        for nm, r in compare(a.out, a.compare, FILES).items():
            print(f"{nm:32s} " + (" ".join(map(str, r)) if isinstance(r[0], str) else
                                  f"{r[0]:7d} {r[1]:11.3e} {r[2]:7d} {r[3]:6d}"))


if __name__ == "__main__":
    main()
