"""Revision analysis: per-log metrics, well-level statistics (reviewer comments R1-3/4/5/10,
R2-1/5).

Everything here is computed from PER-WELL SUFFICIENT STATISTICS of the pooled points,

    S[method, seed, scenario, log, well] = (n, sum y, sum y^2, SSE, SAE)

built once from the per-fold dumps ``pools_seeded/seed{S}/{method}/fold*.npz``.  These
statistics are enough to recompute, exactly, every pooled metric of the paper on any
subset or bootstrap resample of wells:

    R^2 = 1 - SSE / (sum y^2 - (sum y)^2 / n)

so the by-well cluster bootstrap costs a few array additions per replicate.  They contain
no individual log value and are the "per-fold sufficient statistics" requested by
Reviewer 2 (comment 4) -- check with the data owner before releasing them.

Definitions (see the response letter, section 3)
-------------------------------------------------
* ``R2_c``      pooled out-of-fold R^2 of log ``c`` in its own units.
* ``R2_macro``  equal-weight mean of R2_GR, R2_RHOB, R2_NPHI  -> PRIMARY metric.
* ``nRMSE_w``   per-well error standardised by the TRAINING standard deviation of each log
                in that fold:  sqrt( mean_c  SSE_{c,w} / (n_{c,w} sigma_c^2) ).
                Used for per-well summaries and for all inferential tests, because R^2
                inside one short well is unstable (Section 5.3).
* Statistical unit = the well.  Each (method, well) value is first AVERAGED over seeds,
  then Friedman / Wilcoxon are run across the 19 wells.
* Multiple testing: Holm step-down within a declared family (one family per scenario).
"""
import csv
import glob
import itertools
import os
from collections import defaultdict

import numpy as np

from ..models.registry import ORDER, SCENARIOS

__all__ = ["LOGS3", "build_sufficient_stats", "pooled_r2", "macro_r2", "nrmse_per_well",
           "holm", "rank_biserial", "kendall_w", "friedman_wellwise", "wilcoxon_wellwise",
           "boot_delta_macro_r2", "training_sigmas", "fallback_sigmas", "run_revision"]

LOGS3 = ["GR", "RHOB", "NPHI"]
#: Field order of the sufficient-statistics vector.
F_N, F_SY, F_SYY, F_SSE, F_SAE = range(5)


# ----------------------------------------------------------------------------------------
# 1. sufficient statistics
# ----------------------------------------------------------------------------------------
def _suff(p, t, w, n_wells):
    """(n_wells, 5) array of (n, sum y, sum y^2, SSE, SAE) for one pooled vector."""
    out = np.zeros((n_wells, 5))
    if t.size == 0:
        return out
    p = p.astype(np.float64); t = t.astype(np.float64); w = w.astype(np.int64)
    e = p - t
    out[:, F_N] = np.bincount(w, minlength=n_wells)
    out[:, F_SY] = np.bincount(w, weights=t, minlength=n_wells)
    out[:, F_SYY] = np.bincount(w, weights=t * t, minlength=n_wells)
    out[:, F_SSE] = np.bincount(w, weights=e * e, minlength=n_wells)
    out[:, F_SAE] = np.bincount(w, weights=np.abs(e), minlength=n_wells)
    return out


def build_sufficient_stats(pool_dir, n_wells=19, methods=None, seeds=None,
                           scenarios=None):
    """Read every per-fold dump and return ``S[(method, seed, scenario, log)] -> (W, 5)``.

    Also returns the list of seeds and methods found on disk.
    """
    scenarios = scenarios or SCENARIOS
    if seeds is None:
        seeds = sorted(int(os.path.basename(p)[4:])
                       for p in glob.glob(os.path.join(pool_dir, "seed*")))
    found = set()
    for s in seeds:
        found |= {os.path.basename(d) for d in glob.glob(os.path.join(pool_dir, f"seed{s}", "*"))
                  if os.path.isdir(d)}
    if methods is None:
        methods = [m for m in ORDER if m in found] + sorted(found - set(ORDER))
    S = {}
    for m in methods:
        for s in seeds:
            files = sorted(glob.glob(os.path.join(pool_dir, f"seed{s}", m, "fold*.npz")))
            if not files:
                continue
            acc = defaultdict(lambda: np.zeros((n_wells, 5)))
            for fp in files:
                z = np.load(fp)
                for sc in scenarios:
                    for lg in LOGS3:
                        k = f"{sc}__{lg}"
                        if f"{k}__t" not in z.files:
                            continue
                        if f"{k}__w" not in z.files or z[f"{k}__w"].size != z[f"{k}__t"].size:
                            raise SystemExit(f"{fp}: no well label for {k}; re-run with --dump.")
                        acc[(sc, lg)] += _suff(z[f"{k}__p"], z[f"{k}__t"], z[f"{k}__w"], n_wells)
            for (sc, lg), arr in acc.items():
                S[(m, s, sc, lg)] = arr
    return S, seeds, methods


# ----------------------------------------------------------------------------------------
# 2. metrics from sufficient statistics
# ----------------------------------------------------------------------------------------
def pooled_r2(stats, weights=None):
    """Pooled R^2 from a (W, 5) array; ``weights`` = multiplicity of each well (bootstrap)."""
    a = stats if weights is None else stats * weights[:, None]
    n, sy, syy, sse = a[:, F_N].sum(), a[:, F_SY].sum(), a[:, F_SYY].sum(), a[:, F_SSE].sum()
    if n <= 1:
        return np.nan
    sst = syy - sy * sy / n
    return float(1.0 - sse / sst) if sst > 0 else np.nan


def pooled_mae_rmse(stats):
    n = stats[:, F_N].sum()
    if n == 0:
        return np.nan, np.nan
    return float(stats[:, F_SAE].sum() / n), float(np.sqrt(stats[:, F_SSE].sum() / n))


def macro_r2(S, m, s, sc, weights=None):
    """Equal-weight mean of the three per-log pooled R^2 (primary metric)."""
    vals = [pooled_r2(S[(m, s, sc, lg)], weights) for lg in LOGS3 if (m, s, sc, lg) in S]
    return float(np.mean(vals)) if len(vals) == 3 else np.nan


def nrmse_per_well(S, m, s, sc, sigma):
    """Per-well standardised RMSE, shape (W,).  ``sigma[lg]`` is (W,) training std per fold.

    nRMSE_w = sqrt( (1/3) sum_c SSE_{c,w} / (n_{c,w} sigma_{c,w}^2) ).
    A log with no masked point in a well (n = 0, e.g. ``single`` masks one log per window)
    is skipped for that well; the mean runs over the logs present.
    """
    terms = []
    for lg in LOGS3:
        a = S[(m, s, sc, lg)]
        with np.errstate(invalid="ignore", divide="ignore"):
            terms.append(np.where(a[:, F_N] > 0, a[:, F_SSE] / (a[:, F_N] * sigma[lg] ** 2), np.nan))
    T = np.vstack(terms)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(T, axis=0))


def training_sigmas(data_dir=None):
    """Exact fold-local training std per log, ``{log: (W,)}``, from the windowed dataset.

    Matches :class:`~wellog_imputation.evaluation.harness.ChannelScaler` fitted on the
    training wells of each leave-one-well-out fold (same fold order as the harness).
    """
    from ..evaluation.harness import ChannelScaler, load
    X, wells, F = load(data_dir)
    keys = list(F["folds"])
    sig = {lg: np.full(len(keys), np.nan) for lg in LOGS3}
    for wi, fk in enumerate(keys):
        test_w = F["folds"][fk]["test"][0]
        sc_ = ChannelScaler().fit(X[wells != test_w])
        for c, lg in enumerate(LOGS3):
            sig[lg][wi] = sc_.sd[c]
    return sig


def fallback_sigmas(S, m, s, sc):
    """Approximate training std when the dataset is not at hand: std of the ground truth
    at the masked positions of all OTHER wells.  Flagged in the outputs."""
    sig = {}
    for lg in LOGS3:
        a = S[(m, s, sc, lg)]
        tot = a.sum(0)
        out = np.empty(a.shape[0])
        for w in range(a.shape[0]):
            r = tot - a[w]
            n = r[F_N]
            out[w] = np.sqrt(max(r[F_SYY] / n - (r[F_SY] / n) ** 2, 0.0)) if n > 1 else np.nan
        sig[lg] = out
    return sig


# ----------------------------------------------------------------------------------------
# 3. tests
# ----------------------------------------------------------------------------------------
def holm(pvals, alpha=0.05):
    """Holm step-down.  Returns (adjusted p-values, reject mask), NaNs left untouched."""
    p = np.asarray(pvals, float)
    adj = np.full_like(p, np.nan); rej = np.zeros(p.shape, bool)
    ok = np.where(np.isfinite(p))[0]
    m = ok.size
    if m == 0:
        return adj, rej
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    stop = False
    for i, idx in enumerate(order):
        if not stop and p[idx] <= alpha / (m - i):
            rej[idx] = True
        else:
            stop = True
    return adj, rej


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation r = (T+ - T-) / (T+ + T-), zeros dropped."""
    from scipy.stats import rankdata
    d = np.asarray(d, float); d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return np.nan
    r = rankdata(np.abs(d))
    tp, tm = r[d > 0].sum(), r[d < 0].sum()
    return float((tp - tm) / (tp + tm))


def kendall_w(chi2, n_blocks, k):
    """Kendall's W from the Friedman statistic: W = chi2_F / (N (k - 1))."""
    return float(chi2 / (n_blocks * (k - 1)))


def friedman_wellwise(E):
    """Friedman across wells.  ``E`` (W, k): one error value per (well, method), lower =
    better.  Wells with any NaN are dropped (reported in ``n_wells``)."""
    from scipy.stats import friedmanchisquare
    keep = np.all(np.isfinite(E), axis=1)
    X = E[keep]
    if X.shape[0] < 3 or X.shape[1] < 3:
        return dict(chi2=np.nan, p=np.nan, W=np.nan, n_wells=int(X.shape[0]), k=E.shape[1])
    chi2, p = friedmanchisquare(*X.T)
    return dict(chi2=float(chi2), p=float(p), W=kendall_w(chi2, X.shape[0], X.shape[1]),
                n_wells=int(X.shape[0]), k=int(X.shape[1]))


def wilcoxon_wellwise(ea, eb):
    """Paired two-sided Wilcoxon across wells (exact when n is small and no ties)."""
    from scipy.stats import wilcoxon
    keep = np.isfinite(ea) & np.isfinite(eb)
    d = ea[keep] - eb[keep]
    if d.size < 5 or np.all(d == 0):
        return dict(n=int(d.size), stat=np.nan, p=np.nan, r_rb=np.nan,
                    median_diff=float(np.median(d)) if d.size else np.nan)
    res = wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="auto")
    return dict(n=int(d.size), stat=float(res.statistic), p=float(res.pvalue),
                r_rb=rank_biserial(d), median_diff=float(np.median(d)))


def boot_delta_macro_r2(S, a, b, sc, seeds, n_wells, B=10000, rng=None):
    """By-well cluster bootstrap of Delta R2_macro = R2_macro(a) - R2_macro(b).

    The SAME well draw is applied to both methods and to every seed; Delta is averaged
    over seeds within each replicate.  Returns (point estimate, 2.5 %, 97.5 %, p_boot).
    """
    rng = rng or np.random.default_rng(0)
    seeds = [s for s in seeds if all((m, s, sc, lg) in S for m in (a, b) for lg in LOGS3)]
    if not seeds:
        return (np.nan,) * 4
    point = np.mean([macro_r2(S, a, s, sc) - macro_r2(S, b, s, sc) for s in seeds])
    present = np.where(S[(a, seeds[0], sc, LOGS3[0])][:, F_N] +
                       S[(a, seeds[0], sc, LOGS3[1])][:, F_N] +
                       S[(a, seeds[0], sc, LOGS3[2])][:, F_N] > 0)[0]
    deltas = np.empty(B)
    for i in range(B):
        draw = rng.choice(present, size=present.size, replace=True)
        wts = np.bincount(draw, minlength=n_wells).astype(float)
        deltas[i] = np.mean([macro_r2(S, a, s, sc, wts) - macro_r2(S, b, s, sc, wts)
                             for s in seeds])
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    p_boot = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return float(point), float(lo), float(hi), float(min(1.0, p_boot))


# ----------------------------------------------------------------------------------------
# 4. driver
# ----------------------------------------------------------------------------------------
def _w(path, header, rows):
    with open(path, "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(header); cw.writerows(rows)


def _fmt(x, nd=4):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def run_revision(pool_dir, out_dir, n_wells=19, data_dir=None, pairs=None, B=10000,
                 alpha=0.05, release_suff=False):
    """Compute every revision table.  ``pairs``: {scenario: [(a, b), ...]} = the declared
    test family; default = the pairs pre-registered in :mod:`.significance`."""
    os.makedirs(out_dir, exist_ok=True)
    S, seeds, methods = build_sufficient_stats(pool_dir, n_wells=n_wells)
    if not S:
        raise SystemExit(f"no per-fold dumps under {pool_dir}")

    # sigma per fold: exact if the dataset is available, otherwise flagged fallback
    sigma_source = "training_fold"
    try:
        sig_exact = training_sigmas(data_dir)
        if any(v.size != n_wells for v in sig_exact.values()):
            raise ValueError("fold count mismatch")
    except Exception as e:                                      # noqa: BLE001
        sig_exact, sigma_source = None, f"fallback_masked_other_wells ({type(e).__name__})"

    # ---- (a) per-log table + macro ------------------------------------------------------
    rows_log, rows_main = [], []
    macro = {}                      # (m, sc) -> list over seeds
    for m in methods:
        for sc in SCENARIOS:
            for lg in LOGS3:
                r2s, maes, rmses = [], [], []
                for s in seeds:
                    if (m, s, sc, lg) in S:
                        a = S[(m, s, sc, lg)]
                        r2s.append(pooled_r2(a)); mae, rmse = pooled_mae_rmse(a)
                        maes.append(mae); rmses.append(rmse)
                if r2s:
                    rows_log.append([m, sc, lg, len(r2s),
                                     _fmt(np.mean(r2s)), _fmt(np.std(r2s, ddof=1) if len(r2s) > 1 else 0.0),
                                     _fmt(np.mean(maes)), _fmt(np.mean(rmses))])
            macro[(m, sc)] = [macro_r2(S, m, s, sc) for s in seeds if (m, s, sc, "GR") in S]
    _w(os.path.join(out_dir, "table_perlog_revision.csv"),
       ["method", "scenario", "log", "n_seeds", "r2_mean", "r2_sd", "mae_mean", "rmse_mean"],
       rows_log)

    # ---- (b) per-well nRMSE (seed-averaged) -------------------------------------------
    E = {}                          # (m, sc) -> (W,) seed-averaged nRMSE
    rows_well = []
    for m in methods:
        for sc in SCENARIOS:
            per_seed = []
            for s in seeds:
                if not all((m, s, sc, lg) in S for lg in LOGS3):
                    continue
                sig = sig_exact if sig_exact is not None else fallback_sigmas(S, m, s, sc)
                per_seed.append(nrmse_per_well(S, m, s, sc, sig))
            if per_seed:
                with np.errstate(invalid="ignore"):
                    E[(m, sc)] = np.nanmean(np.vstack(per_seed), axis=0)
                for w, v in enumerate(E[(m, sc)]):
                    rows_well.append([m, sc, w, _fmt(v)])
    _w(os.path.join(out_dir, "per_well_nrmse.csv"), ["method", "scenario", "well_index", "nrmse"],
       rows_well)

    # ---- (c) main table: macro R2 (pooled) + well-equal nRMSE median/IQR + ranks ----------
    for sc in SCENARIOS:
        ms = [m for m in methods if macro.get((m, sc))]
        means = {m: np.mean(macro[(m, sc)]) for m in ms}
        rank = {m: i + 1 for i, m in enumerate(sorted(ms, key=lambda x: -means[x]))}
        for m in ms:
            v = macro[(m, sc)]
            e = E.get((m, sc), np.array([np.nan]))
            q1, med, q3 = np.nanpercentile(e, [25, 50, 75]) if np.isfinite(e).any() else (np.nan,) * 3
            rows_main.append([m, sc, len(v), _fmt(np.mean(v)),
                              _fmt(np.std(v, ddof=1) if len(v) > 1 else 0.0), rank[m],
                              _fmt(med), _fmt(q1), _fmt(q3), _fmt(np.nanmean(e))])
    _w(os.path.join(out_dir, "table_main_revision.csv"),
       ["method", "scenario", "n_seeds", "r2_macro_mean", "r2_macro_sd", "rank_r2_macro",
        "nrmse_well_median", "nrmse_well_q1", "nrmse_well_q3", "nrmse_well_mean"], rows_main)

    # ---- (d) Friedman + Kendall W across wells ------------------------------------------
    rows_f = []
    for sc in SCENARIOS:
        ms = [m for m in methods if (m, sc) in E]
        if len(ms) < 3:
            continue
        res = friedman_wellwise(np.column_stack([E[(m, sc)] for m in ms]))
        rows_f.append([sc, res["k"], res["n_wells"], _fmt(res["chi2"]), f"{res['p']:.3e}",
                       _fmt(res["W"])])
    _w(os.path.join(out_dir, "friedman_wellwise.csv"),
       ["scenario", "k_methods", "n_wells", "chi2_F", "p", "kendall_W"], rows_f)

    # ---- (e) Wilcoxon + rank-biserial + Holm + bootstrap CI ------------------------------
    if pairs is None:
        from .significance import WILCOXON_PAIRS
        pairs = WILCOXON_PAIRS
    rows_wx = []
    rng = np.random.default_rng(20261016)
    for sc, plist in pairs.items():
        plist = [(a, b) for a, b in plist if (a, sc) in E and (b, sc) in E]
        res = [wilcoxon_wellwise(E[(a, sc)], E[(b, sc)]) for a, b in plist]
        adj, rej = holm([r["p"] for r in res], alpha=alpha)
        for (a, b), r, pa, rj in zip(plist, res, adj, rej):
            d, lo, hi, pb = boot_delta_macro_r2(S, a, b, sc, seeds, n_wells, B=B, rng=rng)
            rows_wx.append([sc, len(plist), a, b, r["n"], _fmt(r["median_diff"]),
                            _fmt(r["stat"]), f"{r['p']:.4g}" if np.isfinite(r["p"]) else "",
                            f"{pa:.4g}" if np.isfinite(pa) else "", bool(rj), _fmt(r["r_rb"], 3),
                            _fmt(d), _fmt(lo), _fmt(hi), _fmt(pb, 4)])
    _w(os.path.join(out_dir, "wilcoxon_wellwise_holm.csv"),
       ["scenario", "family_size_m", "A", "B", "n_wells", "median_diff_nrmse_A_minus_B",
        "W_stat", "p_exact", "p_holm", "reject_holm_0.05", "rank_biserial",
        "delta_r2_macro_A_minus_B", "boot_ci_lo", "boot_ci_hi", "p_boot"], rows_wx)

    # ---- (f) all-pairs, as a supplementary family per scenario ----------------------------
    rows_all = []
    for sc in SCENARIOS:
        ms = [m for m in methods if (m, sc) in E]
        plist = list(itertools.combinations(ms, 2))
        res = [wilcoxon_wellwise(E[(a, sc)], E[(b, sc)]) for a, b in plist]
        adj, rej = holm([r["p"] for r in res], alpha=alpha)
        for (a, b), r, pa, rj in zip(plist, res, adj, rej):
            rows_all.append([sc, len(plist), a, b, _fmt(r["median_diff"]),
                             f"{r['p']:.4g}" if np.isfinite(r["p"]) else "",
                             f"{pa:.4g}" if np.isfinite(pa) else "", bool(rj), _fmt(r["r_rb"], 3)])
    _w(os.path.join(out_dir, "wilcoxon_allpairs_holm.csv"),
       ["scenario", "family_size_m", "A", "B", "median_diff_nrmse_A_minus_B", "p_exact",
        "p_holm", "reject_holm_0.05", "rank_biserial"], rows_all)

    # ---- (g) sufficient statistics (optional, for release after data-owner approval) ----
    if release_suff:
        rows_s = []
        for (m, s, sc, lg), a in sorted(S.items()):
            for w in range(a.shape[0]):
                rows_s.append([m, s, sc, lg, w] + [_fmt(x, 8) for x in a[w]])
        _w(os.path.join(out_dir, "sufficient_stats_per_well.csv"),
           ["method", "seed", "scenario", "log", "well_index", "n", "sum_y", "sum_y2", "sse",
            "sae"], rows_s)

    with open(os.path.join(out_dir, "REVISION_README.txt"), "w") as f:
        f.write(f"seeds={seeds}\nmethods={methods}\nsigma_source={sigma_source}\n"
                f"bootstrap_B={B}\nalpha={alpha}\nfamily=one per scenario (Holm)\n"
                f"statistical unit=well (values averaged over seeds before testing)\n")
    return dict(seeds=seeds, methods=methods, sigma_source=sigma_source)
