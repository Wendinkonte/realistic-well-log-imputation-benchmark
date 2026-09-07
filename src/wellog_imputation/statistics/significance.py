"""Significance testing (paper Section 5.5).

Three layers, all computed on the PER-FOLD R^2 of the pooled ``all`` channel, with the
19 leave-one-well-out folds as blocks:

* **Friedman** -- omnibus test, per ``(scenario, seed)``;
* **Nemenyi** -- post-hoc pairwise matrix plus a critical-difference diagram, computed
  only when Friedman is significant at 0.05;
* **paired Wilcoxon** -- on the contested pairs listed in :data:`WILCOXON_PAIRS`, with a
  per-pair summary of how many seeds reach significance.

Holm correction is not applied to the Wilcoxon family: the pairs are pre-registered
(they are the ties the article argues about) and each is reported per seed together
with the fraction of seeds reaching p < 0.05, which is the quantity the text uses.
"""
import csv
import os
from collections import defaultdict

import numpy as np

from ..models.registry import LEARNED, SCENARIOS

__all__ = ["WILCOXON_PAIRS", "friedman_table", "nemenyi_and_cd", "wilcoxon_table"]

#: Contested pairs tested by paired Wilcoxon.
#:
#: ``profile``: the three pairs of the top-3 tie, plus XGBoost (nominal 4th) against each
#: member of the podium to decide whether it is statistically separated from it, plus
#: U-Net -- which becomes the nominal leader once eleven methods are compared -- against
#: the top-3 to test whether the tie extends to it.
#: ``blackout``: LOCF against every learned method.
WILCOXON_PAIRS = {
    "profile":  [("stgnn", "gnn"), ("stgnn", "mice"), ("gnn", "mice"),
                 ("xgb", "mice"), ("xgb", "stgnn"), ("xgb", "gnn"),
                 ("unet", "mice"), ("unet", "stgnn"), ("unet", "gnn")],
    "blackout": [("locf", m) for m in LEARNED],
}


def nemenyi_and_cd(M, used, sc, seed, sig_dir):
    """Nemenyi post-hoc matrix (scikit-posthocs) plus a critical-difference diagram.

    Parameters
    ----------
    M : ndarray, shape (n_folds, n_methods)
        Per-fold R^2, methods in the order of ``used``.
    used : list of str
    sc, seed : str, int
    sig_dir : path-like
        Output directory; writes ``nemenyi_{sc}_seed{seed}.csv`` and ``cd_*.{png,pdf}``.

    Failures are recorded in ``NEMENYI_ERRORS.txt`` rather than raised, so one missing
    optional dependency cannot abort a whole aggregation run.
    """
    try:
        import pandas as pd
        import scikit_posthocs as sp
        df = pd.DataFrame(M, columns=used)
        nem = sp.posthoc_nemenyi_friedman(df)
        nem.to_csv(os.path.join(sig_dir, f"nemenyi_{sc}_seed{seed}.csv"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ranks = df.rank(axis=1, ascending=False).mean(axis=0)  # mean rank, 1 = best
        plt.figure(figsize=(8, 2.2))
        sp.critical_difference_diagram(ranks, nem)
        plt.title(f"Nemenyi CD — {sc} (seed {seed})")
        plt.tight_layout()
        plt.savefig(os.path.join(sig_dir, f"cd_{sc}_seed{seed}.png"), dpi=200)
        plt.savefig(os.path.join(sig_dir, f"cd_{sc}_seed{seed}.pdf"))   # vector, for the article
        plt.close()
    except Exception as e:
        with open(os.path.join(sig_dir, "NEMENYI_ERRORS.txt"), "a") as f:
            f.write(f"{sc} seed{seed}: {type(e).__name__}: {e}\n")


def friedman_table(tab, methods, seeds, sig_dir, scenarios=None):
    """Friedman test per ``(scenario, seed)``; triggers Nemenyi when significant.

    Only methods complete over every fold of that cell enter the test.

    Returns
    -------
    list of list
        Rows ``[scenario, seed, n_methods, chi2, p_value, verdict]``.
    """
    from scipy import stats
    scenarios = scenarios or SCENARIOS
    friedman_rows = []
    for sc in scenarios:
        for s in seeds:
            folds = sorted(set().union(*[set(tab.get((s, m, sc, "all"), {})) for m in methods]))
            mat, used = [], []
            for m in methods:
                d = tab.get((s, m, sc, "all"), {})
                col = [d.get(fk, np.nan) for fk in folds]
                if all(c == c for c in col) and len(col) >= 3:
                    mat.append(col); used.append(m)
            if len(used) < 3:
                friedman_rows.append([sc, s, len(used), "", "", "skip(<3 complete methods)"])
                continue
            M = np.array(mat).T  # folds x methods
            chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
            friedman_rows.append([sc, s, len(used), f"{chi2:.3f}", f"{p:.3e}",
                                  "sig" if p < 0.05 else "ns"])
            if p < 0.05:
                nemenyi_and_cd(M, used, sc, s, sig_dir)
    with open(os.path.join(sig_dir, "friedman.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "seed", "n_methods", "chi2", "p_value", "verdict"])
        w.writerows(friedman_rows)
    return friedman_rows


def wilcoxon_table(tab, seeds, sig_dir, pairs=None):
    """Paired Wilcoxon over the 19 folds, for each contested pair and each seed.

    Returns
    -------
    dict
        ``{(scenario, a, b): [n_significant, n_seeds_tested]}``.
    """
    from scipy import stats
    pairs = pairs or WILCOXON_PAIRS
    wx_rows = []
    sig_count = defaultdict(lambda: [0, 0])
    for sc, plist in pairs.items():
        for a, b in plist:
            for s in seeds:
                da = tab.get((s, a, sc, "all"), {}); db = tab.get((s, b, sc, "all"), {})
                common = sorted(set(da) & set(db))
                xa = np.array([da[k] for k in common]); xb = np.array([db[k] for k in common])
                ok = np.isfinite(xa) & np.isfinite(xb)
                xa, xb = xa[ok], xb[ok]
                if xa.size < 5 or np.allclose(xa, xb):
                    wx_rows.append([sc, a, b, s, xa.size, "", "", "skip"]); continue
                try:
                    stat, p = stats.wilcoxon(xa, xb)
                except ValueError:
                    wx_rows.append([sc, a, b, s, xa.size, "", "", "skip"]); continue
                verdict = "sig" if p < 0.05 else "ns"
                wx_rows.append([sc, a, b, s, xa.size, f"{stat:.2f}", f"{p:.3e}",
                                f"{verdict} (median dR2={np.median(xa-xb):+.4f})"])
                sig_count[(sc, a, b)][1] += 1
                sig_count[(sc, a, b)][0] += int(p < 0.05)
    with open(os.path.join(sig_dir, "wilcoxon_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "model_A", "model_B", "seed", "n_folds", "stat", "p_value", "verdict"])
        w.writerows(wx_rows)
        w.writerow([])
        w.writerow(["# summary: fraction of seeds significant (p<0.05) per pair"])
        w.writerow(["scenario", "model_A", "model_B", "n_sig", "n_seeds_tested"])
        for (sc, a, b), (ns, nt) in sorted(sig_count.items()):
            w.writerow([sc, a, b, ns, nt])
    return sig_count
