#!/usr/bin/env python
"""LaTeX tables for the revised manuscript, from the CSVs of reanalyse_revision.py.

Writes into <revision_dir>/latex/:
  tab_main.tex      R2_macro mean +/- sd per method x scenario (nominal best in bold)
  tab_ranks.tex     rank by R2_macro
  tab_perlog.tex    per-log R2 / MAE / RMSE (one block per scenario)
  tab_perwell.tex   per-well nRMSE median [IQR]
  tab_friedman.tex  Friedman across wells + Kendall W
  tab_wilcoxon.tex  declared pairs: median diff, p, p_Holm, r_rb, Delta R2_macro [95% CI]
Usage: python scripts/make_revision_tables.py <revision_dir>
"""
import csv
import os
import sys
from collections import defaultdict

ORDER = ["mean", "locf", "rf", "xgb", "mice", "saits", "brits", "unet", "ae", "gnn", "stgnn"]
DISPLAY = {"mean": "MEAN", "locf": "LOCF", "rf": "RF", "xgb": "XGBoost", "mice": "MICE",
           "saits": "SAITS", "brits": "BRITS", "unet": "U-Net", "ae": "AE", "gnn": "GNN",
           "stgnn": "ST-GNN"}
FAMILY = {"mean": "tabular", "rf": "tabular", "xgb": "tabular", "mice": "tabular",
          "locf": "sequential", "saits": "sequential", "brits": "sequential",
          "unet": "sequential", "ae": "sequential", "gnn": "spatial", "stgnn": "spatial"}
SC = ["single", "block", "profile", "blackout"]
SC_ALL = SC + ["blackout_emp"]
DISPLAY.update({"locf_win": "LOCF (window)", "ridge": "Ridge", "knn": "kNN",
                "interp_linear": "Linear interp.", "interp_pchip": "PCHIP interp.",
                "xgb_miss": "XGBoost (miss.-aware)", "xgb_chain_asc": "XGBoost chain (asc.)",
                "xgb_chain_desc": "XGBoost chain (desc.)", "saits_mix": "SAITS (curriculum)",
                "brits_mix": "BRITS (curriculum)", "idw": "IDW", "nnwell": "Nearest well",
                "okrige": "Ordinary kriging", "gnn_noneigh": "GNN (no neighbours)",
                "stgnn_noneigh": "ST-GNN (no neighbours)", "gnn_shuffle": "GNN (shuffled)",
                "stgnn_shuffle": "ST-GNN (shuffled)"})
FAMILY.update({"locf_win": "sequential", "ridge": "tabular", "knn": "tabular",
               "interp_linear": "sequential", "interp_pchip": "sequential",
               "xgb_miss": "tabular", "xgb_chain_asc": "tabular", "xgb_chain_desc": "tabular",
               "saits_mix": "sequential", "brits_mix": "sequential", "idw": "spatial",
               "nnwell": "spatial", "okrige": "spatial", "gnn_noneigh": "spatial",
               "stgnn_noneigh": "spatial", "gnn_shuffle": "spatial", "stgnn_shuffle": "spatial"})
LOGS = ["GR", "RHOB", "NPHI"]
UNITS = {"GR": "gAPI", "RHOB": "g\\,cm$^{-3}$", "NPHI": "frac."}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("-inf")


def rd(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def f(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    s = f"{v:.{nd}f}"
    return s.replace("-", "$-$", 1) if v < 0 else s


def p_fmt(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if v < 1e-3:
        e = f"{v:.1e}"; m, ex = e.split("e")
        return f"${m}\\times10^{{{int(ex)}}}$"
    return f"{v:.3f}"


def methods_in(rows):
    present = {r["method"] for r in rows}
    return [m for m in ORDER if m in present] + sorted(present - set(ORDER))


def tab_main(rows, out, subset="main"):
    by = {(r["method"], r["scenario"]): r for r in rows}
    allm = methods_in(rows)
    ms = [m for m in allm if (m in ORDER) == (subset == "main")]
    if not ms:
        return
    sc_list = [sc for sc in SC_ALL if any((m, sc) in by for m in ms)]
    suffix = "" if subset == "main" else "_revision"
    best = {sc: max((m for m in ms if (m, sc) in by), key=lambda m: _num(by[(m, sc)]["r2_macro_mean"]))
            for sc in sc_list if any((m, sc) in by for m in ms)}
    hdr = " & ".join(f"\\texttt{{{sc.replace('_', chr(92) + '_')}}}" for sc in sc_list)
    L = [r"\begin{tabular}{@{}ll" + "c" * len(sc_list) + r"@{}}", r"\toprule",
         f"Family & Method & {hdr} \\\\", r"\midrule"]
    for m in ms:
        cells = []
        for sc in sc_list:
            r = by.get((m, sc))
            if r is None:
                cells.append("--"); continue
            c = f"{f(r['r2_macro_mean'])} $\\pm$ {f(r['r2_macro_sd'])}"
            cells.append(f"\\textbf{{{c}}}" if best.get(sc) == m else c)
        L.append(f"{FAMILY.get(m, '')} & {DISPLAY.get(m, m)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, f"tab_main{suffix}.tex"), "w").write("\n".join(L) + "\n")

    L = [r"\begin{tabular}{@{}l" + "c" * len(sc_list) + r"@{}}", r"\toprule",
         f"Method & {hdr} \\\\", r"\midrule"]
    for m in ms:
        L.append(DISPLAY.get(m, m) + " & " + " & ".join(
            by[(m, sc)]["rank_r2_macro"] if (m, sc) in by else "--" for sc in sc_list) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, f"tab_ranks{suffix}.tex"), "w").write("\n".join(L) + "\n")

    L = [r"\begin{tabular}{@{}l" + "c" * len(sc_list) + r"@{}}", r"\toprule",
         f"Method & {hdr} \\\\", r"\midrule"]
    for m in ms:
        cells = []
        for sc in sc_list:
            r = by.get((m, sc))
            cells.append("--" if r is None else
                         f"{f(r['nrmse_well_median'], 2)} [{f(r['nrmse_well_q1'], 2)}, {f(r['nrmse_well_q3'], 2)}]")
        L.append(DISPLAY.get(m, m) + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, f"tab_perwell{suffix}.tex"), "w").write("\n".join(L) + "\n")


def tab_perlog(rows, out):
    by = {(r["method"], r["scenario"], r["log"]): r for r in rows}
    ms = methods_in(rows)
    hdr = " & ".join(f"\\multicolumn{{3}}{{c}}{{{lg} ({UNITS[lg]})}}" for lg in LOGS)
    L = [r"\begin{tabular}{@{}l" + "ccc" * 3 + r"@{}}", r"\toprule",
         f"Method & {hdr} \\\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         " & " + " & ".join(["$R^2$ & MAE & RMSE"] * 3) + r" \\", r"\midrule"]
    for sc in [x for x in SC_ALL if any(k[1] == x for k in by)]:
        L.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{\\texttt{{{sc}}}}}}} \\\\")
        for m in ms:
            cells = []
            for lg in LOGS:
                r = by.get((m, sc, lg))
                nd = 1 if lg == "GR" else 3
                cells += (["--"] * 3 if r is None else
                          [f(r["r2_mean"]), f(r["mae_mean"], nd), f(r["rmse_mean"], nd)])
            L.append(DISPLAY.get(m, m) + " & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tab_perlog.tex"), "w").write("\n".join(L) + "\n")


def tab_tests(fr, wx, out):
    L = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Scenario & $k$ & $n$ wells & $\chi^2_F$ & $p$ & Kendall's $W$ \\", r"\midrule"]
    for r in fr:
        L.append(f"\\texttt{{{r['scenario']}}} & {r['k_methods']} & {r['n_wells']} & "
                 f"{f(r['chi2_F'], 1)} & {p_fmt(r['p'])} & {f(r['kendall_W'], 2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tab_friedman.tex"), "w").write("\n".join(L) + "\n")

    L = [r"\begin{tabular}{@{}llrrrrl@{}}", r"\toprule",
         r"Scenario & Pair (A vs B) & med.\ $\Delta$nRMSE & $p$ & $p_{\mathrm{Holm}}$ & $r_{rb}$ & $\Delta\bar R^2$ [95\% CI] \\",
         r"\midrule"]
    for r in wx:
        sig = r["reject_holm_0.05"] in ("True", "true", "1")
        ph = p_fmt(r["p_holm"]) + ("$^{*}$" if sig else "")
        L.append(f"\\texttt{{{r['scenario']}}} & {DISPLAY.get(r['A'], r['A'])} vs {DISPLAY.get(r['B'], r['B'])} & "
                 f"{f(r['median_diff_nrmse_A_minus_B'])} & {p_fmt(r['p_exact'])} & {ph} & {f(r['rank_biserial'], 2)} & "
                 f"{f(r['delta_r2_macro_A_minus_B'])} [{f(r['boot_ci_lo'])}, {f(r['boot_ci_hi'])}] \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tab_wilcoxon.tex"), "w").write("\n".join(L) + "\n")


def main(rev):
    out = os.path.join(rev, "latex"); os.makedirs(out, exist_ok=True)
    tab_main(rd(os.path.join(rev, "table_main_revision.csv")), out, "main")
    tab_main(rd(os.path.join(rev, "table_main_revision.csv")), out, "revision")
    tab_perlog(rd(os.path.join(rev, "table_perlog_revision.csv")), out)
    tab_tests(rd(os.path.join(rev, "friedman_wellwise.csv")),
              rd(os.path.join(rev, "wilcoxon_wellwise_holm.csv")), out)
    rv = os.path.join(rev, "wilcoxon_revision_holm.csv")
    if os.path.exists(rv):
        tab_tests(rd(os.path.join(rev, "friedman_wellwise.csv")), rd(rv), out)
        os.replace(os.path.join(out, "tab_wilcoxon.tex"), os.path.join(out, "tab_wilcoxon_revision.tex"))
        tab_tests(rd(os.path.join(rev, "friedman_wellwise.csv")),
                  rd(os.path.join(rev, "wilcoxon_wellwise_holm.csv")), out)
    print("->", out, sorted(os.listdir(out)))


if __name__ == "__main__":
    main(sys.argv[1])
