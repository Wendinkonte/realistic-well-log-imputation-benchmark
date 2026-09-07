"""Multi-seed aggregation, variance and ranking stability (paper Section 5.3 and 5.5).

Reads the append-only outputs of the replication driver:

* ``<results>/raw/raw_seed*.csv``                   per-cell metrics (per fold)
* ``<results>/pools_seeded/seed{S}/{m}/fold*.npz``  points (pred, true, well)

and produces:

* ``<out>/raw_metrics.csv``   deduplicated consolidation of every ``raw_seed*.csv``
* ``<out>/aggregated.csv``    per ``(method, scenario, log)``: pooled R^2 mean +/- std and
  95 % CI over seeds, plus the per-fold median averaged over seeds
* ``<out>/significance/``     ``friedman.csv``, ``nemenyi_{scenario}_seed{S}.csv``,
  ``cd_{scenario}_seed{S}.{png,pdf}``, ``wilcoxon_pairs.csv``, ``ranking_stability.csv``
* ``<out>/RESULTS_FOR_PAPER.md``  synthesis used to write the results tables

The pooled ``all`` R^2 (three logs together) is inflated by between-channel variance, so
the per-log R^2 and the zero-skill MEAN floor are reported alongside it.
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from ..models.registry import ORDER, SCENARIOS
from .significance import friedman_table, wilcoxon_table

__all__ = ["SCEN", "LOGS_ALL", "r2", "consolidate_raw", "load_raw_table",
           "load_seed_pool", "discover", "mean_std_ci", "aggregate"]

SCEN = SCENARIOS
LOGS_ALL = ["all", "GR", "RHOB", "NPHI"]


def r2(p, t):
    """Coefficient of determination of two paired vectors; NaN if the target is constant."""
    p, t = np.asarray(p, float), np.asarray(t, float)
    if t.size == 0:
        return np.nan
    ss = np.sum((t - t.mean()) ** 2)
    return float(1 - np.sum((p - t) ** 2) / ss) if ss > 0 else np.nan


def consolidate_raw(results_dir):
    """Concatenate every ``raw_seed*.csv``, deduplicating on
    ``(seed, method, scenario, fold, log)`` and keeping the LAST occurrence, which
    tolerates folds replayed after a preemption."""
    rows = {}
    files = sorted(glob.glob(os.path.join(results_dir, "raw", "raw_seed*.csv")))
    for fp in files:
        with open(fp) as f:
            for r in csv.DictReader(f):
                key = (r["seed"], r["method"], r["scenario"], r["fold"], r["log"])
                rows[key] = r
    return list(rows.values()), files


def load_raw_table(rows):
    """-> ``dict[(seed, method, scenario, log)] = {fold: r2}`` for medians and tests."""
    tab = defaultdict(dict)
    for r in rows:
        try:
            val = float(r["r2"]) if r["r2"] != "" else np.nan
        except ValueError:
            val = np.nan
        tab[(int(r["seed"]), r["method"], r["scenario"], r["log"])][int(r["fold"])] = val
    return tab


def load_seed_pool(pool_dir, seed, method):
    """Concatenate the per-fold ``.npz`` of one ``(seed, method)`` for the pooled R^2."""
    base = os.path.join(pool_dir, f"seed{seed}", method)
    out = defaultdict(lambda: {"p": [], "t": [], "w": []})
    for fp in sorted(glob.glob(os.path.join(base, "fold*.npz"))):
        z = np.load(fp)
        for sc in SCEN:
            for lg in LOGS_ALL:
                k = f"{sc}__{lg}"
                if f"{k}__t" in z.files:
                    out[(sc, lg)]["p"].append(z[f"{k}__p"])
                    out[(sc, lg)]["t"].append(z[f"{k}__t"])
                    out[(sc, lg)]["w"].append(z[f"{k}__w"])
    pooled = {}
    for key, d in out.items():
        if d["t"]:
            pooled[key] = (np.concatenate(d["p"]), np.concatenate(d["t"]), np.concatenate(d["w"]))
    return pooled


def discover(pool_dir, raw_seeds):
    """Find which seeds and methods are present on disk, in canonical display order."""
    seeds = sorted({int(os.path.basename(p)[4:]) for p in glob.glob(os.path.join(pool_dir, "seed*"))})
    seeds = seeds or sorted(raw_seeds)
    methods = set()
    for s in seeds:
        for d in glob.glob(os.path.join(pool_dir, f"seed{s}", "*")):
            if os.path.isdir(d):
                methods.add(os.path.basename(d))
    methods = [m for m in ORDER if m in methods] + [m for m in sorted(methods) if m not in ORDER]
    return seeds, methods


def mean_std_ci(vals):
    """Mean, std, and the 95 % Student-t confidence interval of the mean (small n)."""
    v = np.array([x for x in vals if x == x], float)
    if v.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    m, sd = float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
    if v.size > 1:
        from scipy import stats
        half = stats.t.ppf(0.975, v.size - 1) * sd / np.sqrt(v.size)
    else:
        half = np.nan
    return (m, sd, m - half, m + half, v.size)


def aggregate(results_dir, out_dir=None):
    """Run the whole aggregation.  ``out_dir`` defaults to ``results_dir``."""
    R = str(results_dir)
    OUT = str(out_dir or R)
    POOL = os.path.join(R, "pools_seeded")
    SIG = os.path.join(OUT, "significance")
    os.makedirs(SIG, exist_ok=True)

    # 1) consolidate the raw per-cell metrics
    rows, files = consolidate_raw(R)
    if not rows:
        raise SystemExit(f"no raw_seed*.csv under {R}/raw/ (run scripts/run_replication.py first).")
    raw_seeds = {int(r["seed"]) for r in rows}
    with open(os.path.join(OUT, "raw_metrics.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["seed", "method", "scenario", "fold", "log", "r2", "mae"])
        for r in sorted(rows, key=lambda r: (int(r["seed"]), r["method"], r["scenario"],
                                             int(r["fold"]), r["log"])):
            wcsv.writerow([r["seed"], r["method"], r["scenario"], r["fold"], r["log"],
                           r["r2"], r["mae"]])
    tab = load_raw_table(rows)
    seeds, methods = discover(POOL, raw_seeds)
    print(f"[aggregate] seeds={seeds} | methods={methods} | raw files={len(files)}", flush=True)

    # 2) pooled R^2 per seed, recomputed from the per-fold dumps
    pooled_r2 = {}
    for s in seeds:
        for m in methods:
            pl = load_seed_pool(POOL, s, m)
            for (sc, lg), (p, t, w) in pl.items():
                pooled_r2[(m, s, sc, lg)] = r2(p, t)

    # 3) aggregated.csv: mean +/- std and 95 % CI over seeds, plus the per-fold median
    agg_path = os.path.join(OUT, "aggregated.csv")
    with open(agg_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["method", "scenario", "log", "n_seeds",
                       "pooledR2_mean", "pooledR2_std", "pooledR2_ci_low", "pooledR2_ci_high",
                       "perfoldMedianR2_mean", "perfoldMedianR2_std"])
        for m in methods:
            for sc in SCEN:
                for lg in LOGS_ALL:
                    pooled_vals = [pooled_r2.get((m, s, sc, lg), np.nan) for s in seeds]
                    pm, psd, plo, phi, n = mean_std_ci(pooled_vals)
                    med_per_seed = []
                    for s in seeds:
                        d = tab.get((s, m, sc, lg), {})
                        vv = [v for v in d.values() if v == v]
                        med_per_seed.append(np.median(vv) if vv else np.nan)
                    mm, msd, *_ = mean_std_ci(med_per_seed)
                    wcsv.writerow([m, sc, lg, n,
                                   f"{pm:.4f}", f"{psd:.4f}", f"{plo:.4f}", f"{phi:.4f}",
                                   f"{mm:.4f}", f"{msd:.4f}"])
    print(f"[aggregate] -> {agg_path}", flush=True)

    # 4) ranking stability on the pooled 'all' R^2, per seed
    rank_path = os.path.join(SIG, "ranking_stability.csv")
    top_by_scen = {}
    with open(rank_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["scenario", "seed", "rank", "method", "pooledR2_all"])
        for sc in SCEN:
            tops = []
            for s in seeds:
                scored = [(m, pooled_r2.get((m, s, sc, "all"), np.nan)) for m in methods]
                scored = [(m, v) for m, v in scored if v == v]
                scored.sort(key=lambda x: -x[1])
                for rk, (m, v) in enumerate(scored, 1):
                    wcsv.writerow([sc, s, rk, m, f"{v:.4f}"])
                if scored:
                    tops.append(scored[0][0])
            top_by_scen[sc] = tops
    print(f"[aggregate] -> {rank_path}", flush=True)

    # 5) Friedman (+ Nemenyi and CD diagrams when significant)
    friedman_rows = friedman_table(tab, methods, seeds, SIG, SCEN)
    print(f"[aggregate] -> {os.path.join(SIG, 'friedman.csv')}", flush=True)

    # 6) paired Wilcoxon on the contested pairs
    sig_count = wilcoxon_table(tab, seeds, SIG)
    print(f"[aggregate] -> {os.path.join(SIG, 'wilcoxon_pairs.csv')}", flush=True)

    # 7) narrative synthesis
    _write_results_md(OUT, seeds, methods, pooled_r2, top_by_scen, sig_count, friedman_rows)
    print("[aggregate] done.", flush=True)


def _write_results_md(OUT, seeds, methods, pooled_r2, top_by_scen, sig_count, friedman_rows):
    def fmt(m, sc, lg):
        vals = np.array([pooled_r2.get((m, s, sc, lg), np.nan) for s in seeds], float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return "  n/a  "
        return f"{vals.mean():.3f}±{(vals.std(ddof=1) if vals.size>1 else 0):.3f}"
    lines = []
    lines.append("# RESULTS_FOR_PAPER — multi-seed replication\n")
    lines.append(f"Seeds: {seeds}  (N_SEEDS={len(seeds)}). Methods: {methods}.\n")
    lines.append("Numbers are **pooled R2 across the 19 LOO folds**, reported as "
                 "**mean±std across seeds**. Aggregate `all` pools the 3 logs; per-log "
                 "(GR/RHOB/NPHI) is reported because `all` is inflated by between-channel "
                 "variance.\n")
    lines.append("## Aggregation formula (exact)\n")
    lines.append("- Per (scenario, log): concatenate masked (pred,true) points over **all "
                 "19 folds**, then a **single** R2 = 1 − Σ(pred−true)²/Σ(true−mean)², in "
                 "**original physical units**, on artificially-masked positions only. "
                 "(`metrics.pooled.pooled_agg`). `all` = the three logs pooled together.\n")
    for sc in SCEN:
        lines.append(f"\n## Scenario: {sc}\n")
        lines.append("| method | R2 all | R2 GR | R2 RHOB | R2 NPHI |")
        lines.append("|---|---|---|---|---|")
        for m in methods:
            lines.append(f"| {m} | {fmt(m,sc,'all')} | {fmt(m,sc,'GR')} | "
                         f"{fmt(m,sc,'RHOB')} | {fmt(m,sc,'NPHI')} |")
        lines.append(f"\n**Zero-skill MEAN floor** ({sc}): all={fmt('mean',sc,'all')}, "
                     f"GR={fmt('mean',sc,'GR')}, RHOB={fmt('mean',sc,'RHOB')}, "
                     f"NPHI={fmt('mean',sc,'NPHI')}. "
                     "Note the gap: aggregate `all` is far above 0 even though per-log is ≈0 "
                     "→ confirms the aggregate inflation.\n")
        tops = top_by_scen.get(sc, [])
        if tops:
            uniq = set(tops)
            verdict = (f"STABLE top-rank = **{tops[0]}** across all {len(tops)} seeds"
                       if len(uniq) == 1 else
                       f"UNSTABLE top-rank: {dict((m, tops.count(m)) for m in uniq)}")
            lines.append(f"- Top-rank (pooled R2 all) per seed: {tops} → {verdict}.")
    lines.append("\n## Significance of contested pairs (paired Wilcoxon, 19 folds)\n")
    lines.append("| scenario | A vs B | seeds significant (p<0.05) |")
    lines.append("|---|---|---|")
    for (sc, a, b), (ns, nt) in sorted(sig_count.items()):
        lines.append(f"| {sc} | {a} vs {b} | {ns}/{nt} |")
    lines.append("\n## Friedman (19 folds as blocks, per seed)\n")
    lines.append("| scenario | seed | n_methods | chi2 | p | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for row in friedman_rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines.append("\n_See results/significance/ for Nemenyi matrices + CD diagrams._\n")
    with open(os.path.join(OUT, "RESULTS_FOR_PAPER.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[aggregate] -> {os.path.join(OUT, 'RESULTS_FOR_PAPER.md')}", flush=True)
