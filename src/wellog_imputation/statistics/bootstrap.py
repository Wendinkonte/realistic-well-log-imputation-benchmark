"""Cluster bootstrap of the pooled R^2 and paired method comparisons.

Input: ``<dump_dir>/pool_<method>.npz`` written by the benchmark driver with ``--dump``.
Each file carries keys ``"<scenario>__<log>__p"`` / ``"__t"`` / ``"__w"``, aligned
position by position across ALL methods because the evaluation masks are seeded.

Output: ``bootstrap_ci.csv`` (R^2 with a 95 % CI per method and scenario) and
``bootstrap_pairwise.csv`` (paired delta R^2 with a bootstrap p-value).

Method -- BY-WELL CLUSTER BOOTSTRAP, matching the leave-one-well-out protocol:

* points from the same well are strongly autocorrelated, so an i.i.d. bootstrap over
  POINTS badly underestimates the variance (intervals too tight, p-values too
  optimistic).  WELLS are resampled instead, as clusters.
* CI on R^2: each replicate draws the unique wells (from the ``__w`` array) WITH
  replacement, concatenates every point of the drawn wells, and computes the pooled R^2
  on that sample; the interval is the 2.5/97.5 percentiles.
* paired comparison A vs B within one scenario: THE SAME well draw is applied to both
  methods (positions are aligned by construction), and ``delta = R2_A - R2_B`` is formed
  per replicate, giving a 95 % CI and a two-sided bootstrap p-value
  ``2 * min(P(delta<0), P(delta>0))``.  Pairing cancels the shared variance.

The well label ``__w`` is required.  Pools written before it existed make the script
stop cleanly rather than produce falsely tight intervals.
"""
import glob
import os

import numpy as np

from ..models.registry import SCENARIOS

__all__ = ["SCEN", "ORDER", "PAIRS", "r2", "boot_r2", "boot_delta", "load_pools",
           "run_bootstrap"]

SCEN = SCENARIOS

#: Display order for this script (the nine methods present in the single-seed pools).
ORDER = ["mean", "locf", "rf", "xgb", "mice", "saits", "brits", "gnn", "stgnn"]

#: Paired comparisons to test, chosen as the tight gaps and reversals the paper argues.
PAIRS = [
    ("profile",  "stgnn", "mice"),   # 1st vs 2nd -- the key gap
    ("profile",  "stgnn", "gnn"),    # 1st vs 3rd
    ("profile",  "mice",  "gnn"),    # 2nd vs 3rd
    ("profile",  "stgnn", "saits"),  # 1st vs last (the reversal)
    ("profile",  "stgnn", "locf"),   # spatial vs vertical continuity
    ("single",   "brits", "stgnn"),  # best on the easy regime vs spatial mid-pack
    ("blackout", "locf",  "stgnn"),  # LOCF wins the blackout
    ("blackout", "mice",  "xgb"),    # graceful degradation vs tabular collapse
]


def r2(p, t):
    err = p - t
    ss = np.sum((t - t.mean()) ** 2)
    return float(1 - np.sum(err ** 2) / ss) if ss > 0 else np.nan


def _well_groups(w):
    """Point indices grouped by well.  Returns ``(unique_wells, list_of_index_arrays)``."""
    uniq = np.unique(w)
    groups = [np.flatnonzero(w == u) for u in uniq]
    return uniq, groups


def _resample_idx(groups, rng):
    """Draw ``len(groups)`` wells WITH replacement and concatenate their point indices."""
    pick = rng.integers(0, len(groups), len(groups))
    return np.concatenate([groups[j] for j in pick])


def boot_r2(p, t, w, B, rng):
    """Cluster-bootstrap distribution of the pooled R^2 (wells resampled)."""
    _, groups = _well_groups(w)
    out = np.empty(B)
    for b in range(B):
        idx = _resample_idx(groups, rng)
        out[b] = r2(p[idx], t[idx])
    return out


def boot_delta(pa, pb, t, w, B, rng):
    """PAIRED cluster-bootstrap distribution of ``R2_A - R2_B`` (same well draw for both)."""
    _, groups = _well_groups(w)
    out = np.empty(B)
    for b in range(B):
        idx = _resample_idx(groups, rng)
        ti = t[idx]
        out[b] = r2(pa[idx], ti) - r2(pb[idx], ti)
    return out


def load_pools(dump_dir):
    """Load every ``pool_*.npz`` of a directory, refusing pools without well labels."""
    pools = {}
    for f in sorted(glob.glob(os.path.join(dump_dir, "pool_*.npz"))):
        nm = os.path.basename(f)[len("pool_"):-len(".npz")]
        pools[nm] = np.load(f)
    if not pools:
        raise SystemExit(f"no pool_*.npz in {dump_dir} "
                         f"(run scripts/run_benchmark.py --dump first).")
    no_label = [nm for nm, z in pools.items()
                if not any(k.endswith("__w") for k in z.files)]
    if no_label:
        raise SystemExit(
            "pools without a well label ('__w' missing): "
            f"{', '.join(sorted(no_label))}. "
            "Re-run the benchmark with --dump to regenerate well-labelled pools before "
            "the cluster bootstrap.")
    return pools


def run_bootstrap(dump_dir, out_dir, B=5000, seed=0):
    """Write ``bootstrap_ci.csv`` and ``bootstrap_pairwise.csv`` into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    pools = load_pools(dump_dir)
    models = [m for m in ORDER if m in pools] + [m for m in pools if m not in ORDER]
    print(f"[bootstrap] {len(models)} methods: {', '.join(models)} | B={B}", flush=True)

    # ---- 1) 95 % CI on the pooled R^2 (log 'all') per method and scenario
    ci_path = os.path.join(out_dir, "bootstrap_ci.csv")
    with open(ci_path, "w") as f:
        f.write("modele,scenario,n_points,r2,ci_low,ci_high,boot_mean,boot_sd\n")
        for nm in models:
            z = pools[nm]
            rng = np.random.default_rng(seed)
            for sc in SCEN:
                key = f"{sc}__all"
                if f"{key}__t" not in z:
                    continue
                p, t, w = z[f"{key}__p"], z[f"{key}__t"], z[f"{key}__w"]
                if t.size == 0:
                    continue
                point = r2(p, t)
                bd = boot_r2(p, t, w, B, rng)
                lo, hi = np.percentile(bd, [2.5, 97.5])
                f.write(f"{nm},{sc},{t.size},{point:.4f},{lo:.4f},{hi:.4f},"
                        f"{bd.mean():.4f},{bd.std():.4f}\n")
                print(f"  {nm:6s} {sc:9s} R2={point:6.3f}  CI95=[{lo:6.3f},{hi:6.3f}]  n={t.size}",
                      flush=True)
    print(f"[bootstrap] intervals -> {ci_path}", flush=True)

    # ---- 2) paired comparisons (delta R^2 + bootstrap p-value)
    pw_path = os.path.join(out_dir, "bootstrap_pairwise.csv")
    with open(pw_path, "w") as f:
        f.write("scenario,model_A,model_B,r2_A,r2_B,delta,d_ci_low,d_ci_high,p_value,n_points\n")
        for sc, a, b in PAIRS:
            if a not in pools or b not in pools:
                print(f"  [skip] {sc}: {a} or {b} missing", flush=True)
                continue
            za, zb = pools[a], pools[b]
            key = f"{sc}__all"
            if f"{key}__t" not in za or f"{key}__t" not in zb:
                continue
            ta, tb = za[f"{key}__t"], zb[f"{key}__t"]
            wa, wb = za[f"{key}__w"], zb[f"{key}__w"]
            # verify the pairing: same positions/ground truth AND same well labels
            if ta.shape != tb.shape or not np.allclose(ta, tb, atol=1e-3, equal_nan=True):
                print(f"  [WARN] {sc} {a} vs {b}: ground truth NOT aligned "
                      f"(shapes {ta.shape} vs {tb.shape}) -> paired bootstrap impossible",
                      flush=True)
                continue
            if wa.shape != wb.shape or not np.array_equal(wa, wb):
                print(f"  [WARN] {sc} {a} vs {b}: well labels NOT aligned "
                      f"-> paired cluster bootstrap impossible", flush=True)
                continue
            pa, pb, t, w = za[f"{key}__p"], zb[f"{key}__p"], ta, wa
            ra, rb = r2(pa, t), r2(pb, t)
            rng = np.random.default_rng(seed)
            bd = boot_delta(pa, pb, t, w, B, rng)
            lo, hi = np.percentile(bd, [2.5, 97.5])
            frac_pos = float(np.mean(bd > 0))
            pval = 2.0 * min(frac_pos, 1.0 - frac_pos)
            pval = max(pval, 1.0 / B)  # floor = the bootstrap resolution
            f.write(f"{sc},{a},{b},{ra:.4f},{rb:.4f},{ra-rb:.4f},{lo:.4f},{hi:.4f},"
                    f"{pval:.4g},{t.size}\n")
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            print(f"  {sc:9s} {a:6s}-{b:6s} dR2={ra-rb:+.3f} "
                  f"CI95=[{lo:+.3f},{hi:+.3f}] p={pval:.3g} {sig}", flush=True)
    print(f"[bootstrap] comparisons -> {pw_path}", flush=True)
