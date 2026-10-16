"""Build the clean, windowed dataset from a LAS corpus (paper Section 4.1).

Extracts the measured GR / RHOB / NPHI curves, harmonises units, applies physical
bounds, recomputes the real missingness statistics, defines the leave-one-well-out folds
and cuts the intact windows.

Outputs, written into the configured ``data_dir``:

``well_logs.npz``
    Per well, an ``(L, 4)`` array ``depth, GR, RHOB, NPHI`` (cleaned, NOT normalised).
``quality_report.csv``
    Per well and log: source mnemonic, unit action taken, coverage percentage.
``missingness_real.json``
    Real missingness statistics: co-occurrence over samples and over intervals,
    absent-log combinations, gap-length quartiles (all / interior).
``depth_qc.csv``
    Per well: native depth step, jumps, and whether the well is kept (revision R1).
``folds_loo.json``
    Leave-one-well-out folds over the wells carrying the complete triplet.
``slices.npz``
    Intact windows ``X`` of length 256 plus their ``wells`` and ``starts`` metadata.

Design decisions:

* only MEASURED curves are used -- never composite, modelled or synthesised ones,
  because composite curves blend measured wireline data with model-derived infill and
  make the evaluation circular (paper Section 3.3);
* ``NEUT`` (a count) is EXCLUDED from the NPHI candidates: it is not a porosity;
* the NPHI percent/fraction harmonisation keys off the observed value RANGE, because the
  unit recorded in the LAS header is not reliable;
* no normalisation happens here: the scaler is fitted on the training split of each fold
  by the evaluation harness, which is what keeps the protocol leakage-free.
"""
import csv
import glob
import json
import os
import warnings

import numpy as np

from ..config import get_path

__all__ = ["STEP_M", "SLICE", "STRIDE", "BOUNDS", "PREFS", "build_dataset"]

STEP_M = 0.1524           # depth sampling step (m), i.e. 0.5 ft
SLICE = 256               # window length in samples
STRIDE = 64               # window stride (overlap raises the window count)

#: Physical validity bounds; out-of-range samples become NaN.
BOUNDS = {"GR": (0.0, 400.0), "RHOB": (1.2, 3.1), "NPHI": (-0.05, 1.0)}

#: Accepted MEASURED mnemonics, by priority.  NEUT is excluded (a count, not a porosity).
PREFS = {
    "GR":   ["GR", "SGR", "CGR", "GRD", "GRN"],
    "RHOB": ["RHOB", "RHOZ", "DEN", "DENS"],
    "NPHI": ["NPHI", "TNPH", "NPOR", "PHIN"],
}


def read_las(path):
    import lasio
    try:
        return lasio.read(path)
    except Exception:
        return lasio.read(path, engine="normal")


def pick_curve(curves, prefs):
    """Return ``(mnemonic, data)`` of the first measured mnemonic found, else ``(None, None)``."""
    d = {c.mnemonic.upper(): np.asarray(c.data, float) for c in curves}
    for p in prefs:
        if p in d:
            return p, d[p]
    return None, None


def harmonize(log, mnem, arr):
    """Harmonise units and apply physical bounds.  Returns ``(cleaned_array, action)``."""
    a = arr.copy()
    action = []
    if log == "NPHI":
        fin = a[np.isfinite(a)]
        if fin.size and np.nanpercentile(fin, 99) > 1.5:   # values in % -> fraction
            a = a / 100.0
            action.append("/100(%->frac)")
    if log == "GR":
        fin = a[np.isfinite(a)]
        # A plausible GR has a median of at least ~6 GAPI even in clean carbonates.
        # One well in the study corpus records ~2.7 GAPI (an aberrant 'micro' unit)
        # and its GR is therefore dropped.
        if fin.size and np.nanmedian(fin) < 6.0:
            a[:] = np.nan
            action.append(f"EXCLUDED(GR median={np.nanmedian(fin):.1f}<6, aberrant)")
    lo, hi = BOUNDS[log]
    n_out = int(np.sum((a < lo) | (a > hi)))
    if n_out:
        a[(a < lo) | (a > hi)] = np.nan
        action.append(f"bounds[{lo},{hi}]:-{n_out}")
    return a, ("; ".join(action) if action else "ok")


def gaps(mask_missing):
    """Lengths, in samples, of the consecutive runs of a boolean 'missing' mask."""
    out = []
    m = mask_missing.view(np.int8)
    idx = np.where(np.diff(np.concatenate(([0], m, [0]))) != 0)[0]
    for s, e in zip(idx[::2], idx[1::2]):
        out.append(int(e - s))
    return out


#: Tolerance on the native depth step (m).  Wells whose median step differs from
#: :data:`STEP_M` by more, or whose depth index has a jump > 2 steps or a non-positive
#: step, are EXCLUDED: every window, gap length and grid offset downstream assumes a
#: regular 0.1524 m step, and resampling would turn measured samples into interpolated
#: ones that would then be scored as ground truth.  (Revision R1: one well of the study
#: corpus was recorded at 0.5 m with a 72.5 m jump.)
STEP_TOL = 1e-3


def check_depth_step(depth, step=STEP_M, tol=STEP_TOL):
    """Return ``(ok, info)`` for a depth index; ``info`` is written to depth_qc.csv."""
    d = np.diff(np.asarray(depth, float))
    if d.size == 0:
        return False, dict(median_step=np.nan, min_step=np.nan, max_step=np.nan,
                           n_nonpos=0, n_jumps=0, jump_m=0.0)
    med = float(np.median(d))
    jumps = d > 2 * med
    info = dict(median_step=med, min_step=float(d.min()), max_step=float(d.max()),
                n_nonpos=int(np.sum(d <= 0)), n_jumps=int(jumps.sum()),
                jump_m=float(np.sum(d[jumps] - med)))
    ok = abs(med - step) <= tol and info["n_nonpos"] == 0 and info["n_jumps"] == 0
    return ok, info


def _runs(mask):
    """(start, end) of the True runs of a boolean vector, end exclusive."""
    m = np.asarray(mask, bool).view(np.int8)
    idx = np.where(np.diff(np.concatenate(([0], m, [0]))) != 0)[0]
    return list(zip(idx[::2], idx[1::2]))


def missingness_stats(wells, triplet_wells, step=STEP_M):
    """Observed missingness on the triplet wells (revision R1 definitions).

    * co-occurrence of the number K of absent logs, counted over depth SAMPLES with at
      least one absent log (the original definition) and over maximal INTERVALS on which
      the set of absent logs is constant;
    * the same for every combination of absent logs;
    * per-log gap lengths (runs of an absent log), split into interior gaps and edge
      intervals (touching the top or bottom of the well's depth index), with quartiles
      in metres -- quartiles rather than a median alone, because with ~100 gaps the
      median can fall between two clusters.
    """
    names = ["GR", "RHOB", "NPHI"]
    k_smp = np.zeros(4); k_int = np.zeros(4)
    combo_smp, combo_int, combo_len = {}, {}, {}
    gl_all, gl_int = [], []
    intervals = []          # every observed interval: [absent-log combination, length in samples]
    per_well = {}
    for w in triplet_wells:
        r = wells[w]
        M = np.stack([~np.isfinite(r[n]) for n in names], 1)          # (L, 3)
        code = M[:, 0] * 1 + M[:, 1] * 2 + M[:, 2] * 4
        L = len(code)
        ng = 0
        for k in range(1, 4):
            k_smp[k] += int(np.sum(M.sum(1) == k))
        # maximal intervals of constant non-empty absent set
        change = np.flatnonzero(np.diff(code)) + 1
        bounds = np.concatenate(([0], change, [L]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            c = int(code[a])
            if c == 0:
                continue
            key = "+".join(n for i, n in enumerate(names) if c & (1 << i))
            k_int[bin(c).count("1")] += 1
            combo_int[key] = combo_int.get(key, 0) + 1
            combo_smp[key] = combo_smp.get(key, 0) + (b - a)
            combo_len.setdefault(key, []).append((b - a) * step)
            intervals.append([key, int(b - a)])
        for i in range(3):
            for a, b in _runs(M[:, i]):
                length_m = (b - a) * step
                edge = a == 0 or b == L
                gl_all.append(length_m); ng += 1
                if not edge:
                    gl_int.append(length_m)
        per_well[w] = ng

    def q(v):
        v = np.asarray(v, float)
        if v.size == 0:
            return {}
        return {f"p{p}": round(float(np.percentile(v, p)), 1) for p in (25, 50, 75, 90)} | {"n": int(v.size)}

    def pct(h):
        tot = h[1:].sum()
        return {f"{k}_logs": (round(float(h[k] / tot * 100), 1) if tot else 0.0) for k in range(1, 4)}

    tot_smp = sum(combo_smp.values()) or 1
    tot_int = sum(combo_int.values()) or 1
    return {
        "n_wells_triplet": len(triplet_wells),
        # original key, sample-based (kept for backward compatibility)
        "cooccur_pct": pct(k_smp),
        "cooccur_pct_samples": pct(k_smp),
        "cooccur_pct_intervals": pct(k_int),
        "cooccur_prob_sample": {str(k): float(k_smp[k] / max(k_smp[1:].sum(), 1)) for k in range(1, 4)},
        "combinations": {key: {"share_samples_pct": round(combo_smp[key] / tot_smp * 100, 1),
                               "share_intervals_pct": round(combo_int[key] / tot_int * 100, 1),
                               "length_m": q(combo_len[key])}
                         for key in sorted(combo_smp, key=lambda k: -combo_smp[k])},
        "gap_length_m_all": q(gl_all),
        "gap_length_m_interior": q(gl_int),
        "gaps_per_well": per_well,
        # raw intervals, sampled by the empirical-blackout evaluation (revision R1)
        "intervals": intervals,
        # original key; see gap_length_m_* for the robust summary
        "gap_median_m": round(float(np.median(gl_all)), 1) if gl_all else 0.0,
    }


def build_dataset(las_dir=None, out_dir=None, verbose=True):
    """Run the whole pipeline over ``las_dir`` and write the artefacts into ``out_dir``.

    Returns
    -------
    dict
        Summary counts, also printed when ``verbose``.
    """
    warnings.filterwarnings("ignore")
    las_dir = get_path("las_dir", las_dir)
    out_dir = get_path("data_dir", out_dir)

    files = sorted(glob.glob(os.path.join(las_dir, "*.las")))
    if not files:
        raise SystemExit(
            f"no *.las found in {las_dir}. Point 'las_dir' at your LAS corpus in "
            f"configs/paths.yaml (or set WELLOG_LAS_DIR). The BRGM corpus used in the "
            f"paper is not redistributed -- see docs/DATA_AVAILABILITY.md."
        )
    wells = {}          # well -> dict(depth, GR, RHOB, NPHI)
    qrows = []          # quality-report rows
    dqc = []            # depth-step QC rows (revision R1)
    for f in files:
        name = os.path.basename(f)[:-4]
        las = read_las(f)
        depth = np.asarray(las.index, float)
        ok, info = check_depth_step(depth)
        dqc.append([name, round(info["median_step"], 4), round(info["min_step"], 4),
                    round(info["max_step"], 4), info["n_nonpos"], info["n_jumps"],
                    round(info["jump_m"], 2), "kept" if ok else "EXCLUDED"])
        if not ok:
            qrows.append([name, "DEPTH", "", f"EXCLUDED(step median={info['median_step']:.4f} m, "
                          f"jumps={info['n_jumps']}, nonpos={info['n_nonpos']})", 0.0])
            continue
        rec = {"depth": depth}
        for log in ["GR", "RHOB", "NPHI"]:
            mnem, arr = pick_curve(las.curves, PREFS[log])
            if arr is None:
                rec[log] = np.full(depth.shape, np.nan)
                qrows.append([name, log, "ABSENT", "", 0.0])
                continue
            arr = arr[:len(depth)] if len(arr) >= len(depth) else np.concatenate([arr, np.full(len(depth)-len(arr), np.nan)])
            clean, action = harmonize(log, mnem, arr)
            rec[log] = clean
            cov = float(np.isfinite(clean).mean() * 100)
            qrows.append([name, log, mnem, action, round(cov, 1)])
        wells[name] = rec

    # ---- leave-one-well-out folds: wells with > SLICE samples where all 3 logs co-exist
    triplet_wells = []
    for w, r in wells.items():
        inter = np.isfinite(r["GR"]) & np.isfinite(r["RHOB"]) & np.isfinite(r["NPHI"])
        if inter.sum() > SLICE:
            triplet_wells.append(w)
    triplet_wells.sort()

    # ---- real missingness on the triplet wells (revision R1 definitions)
    miss_stats = missingness_stats(wells, triplet_wells)

    # ---- intact windows (length SLICE, all three logs present)
    X, meta = [], []
    for w in triplet_wells:
        r = wells[w]
        arr = np.stack([r["GR"], r["RHOB"], r["NPHI"]], axis=1)  # (L,3)
        ok = np.all(np.isfinite(arr), axis=1)
        L = len(ok)
        s = 0
        while s + SLICE <= L:
            if ok[s:s+SLICE].all():
                X.append(arr[s:s+SLICE])
                meta.append((w, s))
            s += STRIDE
    X = np.array(X) if X else np.zeros((0, SLICE, 3))

    # ---- write the artefacts
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "well_logs.npz"),
                        **{w: np.stack([wells[w]["depth"], wells[w]["GR"], wells[w]["RHOB"], wells[w]["NPHI"]], 1)
                           for w in wells},
                        logs=np.array(["depth", "GR", "RHOB", "NPHI"]))
    with open(os.path.join(out_dir, "quality_report.csv"), "w", newline="") as fp:
        wcsv = csv.writer(fp); wcsv.writerow(["well", "log", "mnemo_source", "action", "coverage_%"])
        wcsv.writerows(qrows)
    with open(os.path.join(out_dir, "depth_qc.csv"), "w", newline="") as fp:
        wcsv = csv.writer(fp)
        wcsv.writerow(["well", "median_step_m", "min_step_m", "max_step_m", "n_nonpos",
                       "n_jumps", "jump_m", "status"])
        wcsv.writerows(dqc)
    with open(os.path.join(out_dir, "missingness_real.json"), "w") as fp:
        json.dump(miss_stats, fp, indent=2)
    folds = {f"fold_{i}": {"test": [w], "train": [x for x in triplet_wells if x != w]}
             for i, w in enumerate(triplet_wells)}
    with open(os.path.join(out_dir, "folds_loo.json"), "w") as fp:
        json.dump({"wells": triplet_wells, "n_folds": len(triplet_wells), "folds": folds}, fp, indent=2)
    np.savez_compressed(os.path.join(out_dir, "slices.npz"),
                        X=X.astype(np.float32),
                        wells=np.array([m[0] for m in meta]),
                        starts=np.array([m[1] for m in meta]),
                        logs=np.array(["GR", "RHOB", "NPHI"]),
                        slice_len=SLICE, stride=STRIDE)

    summary = {"n_wells_read": len(wells), "n_wells_triplet": len(triplet_wells),
               "n_slices": int(len(X)), "out_dir": str(out_dir), **miss_stats}
    if verbose:
        print(f"Wells read            : {len(wells)}")
        print(f"Triplet wells (LOO)   : {len(triplet_wells)}")
        print(f"Intact {SLICE}-windows  : {len(X)} (stride={STRIDE})")
        print(f"Real missingness      : 1 log={miss_stats['cooccur_pct']['1_logs']}%  "
              f"2 logs={miss_stats['cooccur_pct']['2_logs']}%  3 logs={miss_stats['cooccur_pct']['3_logs']}%  "
              f"| per-log gap length quartiles (m) {miss_stats['gap_length_m_all']}")
        print(f"Written to            : {out_dir}")
    return summary
