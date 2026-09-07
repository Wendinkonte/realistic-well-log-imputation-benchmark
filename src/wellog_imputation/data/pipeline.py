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
    Real missingness statistics (co-occurrence, gap lengths).
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
    for f in files:
        name = os.path.basename(f)[:-4]
        las = read_las(f)
        depth = np.asarray(las.index, float)
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

    # ---- real missingness on the triplet wells: co-occurrence + gap lengths
    hist = np.zeros(4)   # index k = number of logs missing simultaneously (1..3)
    gl = []
    for w in triplet_wells:
        r = wells[w]
        mGR, mR, mN = (~np.isfinite(r["GR"]), ~np.isfinite(r["RHOB"]), ~np.isfinite(r["NPHI"]))
        nm = mGR.astype(int) + mR.astype(int) + mN.astype(int)
        sel = nm > 0
        for k in range(1, 4):
            hist[k] += int(np.sum(nm[sel] == k))
        for m in (mGR, mR, mN):
            gl += gaps(m)
    hist = hist[1:]
    gl = np.array(gl) if gl else np.array([0])
    # A corpus with no gaps at all (the synthetic example) would divide by zero here.
    # On any real corpus hist.sum() > 0, so this guard changes no published number.
    hist_total = hist.sum()
    miss_stats = {
        "n_wells_triplet": len(triplet_wells),
        "cooccur_pct": {f"{k}_logs": (round(float(hist[k-1] / hist_total * 100), 1)
                                      if hist_total else 0.0) for k in range(1, 4)},
        "gap_samples": {q: float(np.percentile(gl, int(q[1:]))) for q in ["p50", "p75", "p90", "p99"]},
        "gap_median_m": round(float(np.percentile(gl, 50) * STEP_M), 1),
    }

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
              f"| median gap {miss_stats['gap_median_m']} m")
        print(f"Written to            : {out_dir}")
    return summary
