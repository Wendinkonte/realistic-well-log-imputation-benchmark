"""Empirical blackout on whole wells (revision R1; Reviewer 1 comment 2, Reviewer 2 comment 2).

The windowed ``blackout`` scenario draws its interval length uniformly in [0.3, 0.9] of a
256-sample window, and every masked interval keeps observed samples of the same logs
inside the window.  This module evaluates instead gaps whose absent-log COMBINATION and
LENGTH are drawn jointly from the intervals observed in the corpus
(``missingness_real.json``, key ``intervals``), applied to the held-out well BEFORE
windowing:

1. the intact segments of the held-out well (GR, RHOB and NPHI all measured) of at least
   one window length are the evaluation support;
2. in each segment and replicate, one observed interval (combination, length) is drawn;
   if it is at least as long as the segment, the whole segment is masked for those logs,
   otherwise it is placed uniformly inside the segment;
3. windowed methods predict the masked segment from overlapping windows (stride T/2,
   predictions averaged on the overlaps); methods exposing ``impute_well`` (whole-well
   LOCF and interpolation) instead see the complete well, so they can use samples of the
   same log outside the window or segment.

Masks are drawn from a seeded generator shared by all methods, so every method is scored
on the same positions.
"""
import json
import os

import numpy as np

__all__ = ["EMP_SCENARIO", "load_intervals", "intact_segments", "draw_masks",
           "predict_segment", "evaluate_well"]

EMP_SCENARIO = "blackout_emp"
LOGS = ["GR", "RHOB", "NPHI"]
_CODE = {"GR": 0, "RHOB": 1, "NPHI": 2}


def load_intervals(data_dir):
    """List of (channel indices, length in samples) from missingness_real.json."""
    with open(os.path.join(str(data_dir), "missingness_real.json")) as fh:
        d = json.load(fh)
    iv = d.get("intervals")
    if not iv:
        raise SystemExit("missingness_real.json has no 'intervals': rebuild the dataset with "
                         "the revision pipeline (scripts/build_dataset.py).")
    return [([_CODE[n] for n in key.split("+")], int(L)) for key, L in iv]


def intact_segments(arr, min_len):
    """(start, end) of runs where the three logs are all finite, length >= min_len."""
    ok = np.all(np.isfinite(arr), axis=1).astype(np.int8)
    idx = np.flatnonzero(np.diff(np.concatenate(([0], ok, [0]))))
    return [(a, b) for a, b in zip(idx[::2], idx[1::2]) if b - a >= min_len]


def draw_masks(segments, intervals, rng, n_rep):
    """Per replicate, one boolean (L_seg, 3) mask per segment."""
    out = []
    for _ in range(n_rep):
        rep = []
        for a, b in segments:
            chans, L = intervals[rng.integers(len(intervals))]
            n = b - a
            m = np.zeros((n, 3), bool)
            if L >= n:
                m[:, chans] = True
            else:
                s = int(rng.integers(0, n - L + 1))
                m[s:s + L, chans] = True
            rep.append(m)
        out.append(rep)
    return out


def predict_segment(Zseg_in, seg_start, predict_windows, T):
    """Overlapping-window reconstruction of one segment (stride T/2, mean on overlaps)."""
    L = len(Zseg_in)
    starts = list(range(0, L - T + 1, T // 2))
    if starts[-1] != L - T:
        starts.append(L - T)
    W = np.stack([Zseg_in[s:s + T] for s in starts])
    P = predict_windows(W, [seg_start + s for s in starts])
    acc = np.zeros_like(Zseg_in, dtype=np.float64); cnt = np.zeros((L, 1))
    for s, p in zip(starts, P):
        acc[s:s + T] += p; cnt[s:s + T] += 1
    return (acc / cnt).astype(np.float32)


def evaluate_well(well_arr, scaler, intervals, rng, n_rep, T, predict_windows,
                  impute_well=None):
    """Yield (pred, true, mask) arrays of shape (1, L_seg, 3) in physical units.

    ``well_arr``: (L, 3) cleaned logs of the held-out well (NaN where not measured).
    ``predict_windows(Z, starts)``: normalised windows (N, T, 3) -> predictions.
    ``impute_well(Zwell)``: optional whole-well imputer, (L, 3) -> (L, 3).
    """
    segs = intact_segments(well_arr, T)
    if not segs:
        return
    Zwell = scaler.transform(well_arr)
    for rep in draw_masks(segs, intervals, rng, n_rep):
        for (a, b), m in zip(segs, rep):
            if impute_well is not None:
                Zin = Zwell.copy(); Zin[a:b][m] = np.nan
                pz = impute_well(Zin)[a:b]
            else:
                Zseg = Zwell[a:b].copy(); Zseg[m] = np.nan
                pz = predict_segment(Zseg, a, predict_windows, T)
            pred = scaler.inverse(pz); true = well_arr[a:b]
            yield pred[None], true[None], m[None].astype(np.float32)
