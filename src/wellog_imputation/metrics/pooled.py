"""Metrics and the pooled aggregation used throughout the paper (Section 5.3).

All metrics are computed in the ORIGINAL log units (after inverse scaling) and ONLY at
the artificially masked positions carried by ``indicating_mask``.

Aggregation across leave-one-well-out folds is **pooled**, not averaged per fold: the
masked positions of every fold are accumulated and a single metric is computed over the
concatenation.  Averaging R^2 per fold instead lets a well contributing one window weigh
as much as a well contributing 237, which fabricates an apparent collapse of otherwise
healthy models -- the reporting pitfall demonstrated in Section 5.3.
"""
import os

import numpy as np

__all__ = ["LOGS", "point_stats", "metrics", "new_pool", "add_to_pool", "pooled_agg",
           "save_pool"]

#: Channel order of every ``(N, T, C)`` array in the package.
LOGS = ["GR", "RHOB", "NPHI"]


def point_stats(p, t):
    """MAE / RMSE / R^2 / Pearson r / n for two paired vectors.  NaN metrics if empty."""
    if t.size == 0:
        return dict(mae=np.nan, rmse=np.nan, r2=np.nan, cc=np.nan, n=0)
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss = float(np.sum((t - t.mean())**2))
    r2 = float(1 - np.sum(err**2)/ss) if ss > 0 else np.nan
    cc = float(np.corrcoef(p, t)[0, 1]) if p.size > 1 and np.std(p) > 0 and np.std(t) > 0 else np.nan
    return dict(mae=mae, rmse=rmse, r2=r2, cc=cc, n=int(t.size))


# Backwards-compatible private alias (the original name inside harness.py).
_point_stats = point_stats


def metrics(pred, true, mask):
    """Per-log and pooled metrics for one batch.

    Parameters
    ----------
    pred, true, mask : ndarray, shape (N, T, C)
        In original units; ``mask`` is 1 at the evaluated positions.

    Returns
    -------
    dict
        ``{'all': {...}, 'GR': {...}, 'RHOB': {...}, 'NPHI': {...}}``.
    """
    out = {}
    m = mask > 0.5
    out["all"] = point_stats(pred[m], true[m])
    for c, name in enumerate(LOGS):
        mc = m[:, :, c]
        out[name] = point_stats(pred[:, :, c][mc], true[:, :, c][mc])
    return out


def new_pool(scenarios):
    """Empty accumulator for the pooled aggregation.

    ``'w'`` holds, for every pooled point, the index of the test well (fold) it came
    from -- required by the by-well cluster bootstrap.
    """
    return {sc: {lg: {"p": [], "t": [], "w": []} for lg in (["all"] + LOGS)} for sc in scenarios}


def add_to_pool(pool, sc, pred, true, mask, well_idx=None):
    """Append the masked positions of one fold (original units) to a scenario pool."""
    m = mask > 0.5

    def _add(lg, pvals, tvals):
        pool[sc][lg]["p"].append(pvals); pool[sc][lg]["t"].append(tvals)
        if well_idx is not None:
            pool[sc][lg]["w"].append(np.full(tvals.shape[0], well_idx, dtype=np.int64))

    _add("all", pred[m], true[m])
    for c, name in enumerate(LOGS):
        mc = m[:, :, c]
        _add(name, pred[:, :, c][mc], true[:, :, c][mc])


def pooled_agg(pool, scenarios):
    """One R^2/MAE/RMSE/r per ``(scenario, log)`` over all pooled points."""
    agg = {}
    for sc in scenarios:
        agg[sc] = {}
        for lg in (["all"] + LOGS):
            ps, ts = pool[sc][lg]["p"], pool[sc][lg]["t"]
            p = np.concatenate(ps) if ps else np.array([])
            t = np.concatenate(ts) if ts else np.array([])
            agg[sc][lg] = point_stats(p, t)
    return agg


def save_pool(pool, scenarios, path):
    """Persist the pooled ``(pred, true, well)`` triples of one method to a ``.npz``.

    Positions are concatenated in the SAME order (folds, then masked positions) for every
    method.  Because the evaluation masks are seeded, the resulting arrays are aligned
    position by position across methods, which is what makes the PAIRED bootstrap of R^2
    differences valid.

    .. warning::
       These files contain the ground-truth log values at every evaluated position and
       are therefore derived data.  For the BRGM corpus they are proprietary and must not
       be redistributed; the public ``.gitignore`` excludes them.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {}
    for sc in scenarios:
        for lg in (["all"] + LOGS):
            ps, ts = pool[sc][lg]["p"], pool[sc][lg]["t"]
            ws = pool[sc][lg].get("w", [])
            out[f"{sc}__{lg}__p"] = (np.concatenate(ps) if ps else np.array([])).astype(np.float32)
            out[f"{sc}__{lg}__t"] = (np.concatenate(ts) if ts else np.array([])).astype(np.float32)
            out[f"{sc}__{lg}__w"] = (np.concatenate(ws) if ws else np.array([], dtype=np.int64)).astype(np.int64)
    np.savez_compressed(path, **out)
    return path
