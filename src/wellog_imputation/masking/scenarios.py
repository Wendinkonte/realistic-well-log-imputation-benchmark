"""Evaluation-mask generators for the four missingness scenarios (paper Section 4.3).

Two families of mask:

1. **Mono-log** -- one log is masked per window, as in the reference benchmarks:

   * ``single``  : ``n_points`` scattered samples;
   * ``block``   : one contiguous block;
   * ``profile`` : the whole log over the window.

2. **Partial blackout** (``blackout``) -- the realistic mechanism: *several* logs are
   masked simultaneously over the *same* contiguous interval, with the number of
   co-missing logs drawn from the co-occurrence distribution measured on the real
   corpus (see :data:`COOCCUR_DEFAULT`) and long gaps.

Output convention for a batch ``X`` of shape ``(N, T, C)``:

* ``X_true``          -- ground truth (a copy of ``X``);
* ``X_input``         -- ``X`` with the artificially masked positions set to NaN;
* ``indicating_mask`` -- 1.0 at the artificially masked positions (those to be scored),
  0.0 elsewhere.

Metrics are computed **only** where ``indicating_mask`` is 1.

Determinism: pass a seeded :class:`numpy.random.Generator` to reproduce a mask exactly.
The benchmark harness derives that generator from a fixed per-scenario seed table so
that every method sees identical masked positions within a run
(:func:`wellog_imputation.evaluation.harness.eval_seed`).
"""
import numpy as np

__all__ = ["COOCCUR_DEFAULT", "mask_mono", "mask_partial_blackout", "make_eval_set"]

# Empirical distribution of the number of logs missing simultaneously, measured on the
# clean Paris Basin corpus (3.5 / 75.9 / 20.6 %).
# Source: <data_dir>/missingness_real.json, produced by the data pipeline;
# cf. article Eq. (1) and the missingness table of Section 5.3.
#
# NOTE: the originally published single-seed run (job 316552) used the slightly earlier
# values {2: 0.761, 3: 0.204}.  The difference (~0.2 percentage point) is below the
# precision at which results are reported and does not change any R^2 at three decimals.
# See docs/REPRODUCIBILITY.md.
COOCCUR_DEFAULT = {1: 0.035, 2: 0.759, 3: 0.206}


def _empty_mask(shape):
    return np.zeros(shape, dtype=bool)


def mask_mono(X, mode="block", n_points=1, block_len=(20, 100), rng=None):
    """Mask ONE randomly chosen log per window.

    Parameters
    ----------
    X : ndarray, shape (N, T, C)
        Batch of windows.
    mode : {'single', 'block', 'profile'}
        Scattered points, one contiguous block, or the entire log of the window.
    n_points : int
        Number of scattered samples, used by ``mode='single'``.
    block_len : int or sequence of int
        Candidate block lengths in samples, used by ``mode='block'``.
    rng : numpy.random.Generator, optional

    Returns
    -------
    ndarray of bool, shape (N, T, C)
    """
    rng = rng or np.random.default_rng()
    N, T, C = X.shape
    m = _empty_mask(X.shape)
    for i in range(N):
        c = rng.integers(0, C)
        if mode == "single":
            idx = rng.integers(0, T, size=n_points)
            m[i, idx, c] = True
        elif mode == "block":
            bl = int(rng.choice(np.atleast_1d(block_len)))
            bl = min(bl, T)
            s = rng.integers(0, T - bl + 1)
            m[i, s:s+bl, c] = True
        elif mode == "profile":
            m[i, :, c] = True
        else:
            raise ValueError(mode)
    return m


def mask_partial_blackout(X, cooccur=None, block_frac=(0.3, 0.9), rng=None,
                          allow_full=True):
    """Mask ``k`` logs -- ``k`` drawn from ``cooccur`` -- over the SAME contiguous interval.

    Parameters
    ----------
    X : ndarray, shape (N, T, C)
    cooccur : dict, optional
        ``{n_logs: probability}``.  Defaults to :data:`COOCCUR_DEFAULT`.
    block_frac : (float, float)
        Fraction of the window covered by the blackout, drawn uniformly in this range.
    rng : numpy.random.Generator, optional
    allow_full : bool
        If False, force ``k < C`` so at least one log stays observed (making the window
        solvable from within-well information alone).

    Returns
    -------
    ndarray of bool, shape (N, T, C)
    """
    rng = rng or np.random.default_rng()
    cooccur = cooccur or COOCCUR_DEFAULT
    N, T, C = X.shape
    ks = np.array(sorted(cooccur))
    pk = np.array([cooccur[k] for k in ks], float); pk /= pk.sum()
    m = _empty_mask(X.shape)
    for i in range(N):
        k = int(rng.choice(ks, p=pk))
        if not allow_full:
            k = min(k, C - 1)
        k = min(k, C)
        cols = rng.choice(C, size=k, replace=False)
        f = rng.uniform(*block_frac)
        bl = max(1, min(T, int(round(f * T))))
        s = rng.integers(0, T - bl + 1)
        for c in cols:
            m[i, s:s+bl, c] = True
    return m


def make_eval_set(X, scenario, rng=None, **kw):
    """Build ``(X_true, X_input, indicating_mask)`` for one scenario.

    Parameters
    ----------
    X : ndarray, shape (N, T, C)
    scenario : {'single', 'block', 'profile', 'blackout'}
    rng : numpy.random.Generator, optional
    **kw
        Forwarded to :func:`mask_mono` or :func:`mask_partial_blackout`.

    Returns
    -------
    X_true, X_input, indicating_mask : ndarray
        ``indicating_mask`` is float32, 1.0 at the positions to be scored.
    """
    rng = rng or np.random.default_rng()
    X = np.asarray(X, float)
    if scenario in ("single", "block", "profile"):
        mask = mask_mono(X, mode=scenario, rng=rng, **kw)
    elif scenario == "blackout":
        mask = mask_partial_blackout(X, rng=rng, **kw)
    else:
        raise ValueError(scenario)
    X_true = X.copy()
    X_input = X.copy(); X_input[mask] = np.nan
    return X_true, X_input, mask.astype(np.float32)
