"""Shared evaluation harness: model interface, fold-local scaling, LOO loop.

Anti-leakage design (paper Section 4.5):

* the channel scaler is fitted ONLY on the training windows of the current fold;
* training uses a *mixed* missingness mask (a quarter of the windows under each of the
  four scenarios), so no model is tuned to one regime;
* evaluation uses a fixed scenario whose masks are generated from a seeded generator, so
  ALL models see exactly the same masked positions -- a fair comparison and a valid
  paired significance test;
* metrics are computed in original units (after inverse scaling), per log and pooled,
  ONLY at the artificially masked positions.

Model interface expected by :func:`run_model`::

    m.fit(Xtr_in, Xtr_true, Mtr)   # normalised (N, T, C); Xtr_in has NaN where masked
    m.impute(Xev_in) -> (N, T, C)  # fills the NaN
"""
import json
import os

import numpy as np

from ..config import get_path
from ..masking.scenarios import make_eval_set
from ..metrics.pooled import (LOGS, add_to_pool, metrics, new_pool, point_stats,
                              pooled_agg, save_pool)
from ..utils.determinism import set_determinism

__all__ = ["LOGS", "ChannelScaler", "EVAL_SCENARIOS", "EVAL_SEED", "eval_seed", "load",
           "metrics", "new_pool", "add_to_pool", "pooled_agg", "save_pool",
           "set_determinism", "training_mask", "run_model", "point_stats"]


# ---------- evaluation-mask seeding, derived from the master seed ----------
def eval_seed(scenario, master_seed=0):
    """Seed of the evaluation-mask generator for ``(scenario, master_seed)``.

    Combines the fixed :data:`EVAL_SEED` table -- which guarantees IDENTICAL masked
    positions for ALL models within a run, giving a fair comparison and valid paired
    statistics -- with the master seed of the run, so that positions VARY from one seed
    to the next (inter-seed variability is a wanted output, not noise to be suppressed).

    ``master_seed=0`` yields ``EVAL_SEED[scenario]``, matching the originally published
    single-seed run at the precision reported.  See docs/REPRODUCIBILITY.md.
    """
    return EVAL_SEED[scenario] + int(master_seed) * 100000


# ---------- data ----------
def load(data_dir=None):
    """Load the windowed dataset and the leave-one-well-out fold definition.

    Parameters
    ----------
    data_dir : path-like, optional
        Directory holding ``slices.npz`` and ``folds_loo.json``.  Defaults to the
        configured ``data_dir`` (see :mod:`wellog_imputation.config`).

    Returns
    -------
    X : ndarray (N, 256, 3) float32
    wells : ndarray of str, shape (N,)
        Well identifier of each window.
    folds : dict
        As written by the data pipeline: ``{'wells': [...], 'n_folds': int,
        'folds': {'fold_0': {'test': [w], 'train': [...]}, ...}}``.
    """
    data_dir = get_path("data_dir", data_dir)
    s = np.load(os.path.join(data_dir, "slices.npz"), allow_pickle=True)
    X = s["X"].astype(np.float32)               # (N, 256, 3)
    wells = s["wells"].astype(str)
    with open(os.path.join(data_dir, "folds_loo.json")) as fh:
        folds = json.load(fh)
    return X, wells, folds


# ---------- per-channel scaler ----------
class ChannelScaler:
    """Standardise each log channel.  Fitted on the TRAIN windows of a fold only."""

    def fit(self, X):                            # X (N, T, C)
        flat = X.reshape(-1, X.shape[-1])
        self.mu = np.nanmean(flat, 0)
        self.sd = np.nanstd(flat, 0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd

    def inverse(self, X):
        return X * self.sd + self.mu


# ---------- scenarios ----------
#: Evaluation configuration of each scenario, passed to
#: :func:`~wellog_imputation.masking.scenarios.make_eval_set`.
EVAL_SCENARIOS = {
    "single":   dict(scenario="single", n_points=5),
    "block":    dict(scenario="block", block_len=(20, 100)),
    "profile":  dict(scenario="profile"),
    "blackout": dict(scenario="blackout"),
}

#: Fixed per-scenario seed of the evaluation masks.
#:
#: An explicit table, not ``hash(scenario)``: Python's string hash is salted per process
#: (PYTHONHASHSEED), so hashing would silently change the masked positions on every run.
EVAL_SEED = {"single": 1000, "block": 1001, "profile": 1002, "blackout": 1003}


def training_mask(X, rng):
    """Mixed training missingness: a quarter of the windows under each scenario."""
    N = X.shape[0]
    Mtr = np.zeros_like(X, dtype=np.float32)
    Xin = X.copy()
    order = rng.permutation(N)
    chunks = np.array_split(order, 4)
    specs = [dict(scenario="single", n_points=5), dict(scenario="block", block_len=(20, 100)),
             dict(scenario="profile"), dict(scenario="blackout")]
    for idx, spec in zip(chunks, specs):
        if len(idx) == 0:
            continue
        _, xi, m = make_eval_set(X[idx], rng=rng, **spec)
        Xin[idx] = xi
        Mtr[idx] = m
    return Xin, Mtr


# ---------- leave-one-well-out loop ----------
def _save_curve_generic(curve_dir, model_name, seed, wi, fk, history):
    """Dump the per-epoch loss history of a non-spatial model exposing ``.history``.

    Format is aligned with the spatial equivalent so one plotting routine reads both.
    """
    os.makedirs(curve_dir, exist_ok=True)
    p = os.path.join(curve_dir, f"curve_{model_name.lower()}_seed{seed}_fold{wi:02d}.npz")
    np.savez(p, epoch=np.array(history.get("epoch", [])),
             train_mse=np.array(history.get("train_mse", []), dtype=float),
             val_mse=np.array(history.get("val_mse", []), dtype=float),
             kind=model_name.lower(), seed=seed, fold=wi, fold_key=str(fk))
    return p


def run_model(make_model, model_name, scenarios=None, folds_subset=None,
              seed=0, verbose=True, on_fold=None, skip_fold=None, curve_dir=None,
              fold_indices=None, data_dir=None):
    """Train and evaluate one model over every leave-one-well-out fold.

    Parameters
    ----------
    make_model : callable
        Zero-argument factory returning a fresh model (see the module docstring for the
        expected interface).  A NEW model is built for every fold.
    model_name : str
        Label used in logs and curve filenames.
    scenarios : list of str, optional
        Defaults to all four.
    folds_subset : int, optional
        Use only the first N folds (smoke tests).
    seed : int
        Master seed of the run.  Drives the training masks, model initialisation and --
        by derivation through :func:`eval_seed` -- the evaluation masks.
    on_fold : callable, optional
        ``on_fold(wi, fold_key, test_well, fold_metrics, fold_pool)``, called at the end
        of each fold (all scenarios computed).  Used to append per-cell metrics and dump
        per-fold points so a preempted run can resume.
    skip_fold : callable, optional
        ``skip_fold(wi, fold_key) -> bool``.  True means the fold is already on disk and
        training (the expensive part) is skipped.  The returned pool/agg is then PARTIAL;
        the replication driver rebuilds the aggregate from the per-fold dumps instead.
    curve_dir : path-like, optional
        Where to write per-epoch loss curves.
    fold_indices : container of int, optional
        Run only these fold indices (per-fold parallelism).
    data_dir : path-like, optional
        Overrides the configured dataset directory.

    Returns
    -------
    dict
        ``{'model', 'per_fold', 'agg', 'pool'}``.
    """
    X, wells, F = load(data_dir)
    scenarios = scenarios or list(EVAL_SCENARIOS)
    results = {sc: {lg: {k: [] for k in ["mae", "rmse", "r2", "cc", "n"]} for lg in (["all"]+LOGS)}
               for sc in scenarios}
    pool = new_pool(scenarios)          # accumulator for the pooled R^2
    fold_keys = list(F["folds"])
    if folds_subset is not None:
        fold_keys = fold_keys[:folds_subset]

    for wi, fk in enumerate(fold_keys):     # wi = index of the held-out well (pool label)
        if fold_indices is not None and wi not in fold_indices:
            continue
        test_w = F["folds"][fk]["test"][0]
        tr = wells != test_w
        te = wells == test_w
        if te.sum() == 0 or tr.sum() == 0:
            continue
        if skip_fold is not None and skip_fold(wi, fk):
            if verbose:
                print(f"  [{model_name}] fold {fk} already done -> skipped", flush=True)
            continue
        Xtr, Xte = X[tr], X[te]
        sc_ = ChannelScaler().fit(Xtr)          # fold-local scaling: TRAIN windows only
        Ztr, Zte = sc_.transform(Xtr), sc_.transform(Xte)

        rng = np.random.default_rng(seed)
        Xtr_in, Mtr = training_mask(Ztr, rng)
        Xtr_true = Ztr

        model = make_model()
        model.fit(Xtr_in, Xtr_true, Mtr)
        if curve_dir is not None and getattr(model, "history", None) is not None:
            _save_curve_generic(curve_dir, model_name, seed, wi, fk, model.history)

        fold_pool = new_pool(scenarios)     # points of THIS fold only (per-cell dump)
        fold_metrics = {}
        for sc in scenarios:
            rng_e = np.random.default_rng(eval_seed(sc, seed))
            _, Zte_in, Mte = make_eval_set(Zte, rng=rng_e, **EVAL_SCENARIOS[sc])
            pred_z = model.impute(Zte_in)
            pred = sc_.inverse(pred_z)
            true = sc_.inverse(Zte)
            mm = metrics(pred, true, Mte)                 # per-fold (diagnostic only)
            fold_metrics[sc] = mm
            add_to_pool(pool, sc, pred, true, Mte, well_idx=wi)        # global aggregate
            add_to_pool(fold_pool, sc, pred, true, Mte, well_idx=wi)   # per-fold dump
            for lg in (["all"]+LOGS):
                for k in ["mae", "rmse", "r2", "cc", "n"]:
                    results[sc][lg][k].append(mm[lg][k])
        if on_fold is not None:
            on_fold(wi, fk, test_w, fold_metrics, fold_pool)
        if verbose:
            r2b = pooled_agg(pool, ["blackout"])["blackout"]["all"]["r2"] if "blackout" in scenarios else float("nan")
            print(f"  [{model_name}] fold {fk} done | pooled R2(blackout,all)={r2b:.3f}", flush=True)

    # Aggregation is the POOLED metric over all folds; per_fold keeps the detail.
    agg = pooled_agg(pool, scenarios)
    return {"model": model_name, "per_fold": results, "agg": agg, "pool": pool}
