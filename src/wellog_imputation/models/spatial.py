"""Spatial family (GNN / ST-GNN): impute a well by borrowing from its neighbours.

Design
------
* **Common depth grid** -- all wells are resampled onto one shared 0.1524 m axis, giving
  a tensor ``G`` of shape ``(W, Dtot, C)`` padded with NaN.
* **Well graph** -- built from the surface coordinates: k nearest neighbours with weights
  ``~ exp(-d^2)``, row-normalised.
* **Model** -- plain PyTorch:
    * a SPATIAL convolution mixes wells through the adjacency,
      ``einsum('wv,bvtc->bwtc', A, X)``;
    * a TEMPORAL convolution (``Conv1d``) runs along depth.
* **Evaluation** -- on exactly the same ``(well, start)`` windows as the within-well
  families, but with the ``(W, 256, C)`` neighbourhood supplied, which is what makes the
  comparison fair.
* **Leave-one-well-out** -- during training the row of the held-out well is set to NaN,
  so it is not even available as a neighbour.  At test time its observed positions and
  its neighbours are both available.

``stgnn`` uses the spatial AND temporal convolutions; ``gnn`` uses the spatial one only,
which isolates the contribution of the depth-wise component.

Anti-leakage
------------
:func:`run_spatial` takes a ``leak`` flag.  ``leak=False`` (default) is the protocol used
throughout the paper.  ``leak=True`` deliberately leaves the held-out well's row in the
training grid, so the target can read its own measurements back through the graph; this
is the controlled ablation of Section 4.3, not a usable configuration.
"""
import csv
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn

from ..config import PROJECT_ROOT, get_path
from ..evaluation.harness import (EVAL_SCENARIOS, ChannelScaler, eval_seed,
                                  training_mask)
from ..masking.scenarios import make_eval_set
from ..metrics.pooled import LOGS, add_to_pool, metrics, new_pool, pooled_agg
from ..utils.determinism import set_determinism

__all__ = ["SpatialGrid", "STGNN", "SpatialImputer", "run_spatial", "build_adjacency",
           "load_coords"]

#: Native depth sampling step of the corpus, in metres (0.5 ft).
STEP = 0.1524


# ---------- coordinates ----------
def _load_aliases():
    """Optional corpus-specific identifier aliases.

    Some archives spell the same well differently in the log files and in the coordinate
    table. Rather than hard-coding any real well name in the source, the substitutions
    are read from ``configs/well_aliases.yaml`` when that file exists::

        aliases:
          SOME-SPELLING: CANONICAL-SPELLING

    The file is git-ignored, because a list of well names is itself identifying. With no
    file the mapping is empty, which is correct for any corpus that spells its wells
    consistently.
    """
    cfg = PROJECT_ROOT / "configs" / "well_aliases.yaml"
    if not cfg.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(cfg) as fh:
        loaded = yaml.safe_load(fh) or {}
    table = loaded.get("aliases", loaded) if isinstance(loaded, dict) else {}
    return {str(k).upper(): str(v).upper() for k, v in table.items()}


_ALIASES = _load_aliases()


def _key(s):
    """Normalise a well identifier so the log corpus and the coordinate table agree.

    Strips a trailing archive-index suffix, collapses separators, drops a trailing
    deviation letter, expands the French abbreviation ``SS`` to ``SOUS``, and finally
    applies any corpus-specific alias from ``configs/well_aliases.yaml``.
    """
    s = s.upper(); s = re.sub(r'-?14-\d+.*$', '', s); s = s.replace('_', '-')
    s = re.sub(r'-+', '-', s).strip('-'); s = re.sub(r'(\d+)D\b', r'\1', s)
    s = re.sub(r'\bSS\b', 'SOUS', s)
    for src, dst in _ALIASES.items():
        s = s.replace(src, dst)
    return s


def load_coords(wells, coords_csv=None):
    """Load projected surface coordinates for ``wells``.

    The CSV must carry the columns ``Name``, ``Surf_LocX_L93``, ``Surf_LocY_L93`` (see
    docs/DATA_FORMAT.md).  Its location comes from the configured ``coords_csv`` path;
    for the BRGM corpus this file is proprietary and is not redistributed.

    Returns
    -------
    ndarray, shape (W, 2)
    """
    coords_csv = get_path("coords_csv", coords_csv)
    if not os.path.isfile(coords_csv):
        raise FileNotFoundError(
            f"well coordinate table not found at {coords_csv}. The spatial models need "
            f"surface coordinates; set 'coords_csv' in configs/paths.yaml or the "
            f"WELLOG_COORDS_CSV environment variable. See docs/DATA_FORMAT.md."
        )
    raw = {}
    with open(coords_csv) as f:
        for row in csv.DictReader(f):
            raw[_key(row['Name'])] = (float(row['Surf_LocX_L93']), float(row['Surf_LocY_L93']))
    return np.array([raw[_key(w)] for w in wells], float)   # (W, 2)


def build_adjacency(P, k=8, self_loop=True):
    """Row-normalised kNN adjacency with Gaussian distance weights.

    Parameters
    ----------
    P : ndarray, shape (W, 2)
        Projected coordinates in metres.
    k : int
        Number of neighbours kept per well.
    self_loop : bool
        Keep a unit self-weight before normalisation.
    """
    W = len(P)
    D = np.sqrt(((P[:, None, :] - P[None, :, :])**2).sum(-1)) / 1000.0  # km
    scale = np.median(D[D > 0])
    A = np.exp(-(D**2) / (2 * scale**2))
    Ak = np.zeros_like(A)
    for i in range(W):
        nn_idx = np.argsort(D[i])[1:k+1]
        Ak[i, nn_idx] = A[i, nn_idx]
    if self_loop:
        np.fill_diagonal(Ak, 1.0)
    Ak /= (Ak.sum(1, keepdims=True) + 1e-8)                  # row normalisation
    return Ak.astype(np.float32)


# ---------- common depth grid ----------
class SpatialGrid:
    """All wells resampled onto one shared depth axis, plus the well graph."""

    def __init__(self, data_dir=None, coords_csv=None):
        data_dir = get_path("data_dir", data_dir)
        wl = np.load(os.path.join(data_dir, "well_logs.npz"), allow_pickle=True)
        with open(os.path.join(data_dir, "folds_loo.json")) as fh:
            folds = json.load(fh)
        self.wells = folds["wells"]                           # the complete-triplet wells
        arrs = {w: wl[w] for w in self.wells}                 # (L,4): depth, GR, RHOB, NPHI
        d0 = min(a[0, 0] for a in arrs.values())
        dmax = max(a[-1, 0] for a in arrs.values())
        self.Dtot = int(round((dmax - d0) / STEP)) + 1
        self.d0 = d0
        W = len(self.wells)
        G = np.full((W, self.Dtot, 3), np.nan, np.float32)
        self.offset = {}
        for i, w in enumerate(self.wells):
            a = arrs[w]; off = int(round((a[0, 0] - d0) / STEP))
            self.offset[w] = off
            L = a.shape[0]
            G[i, off:off+L, :] = a[:, 1:4]
        self.G = G                                            # (W, Dtot, 3)
        self.widx = {w: i for i, w in enumerate(self.wells)}
        self.A = build_adjacency(load_coords(self.wells, coords_csv))

    def gstart(self, well, local_start):
        """Convert a within-well window start into a grid index."""
        return self.offset[well] + local_start

    def block(self, gstart, T):                               # (W, T, 3) neighbourhood
        return self.G[:, gstart:gstart+T, :].copy()


# ---------- ST-GNN ----------
class STGNN(nn.Module):
    """Spatial (+ optionally temporal) graph network over the well grid."""

    def __init__(self, C=3, hidden=64, layers=3, temporal=True):
        super().__init__()
        self.temporal = temporal
        self.inp = nn.Linear(2 * C, hidden)
        self.sp = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.tp = nn.ModuleList([nn.Conv1d(hidden, hidden, 5, padding=2) for _ in range(layers)]) \
            if temporal else None
        self.out = nn.Linear(hidden, C)
        self.act = nn.GELU()

    def forward(self, X, M, A):
        # X, M: (B, W, T, C); A: (W, W)
        h = self.act(self.inp(torch.cat([torch.nan_to_num(X), M], -1)))   # (B,W,T,H)
        for l in range(len(self.sp)):
            hs = torch.einsum('wv,bvtc->bwtc', A, h)        # spatial: mix over wells
            hs = self.act(self.sp[l](hs))
            h = h + hs
            if self.temporal:
                B, W, T, H = h.shape
                ht = h.permute(0, 1, 3, 2).reshape(B * W, H, T)
                ht = self.act(self.tp[l](ht)).reshape(B, W, H, T).permute(0, 1, 3, 2)
                h = h + ht
        return self.out(h)                                    # (B,W,T,C)


# ---------- spatial imputer ----------
class SpatialImputer:
    """Fit an :class:`STGNN` on the well grid and impute one window at a time."""

    def __init__(self, kind="stgnn", epochs=30, batch=16, lr=1e-3, device=None, seed=0):
        self.kind = kind; self.name = kind.upper()
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, grid, train_wells, slices_idx, X_slices, wells_of_slice, starts, scaler,
            test_well=None, train_slices=None):
        """Fit on the training windows of the fold.

        Parameters
        ----------
        test_well : str or None
            If given, that well's row is blanked in the training grid (anti-leakage).
            ``None`` leaves the grid intact -- the leaky configuration of the ablation.
        train_slices : sequence of int, optional
            If given, training targets are EXACTLY these window indices (a window-level
            split) instead of the by-well filter ``train_wells`` (leave-one-well-out).
            Used by the window-split leak reproduction of Section 5.4.
        """
        set_determinism(self.seed)
        self.scaler = scaler
        Gz_full = self.scaler.transform(grid.G)
        # ANTI-LEAKAGE: during training the held-out well's row is blanked, so it is not
        # even usable as a neighbour.  At inference the full grid is restored.
        Gz = Gz_full.copy()
        if test_well is not None:
            Gz[grid.widx[test_well]] = np.nan
        A = torch.tensor(grid.A, device=self.device)
        net = STGNN(temporal=(self.kind == "stgnn")).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        if train_slices is not None:
            idx = list(train_slices)
        else:
            idx = [i for i in slices_idx if wells_of_slice[i] in train_wells]
        T = X_slices.shape[1]
        # Windows of the held-out well (already EXCLUDED from training) -- convergence
        # monitoring only.  READ-ONLY: the loss is measured each epoch but never used for
        # the gradient nor for model selection (no early stopping), so it changes neither
        # the fit nor any reported metric.  Empty when leaking (test_well is None).
        val_idx = ([i for i in slices_idx if wells_of_slice[i] == test_well]
                   if test_well is not None else [])
        self.history = {"epoch": [], "train_mse": [], "val_mse": []}

        def _val_mse():
            if not val_idx:
                return float("nan")
            vrng = np.random.default_rng(self.seed + 777)   # fixed monitoring mask
            net.eval()
            tot_v, n_v = 0.0, 0
            with torch.no_grad():
                for b0 in range(0, len(val_idx), self.batch):
                    bi = val_idx[b0:b0+self.batch]
                    blocks = np.stack([Gz_full[:, grid.gstart(wells_of_slice[i], int(starts[i])):
                                               grid.gstart(wells_of_slice[i], int(starts[i]))+T, :]
                                       for i in bi])
                    tw = np.array([grid.widx[wells_of_slice[i]] for i in bi])
                    Xb = torch.tensor(blocks, device=self.device)
                    target_rows = Xb[np.arange(len(bi)), tw]
                    xin, mse = training_mask(target_rows.cpu().numpy(), vrng)
                    Xb2 = Xb.clone(); Xb2[np.arange(len(bi)), tw] = torch.tensor(xin, device=self.device)
                    # blank the held-out well's row, as during anti-leakage training
                    Xb2[:, grid.widx[test_well]] = torch.tensor(xin, device=self.device)
                    Mb = (~torch.isnan(Xb2)).float()
                    pr = net(Xb2, Mb, A)[np.arange(len(bi)), tw]
                    msk = torch.tensor(mse, device=self.device); tgt = torch.nan_to_num(target_rows)
                    tot_v += float(((pr - tgt)**2 * msk).sum() / (msk.sum() + 1e-6)); n_v += 1
            net.train()
            return tot_v / max(n_v, 1)

        for ep in range(self.epochs):
            rng.shuffle(idx)
            tot = 0.0; nb = 0
            for b0 in range(0, len(idx), self.batch):
                bi = idx[b0:b0+self.batch]
                blocks = np.stack([Gz[:, grid.gstart(wells_of_slice[i], int(starts[i])):
                                          grid.gstart(wells_of_slice[i], int(starts[i]))+T, :]
                                   for i in bi])               # (B,W,T,C)
                tw = np.array([grid.widx[wells_of_slice[i]] for i in bi])
                Xb = torch.tensor(blocks, device=self.device)
                # artificial masking on the target well's row (self-supervision)
                target_rows = Xb[np.arange(len(bi)), tw]       # (B,T,C)
                xin, mse = training_mask(target_rows.cpu().numpy(), rng)
                Xb2 = Xb.clone()
                Xb2[np.arange(len(bi)), tw] = torch.tensor(xin, device=self.device)
                Mb = (~torch.isnan(Xb2)).float()
                pred = net(Xb2, Mb, A)
                pr = pred[np.arange(len(bi)), tw]
                msk = torch.tensor(mse, device=self.device)
                tgt = torch.nan_to_num(target_rows)
                loss = ((pr - tgt)**2 * msk).sum() / (msk.sum() + 1e-6)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()); nb += 1   # value only, for the curve
            self.history["epoch"].append(ep)
            self.history["train_mse"].append(tot / max(nb, 1))
            self.history["val_mse"].append(_val_mse())
        self.net = net; self.A = A; self.grid = grid; self.Gz = Gz; self.T = T
        return self

    @torch.no_grad()
    def impute_slice(self, well, start, x_eval_in):
        """Impute one window.

        Parameters
        ----------
        x_eval_in : ndarray, shape (T, C)
            Normalised window with NaN at the positions to impute.

        Returns
        -------
        ndarray, shape (T, C)
        """
        self.net.eval()
        T = self.T; gs = self.grid.gstart(well, start)
        block = self.Gz[:, gs:gs+T, :].copy()                 # normalised neighbourhood
        wi = self.grid.widx[well]
        block[wi] = x_eval_in                                  # target row = masked input
        Xb = torch.tensor(block[None], device=self.device)
        Mb = (~torch.isnan(Xb)).float()
        pred = self.net(Xb, Mb, self.A)[0, wi].cpu().numpy()   # (T, C)
        out = x_eval_in.copy(); nanm = np.isnan(out); out[nanm] = pred[nanm]
        return out


def _save_curve(curve_dir, kind, seed, wi, fk, history):
    """Persist the per-epoch train/val loss history of one spatial fold."""
    os.makedirs(curve_dir, exist_ok=True)
    p = os.path.join(curve_dir, f"curve_{kind}_seed{seed}_fold{wi:02d}.npz")
    np.savez(p, epoch=np.array(history["epoch"]),
             train_mse=np.array(history["train_mse"]),
             val_mse=np.array(history["val_mse"]),
             kind=kind, seed=seed, fold=wi, fold_key=str(fk))
    return p


def run_spatial(kind="stgnn", folds_subset=None, epochs=30, seed=0, verbose=True, leak=False,
                on_fold=None, skip_fold=None, curve_dir=None, fold_indices=None,
                data_dir=None, coords_csv=None):
    """Train and evaluate a spatial model over every leave-one-well-out fold.

    Parameters
    ----------
    leak : bool
        ``False`` (default) is the protocol of the paper: anti-leakage ON, the held-out
        well's row blanked during training.  ``True`` deliberately enables the spatial
        leak (the target reads its own measurements through the graph) and exists only
        for the controlled ablation of Section 4.3.
    on_fold, skip_fold, curve_dir, fold_indices
        As in :func:`wellog_imputation.evaluation.harness.run_model`.

    Returns
    -------
    dict
        ``{'model', 'agg', 'per_fold', 'pool'}``.
    """
    data_dir = get_path("data_dir", data_dir)
    s = np.load(os.path.join(data_dir, "slices.npz"), allow_pickle=True)
    X = s["X"].astype(np.float32); wells = s["wells"].astype(str); starts = s["starts"]
    grid = SpatialGrid(data_dir=data_dir, coords_csv=coords_csv)
    with open(os.path.join(data_dir, "folds_loo.json")) as fh:
        folds = json.load(fh)
    fold_keys = list(folds["folds"])[: (folds_subset or len(folds["folds"]))]
    scenarios = list(EVAL_SCENARIOS)
    res = {sc: {lg: {k: [] for k in ["mae", "rmse", "r2", "cc", "n"]} for lg in (["all"]+LOGS)}
           for sc in scenarios}
    pool = new_pool(scenarios)          # accumulator for the pooled R^2

    for wi, fk in enumerate(fold_keys):     # wi = index of the held-out well (pool label)
        if fold_indices is not None and wi not in fold_indices:
            continue
        test_w = folds["folds"][fk]["test"][0]
        train_w = folds["folds"][fk]["train"]
        if skip_fold is not None and skip_fold(wi, fk):
            if verbose:
                print(f"  [{kind}] fold {fk} already done -> skipped", flush=True)
            continue
        tr_mask = wells != test_w
        sc_ = ChannelScaler().fit(X[tr_mask])   # fold-local scaling: TRAIN windows only
        imp = SpatialImputer(kind=kind, epochs=epochs, seed=seed)
        # leak ON -> test_well=None: the held-out well's row stays in the training grid,
        # so the target can read its own values back through the graph.
        imp.fit(grid, set(train_w), list(range(len(X))), X, wells, starts, sc_,
                test_well=(None if leak else test_w))
        if curve_dir is not None and getattr(imp, "history", None) is not None:
            _save_curve(curve_dir, kind, seed, wi, fk, imp.history)

        te = np.where(wells == test_w)[0]
        if len(te) == 0:
            continue
        fold_pool = new_pool(scenarios)      # points of THIS fold only
        fold_metrics = {}
        for sc in scenarios:
            rng_e = np.random.default_rng(eval_seed(sc, seed))
            Zte = sc_.transform(X[te])
            _, Zte_in, Mte = make_eval_set(Zte, rng=rng_e, **EVAL_SCENARIOS[sc])
            preds = np.stack([imp.impute_slice(test_w, int(starts[i]), Zte_in[j])
                              for j, i in enumerate(te)])
            pred = sc_.inverse(preds); true = sc_.inverse(Zte)
            mm = metrics(pred, true, Mte)                  # per-fold (diagnostic only)
            fold_metrics[sc] = mm
            add_to_pool(pool, sc, pred, true, Mte, well_idx=wi)        # global aggregate
            add_to_pool(fold_pool, sc, pred, true, Mte, well_idx=wi)   # per-fold dump
            for lg in (["all"]+LOGS):
                for k in ["mae", "rmse", "r2", "cc", "n"]:
                    res[sc][lg][k].append(mm[lg][k])
        if on_fold is not None:
            on_fold(wi, fk, test_w, fold_metrics, fold_pool)
        if verbose:
            r2b = pooled_agg(pool, ["blackout"])["blackout"]["all"]["r2"]
            print(f"  [{kind}] fold {fk} done | pooled R2(blackout)={r2b:.3f}", flush=True)

    agg = pooled_agg(pool, scenarios)
    return {"model": kind.upper(), "agg": agg, "per_fold": res, "pool": pool}
