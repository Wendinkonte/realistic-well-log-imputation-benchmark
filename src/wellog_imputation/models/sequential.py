"""Sequential imputers: LOCF, SAITS, BRITS, AE, U-Net.

All of them are WITHIN-WELL models: they see the window of the target well only, never a
neighbouring well.  SAITS and BRITS come from PyPOTS; AE and U-Net are the two deep
reconstruction autoencoders, implemented directly in PyTorch.

Common interface (see :mod:`wellog_imputation.evaluation.harness`)::

    m.fit(Xtr_in, Xtr_true, Mtr)
    m.impute(Xev_in) -> (N, T, C)
"""
import logging
import re

import numpy as np

from ..utils.determinism import set_determinism

__all__ = ["LOCFImputer", "SeqImputer", "AEImputer", "UNetImputer"]


def _capture_pypots_losses(fit_call):
    """Run ``fit_call()`` while capturing the per-epoch losses PyPOTS logs.

    PyPOTS emits lines such as ``Epoch 001 - training loss (MAE): 0.1234``; they are
    parsed into ``{'epoch': [...], 'train_mse': [...], 'val_mse': [nan, ...]}``.

    ``train_mse`` actually holds the PyPOTS training loss (a self-supervised MAE); the
    key name is kept homogeneous with the spatial curves so one plotting routine reads
    both.  There is no validation loss because no ``val_set`` is passed -- see
    :meth:`SeqImputer.fit`.
    """
    losses = []
    pat = re.compile(r"training loss[^0-9\-]*([-+0-9.eE]+)")

    class _H(logging.Handler):
        def emit(self, record):
            m = pat.search(record.getMessage())
            if m:
                try:
                    losses.append(float(m.group(1)))
                except ValueError:
                    pass

    # PyPOTS logs through a logger object named 'PyPOTS running log' with propagate=False,
    # so we attach to THAT object rather than to getLogger('pypots').
    try:
        from pypots.utils.logging import logger as lg
    except Exception:
        lg = logging.getLogger("PyPOTS running log")
    h = _H(); old_level = lg.level
    lg.addHandler(h); lg.setLevel(logging.INFO)
    try:
        fit_call()
    finally:
        lg.removeHandler(h); lg.setLevel(old_level)
    return {"epoch": list(range(len(losses))),
            "train_mse": losses,
            "val_mse": [float("nan")] * len(losses)}


class LOCFImputer:
    """Last-observation-carry-forward along depth.

    A parameter-free within-well baseline: each gap is filled with the last observed
    sample above it (0 if the window starts on a gap).
    """

    name = "LOCF"

    def fit(self, Xin, Xtrue, M):
        pass

    def impute(self, Xev):
        Y = Xev.copy()                 # (N, T, C); forward-fill along depth (axis T)
        N, T, C = Y.shape
        for i in range(N):
            for c in range(C):
                col = Y[i, :, c]
                last = 0.0
                for t in range(T):
                    if np.isnan(col[t]):
                        col[t] = last
                    else:
                        last = col[t]
        return Y


class SeqImputer:
    """SAITS / BRITS through PyPOTS, self-supervised on the intact (NaN-free) windows.

    Hyperparameters are listed in configs/models/sequential.yaml.
    """

    def __init__(self, kind="saits", epochs=20, batch_size=32, device=None, seed=0):
        self.kind = kind; self.name = kind.upper()
        self.epochs = epochs; self.batch_size = batch_size; self.seed = seed
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _build(self, T, C):
        from pypots.optim import Adam
        # verbose=True only makes PyPOTS log the per-epoch loss (captured above for the
        # convergence curves); it does not affect the fit.
        if self.kind == "saits":
            from pypots.imputation import SAITS
            return SAITS(n_steps=T, n_features=C, n_layers=2, d_model=128, n_heads=4,
                         d_k=32, d_v=32, d_ffn=128, dropout=0.1,
                         batch_size=self.batch_size, epochs=self.epochs,
                         patience=max(3, self.epochs//3),
                         optimizer=Adam(lr=1e-3), num_workers=0, device=self.device,
                         saving_path=None, verbose=True)
        if self.kind == "brits":
            from pypots.imputation import BRITS
            return BRITS(n_steps=T, n_features=C, rnn_hidden_size=128,
                         batch_size=self.batch_size, epochs=self.epochs,
                         patience=max(3, self.epochs//3),
                         optimizer=Adam(lr=1e-3), num_workers=0, device=self.device,
                         saving_path=None, verbose=True)
        raise ValueError(self.kind)

    def fit(self, Xin, Xtrue, M):
        # This PyPOTS version does not expose a seed argument on SAITS/BRITS, so the
        # global torch/numpy state is set immediately before construction and fit.
        set_determinism(self.seed)
        T, C = Xtrue.shape[1], Xtrue.shape[2]
        self.model = self._build(T, C)
        # No val_set is passed on purpose: with one, SAITS/BRITS would early-stop on the
        # validation loss, which would change the fitted model.  Without it PyPOTS
        # monitors the TRAINING loss for patience.  The loss is captured through the
        # PyPOTS logger, changing nothing in the fit itself.
        self.history = _capture_pypots_losses(lambda:
            self.model.fit({"X": Xtrue.astype(np.float32)}))   # PyPOTS masks internally

    def impute(self, Xev):
        out = self.model.impute({"X": Xev.astype(np.float32)})
        return np.asarray(out)


# AE and U-Net are within-well models (the target well's window only, never a neighbour)
# registered in the SEQUENTIAL family next to SAITS/BRITS.  They read all C logs as input
# channels -- so they do exploit cross-log structure -- but no other well.
#
# The self-supervision convention is IDENTICAL to SAITS/BRITS and to the spatial family:
#   * trained on intact windows with artificial masking; the harness' mixed training mask
#     (a quarter of the windows under each of the four scenarios) is reused as delivered
#     through fit(Xin, Xtrue, M).  Xin is the normalised window with NaN at masked
#     positions, M is 1 there, Xtrue is the intact window.
#   * loss computed ONLY at the artificially masked positions -- the same masked
#     reconstruction objective as SAITS/BRITS.
#   * input = window (T, C) with missing set to 0 (normalised space) concatenated with a
#     binary observed-mask channel (1 = observed) -> 2*C channels.  Output = (T, C).
#   * hyperparameters matched to SAITS/BRITS: Adam lr 1e-3, 100 epochs,
#     patience = epochs // 3, batch 16; torch/numpy/cuda seeded via set_determinism.
class _DeepReconImputer:
    """Shared training loop of AE and U-Net: masked self-supervised reconstruction."""

    def __init__(self, epochs=100, batch_size=16, lr=1e-3, device=None, seed=0):
        self.epochs = epochs; self.batch_size = batch_size; self.lr = lr; self.seed = seed
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None; self.history = None

    def _build(self, T, C):
        raise NotImplementedError

    @staticmethod
    def _to_input(Xwin):
        """(N,T,C) -> (2C-channel input: 0-fill || observed-mask, observed-mask).  NaN -> 0."""
        obs = np.isfinite(Xwin).astype(np.float32)        # 1 = observed
        x0 = np.nan_to_num(Xwin, nan=0.0).astype(np.float32)
        return np.concatenate([x0, obs], axis=-1), obs    # (N,T,2C), (N,T,C)

    def fit(self, Xin, Xtrue, M):
        import torch
        set_determinism(self.seed)
        N, T, C = Xtrue.shape
        self.C = C; self.T = T
        Xnet, _ = self._to_input(Xin)                     # 0-fill + observed-mask channel
        target = np.nan_to_num(Xtrue, nan=0.0).astype(np.float32)
        # loss mask = artificially masked AND finite ground truth
        lossm = (np.asarray(M, np.float32) > 0.5) & np.isfinite(Xtrue)
        lossm = lossm.astype(np.float32)

        dev = self.device
        Xt = torch.tensor(Xnet, device=dev)
        Tt = torch.tensor(target, device=dev)
        Lt = torch.tensor(lossm, device=dev)
        net = self._build(T, C).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        idx = np.arange(N)
        patience = max(3, self.epochs // 3)               # as for SAITS/BRITS
        best = float("inf"); best_state = None; bad = 0
        self.history = {"epoch": [], "train_mse": [], "val_mse": []}
        for ep in range(self.epochs):
            net.train(); rng.shuffle(idx)
            tot = 0.0; nb = 0
            for b0 in range(0, N, self.batch_size):
                bi = idx[b0:b0 + self.batch_size]
                xb = Xt[bi]; tb = Tt[bi]; mb = Lt[bi]
                pred = net(xb)                            # (B,T,C)
                denom = mb.sum() + 1e-6
                loss = ((pred - tb) ** 2 * mb).sum() / denom
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()); nb += 1
            tr = tot / max(nb, 1)
            self.history["epoch"].append(ep)
            self.history["train_mse"].append(tr)
            self.history["val_mse"].append(float("nan"))  # no val_set (as for SAITS/BRITS)
            # early stopping on the TRAINING loss, as for SAITS/BRITS
            if tr < best - 1e-6:
                best = tr; bad = 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net
        return self

    def impute(self, Xev):
        import torch
        Xnet, _ = self._to_input(Xev)
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(Xnet, device=self.device)).cpu().numpy()
        Y = np.asarray(Xev, np.float32).copy()
        nanm = np.isnan(Y)
        Y[nanm] = out.astype(np.float32)[nanm]
        return Y


class AEImputer(_DeepReconImputer):
    """AE -- MLP autoencoder over the flattened window (the simplest deep reconstructor).

    Encoder ``Flatten(T*2C) -> 256 -> 128 -> 64`` (GELU, no batch norm); symmetric decoder
    ``64 -> 128 -> 256 -> T*C``, reshaped to ``(T, C)``.
    """

    name = "AE"

    def _build(self, T, C):
        import torch.nn as nn
        din = T * 2 * C; dout = T * C

        class _AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(din, 256), nn.GELU(),
                                         nn.Linear(256, 128), nn.GELU(),
                                         nn.Linear(128, 64), nn.GELU())
                self.dec = nn.Sequential(nn.Linear(64, 128), nn.GELU(),
                                         nn.Linear(128, 256), nn.GELU(),
                                         nn.Linear(256, dout))

            def forward(self, x):                          # x (B,T,2C)
                B = x.shape[0]
                h = self.dec(self.enc(x.reshape(B, -1)))
                return h.reshape(B, T, C)

        return _AE()


class UNetImputer(_DeepReconImputer):
    """U-Net -- 1D convolutional autoencoder along DEPTH, with skip connections.

    Three encoder levels ``Conv1d(k=5, same padding) + GELU + maxpool(2)``, base width 64
    doubled per level (64 -> 128 -> 256); a 256-channel bottleneck; three decoder levels
    (``ConvTranspose1d``) each concatenating the matching encoder feature map (skip) and
    halving the width; a final ``Conv1d(k=1)`` to C channels.

    Kernel size 5 matches the temporal kernel of the ST-GNN, for consistency.  If T is
    not a multiple of 8 (three downsamplings) the input is zero-padded up to the next
    multiple and the padding is stripped from the output.
    """

    name = "UNET"

    def _build(self, T, C):
        import torch
        import torch.nn as nn
        cin = 2 * C

        class _UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = nn.GELU()
                self.pool = nn.MaxPool1d(2)
                self.e1 = nn.Conv1d(cin, 64, 5, padding=2)
                self.e2 = nn.Conv1d(64, 128, 5, padding=2)
                self.e3 = nn.Conv1d(128, 256, 5, padding=2)
                self.bott = nn.Conv1d(256, 256, 5, padding=2)
                self.u3 = nn.ConvTranspose1d(256, 256, 2, stride=2)
                self.d3 = nn.Conv1d(256 + 256, 128, 5, padding=2)
                self.u2 = nn.ConvTranspose1d(128, 128, 2, stride=2)
                self.d2 = nn.Conv1d(128 + 128, 64, 5, padding=2)
                self.u1 = nn.ConvTranspose1d(64, 64, 2, stride=2)
                self.d1 = nn.Conv1d(64 + 64, 64, 5, padding=2)
                self.outc = nn.Conv1d(64, C, 1)

            def forward(self, x):                          # x (B,T,2C)
                B, Torig, _ = x.shape
                h = x.permute(0, 2, 1)                     # (B,2C,T)
                pad = (-Torig) % 8                         # pad to a multiple of 8
                if pad:
                    h = nn.functional.pad(h, (0, pad))
                e1 = self.act(self.e1(h))                  # (B,64,T)
                e2 = self.act(self.e2(self.pool(e1)))      # (B,128,T/2)
                e3 = self.act(self.e3(self.pool(e2)))      # (B,256,T/4)
                b = self.act(self.bott(self.pool(e3)))     # (B,256,T/8)
                d = self.act(self.d3(torch.cat([self.u3(b), e3], 1)))   # (B,128,T/4)
                d = self.act(self.d2(torch.cat([self.u2(d), e2], 1)))   # (B,64,T/2)
                d = self.act(self.d1(torch.cat([self.u1(d), e1], 1)))   # (B,64,T)
                out = self.outc(d)                         # (B,C,T)
                out = out[:, :, :Torig]                    # strip the padding
                return out.permute(0, 2, 1)                # (B,T,C)

        return _UNet()
