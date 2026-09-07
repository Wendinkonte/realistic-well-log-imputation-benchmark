"""Reproducible random state for the neural models.

Extracted verbatim from the original ``harness.set_determinism`` so that both the
evaluation harness and the model implementations can seed without importing each other.
"""
import os
import random

import numpy as np

__all__ = ["set_determinism"]


def set_determinism(seed=0):
    """Seed numpy and torch so a re-run reproduces the same numbers to ~3 decimals.

    Called before fitting each neural model (SAITS/BRITS via PyPOTS, AE/U-Net, GNN/ST-GNN).

    cuDNN determinism is deliberately NOT enabled.  Forcing it pushes the bidirectional
    RNN of BRITS onto an unoptimised cuDNN kernel and makes it roughly 80x slower
    (33 min per fold instead of 24 s).  Seeding alone makes runs reproducible at the
    precision reported in the paper; the residual cuDNN nondeterminism on conv/RNN
    kernels stays below 1e-3 on pooled R^2.  This caveat is documented in the article
    and in docs/REPRODUCIBILITY.md.

    ``benchmark=True`` lets cuDNN autotune, which speeds up the convolutions of the
    spatial models.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
