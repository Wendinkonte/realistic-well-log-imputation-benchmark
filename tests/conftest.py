"""Shared fixtures.  No test in this suite touches proprietary data."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def windows():
    """A small artificial batch ``(N, T, C)`` with no missing values."""
    r = np.random.default_rng(12345)
    return r.normal(size=(24, 64, 3)).astype(np.float32)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Write a miniature dataset in the pipeline's own output format.

    Returns the directory, usable as ``data_dir`` by the harness.
    """
    wells = [f"SYN{i+1:02d}" for i in range(4)]
    r = np.random.default_rng(7)
    T, per_well = 64, 5
    X, wcol, starts = [], [], []
    for w in wells:
        for k in range(per_well):
            X.append(r.normal(size=(T, 3)))
            wcol.append(w)
            starts.append(k * 16)
    X = np.asarray(X, dtype=np.float32)
    np.savez_compressed(tmp_path / "slices.npz",
                        X=X, wells=np.array(wcol), starts=np.array(starts),
                        logs=np.array(["GR", "RHOB", "NPHI"]),
                        slice_len=T, stride=16)
    folds = {f"fold_{i}": {"test": [w], "train": [x for x in wells if x != w]}
             for i, w in enumerate(wells)}
    with open(tmp_path / "folds_loo.json", "w") as f:
        json.dump({"wells": wells, "n_folds": len(wells), "folds": folds}, f)
    return tmp_path
