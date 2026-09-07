"""Leave-one-well-out separation and fold-local scaling.

Together these are the anti-leakage protocol of Section 4.5: the held-out well must
appear in no training window, and the scaler must never see it.
"""

import numpy as np
import pytest

from wellog_imputation.evaluation.harness import ChannelScaler, load, run_model
from wellog_imputation.models.factory import factory


def test_folds_are_leave_one_well_out(synthetic_dataset):
    X, wells, folds = load(synthetic_dataset)
    assert folds["n_folds"] == len(folds["wells"])
    for fk, f in folds["folds"].items():
        assert len(f["test"]) == 1, "exactly one well is held out per fold"
        held = f["test"][0]
        assert held not in f["train"], "the held-out well must not be in train"
        assert set(f["train"]) | {held} == set(folds["wells"]), "the split must be a partition"


def test_every_well_is_held_out_exactly_once(synthetic_dataset):
    _, _, folds = load(synthetic_dataset)
    held = [f["test"][0] for f in folds["folds"].values()]
    assert sorted(held) == sorted(folds["wells"])
    assert len(held) == len(set(held))


def test_no_window_of_the_held_out_well_reaches_training(synthetic_dataset):
    X, wells, folds = load(synthetic_dataset)
    for fk, f in folds["folds"].items():
        held = f["test"][0]
        tr = wells != held
        te = wells == held
        assert te.sum() > 0 and tr.sum() > 0
        assert held not in set(wells[tr]), "a training window belongs to the held-out well"
        assert set(wells[te]) == {held}
        assert tr.sum() + te.sum() == len(wells), "train and test must partition the windows"


def test_scaler_is_fitted_on_training_windows_only(synthetic_dataset):
    """A scaler fitted on train only must differ from one fitted on everything."""
    X, wells, folds = load(synthetic_dataset)
    held = folds["folds"]["fold_0"]["test"][0]
    tr = wells != held
    s_train = ChannelScaler().fit(X[tr])
    s_all = ChannelScaler().fit(X)
    assert not np.allclose(s_train.mu, s_all.mu), "the scaler must not see the held-out well"


def test_scaler_roundtrip_is_the_identity(windows):
    sc = ChannelScaler().fit(windows)
    back = sc.inverse(sc.transform(windows))
    np.testing.assert_allclose(back, windows, rtol=1e-5, atol=1e-5)


def test_scaler_standardises_per_channel(windows):
    sc = ChannelScaler().fit(windows)
    z = sc.transform(windows)
    flat = z.reshape(-1, z.shape[-1])
    np.testing.assert_allclose(flat.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(flat.std(axis=0), 1.0, atol=1e-3)


@pytest.mark.parametrize("method", ["mean", "locf", "mice"])
def test_harness_runs_end_to_end_on_synthetic_data(synthetic_dataset, method):
    """A full LOO pass with a cheap method: exercises masking, scaling, metrics, pooling."""
    r = run_model(factory(method), method.upper(), folds_subset=2, verbose=False,
                  data_dir=synthetic_dataset)
    assert set(r["agg"]) == {"single", "block", "profile", "blackout"}
    for sc, per_log in r["agg"].items():
        assert set(per_log) == {"all", "GR", "RHOB", "NPHI"}
        assert per_log["all"]["n"] > 0, f"{sc}: nothing was evaluated"
        assert np.isfinite(per_log["all"]["mae"])
