"""Metric computation and the pooled aggregation."""
import numpy as np
import pytest

from wellog_imputation.metrics.pooled import (LOGS, add_to_pool, metrics, new_pool,
                                              point_stats, pooled_agg)

SCEN = ["single", "profile"]


def test_perfect_prediction_scores_one():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    s = point_stats(t.copy(), t)
    assert s["r2"] == pytest.approx(1.0)
    assert s["mae"] == pytest.approx(0.0)
    assert s["rmse"] == pytest.approx(0.0)
    assert s["n"] == 4


def test_mean_prediction_scores_zero():
    """Predicting the target mean is the zero-skill floor: R^2 = 0 by construction."""
    t = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    p = np.full_like(t, t.mean())
    assert point_stats(p, t)["r2"] == pytest.approx(0.0, abs=1e-12)


def test_worse_than_the_mean_is_negative():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([100.0, -100.0, 100.0, -100.0])
    assert point_stats(p, t)["r2"] < 0


def test_empty_input_gives_nan_not_a_crash():
    s = point_stats(np.array([]), np.array([]))
    assert s["n"] == 0
    assert np.isnan(s["r2"]) and np.isnan(s["mae"])


def test_constant_target_gives_nan_r2():
    t = np.full(5, 3.0)
    assert np.isnan(point_stats(t + 0.1, t)["r2"])


def test_metrics_score_only_masked_positions():
    """Errors outside the mask must not affect the score."""
    rng = np.random.default_rng(0)
    true = rng.normal(size=(4, 16, 3))
    pred = true.copy()
    mask = np.zeros_like(true, dtype=np.float32)
    mask[:, :8, 0] = 1.0
    pred[:, 8:, :] = 1e6          # huge errors, all OUTSIDE the mask
    out = metrics(pred, true, mask)
    assert out["all"]["r2"] == pytest.approx(1.0)
    assert out["all"]["n"] == 4 * 8
    assert out["GR"]["n"] == 4 * 8
    assert out["RHOB"]["n"] == 0


def test_metrics_reports_every_log():
    rng = np.random.default_rng(1)
    true = rng.normal(size=(3, 10, 3))
    pred = true + 0.1
    mask = np.ones_like(true, dtype=np.float32)
    out = metrics(pred, true, mask)
    assert set(out) == {"all", *LOGS}
    assert out["all"]["n"] == 3 * 10 * 3
    for lg in LOGS:
        assert out[lg]["n"] == 3 * 10


def test_pooling_equals_one_metric_over_the_concatenation():
    """The pooled R^2 must equal the R^2 of all folds concatenated -- not their average."""
    rng = np.random.default_rng(2)
    pool = new_pool(SCEN)
    all_p, all_t = [], []
    for fold in range(3):
        true = rng.normal(size=(2, 8, 3)) * (fold + 1)      # deliberately unequal folds
        pred = true + rng.normal(scale=0.3, size=true.shape)
        mask = np.ones_like(true, dtype=np.float32)
        add_to_pool(pool, "single", pred, true, mask, well_idx=fold)
        all_p.append(pred.reshape(-1)); all_t.append(true.reshape(-1))
    agg = pooled_agg(pool, SCEN)["single"]["all"]
    expected = point_stats(np.concatenate(all_p), np.concatenate(all_t))
    assert agg["r2"] == pytest.approx(expected["r2"])
    assert agg["n"] == expected["n"]


def test_pooling_differs_from_per_fold_averaging():
    """The reporting pitfall of Section 5.3, demonstrated on a tiny case.

    One fold contributes many points, another contributes few. Averaging R^2 per fold
    lets the small fold weigh as much as the large one; pooling does not.
    """
    rng = np.random.default_rng(3)
    pool = new_pool(SCEN)
    per_fold = []
    for fold, n_win in enumerate([20, 1]):
        true = rng.normal(size=(n_win, 8, 3))
        noise = 0.05 if fold == 0 else 3.0        # the tiny fold is much noisier
        pred = true + rng.normal(scale=noise, size=true.shape)
        mask = np.ones_like(true, dtype=np.float32)
        add_to_pool(pool, "single", pred, true, mask, well_idx=fold)
        per_fold.append(metrics(pred, true, mask)["all"]["r2"])
    pooled = pooled_agg(pool, SCEN)["single"]["all"]["r2"]
    assert pooled > np.mean(per_fold) + 0.05, (
        "per-fold averaging must be dragged down by the one-window fold, "
        "while pooling is not"
    )


def test_pool_records_the_well_index_for_the_cluster_bootstrap():
    rng = np.random.default_rng(4)
    pool = new_pool(SCEN)
    for fold in range(3):
        true = rng.normal(size=(2, 8, 3))
        mask = np.ones_like(true, dtype=np.float32)
        add_to_pool(pool, "single", true, true, mask, well_idx=fold)
    w = np.concatenate(pool["single"]["all"]["w"])
    assert set(np.unique(w)) == {0, 1, 2}
    assert w.size == pooled_agg(pool, SCEN)["single"]["all"]["n"]
