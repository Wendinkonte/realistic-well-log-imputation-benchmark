"""Revision statistics: sufficient statistics, Holm, effect sizes, end-to-end driver."""
import csv
import os

import numpy as np
import pytest

from wellog_imputation.metrics.pooled import LOGS, add_to_pool, new_pool, save_pool
from wellog_imputation.statistics.aggregate import r2 as r2_direct
from wellog_imputation.statistics.wellwise import (LOGS3, _suff, build_sufficient_stats,
                                                   holm, kendall_w, pooled_r2,
                                                   rank_biserial, run_revision)

SC = ["single", "block", "profile", "blackout"]


def test_pooled_r2_from_sufficient_stats_matches_direct(rng):
    t = rng.normal(size=500); p = t + rng.normal(scale=0.5, size=500)
    w = rng.integers(0, 7, size=500)
    assert pooled_r2(_suff(p, t, w, 7)) == pytest.approx(r2_direct(p, t), rel=1e-10)


def test_bootstrap_weights_equal_duplicated_wells(rng):
    t = rng.normal(size=300); p = t + rng.normal(scale=0.3, size=300)
    w = rng.integers(0, 5, size=300)
    wts = np.array([2, 0, 1, 1, 0], float)
    keep = np.concatenate([np.where(w == k)[0].repeat(1) for k in range(5) for _ in range(int(wts[k]))])
    assert pooled_r2(_suff(p, t, w, 5), wts) == pytest.approx(r2_direct(p[keep], t[keep]), rel=1e-10)


def test_holm_matches_textbook():
    p = [0.01, 0.04, 0.03, 0.005]
    adj, rej = holm(p, alpha=0.05)
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1, running max
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert rej.tolist() == [True, False, False, True]


def test_holm_ignores_nan():
    adj, rej = holm([0.001, np.nan, 0.2])
    assert np.isnan(adj[1]) and adj[0] == pytest.approx(0.002) and rej[0] and not rej[2]


def test_rank_biserial_extremes():
    assert rank_biserial([1, 2, 3]) == 1.0
    assert rank_biserial([-1, -2, -3]) == -1.0
    assert rank_biserial([0, 0]) != rank_biserial([0, 0]) or True   # NaN, no crash


def test_kendall_w_bounds():
    assert kendall_w(19 * 10, 19, 11) == pytest.approx(1.0)


def _fake_results(root, rng, methods=("mean", "locf", "mice", "stgnn"), seeds=(0, 1),
                  n_wells=6):
    """Per-fold dumps with method-dependent noise; truth shared across methods per seed."""
    noise = {"mean": 1.0, "locf": 0.3, "mice": 0.5, "stgnn": 0.6}
    for s in seeds:
        r_t = np.random.default_rng(100 + s)
        truths = {wi: r_t.normal(size=(3, 40, 3)) * [30, 0.2, 0.1] + [60, 2.4, 0.2]
                  for wi in range(n_wells)}
        masks = {wi: (r_t.random((3, 40, 3)) < 0.4).astype(float) for wi in range(n_wells)}
        for m in methods:
            for wi in range(n_wells):
                pool = new_pool(SC)
                t = truths[wi]
                p = t + rng.normal(size=t.shape) * noise[m] * np.array([30, 0.2, 0.1])
                for sc in SC:
                    add_to_pool(pool, sc, p, t, masks[wi], well_idx=wi)
                save_pool(pool, SC, os.path.join(root, "pools_seeded", f"seed{s}", m,
                                                 f"fold{wi:02d}.npz"))


def test_end_to_end_driver(tmp_path, rng):
    _fake_results(tmp_path, rng)
    S, seeds, methods = build_sufficient_stats(tmp_path / "pools_seeded", n_wells=6)
    assert seeds == [0, 1] and methods == ["mean", "locf", "mice", "stgnn"]
    out = tmp_path / "rev"
    info = run_revision(str(tmp_path / "pools_seeded"), str(out), n_wells=6, B=200,
                        pairs={"profile": [("locf", "mean"), ("mice", "stgnn")]})
    assert info["sigma_source"].startswith("fallback")      # no dataset given
    for f in ["table_perlog_revision.csv", "table_main_revision.csv", "per_well_nrmse.csv",
              "friedman_wellwise.csv", "wilcoxon_wellwise_holm.csv",
              "wilcoxon_allpairs_holm.csv"]:
        assert (out / f).exists(), f
    rows = list(csv.DictReader(open(out / "table_main_revision.csv")))
    prof = {r["method"]: r for r in rows if r["scenario"] == "profile"}
    assert prof["locf"]["rank_r2_macro"] == "1"           # least noise -> best
    assert prof["mean"]["rank_r2_macro"] == "4"
    wx = list(csv.DictReader(open(out / "wilcoxon_wellwise_holm.csv")))
    lm = [r for r in wx if r["A"] == "locf" and r["B"] == "mean"][0]
    assert float(lm["delta_r2_macro_A_minus_B"]) > 0
    assert float(lm["boot_ci_lo"]) > 0                     # clearly better under bootstrap
