"""Masking scenarios: shapes, determinism, and the behaviour each regime promises."""
import numpy as np
import pytest

from wellog_imputation.masking.scenarios import (COOCCUR_DEFAULT, make_eval_set,
                                                 mask_mono, mask_partial_blackout)

SCENARIOS = ["single", "block", "profile", "blackout"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_shapes_and_dtypes(windows, scenario):
    Xt, Xi, ind = make_eval_set(windows, scenario, rng=np.random.default_rng(0))
    assert Xt.shape == Xi.shape == ind.shape == windows.shape
    assert ind.dtype == np.float32
    assert set(np.unique(ind)) <= {0.0, 1.0}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_masked_positions_are_nan_and_only_those(windows, scenario):
    Xt, Xi, ind = make_eval_set(windows, scenario, rng=np.random.default_rng(0))
    m = ind > 0.5
    assert np.isnan(Xi[m]).all(), "every masked position must be NaN in the input"
    assert not np.isnan(Xi[~m]).any(), "no unmasked position may be NaN"
    np.testing.assert_array_equal(Xt, windows)   # ground truth is untouched


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_deterministic_for_a_given_seed(windows, scenario):
    a = make_eval_set(windows, scenario, rng=np.random.default_rng(42))[2]
    b = make_eval_set(windows, scenario, rng=np.random.default_rng(42))[2]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_different_seeds_give_different_masks(windows, scenario):
    a = make_eval_set(windows, scenario, rng=np.random.default_rng(1))[2]
    b = make_eval_set(windows, scenario, rng=np.random.default_rng(2))[2]
    assert not np.array_equal(a, b)


def test_single_masks_one_log_and_n_points(windows):
    ind = mask_mono(windows, mode="single", n_points=5, rng=np.random.default_rng(0))
    touched = (ind.sum(axis=1) > 0).sum(axis=1)
    assert (touched == 1).all(), "single must touch exactly one log per window"
    # at most n_points, fewer only if the random draw repeats an index
    assert ind.sum(axis=(1, 2)).max() <= 5


def test_block_is_contiguous_within_one_log(windows):
    ind = mask_mono(windows, mode="block", block_len=(20, 100),
                    rng=np.random.default_rng(0))
    touched = (ind.sum(axis=1) > 0).sum(axis=1)
    assert (touched == 1).all()
    for i in range(ind.shape[0]):
        c = int(np.argmax(ind[i].sum(axis=0) > 0))
        pos = np.flatnonzero(ind[i, :, c])
        assert pos.size > 0
        assert np.array_equal(pos, np.arange(pos[0], pos[-1] + 1)), "block must be contiguous"


def test_profile_masks_a_whole_log(windows):
    ind = mask_mono(windows, mode="profile", rng=np.random.default_rng(0))
    T = windows.shape[1]
    per_log = ind.sum(axis=1)                      # (N, C)
    assert ((per_log == T).sum(axis=1) == 1).all(), "exactly one log fully masked"
    assert ((per_log == 0).sum(axis=1) == windows.shape[2] - 1).all()


def test_blackout_is_multilog_over_the_same_interval(windows):
    ind = mask_partial_blackout(windows, rng=np.random.default_rng(0)).astype(bool)
    n_multi = 0
    for i in range(ind.shape[0]):
        cols = np.flatnonzero(ind[i].any(axis=0))
        assert cols.size >= 1
        spans = set()
        for c in cols:
            pos = np.flatnonzero(ind[i, :, c])
            assert np.array_equal(pos, np.arange(pos[0], pos[-1] + 1)), "contiguous"
            spans.add((pos[0], pos[-1]))
        assert len(spans) == 1, "all masked logs must share the SAME interval"
        n_multi += int(cols.size >= 2)
    # the co-occurrence distribution puts ~96 % of the mass on 2-3 logs
    assert n_multi > 0.5 * ind.shape[0]


def test_blackout_respects_allow_full_false(windows):
    ind = mask_partial_blackout(windows, rng=np.random.default_rng(0), allow_full=False)
    C = windows.shape[2]
    n_cols = (ind.sum(axis=1) > 0).sum(axis=1)
    assert (n_cols <= C - 1).all(), "at least one log must stay observed"


def test_cooccurrence_distribution_is_the_measured_one():
    assert COOCCUR_DEFAULT == {1: 0.035, 2: 0.759, 3: 0.206}
    assert abs(sum(COOCCUR_DEFAULT.values()) - 1.0) < 1e-9


def test_unknown_scenario_raises(windows):
    with pytest.raises(ValueError):
        make_eval_set(windows, "not_a_scenario")
