"""The fairness guarantee: every method must see EXACTLY the same masked positions.

This is what makes the cross-method comparison fair and the paired statistics valid, so
it is tested directly rather than assumed.
"""
import numpy as np

from wellog_imputation.evaluation.harness import (EVAL_SCENARIOS, EVAL_SEED, eval_seed)
from wellog_imputation.masking.scenarios import make_eval_set


def _mask_for(scenario, windows, master_seed=0):
    rng = np.random.default_rng(eval_seed(scenario, master_seed))
    return make_eval_set(windows, rng=rng, **EVAL_SCENARIOS[scenario])[2]


def test_all_methods_would_see_identical_masks(windows):
    """Two independent draws with the same (scenario, seed) coincide exactly.

    In the harness each method re-derives its evaluation mask this way, so identical
    derivation means identical masked positions across methods.
    """
    for sc in EVAL_SCENARIOS:
        a = _mask_for(sc, windows, 0)
        b = _mask_for(sc, windows, 0)
        np.testing.assert_array_equal(a, b, err_msg=f"masks differ within scenario {sc}")


def test_masks_vary_across_master_seeds(windows):
    for sc in EVAL_SCENARIOS:
        a = _mask_for(sc, windows, 0)
        b = _mask_for(sc, windows, 1)
        assert not np.array_equal(a, b), f"{sc}: masks must vary across master seeds"


def test_seed_table_is_explicit_and_distinct():
    """An explicit table, never hash(scenario) -- Python's string hash is salted."""
    assert EVAL_SEED == {"single": 1000, "block": 1001, "profile": 1002, "blackout": 1003}
    assert len(set(EVAL_SEED.values())) == len(EVAL_SEED)


def test_eval_seed_derivation():
    for sc, base in EVAL_SEED.items():
        assert eval_seed(sc, 0) == base           # seed 0 reproduces the published table
        assert eval_seed(sc, 3) == base + 300000
    # different scenarios never collide across the seeds actually used
    seen = {eval_seed(sc, s) for sc in EVAL_SEED for s in range(5)}
    assert len(seen) == len(EVAL_SEED) * 5
