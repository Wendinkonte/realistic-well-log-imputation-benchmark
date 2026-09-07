#!/usr/bin/env python
"""Reproduce the window-level split leak (paper Section 5.4).

The Kapoor & Narayanan pitfall: a NON-well-wise split.  The windows are partitioned at
random (80/20) instead of leave-one-well-out, so windows of the SAME well land on both
sides.  The spatial model is then trained on the very wells it is evaluated on -- every
grid row visible, nothing blanked -- and it memorises each well's signature, "imputing"
by reading neighbouring depths of the same well.

Everything else (architecture, scenarios, evaluation seeds, pooled metrics) is identical
to the clean protocol, so the gap against the clean LOO run measures the artefact alone.

Usage:
    python scripts/run_window_split_leak.py [--models gnn,stgnn] [--epochs E]
                                            [--test-frac F]

Outputs: ``<results>/ablations/leak_randomsplit.json`` and ``leak_randomsplit.csv``.
"""
import argparse
import json
import os
import time

import numpy as np

import _bootstrap_path  # noqa: F401
from wellog_imputation.config import get_path
from wellog_imputation.evaluation.harness import (EVAL_SCENARIOS, EVAL_SEED,
                                                  ChannelScaler)
from wellog_imputation.masking.scenarios import make_eval_set
from wellog_imputation.metrics.pooled import add_to_pool, new_pool, pooled_agg
from wellog_imputation.models.registry import SCENARIOS

SCEN = SCENARIOS


def run_leaky(kind="stgnn", epochs=100, seed=0, test_frac=0.2, verbose=True,
              data_dir=None, coords_csv=None):
    """Train on a random window split (leaky) and evaluate with the clean metrics."""
    from wellog_imputation.models.spatial import SpatialGrid, SpatialImputer
    data_dir = get_path("data_dir", data_dir)
    s = np.load(os.path.join(data_dir, "slices.npz"), allow_pickle=True)
    X = s["X"].astype(np.float32); wells = s["wells"].astype(str); starts = s["starts"]
    grid = SpatialGrid(data_dir=data_dir, coords_csv=coords_csv)
    N = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_test = int(round(test_frac * N))
    test_idx = np.sort(perm[:n_test]); train_idx = np.sort(perm[n_test:])
    if verbose:
        nw_overlap = len(set(wells[test_idx]) & set(wells[train_idx]))
        print(f"[leak] window split: {len(train_idx)} train / {len(test_idx)} test; "
              f"{nw_overlap}/{len(set(wells))} wells present on BOTH sides (= the leak)",
              flush=True)

    # The scaler is fitted on the training windows: the leak is in the split, not here.
    sc_ = ChannelScaler().fit(X[train_idx])
    # Full grid, nothing blanked (test_well=None); targets are the training windows.
    imp = SpatialImputer(kind=kind, epochs=epochs, seed=seed)
    imp.fit(grid, set(wells), list(range(N)), X, wells, starts, sc_,
            test_well=None, train_slices=list(train_idx))

    pool = new_pool(SCEN)
    for sc in SCEN:
        rng_e = np.random.default_rng(EVAL_SEED[sc])
        Zte = sc_.transform(X[test_idx])
        _, Zte_in, Mte = make_eval_set(Zte, rng=rng_e, **EVAL_SCENARIOS[sc])
        preds = np.stack([imp.impute_slice(wells[i], int(starts[i]), Zte_in[j])
                          for j, i in enumerate(test_idx)])
        pred = sc_.inverse(preds); true = sc_.inverse(Zte)
        add_to_pool(pool, sc, pred, true, Mte)
    agg = pooled_agg(pool, SCEN)
    return {"model": kind.upper(), "agg": agg}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--models", type=str, default="gnn,stgnn")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    args = ap.parse_args()

    OUT = os.path.join(str(get_path("results_dir", args.out)), "ablations")
    os.makedirs(OUT, exist_ok=True)
    names = args.models.split(",")

    allres = {}
    for nm in names:
        t0 = time.time()
        print(f"\n########## {nm.upper()} (LEAK random split) ##########", flush=True)
        r = run_leaky(kind=nm, epochs=args.epochs, test_frac=args.test_frac,
                      data_dir=args.data_dir)
        allres[nm] = {"model": nm, "agg": r["agg"]}
        print(f"  -> {nm} (leaky) finished in {int(time.time()-t0)}s", flush=True)
        with open(os.path.join(OUT, "leak_randomsplit.json"), "w") as f:
            json.dump(allres, f, indent=2)

    with open(os.path.join(OUT, "leak_randomsplit.csv"), "w") as f:
        f.write("model," + ",".join(SCEN) + "\n")
        for nm in names:
            f.write(f"{nm}," + ",".join(f"{allres[nm]['agg'][sc]['all']['r2']:.3f}"
                                        for sc in SCEN) + "\n")

    print("\n=== leaky pooled R2 (random split) — compare with the clean LOO run ===")
    print(f"{'model':8s} " + " ".join(f"{s:>9s}" for s in SCEN))
    for nm in names:
        print(f"{nm:8s} " + " ".join(f"{allres[nm]['agg'][sc]['all']['r2']:9.3f}" for sc in SCEN))
    print(f"\nWritten: {OUT}/leak_randomsplit.{{json,csv}}")


if __name__ == "__main__":
    main()
