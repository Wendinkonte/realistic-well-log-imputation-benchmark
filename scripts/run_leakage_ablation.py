#!/usr/bin/env python
"""Controlled spatial-leakage ablation (paper Section 4.3).

A single matched toggle: same architecture, same LOO folds, same evaluation seeds, same
epochs.  The ONLY difference is whether the held-out well's row is blanked in the
training grid (anti-leakage ON) or left in place (leak ON).

With the leak the scores inflate -- most of all on ``profile`` and ``blackout``, where
the target reads its own measurements back through the graph -- and without it they fall
back to their honest level.  This is the direct demonstration of the ST-GNN
``R^2 ~ 0.98`` artefact.

Usage:
    python scripts/run_leakage_ablation.py [--models gnn,stgnn] [--epochs E] [--folds N]

Outputs: ``<results>/ablations/ablation_leak.json`` and ``ablation_leak.csv``
(the matched leak OFF vs ON delta).
"""
import argparse
import json
import os
import time

import _bootstrap_path  # noqa: F401
from wellog_imputation.config import get_path
from wellog_imputation.models.registry import SCENARIOS

SCEN = SCENARIOS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--models", type=str, default="gnn,stgnn")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    args = ap.parse_args()

    from wellog_imputation.models.spatial import run_spatial
    OUT = os.path.join(str(get_path("results_dir", args.out)), "ablations")
    os.makedirs(OUT, exist_ok=True)
    names = args.models.split(",")

    allres = {}
    for nm in names:
        for leak in (False, True):
            tag = f"{nm}_{'leakON' if leak else 'leakOFF'}"
            t0 = time.time()
            print(f"\n########## {tag} ##########", flush=True)
            r = run_spatial(kind=nm, folds_subset=args.folds, epochs=args.epochs, leak=leak,
                            data_dir=args.data_dir)
            allres[tag] = {"model": nm, "leak": leak, "agg": r["agg"], "per_fold": r["per_fold"]}
            print(f"  -> {tag} finished in {int(time.time()-t0)}s", flush=True)
            with open(os.path.join(OUT, "ablation_leak.json"), "w") as f:
                json.dump(allres, f, indent=2)

    with open(os.path.join(OUT, "ablation_leak.csv"), "w") as f:
        f.write("model,scenario,r2_leakOFF,r2_leakON,delta\n")
        for nm in names:
            off, on = allres.get(f"{nm}_leakOFF"), allres.get(f"{nm}_leakON")
            if not (off and on):
                continue
            for sc in SCEN:
                ro = off["agg"][sc]["all"]["r2"]
                rn = on["agg"][sc]["all"]["r2"]
                f.write(f"{nm},{sc},{ro:.3f},{rn:.3f},{rn-ro:+.3f}\n")

    print(f"\nWritten to {OUT}/ (ablation_leak.json, ablation_leak.csv)")
    print("\n=== pooled R2: leak OFF -> ON (delta) ===")
    print(f"{'model':8s} {'scenario':10s} {'OFF':>8s} {'ON':>8s} {'delta':>8s}")
    for nm in names:
        off, on = allres.get(f"{nm}_leakOFF"), allres.get(f"{nm}_leakON")
        if not (off and on):
            continue
        for sc in SCEN:
            ro = off["agg"][sc]["all"]["r2"]; rn = on["agg"][sc]["all"]["r2"]
            print(f"{nm:8s} {sc:10s} {ro:8.3f} {rn:8.3f} {rn-ro:+8.3f}")


if __name__ == "__main__":
    main()
