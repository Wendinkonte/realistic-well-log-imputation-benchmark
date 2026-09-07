#!/usr/bin/env python
"""Single-seed benchmark driver: every method over every leave-one-well-out fold.

Runs the requested methods across all folds and the four scenarios, aggregates with the
pooled R^2, and writes ``results.json`` plus the two summary CSVs.

Usage:
    python scripts/run_benchmark.py [--models a,b,c] [--epochs E] [--folds N]
                                    [--out DIR] [--dump] [--resume]

Methods: mean, locf, rf, xgb, mice, saits, brits, unet, ae, gnn, stgnn
"""
import argparse
import json
import os
import time

import _bootstrap_path  # noqa: F401
from wellog_imputation.config import get_path
from wellog_imputation.evaluation.harness import run_model
from wellog_imputation.metrics.pooled import LOGS, save_pool
from wellog_imputation.models.factory import factory
from wellog_imputation.models.registry import FAMILY, ORDER, SCENARIOS

SCEN = SCENARIOS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=None,
                    help="use only the first N folds (smoke test)")
    ap.add_argument("--epochs", type=int, default=50,
                    help="epochs for the neural models (the paper uses 100)")
    ap.add_argument("--models", type=str,
                    default="mean,locf,rf,xgb,saits,brits,gnn,stgnn")
    ap.add_argument("--resume", action="store_true",
                    help="merge into an existing results.json instead of overwriting it")
    ap.add_argument("--out", type=str, default=None,
                    help="output directory (default: configured results_dir)")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="dataset directory (default: configured data_dir)")
    ap.add_argument("--dump", action="store_true",
                    help="dump the pooled (pred,true,well) triples per method to "
                         "<dump-dir>/pool_<m>.npz, for the cluster bootstrap")
    ap.add_argument("--dump-dir", type=str, default=None,
                    help="directory for the pool dumps (default: <out>/pools)")
    args = ap.parse_args()

    OUT_DIR = str(get_path("results_dir", args.out))
    DUMP_DIR = args.dump_dir or os.path.join(OUT_DIR, "pools")
    os.makedirs(OUT_DIR, exist_ok=True)
    names = args.models.split(",")
    allres = {}
    if args.resume:
        rp = os.path.join(OUT_DIR, "results.json")
        if os.path.exists(rp):
            with open(rp) as f:
                allres = json.load(f)
            print(f"[resume] {len(allres)} methods loaded from results.json: "
                  f"{', '.join(allres)}", flush=True)

    for nm in names:
        t0 = time.time()
        print(f"\n########## {nm.upper()} ({FAMILY[nm]}) ##########", flush=True)
        if nm in ("gnn", "stgnn"):
            from wellog_imputation.models.spatial import run_spatial
            r = run_spatial(kind=nm, folds_subset=args.folds, epochs=args.epochs,
                            data_dir=args.data_dir)
        elif nm in ("saits", "brits", "unet", "ae"):
            r = run_model(factory(nm, epochs=args.epochs), nm.upper(),
                          folds_subset=args.folds, data_dir=args.data_dir)
        else:
            r = run_model(factory(nm), nm.upper(), folds_subset=args.folds,
                          data_dir=args.data_dir)
        allres[nm] = {"family": FAMILY[nm], "agg": r["agg"], "per_fold": r["per_fold"]}
        print(f"  -> {nm} finished in {int(time.time()-t0)}s", flush=True)
        if args.dump and "pool" in r:
            p = save_pool(r["pool"], SCEN, os.path.join(DUMP_DIR, f"pool_{nm}.npz"))
            print(f"     pool dumped -> {p}", flush=True)
        with open(os.path.join(OUT_DIR, "results.json"), "w") as f:   # incremental save
            json.dump(allres, f, indent=2)

    report = [m for m in ORDER if m in allres]
    with open(os.path.join(OUT_DIR, "summary_r2.csv"), "w") as f:
        f.write("famille,modele," + ",".join(SCEN) + "\n")
        for nm in report:
            row = [FAMILY[nm], nm] + [f"{allres[nm]['agg'][sc]['all']['r2']:.3f}" for sc in SCEN]
            f.write(",".join(row) + "\n")
    with open(os.path.join(OUT_DIR, "summary_full.csv"), "w") as f:
        f.write("famille,modele,scenario,log,mae,rmse,r2,cc\n")
        for nm in report:
            for sc in SCEN:
                for lg in (["all"] + LOGS):
                    m = allres[nm]["agg"][sc][lg]
                    f.write(f"{FAMILY[nm]},{nm},{sc},{lg},{m['mae']:.4f},{m['rmse']:.4f},"
                            f"{m['r2']:.4f},{m['cc']:.4f}\n")
    print(f"\nResults written to {OUT_DIR}/ (results.json, summary_r2.csv, summary_full.csv)")
    print("\n=== pooled R2 by scenario ===")
    print(f"{'method':10s} " + " ".join(f"{s:>9s}" for s in SCEN))
    for nm in report:
        print(f"{nm:10s} " + " ".join(f"{allres[nm]['agg'][sc]['all']['r2']:9.3f}" for sc in SCEN))


if __name__ == "__main__":
    main()
