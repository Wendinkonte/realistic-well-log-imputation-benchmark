#!/usr/bin/env python
"""Run ONE method on ONE scenario, and print the pooled metrics.

The smallest useful entry point: handy for a smoke test, for inspecting a single
method's behaviour, or as a template for a new imputer.

Usage:
    python scripts/run_single_experiment.py --model mice --scenario profile
    python scripts/run_single_experiment.py --model unet --scenario blackout \
        --folds 2 --epochs 5 --seed 0

Note: SAITS and BRITS require ``--epochs >= 4``. Their patience is
``max(3, epochs // 3)`` and PyPOTS asserts ``patience < epochs``. The published run uses
100 epochs, so this only bites on very short smoke tests.
"""
import argparse
import json

import _bootstrap_path  # noqa: F401
from wellog_imputation.evaluation.harness import run_model
from wellog_imputation.metrics.pooled import LOGS
from wellog_imputation.models.factory import factory
from wellog_imputation.models.registry import FAMILY, SCENARIOS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="mean, locf, rf, xgb, mice, saits, brits, unet, ae, gnn, stgnn")
    ap.add_argument("--scenario", default=None, choices=SCENARIOS + [None],
                    help="one scenario (default: all four)")
    ap.add_argument("--folds", type=int, default=None, help="use only the first N folds")
    ap.add_argument("--epochs", type=int, default=100,
                    help="neural models only. SAITS and BRITS use patience = "
                         "max(3, epochs // 3) and PyPOTS requires patience < epochs, so "
                         "they need --epochs >= 4; the paper uses 100.")
    ap.add_argument("--seed", type=int, default=0, help="master seed of the run")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--json", action="store_true",
                    help="print the aggregate as JSON on stdout instead of a table. "
                         "SAITS and BRITS: PyPOTS prints a banner and its training log to "
                         "stdout on its own, so use --json-out for machine-readable output "
                         "with those two.")
    ap.add_argument("--json-out", default=None, metavar="PATH",
                    help="write the aggregate as JSON to PATH. Unlike --json this is "
                         "unaffected by third-party logging, so it is the reliable option "
                         "for scripting. Missing values are written as the Python-style "
                         "literal NaN, matching results.json.")
    args = ap.parse_args()

    scenarios = [args.scenario] if args.scenario else list(SCENARIOS)
    nm = args.model.lower()
    # Keep our own progress output off stdout when it is carrying machine-readable data.
    verbose = not (args.json or args.json_out)
    if nm in ("gnn", "stgnn"):
        from wellog_imputation.models.spatial import run_spatial
        r = run_spatial(kind=nm, folds_subset=args.folds, epochs=args.epochs,
                        seed=args.seed, data_dir=args.data_dir, verbose=verbose)
    elif nm in ("saits", "brits"):
        r = run_model(factory(nm, epochs=args.epochs), nm.upper(), scenarios=scenarios,
                      folds_subset=args.folds, seed=args.seed, data_dir=args.data_dir,
                      verbose=verbose)
    elif nm in ("unet", "ae"):
        r = run_model(factory(nm, epochs=args.epochs, seed=args.seed), nm.upper(),
                      scenarios=scenarios, folds_subset=args.folds, seed=args.seed,
                      data_dir=args.data_dir, verbose=verbose)
    else:
        r = run_model(factory(nm), nm.upper(), scenarios=scenarios,
                      folds_subset=args.folds, seed=args.seed, data_dir=args.data_dir,
                      verbose=verbose)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(r["agg"], f, indent=2)
        print(f"aggregate written to {args.json_out}")
        return
    if args.json:
        print(json.dumps(r["agg"], indent=2))
        return
    print(f"\n=== {nm.upper()} ({FAMILY[nm]}) | seed={args.seed} ===")
    print(f"{'scenario':10s} {'log':6s} {'R2':>8s} {'MAE':>10s} {'RMSE':>10s} {'n':>8s}")
    for sc in r["agg"]:
        for lg in (["all"] + LOGS):
            a = r["agg"][sc][lg]
            print(f"{sc:10s} {lg:6s} {a['r2']:8.3f} {a['mae']:10.4f} {a['rmse']:10.4f} {a['n']:8d}")


if __name__ == "__main__":
    main()
