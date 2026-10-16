"""Revision R1: depth-step quality control and the revised missingness statistics."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wellog_imputation.data.pipeline import (STEP_M, check_depth_step, missingness_stats)

ROOT = Path(__file__).resolve().parent.parent


def test_regular_step_is_kept():
    ok, info = check_depth_step(1000 + np.arange(500) * STEP_M)
    assert ok and info["n_jumps"] == 0


def test_coarse_step_with_jump_is_excluded():
    d = np.concatenate([1741 + np.arange(200) * 0.5, 1741 + 200 * 0.5 + 72.5 + np.arange(400) * 0.5])
    ok, info = check_depth_step(d)
    assert not ok
    assert info["median_step"] == pytest.approx(0.5)
    assert info["n_jumps"] == 1 and info["jump_m"] == pytest.approx(72.5, abs=0.6)


def test_zero_step_is_excluded():
    d = 1000 + np.arange(100) * STEP_M
    d[50] = d[49]
    assert not check_depth_step(d)[0]


def _well(L, gr=(), rhob=(), nphi=()):
    """Arrays of length L with NaN on the given (start, end) runs."""
    r = {"depth": np.arange(L) * STEP_M}
    for name, runs in (("GR", gr), ("RHOB", rhob), ("NPHI", nphi)):
        a = np.ones(L)
        for s, e in runs:
            a[s:e] = np.nan
        r[name] = a
    return r


def test_missingness_samples_vs_intervals():
    # one long RHOB+NPHI interval (100 samples) and three short GR-only gaps (5 each)
    w = _well(400, gr=[(20, 25), (40, 45), (60, 65)], rhob=[(200, 300)], nphi=[(200, 300)])
    st = missingness_stats({"A": w}, ["A"])
    # samples: K=1 -> 15, K=2 -> 100  => 13.0 % / 87.0 %
    assert st["cooccur_pct_samples"]["1_logs"] == pytest.approx(13.0, abs=0.1)
    assert st["cooccur_pct_samples"]["2_logs"] == pytest.approx(87.0, abs=0.1)
    # intervals: 3 single-log, 1 two-log  => 75 % / 25 %
    assert st["cooccur_pct_intervals"]["1_logs"] == pytest.approx(75.0)
    assert st["cooccur_pct_intervals"]["2_logs"] == pytest.approx(25.0)
    assert st["cooccur_pct"] == st["cooccur_pct_samples"]           # backward-compatible key
    assert list(st["combinations"])[0] == "RHOB+NPHI"
    assert st["cooccur_prob_sample"]["2"] == pytest.approx(100 / 115)


def test_edge_intervals_are_separated_from_interior_gaps():
    # NPHI absent over the top 50 samples (edge) + one interior gap of 10
    w = _well(300, nphi=[(0, 50), (150, 160)])
    st = missingness_stats({"A": w}, ["A"])
    assert st["gap_length_m_all"]["n"] == 2
    assert st["gap_length_m_interior"]["n"] == 1
    assert st["gap_length_m_interior"]["p50"] == pytest.approx(round(10 * STEP_M, 1))


def test_pipeline_excludes_irregular_well(tmp_path):
    pytest.importorskip("lasio")
    spec = importlib.util.spec_from_file_location(
        "make_synthetic", ROOT / "examples" / "synthetic" / "make_synthetic.py")
    ms = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
    las_dir = tmp_path / "las"; las_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        d = 1000 + np.arange(600) * STEP_M
        ms.write_las(las_dir / f"SYN{i:02d}.las", f"SYN{i:02d}", d, ms.synth_well(rng, 600, i))
    # irregular well: 0.5 m step with a 72.5 m jump
    d = np.concatenate([1741 + np.arange(300) * 0.5, 1741 + 300 * 0.5 + 72.5 + np.arange(300) * 0.5])
    ms.write_las(las_dir / "SYNBAD.las", "SYNBAD", d, ms.synth_well(rng, 600, 9))

    from wellog_imputation.data.pipeline import build_dataset
    out = tmp_path / "proc"
    build_dataset(las_dir=str(las_dir), out_dir=str(out), verbose=False)
    folds = json.loads((out / "folds_loo.json").read_text())
    assert "SYNBAD" not in folds["wells"] and len(folds["wells"]) == 4
    rows = {r["well"]: r for r in csv.DictReader(open(out / "depth_qc.csv"))}
    assert rows["SYNBAD"]["status"] == "EXCLUDED" and rows["SYN00"]["status"] == "kept"
    ms_json = json.loads((out / "missingness_real.json").read_text())
    assert "cooccur_pct_intervals" in ms_json and "gap_length_m_interior" in ms_json
