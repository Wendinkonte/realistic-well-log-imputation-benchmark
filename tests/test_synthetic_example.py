"""The synthetic corpus: it must generate, load, and drive the pipeline."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "examples" / "synthetic" / "make_synthetic.py"


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    subprocess.run([sys.executable, str(GEN), "--out", str(out), "--wells", "4",
                    "--samples", "600"], check=True, capture_output=True)
    return out


def test_generator_writes_every_artefact(generated):
    proc = generated / "processed"
    for name in ["well_logs.npz", "slices.npz", "folds_loo.json", "well_coordinates.csv"]:
        assert (proc / name).is_file(), f"{name} was not written"
    las = sorted((generated / "las").glob("*.las"))
    assert len(las) == 4


def test_synthetic_wells_are_obviously_not_real(generated):
    """Nothing in the synthetic corpus may look like a real identifier or location."""
    with open(generated / "processed" / "folds_loo.json") as f:
        wells = json.load(f)["wells"]
    assert all(w.startswith("SYN") for w in wells)
    coords = (generated / "processed" / "well_coordinates.csv").read_text()
    assert "SYN01" in coords
    # round, grid-placed coordinates, nowhere near a real projected extent
    for line in coords.strip().splitlines()[1:]:
        x, y = (float(v) for v in line.split(",")[2:4])
        assert x % 100 == 0 and y % 100 == 0


def test_slices_have_the_expected_shape(generated):
    z = np.load(generated / "processed" / "slices.npz", allow_pickle=True)
    X = z["X"]
    assert X.ndim == 3 and X.shape[1] == 256 and X.shape[2] == 3
    assert X.shape[0] == len(z["wells"]) == len(z["starts"])
    assert np.isfinite(X).all(), "windows must be intact by construction"


def test_las_files_are_readable_and_round_trip(generated):
    lasio = pytest.importorskip("lasio")
    f = sorted((generated / "las").glob("*.las"))[0]
    las = lasio.read(str(f))
    mnemonics = {c.mnemonic.upper() for c in las.curves}
    assert {"GR", "RHOB", "NPHI"} <= mnemonics
    assert len(las.index) > 0


def test_pipeline_rebuilds_the_dataset_from_the_las(generated, tmp_path):
    pytest.importorskip("lasio")
    from wellog_imputation.data.pipeline import build_dataset
    out = tmp_path / "processed"
    summary = build_dataset(las_dir=generated / "las", out_dir=out, verbose=False)
    assert summary["n_wells_read"] == 4
    assert summary["n_wells_triplet"] == 4
    assert summary["n_slices"] > 0
    assert (out / "slices.npz").is_file()
    assert (out / "folds_loo.json").is_file()


def test_end_to_end_run_on_the_synthetic_corpus(generated):
    """The smallest complete experiment: masking, scaling, imputation, metrics."""
    from wellog_imputation.evaluation.harness import run_model
    from wellog_imputation.models.factory import factory
    r = run_model(factory("mean"), "MEAN", folds_subset=2, verbose=False,
                  data_dir=generated / "processed")
    assert r["agg"]["profile"]["all"]["n"] > 0
    assert np.isfinite(r["agg"]["profile"]["all"]["r2"])
