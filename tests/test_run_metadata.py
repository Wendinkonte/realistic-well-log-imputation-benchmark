"""Replication driver: fold count read from the dataset, provenance metadata."""
import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_run_replication():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("run_replication", SCRIPTS / "run_replication.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_progress_counts_the_real_number_of_folds(synthetic_dataset, tmp_path):
    rr = _load_run_replication()
    n = rr.n_folds_of(synthetic_dataset)
    assert n == 4                                   # not the 19 of the original corpus
    pool = tmp_path / "pools_seeded"
    (pool / "seed0" / "mean").mkdir(parents=True)
    (pool / "seed0" / "mean" / "fold02.npz").write_bytes(b"")
    line = rr.progress_line(0, "mean", str(pool), range(n))
    assert "1/4 target folds already done" in line


def test_run_metadata_records_the_commit(tmp_path):
    from wellog_imputation.utils.provenance import git_commit, write_run_metadata
    p = tmp_path / "meta" / "run_metadata.json"
    meta = write_run_metadata(str(p), seed=3)
    assert json.loads(p.read_text()) == meta
    assert meta["seed"] == 3 and meta["git_commit"] == git_commit()[0]
    if meta["git_commit"] is not None:              # outside a checkout it is None
        assert len(meta["git_commit"]) == 40
