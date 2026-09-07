"""Spatial family smoke test, on synthetic data only.

The spatial models are the only ones that read a neighbouring well, so they are the only
ones needing a coordinate table and the well graph. These tests check that the whole
spatial path — grid construction, identifier normalisation, coordinate lookup, adjacency,
the anti-leakage blanking and one full fold — works from a fresh clone with no
proprietary data and no alias file.

They skip when PyTorch is absent, so the light CI job stays green.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the spatial family needs PyTorch")

from wellog_imputation.models.spatial import (SpatialGrid, _key, _load_aliases,  # noqa: E402
                                              build_adjacency, load_coords)


@pytest.fixture
def synthetic_spatial(tmp_path):
    """A miniature corpus with a coordinate table, in the pipeline's output format."""
    wells = [f"SYN{i+1:02d}" for i in range(5)]
    r = np.random.default_rng(11)
    T, L, per_well = 64, 400, 4
    logs_by_well, X, wcol, starts = {}, [], [], []
    for wi, w in enumerate(wells):
        depth = 100.0 + np.arange(L) * 0.1524
        arr = r.normal(size=(L, 3))
        logs_by_well[w] = np.concatenate([depth[:, None], arr], axis=1)
        for k in range(per_well):
            s = k * 16
            X.append(arr[s:s+T]); wcol.append(w); starts.append(s)
    np.savez_compressed(tmp_path / "well_logs.npz", **logs_by_well,
                        logs=np.array(["depth", "GR", "RHOB", "NPHI"]))
    np.savez_compressed(tmp_path / "slices.npz",
                        X=np.asarray(X, np.float32), wells=np.array(wcol),
                        starts=np.array(starts), logs=np.array(["GR", "RHOB", "NPHI"]),
                        slice_len=T, stride=16)
    folds = {f"fold_{i}": {"test": [w], "train": [x for x in wells if x != w]}
             for i, w in enumerate(wells)}
    with open(tmp_path / "folds_loo.json", "w") as f:
        json.dump({"wells": wells, "n_folds": len(wells), "folds": folds}, f)
    coords = tmp_path / "well_coordinates.csv"
    with open(coords, "w") as f:
        f.write("Name,SIGLE,Surf_LocX_L93,Surf_LocY_L93\n")
        for i, w in enumerate(wells):
            f.write(f"{w},{w},{100000.0 + 2000.0*(i % 3):.1f},{200000.0 + 2000.0*(i // 3):.1f}\n")
    return tmp_path, coords, wells


def test_no_alias_file_is_required():
    """A corpus that spells its wells consistently needs no alias table at all."""
    assert _load_aliases() == {}, (
        "an alias file is present in this checkout; the public repository must ship none"
    )


def test_identifier_normalisation_is_generic():
    """_key applies only generic rules -- no real well name is baked into the source."""
    assert _key("SYN01") == "SYN01"
    assert _key("SOME-WELL-1-14-9999-") == "SOME-WELL-1"   # archive suffix stripped
    assert _key("some_well-2") == "SOME-WELL-2"            # separators normalised
    assert _key("SOME-WELL-3D") == "SOME-WELL-3"           # deviation letter dropped
    assert _key("A-SS-B-1") == "A-SOUS-B-1"                # generic French abbreviation


def test_coordinates_load_and_missing_file_raises_clearly(synthetic_spatial, tmp_path):
    data_dir, coords, wells = synthetic_spatial
    P = load_coords(wells, coords)
    assert P.shape == (len(wells), 2)
    assert np.isfinite(P).all()
    with pytest.raises(FileNotFoundError, match="coordinate table"):
        load_coords(wells, tmp_path / "does_not_exist.csv")


def test_adjacency_is_row_normalised_knn():
    P = np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0], [1000.0, 1000.0]])
    A = build_adjacency(P, k=2)
    assert A.shape == (4, 4)
    np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-5)
    assert (A >= 0).all()
    assert (np.diag(A) > 0).all(), "self-loops must survive normalisation"


def test_grid_builds_from_synthetic_corpus(synthetic_spatial):
    data_dir, coords, wells = synthetic_spatial
    g = SpatialGrid(data_dir=data_dir, coords_csv=coords)
    assert g.wells == wells
    assert g.G.shape[0] == len(wells) and g.G.shape[2] == 3
    assert g.A.shape == (len(wells), len(wells))
    assert set(g.widx) == set(wells)
    block = g.block(g.gstart(wells[0], 0), 64)
    assert block.shape == (len(wells), 64, 3)


def test_full_spatial_fold_runs_and_respects_anti_leakage(synthetic_spatial):
    """One complete fold of the spatial model, end to end, on synthetic data."""
    from wellog_imputation.models.spatial import run_spatial
    data_dir, coords, _ = synthetic_spatial
    r = run_spatial(kind="stgnn", folds_subset=1, epochs=2, verbose=False,
                    data_dir=data_dir, coords_csv=coords)
    assert set(r["agg"]) == {"single", "block", "profile", "blackout"}
    assert r["agg"]["profile"]["all"]["n"] > 0
    assert np.isfinite(r["agg"]["profile"]["all"]["mae"])


def test_leak_toggle_changes_what_the_model_can_see(synthetic_spatial):
    """leak=False blanks the held-out well's row; leak=True leaves it in place."""
    from wellog_imputation.evaluation.harness import ChannelScaler
    from wellog_imputation.models.spatial import SpatialImputer
    data_dir, coords, wells = synthetic_spatial
    g = SpatialGrid(data_dir=data_dir, coords_csv=coords)
    s = np.load(data_dir / "slices.npz", allow_pickle=True)
    X = s["X"].astype(np.float32); w = s["wells"].astype(str); st = s["starts"]
    held = wells[0]
    sc = ChannelScaler().fit(X[w != held])

    clean = SpatialImputer(kind="gnn", epochs=1).fit(
        g, set(wells[1:]), list(range(len(X))), X, w, st, sc, test_well=held)
    leaky = SpatialImputer(kind="gnn", epochs=1).fit(
        g, set(wells), list(range(len(X))), X, w, st, sc, test_well=None)

    row = g.widx[held]
    assert np.isnan(clean.Gz[row]).all(), \
        "anti-leakage: the held-out well's row must be blanked during training"
    assert not np.isnan(leaky.Gz[row]).all(), \
        "leak=True: the row must remain visible (this is the ablation, never the protocol)"
