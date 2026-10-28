"""The released tables must describe the published protocol exactly.

Three things are checked here, all cheap and all easy to break by accident when the
repository is pruned for release:

* the 19-fold leave-one-well-out structure survives in the v1.0 per-fold metrics
  (original submission, kept for transparency);
* the 18-fold structure of the revision R1 survives in results/revision_r1/;
* no manuscript or LaTeX source has crept back into a code-and-reproducibility tree.
"""
import csv
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
R1 = RESULTS / "revision_r1"
FOLD_SPECS = ROOT / "configs" / "benchmark"

# v1.0, original submission
N_WELLS = 19
N_SEEDS = 5
N_SCENARIOS = 4

# revision R1
N_WELLS_R1 = 18
R1_SCENARIOS = {"single", "block", "profile", "blackout", "blackout_emp"}


def _rows(path):
    return list(csv.DictReader(path.open(newline="")))


@pytest.mark.parametrize("name,n", [("folds_loo_19wells.yaml", N_WELLS),
                                    ("folds_loo_18wells.yaml", N_WELLS_R1)])
def test_fold_specification_describes_leave_one_well_out_folds(name, n):
    spec = yaml.safe_load((FOLD_SPECS / name).read_text())
    assert spec["n_wells"] == n
    assert spec["n_folds"] == n
    wells = spec["wells"]
    assert wells == [f"W{i:02d}" for i in range(1, n + 1)]
    assert len(spec["folds"]) == n
    for fold in spec["folds"].values():
        assert len(fold["test"]) == 1
        assert len(fold["train"]) == n - 1
        # a well is held out or trained on, never both: that is the whole point
        assert set(fold["test"]).isdisjoint(fold["train"])
        assert set(fold["test"]) | set(fold["train"]) == set(wells)


def test_r1_fold_specification_matches_the_error_stats_numbering():
    """W{fold+1:02d} is the well held out in fold ``fold``, as in the R1 export."""
    spec = yaml.safe_load((FOLD_SPECS / "folds_loo_18wells.yaml").read_text())
    for f in range(N_WELLS_R1):
        assert spec["folds"][f"fold_{f}"]["test"] == [f"W{f + 1:02d}"]
    pairs = {(int(r["fold"]), r["well"]) for r in _rows(R1 / "error_stats" / "fold_train_sigma.csv")}
    assert pairs == {(f, f"W{f + 1:02d}") for f in range(N_WELLS_R1)}


@pytest.mark.parametrize("name", ["raw_metrics.csv", "raw_metrics_full.csv"])
def test_released_per_fold_metrics_cover_all_19_folds(name):
    rows = list(csv.DictReader((RESULTS / name).open(newline="")))
    folds = {int(r["fold"]) for r in rows}
    assert folds == set(range(N_WELLS)), f"{name} does not carry all 19 folds"
    assert {int(r["seed"]) for r in rows} == set(range(N_SEEDS))
    assert len({r["scenario"] for r in rows}) == N_SCENARIOS
    # every (seed, method, scenario, log) cell is complete over the folds
    cells = {}
    for r in rows:
        cells.setdefault((r["seed"], r["method"], r["scenario"], r["log"]), set()).add(int(r["fold"]))
    incomplete = [k for k, v in cells.items() if v != set(range(N_WELLS))]
    assert not incomplete, f"{name}: incomplete folds for {incomplete[:5]}"


def test_significance_blocks_on_19_folds():
    rows = list(csv.DictReader((RESULTS / "significance" / "friedman.csv").open(newline="")))
    assert rows, "friedman.csv is empty"
    assert {int(r["seed"]) for r in rows} == set(range(N_SEEDS))
    assert len({r["scenario"] for r in rows}) == N_SCENARIOS


def test_r1_per_well_tables_cover_all_18_wells():
    rows = _rows(R1 / "per_well_nrmse.csv")
    assert {int(r["well_index"]) for r in rows} == set(range(N_WELLS_R1))
    assert {r["scenario"] for r in rows} == R1_SCENARIOS
    cells = {}
    for r in rows:
        cells.setdefault((r["method"], r["scenario"]), set()).add(int(r["well_index"]))
    incomplete = [k for k, v in cells.items() if v != set(range(N_WELLS_R1))]
    assert not incomplete, f"per_well_nrmse.csv: incomplete wells for {incomplete[:5]}"


@pytest.mark.parametrize("name", ["friedman_wellwise.csv", "wilcoxon_wellwise_holm.csv",
                                  "wilcoxon_revision_holm.csv"])
def test_r1_significance_blocks_on_18_wells(name):
    rows = _rows(R1 / name)
    assert rows, f"{name} is empty"
    assert {int(r["n_wells"]) for r in rows} == {N_WELLS_R1}
    assert {r["scenario"] for r in rows} <= R1_SCENARIOS


def test_r1_main_table_covers_five_scenarios_and_five_seeds():
    rows = _rows(R1 / "table_main_revision.csv")
    assert {r["scenario"] for r in rows} == R1_SCENARIOS
    assert {int(r["n_seeds"]) for r in rows} == {N_SEEDS}


def test_no_manuscript_or_latex_source_in_the_tree():
    """This is a code and reproducibility repository; article sources live elsewhere."""
    bad_suffixes = {".tex", ".bib", ".bst", ".sty", ".cls", ".aux", ".toc", ".fls",
                    ".fdb_latexmk", ".bbl", ".blg", ".synctex", ".dvi"}
    skip = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".venv", "venv"}
    offenders = [
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file()
        and not any(part in skip for part in p.parts)
        and (p.suffix.lower() in bad_suffixes or p.name.endswith(".synctex.gz"))
    ]
    assert not offenders, f"manuscript/LaTeX source present: {offenders}"


def test_every_released_pdf_is_a_generated_figure():
    """Only matplotlib output belongs here: no article, supplement or cover letter."""
    header = re.compile(rb"Matplotlib")
    offenders = []
    for p in RESULTS.rglob("*.pdf"):
        blob = p.read_bytes()
        if not blob.startswith(b"%PDF"):
            offenders.append(f"{p.relative_to(ROOT)}: not a PDF")
        elif not header.search(blob[:4096]) and not header.search(blob[-4096:]):
            offenders.append(f"{p.relative_to(ROOT)}: not produced by matplotlib")
    assert not offenders, "unexpected PDFs:\n" + "\n".join(offenders)
