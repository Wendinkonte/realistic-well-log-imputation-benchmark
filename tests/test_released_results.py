"""The released tables must describe the published protocol exactly.

Two things are checked here, both cheap and both easy to break by accident when the
repository is pruned for release:

* the 19-fold leave-one-well-out structure survives in the released per-fold metrics;
* no manuscript or LaTeX source has crept back into a code-and-reproducibility tree.
"""
import csv
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FOLD_SPEC = ROOT / "configs" / "benchmark" / "folds_loo_19wells.yaml"

N_WELLS = 19
N_SEEDS = 5
N_SCENARIOS = 4


def test_fold_specification_describes_19_leave_one_well_out_folds():
    spec = yaml.safe_load(FOLD_SPEC.read_text())
    assert spec["n_wells"] == N_WELLS
    assert spec["n_folds"] == N_WELLS
    wells = spec["wells"]
    assert wells == [f"W{i:02d}" for i in range(1, N_WELLS + 1)]
    assert len(spec["folds"]) == N_WELLS
    for fold in spec["folds"].values():
        assert len(fold["test"]) == 1
        assert len(fold["train"]) == N_WELLS - 1
        # a well is held out or trained on, never both: that is the whole point
        assert set(fold["test"]).isdisjoint(fold["train"])
        assert set(fold["test"]) | set(fold["train"]) == set(wells)


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
