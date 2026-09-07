"""Repository hygiene: the released tree must carry no proprietary artefact.

These tests are a safety net, not a substitute for review. They run over the working
tree, so they also catch a file added by accident after the initial release.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".venv", "venv"}

TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".cff",
                 ".toml", ".sbatch", ".sh", ".cfg", ".ini"}


def iter_files():
    for p in ROOT.rglob("*"):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            yield p


def test_no_las_family_file_in_the_tree():
    bad = [p for p in iter_files()
           if p.suffix.lower() in {".las", ".dlis", ".lis"}
           and "examples/synthetic" not in p.as_posix()]
    assert not bad, f"LAS-family files present outside the synthetic example: {bad}"


def test_no_reidentification_table():
    pat = re.compile(r"(anonymi[sz]ation.*map|well_matching)", re.I)
    bad = [p for p in iter_files() if pat.search(p.name)]
    assert not bad, f"re-identification table present: {bad}"


def test_no_machine_specific_absolute_paths():
    """No /gpfs, /home/<user>, C:\\Users or /Users path may survive in the sources."""
    pat = re.compile(r"(/gpfs/|/home/[a-z0-9_-]+/|[A-Z]:\\\\Users\\\\|/Users/[a-z0-9_-]+/)", re.I)
    offenders = []
    for p in iter_files():
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if p.name == "test_privacy_guards.py":       # this file names the patterns
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:100]}")
    assert not offenders, "absolute paths found:\n" + "\n".join(offenders)


def test_released_results_carry_no_well_identifier():
    """Every released CSV must be keyed by fold index or by an anonymous W-id."""
    real_name = re.compile(r"\b[A-Z][A-Z-]{4,}-\d+[A-Z]?\b")
    anon_ok = re.compile(r"^W\d{2}$")
    offenders = []
    for p in (ROOT / "results").rglob("*.csv"):
        with open(p, newline="") as f:
            for row in csv.reader(f):
                for cell in row:
                    c = cell.strip()
                    if anon_ok.match(c) or not c:
                        continue
                    if real_name.match(c):
                        offenders.append(f"{p.relative_to(ROOT)}: {c}")
    assert not offenders, "well-like identifiers in released results:\n" + "\n".join(
        sorted(set(offenders))[:20])


def test_no_obvious_secret_material():
    pat = re.compile(
        r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[bap]-[A-Za-z0-9-]{10,}"
        r"|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY)")
    offenders = []
    for p in iter_files():
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if p.name == "test_privacy_guards.py":
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if pat.search(text):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"possible secret material: {offenders}"


def test_gitignore_blocks_the_dangerous_classes():
    ig = (ROOT / ".gitignore").read_text()
    for pattern in ["*.las", "well_matching.csv", "well_anonymization_map.csv",
                    "results/pools/", "results/pools_seeded/", ".env", "*.pkl", "*.pt"]:
        assert pattern in ig, f".gitignore is missing {pattern!r}"
