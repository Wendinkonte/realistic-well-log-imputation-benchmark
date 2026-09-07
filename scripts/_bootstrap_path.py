"""Make ``wellog_imputation`` importable when the package has not been pip-installed.

Every script imports this first, so ``python scripts/run_benchmark.py`` works straight
from a fresh clone.  After ``pip install -e .`` it is a no-op.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
