"""Path and run configuration.

Every filesystem location used by the package is resolved here, so that no module
carries a machine-specific absolute path.  Resolution order, highest priority first:

1. an explicit argument passed to the calling function;
2. an environment variable (``WELLOG_DATA_DIR``, ``WELLOG_RESULTS_DIR``, ...);
3. the ``configs/paths.yaml`` file at the project root, if present;
4. a default relative to the project root (``data/processed``, ``results``, ...).

Copy ``configs/paths.example.yaml`` to ``configs/paths.yaml`` and edit it to point at
your own LAS corpus.  ``paths.yaml`` is git-ignored, so local paths never leave the
machine they were written on.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["PROJECT_ROOT", "paths", "get_path", "Paths"]


def _find_project_root() -> Path:
    """Walk upwards from this file until a directory holding ``configs/`` is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    # installed as a package with no source tree alongside: fall back to the cwd
    return Path.cwd()


PROJECT_ROOT = _find_project_root()

# key -> (environment variable, default relative to PROJECT_ROOT)
_SPEC = {
    "las_dir":     ("WELLOG_LAS_DIR",     "data/raw"),
    "data_dir":    ("WELLOG_DATA_DIR",    "data/processed"),
    "results_dir": ("WELLOG_RESULTS_DIR", "results"),
    "figures_dir": ("WELLOG_FIGURES_DIR", "results/figures"),
    "coords_csv":  ("WELLOG_COORDS_CSV",  "data/processed/well_coordinates.csv"),
    "anon_map":    ("WELLOG_ANON_MAP",    "data/processed/well_anonymization_map.csv"),
}


def _load_yaml_paths() -> dict:
    cfg = PROJECT_ROOT / "configs" / "paths.yaml"
    if not cfg.is_file():
        return {}
    try:
        import yaml
    except ImportError:  # PyYAML is optional; env vars and defaults still work
        return {}
    with open(cfg) as fh:
        loaded = yaml.safe_load(fh) or {}
    return loaded.get("paths", loaded) if isinstance(loaded, dict) else {}


_YAML = _load_yaml_paths()


class Paths:
    """Lazily resolved project paths.  Attribute access returns a :class:`pathlib.Path`."""

    def __getattr__(self, key: str) -> Path:
        if key not in _SPEC:
            raise AttributeError(f"unknown path key {key!r}; known keys: {sorted(_SPEC)}")
        return get_path(key)

    def __dir__(self):
        return sorted(_SPEC)

    def as_dict(self) -> dict:
        return {k: get_path(k) for k in _SPEC}


def get_path(key: str, override: str | os.PathLike | None = None) -> Path:
    """Resolve one configured path.

    Parameters
    ----------
    key
        One of ``las_dir``, ``data_dir``, ``results_dir``, ``figures_dir``,
        ``coords_csv``, ``anon_map``.
    override
        If given, returned as-is (expanded).  Lets a caller pass ``--data-dir`` straight
        through without having to branch on ``None``.
    """
    if override is not None:
        return Path(override).expanduser()
    if key not in _SPEC:
        raise KeyError(f"unknown path key {key!r}; known keys: {sorted(_SPEC)}")
    env_var, default = _SPEC[key]
    value = os.environ.get(env_var) or _YAML.get(key)
    if value is None:
        return PROJECT_ROOT / default
    value = Path(str(value)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


paths = Paths()
