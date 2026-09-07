"""Shared figure style and well anonymisation.

Purpose: consistency across every figure of the paper, plus submission compliance.

* fonts >= 8 pt at final size (a ~13.5 cm text width in preprint layout);
* ONE colour per method, identical in every figure (:data:`METHOD_COLOR`);
* ONE marker per method, so figures survive black-and-white printing
  (:data:`METHOD_MARKER`);
* a colour-blind-safe qualitative palette (Okabe-Ito, extended after Paul Tol);
* well anonymisation through :func:`anon`.

Typical use::

    from wellog_imputation.utils import figstyle
    figstyle.apply()
    ax.plot(..., color=figstyle.METHOD_COLOR[m], marker=figstyle.METHOD_MARKER[m])
    ax.set_title(figstyle.anon(well_id))     # never a raw well identifier

Anonymisation
-------------
:func:`anon` maps a well identifier to a short label (``W01`` .. ``W19``) using the CSV
at the configured ``anon_map`` path, whose columns are ``anon_id`` and ``well_name``.

That table is a re-identification key.  It is NOT part of this repository and is
git-ignored; without it :func:`anon` simply returns the identifier it was given, which
is the right behaviour when the identifiers in your own data are not sensitive.

When a map IS present, an identifier missing from it raises by default
(``strict=True``), rather than silently falling through to the raw name.  That closes
the path by which a real identifier could reach a published figure.  Set the environment
variable ``WELLOG_ANON_STRICT=0`` to restore the permissive behaviour.
"""
import csv
import os

import matplotlib as mpl

from ..config import get_path
from ..models.registry import DISPLAY, FAMILY, ORDER

__all__ = ["apply", "ORDER", "DISPLAY", "DISP", "FAMILY", "METHOD_COLOR", "METHOD_MARKER",
           "FAMILY_MARKER", "anon", "load_anon_map", "has_anon_map"]


# ------------------------------------------------------------------ rcParams
def apply():
    """Common typography: >= 8 pt, serif to match the body text of the article."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#222222",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.dpi": 300,      # rasterised elements (hexbins) at 300 dpi
        "figure.dpi": 150,
        "pdf.fonttype": 42,      # editable vector fonts (TrueType)
    })


# ------------------------------------------------------------- method palette
# Eleven methods -> eleven distinct, colour-blind-safe hues (Okabe-Ito + Tol muted).
# One colour per method, frozen and identical across all figures.
METHOD_COLOR = {
    "mean":  "#888888",   # grey       (baseline)
    "locf":  "#000000",   # black      (baseline)
    "rf":    "#DDCC77",   # sand       (tabular)
    "xgb":   "#E69F00",   # orange O-I (tabular)
    "mice":  "#117733",   # dark green (tabular)
    "saits": "#56B4E9",   # sky blue   (sequential)
    "brits": "#CC79A7",   # mauve O-I  (sequential)
    "unet":  "#D55E00",   # vermillion (sequential)
    "ae":    "#332288",   # indigo     (sequential)
    "gnn":   "#44AA99",   # teal       (spatial)
    "stgnn": "#882255",   # burgundy   (spatial)
}

#: One distinct marker per method, to disambiguate in greyscale.
METHOD_MARKER = {
    "mean":  "o", "locf": "s", "rf": "D", "xgb": "^", "mice": "v",
    "saits": "P", "brits": "X", "unet": "*", "ae": "h", "gnn": "<", "stgnn": ">",
}

#: Marker by FAMILY, for figures whose message is the family rather than the method.
FAMILY_MARKER = {"baseline": "o", "tabular": "s", "sequential": "^", "spatial": "D"}

#: Backwards-compatible alias of the display-name table.
DISP = DISPLAY


# ---------------------------------------------------------------- anonymisation
def load_anon_map(path=None):
    """Load ``{well_name: anon_id}``.  Returns an empty dict when no map is configured."""
    p = get_path("anon_map", path)
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        return {row["well_name"]: row["anon_id"] for row in csv.DictReader(f)}


_ANON = load_anon_map()

#: True when an anonymisation map was found at import time.
ANON = _ANON          # kept under the original name for compatibility


def has_anon_map():
    """Whether an anonymisation map is configured and was loaded."""
    return bool(_ANON)


def _strict_default():
    return os.environ.get("WELLOG_ANON_STRICT", "1") not in ("0", "false", "False")


def anon(well, strict=None):
    """Map a well identifier to its short anonymous label.

    Handles the identifier variants found in the corpus by stripping an archive index
    suffix before lookup, then falling back to the full identifier.

    Parameters
    ----------
    well : str
    strict : bool, optional
        When a map IS loaded and the identifier is absent from it, raise ``KeyError``
        instead of returning the raw identifier.  Defaults to True, or to the value of
        the ``WELLOG_ANON_STRICT`` environment variable.

    Returns
    -------
    str
        The anonymous label, or -- when no map is configured -- ``well`` unchanged.
    """
    if not _ANON:
        return str(well)                      # no map: identifiers are used as they are
    key = str(well).split("-14-")[0]
    hit = _ANON.get(key, _ANON.get(str(well)))
    if hit is not None:
        return hit
    if strict is None:
        strict = _strict_default()
    if strict:
        raise KeyError(
            f"well identifier {well!r} is absent from the anonymisation map; refusing to "
            f"fall back to the raw identifier in a figure. Add it to the map, or pass "
            f"strict=False / set WELLOG_ANON_STRICT=0 if the raw identifier is not "
            f"sensitive."
        )
    return key
