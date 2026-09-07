"""Factory mapping a method name to a zero-argument model constructor."""
from .sequential import AEImputer, LOCFImputer, SeqImputer, UNetImputer
from .tabular import MeanImputer, MICEImputer, TabularImputer

__all__ = ["factory", "WITHIN_WELL_METHODS", "SPATIAL_METHODS"]

#: Methods driven by :func:`wellog_imputation.evaluation.harness.run_model`.
WITHIN_WELL_METHODS = ["mean", "locf", "rf", "xgb", "mice", "saits", "brits", "unet", "ae"]

#: Methods driven by :func:`wellog_imputation.models.spatial.run_spatial`.
SPATIAL_METHODS = ["gnn", "stgnn"]


def factory(name, **kw):
    """Return a callable building a fresh imputer for ``name``.

    Extra keyword arguments (``epochs``, ``seed``, ...) are forwarded to the neural
    models only; the parameter-free and scikit-learn models ignore them, exactly as in
    the original implementation.
    """
    name = name.lower()
    if name == "mean":  return lambda: MeanImputer()
    if name == "locf":  return lambda: LOCFImputer()
    if name == "rf":    return lambda: TabularImputer("rf")
    if name == "xgb":   return lambda: TabularImputer("xgb")
    if name == "mice":  return lambda: MICEImputer()
    if name == "saits": return lambda: SeqImputer("saits", **kw)
    if name == "brits": return lambda: SeqImputer("brits", **kw)
    if name == "ae":    return lambda: AEImputer(**kw)
    if name == "unet":  return lambda: UNetImputer(**kw)
    raise ValueError(name)
