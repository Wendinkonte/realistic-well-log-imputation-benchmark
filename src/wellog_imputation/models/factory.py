"""Factory mapping a method name to a zero-argument model constructor."""
from .sequential import (AEImputer, InterpImputer, LOCFImputer, LOCFWindowImputer,
                         SeqImputer, UNetImputer)
from .tabular import MeanImputer, MICEImputer, TabularImputer, XGBChainImputer

__all__ = ["factory", "WITHIN_WELL_METHODS", "SPATIAL_METHODS", "REVISION_WITHIN_WELL",
           "REVISION_SPATIAL"]

#: Methods driven by :func:`wellog_imputation.evaluation.harness.run_model`.
WITHIN_WELL_METHODS = ["mean", "locf", "rf", "xgb", "mice", "saits", "brits", "unet", "ae"]

#: Methods driven by :func:`wellog_imputation.models.spatial.run_spatial`.
SPATIAL_METHODS = ["gnn", "stgnn"]

#: Revision R1 methods driven by run_model (baselines, training-protocol tests).
REVISION_WITHIN_WELL = ["locf_win", "ridge", "knn", "interp_linear", "interp_pchip", "xgb_miss",
                        "xgb_chain_asc", "xgb_chain_desc", "saits_mix", "brits_mix"]
#: Revision R1 methods driven by run_spatial (spatial baselines and graph controls).
REVISION_SPATIAL = ["idw", "nnwell", "okrige", "gnn_noneigh", "stgnn_noneigh",
                    "gnn_shuffle", "stgnn_shuffle"]


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
    # ---- revision R1
    if name == "locf_win": return lambda: LOCFWindowImputer()
    if name == "ridge": return lambda: TabularImputer("ridge")
    if name == "knn":   return lambda: TabularImputer("knn")
    if name == "interp_linear": return lambda: InterpImputer("linear")
    if name == "interp_pchip":  return lambda: InterpImputer("pchip")
    if name == "xgb_miss":      return lambda: TabularImputer("xgb", train_on="corrupted")
    if name == "xgb_chain_asc":  return lambda: XGBChainImputer("ascending")
    if name == "xgb_chain_desc": return lambda: XGBChainImputer("descending")
    if name == "saits_mix": return lambda: SeqImputer("saits", train_on="corrupted", **kw)
    if name == "brits_mix": return lambda: SeqImputer("brits", train_on="corrupted", **kw)
    raise ValueError(name)
