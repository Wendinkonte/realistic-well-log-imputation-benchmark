"""The eleven imputation methods, grouped by family.

``tabular``    MEAN, RF, XGBoost, MICE
``sequential`` LOCF, SAITS, BRITS, U-Net, AE
``spatial``    GNN, ST-GNN

Note that :mod:`wellog_imputation.models.spatial` pulls in PyTorch at import time; it is
therefore NOT imported eagerly here, so the tabular and masking code stays importable in
a minimal environment.
"""
from .factory import SPATIAL_METHODS, WITHIN_WELL_METHODS, factory
from .registry import DISPLAY, FAMILY, LEARNED, ORDER, PAPER_FAMILY, SCENARIOS
from .sequential import AEImputer, LOCFImputer, SeqImputer, UNetImputer
from .tabular import MeanImputer, MICEImputer, TabularImputer

__all__ = ["factory", "WITHIN_WELL_METHODS", "SPATIAL_METHODS",
           "ORDER", "FAMILY", "PAPER_FAMILY", "DISPLAY", "LEARNED", "SCENARIOS",
           "MeanImputer", "TabularImputer", "MICEImputer",
           "LOCFImputer", "SeqImputer", "AEImputer", "UNetImputer"]
