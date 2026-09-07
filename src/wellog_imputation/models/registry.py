"""Single source of truth for method names, ordering and family labels.

In the original code base this table was duplicated in seven modules.  It is centralised
here; the VALUES are unchanged, so every table, figure and CSV keeps the exact labels of
the published run.

.. note::
   ``FAMILY`` labels MEAN and LOCF as ``"baseline"``, which is what every released CSV
   contains.  The article groups MEAN with the tabular family and LOCF with the
   sequential family, on the grounds of the information each one uses.  The mapping
   between the two conventions is given in docs/METHODS_MAPPING.md.  The code labels are
   deliberately left untouched so released results stay byte-comparable.
"""

__all__ = ["ORDER", "FAMILY", "DISPLAY", "PAPER_FAMILY", "LEARNED", "SCENARIOS"]

#: Display order of the eleven methods, used by every table and figure.
ORDER = ["mean", "locf", "rf", "xgb", "mice", "saits", "brits", "unet", "ae", "gnn", "stgnn"]

#: Family label as written into every released results file.
FAMILY = {"mean": "baseline", "locf": "baseline", "rf": "tabular", "xgb": "tabular",
          "mice": "tabular", "saits": "sequential", "brits": "sequential",
          "unet": "sequential", "ae": "sequential", "gnn": "spatial", "stgnn": "spatial"}

#: Family grouping as presented in the article (documentation only; not used to
#: compute or aggregate anything).
PAPER_FAMILY = {"mean": "tabular", "rf": "tabular", "xgb": "tabular", "mice": "tabular",
                "locf": "sequential", "saits": "sequential", "brits": "sequential",
                "unet": "sequential", "ae": "sequential",
                "gnn": "spatial", "stgnn": "spatial"}

#: Human-readable names used in figures and result tables.
DISPLAY = {"mean": "MEAN", "locf": "LOCF", "rf": "RF", "xgb": "XGBoost", "mice": "MICE",
           "saits": "SAITS", "brits": "BRITS", "unet": "U-Net", "ae": "AE",
           "gnn": "GNN", "stgnn": "ST-GNN"}

#: The nine learned methods (everything except the two parameter-free baselines).
LEARNED = ["rf", "xgb", "mice", "saits", "brits", "unet", "ae", "gnn", "stgnn"]

#: The four missingness regimes, in reporting order.
SCENARIOS = ["single", "block", "profile", "blackout"]
