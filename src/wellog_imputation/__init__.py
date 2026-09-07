"""Realistic Missingness Changes the Ranking of Well-Log Imputation Models.

Reference implementation of the benchmark: eleven imputation methods in three families,
four missingness regimes, leave-one-well-out cross-validation with fold-local scaling
and identical evaluation masks across methods.

Sub-packages
------------
``data``        LAS ingestion, cleaning, windowing, fold construction
``masking``     the four missingness-scenario generators
``models``      the eleven imputers (tabular / sequential / spatial)
``evaluation``  the leave-one-well-out harness
``metrics``     metric computation and pooled aggregation
``statistics``  multi-seed aggregation, significance testing, cluster bootstrap
``utils``       determinism, figure style, anonymisation
``config``      path resolution (no absolute path is hard-coded anywhere else)
"""
__version__ = "1.0.0"

__all__ = ["__version__", "config"]
