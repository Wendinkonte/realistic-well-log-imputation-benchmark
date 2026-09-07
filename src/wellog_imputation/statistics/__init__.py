"""Multi-seed aggregation, significance testing and bootstrap confidence intervals."""
from .aggregate import aggregate
from .bootstrap import run_bootstrap
from .significance import WILCOXON_PAIRS, friedman_table, wilcoxon_table

__all__ = ["aggregate", "run_bootstrap", "WILCOXON_PAIRS", "friedman_table", "wilcoxon_table"]
