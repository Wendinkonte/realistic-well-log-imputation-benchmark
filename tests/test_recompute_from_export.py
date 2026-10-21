"""scripts/recompute_from_export.py embeds the design constants so that it runs on the
export alone; they must stay identical to the package's."""
import importlib.util
from pathlib import Path

from wellog_imputation.models.registry import LEARNED, ORDER
from wellog_imputation.statistics.significance import WILCOXON_PAIRS
from wellog_imputation.statistics.wellwise import ALL_SCENARIOS, LOGS3, REVISION_PAIRS

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "recompute_from_export.py"


def test_embedded_constants_match_the_package():
    spec = importlib.util.spec_from_file_location("recompute_from_export", SCRIPT)
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    assert rc.ORDER == ORDER and rc.LEARNED == LEARNED and rc.LOGS3 == LOGS3
    assert rc.SCENARIOS == ALL_SCENARIOS
    assert rc.WILCOXON_PAIRS == WILCOXON_PAIRS and rc.REVISION_PAIRS == REVISION_PAIRS
