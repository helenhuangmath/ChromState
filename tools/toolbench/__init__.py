"""toolbench -- run other segmentation tools on ChromState's bins and score them alike.

See ``tools/README.md``. Entry point: ``python tools/run_tools.py``.
"""

from __future__ import annotations

from .core import UNASSIGNED, Availability, ToolAdapter, ToolRun, get_adapter, register, registry
from .evaluate import agreement_matrix, availability_table, evaluate_runs, run_tools
from .metrics import ALL_METRICS, METRICS, TRUTH_FREE, EvalContext, Metric
from .report import write_report

__all__ = [
    "ALL_METRICS",
    "METRICS",
    "TRUTH_FREE",
    "UNASSIGNED",
    "Availability",
    "EvalContext",
    "Metric",
    "ToolAdapter",
    "ToolRun",
    "agreement_matrix",
    "availability_table",
    "evaluate_runs",
    "get_adapter",
    "register",
    "registry",
    "run_tools",
    "write_report",
]

from . import methods  # noqa: E402,F401  -- registers every compared method on import
