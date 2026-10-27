"""Every method the harness can compare, in one folder.

| Module | Methods | What a row from it is evidence about |
|---|---|---|
| :mod:`~toolbench.methods.internal` | ChromState, ChromHMM-like, Segway-like, Gaussian mixture, k-means, naive signal | the *statistical model*, on identical bins |
| :mod:`~toolbench.methods.chromhmm` | real ChromHMM (``LearnModel``) | the *published software* |
| :mod:`~toolbench.methods.epicseg` | real EpiCSeg (``normalizecounts`` + ``segment``) | the *published software* |
| :mod:`~toolbench.methods.normr` | real normR (``enrichR`` per mark) | the published software, on patterns and bivalency only |
| :mod:`~toolbench.methods.dihmm` | real diHMM (ykai16 C++ build) | the *published software*, nucleosome level |
| :mod:`~toolbench.methods.configured` | anything in ``tools.json`` — Segway, epigenome-ssm, your own | the *published software*, once you have checked its command line |

Importing this package registers all of them, so ``toolbench check`` and the runner see
one flat namespace of method keys regardless of which module a method came from.

Adding a method: if it is a command line, add an entry to ``tools.json`` and write no
Python at all. If it needs real logic -- a multi-step tool, a bespoke output format --
add a module here that defines an adapter and a ``build()`` returning instances of it,
then list it in :data:`METHOD_MODULES`.
"""

from __future__ import annotations

from typing import List

from ..core import ToolAdapter, register
from . import chromhmm, configured, dihmm, epicseg, internal, normr
from .chromhmm import ChromHMMAdapter
from .configured import TOOLS_JSON, ExternalCommandAdapter, load_specs
from .internal import INTERNAL_METHODS, InternalAdapter

#: Modules contributing methods, in the order they appear in every table and figure.
#: In-process first (they always run), then real software.
METHOD_MODULES = [internal, chromhmm, epicseg, normr, dihmm, configured]

__all__ = [
    "INTERNAL_METHODS",
    "METHOD_MODULES",
    "TOOLS_JSON",
    "ChromHMMAdapter",
    "ExternalCommandAdapter",
    "InternalAdapter",
    "build_all",
    "load_specs",
    "register_all",
]


def build_all() -> List[ToolAdapter]:
    """Instantiate every method, without touching the registry."""
    adapters: List[ToolAdapter] = []
    for module in METHOD_MODULES:
        adapters.extend(module.build())
    return adapters


def register_all() -> List[str]:
    """Register every method. Returns the keys, in table order."""
    keys = []
    for adapter in build_all():
        register(adapter)
        keys.append(adapter.key)
    return keys


register_all()
