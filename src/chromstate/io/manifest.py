"""Sample manifests.

ChromState reads ChromHMM's ``cellmarkfiletable`` format unchanged, so an existing
ChromHMM setup can be pointed at ChromState without rewriting any config:

    sample <TAB> mark <TAB> signal_file [<TAB> control_file]

Blank lines and ``#`` comments are ignored. A fifth optional column may give the input
format (``bam``/``bed``/``peak``); when absent it is inferred from the file extension.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .counts import infer_format


@dataclass
class MarkFile:
    sample: str
    mark: str
    path: str
    control: Optional[str] = None
    fmt: Optional[str] = None

    def resolved_format(self) -> str:
        return self.fmt if self.fmt is not None else infer_format(self.path)


def read_manifest(path: str, root: Optional[str] = None) -> List[MarkFile]:
    """Parse a ChromHMM-style cellmarkfiletable into ``MarkFile`` records."""
    root = root if root is not None else os.path.dirname(os.path.abspath(path))
    entries: List[MarkFile] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            # Fields are positional: an empty column means "absent", never "skip".
            # Dropping blanks here would let a 5-column row with no control silently
            # shift its format string into the control-file slot.
            f = [x.strip() for x in line.split("\t")]
            if len(f) < 3 or not all(f[:3]):
                raise ValueError(
                    f"{path}:{lineno}: expected at least 3 non-empty tab-separated fields "
                    f"(sample, mark, file), got {f!r}"
                )
            sample, mark, signal = f[0], f[1], f[2]
            control = f[3] if len(f) > 3 and f[3] else None
            fmt = f[4] if len(f) > 4 and f[4] else None
            entries.append(
                MarkFile(
                    sample=sample,
                    mark=mark,
                    path=_resolve(signal, root),
                    control=_resolve(control, root) if control else None,
                    fmt=fmt,
                )
            )
    if not entries:
        raise ValueError(f"No entries found in manifest {path!r}")
    return entries


def _resolve(path: str, root: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def manifest_to_frame(entries: List[MarkFile]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample": e.sample,
                "mark": e.mark,
                "path": e.path,
                "control": e.control,
                "format": e.resolved_format(),
            }
            for e in entries
        ]
    )


def validate_manifest(entries: List[MarkFile]) -> None:
    """Check every sample carries the same mark panel, and that files exist.

    A shared mark panel is required because ChromState fits one joint model across all
    samples; a mark present in one sample and missing in another would make the states
    non-comparable, which is the whole point of the joint fit.
    """
    frame = manifest_to_frame(entries)
    panels = frame.groupby("sample")["mark"].apply(lambda s: tuple(sorted(s)))
    if panels.nunique() != 1:
        detail = "\n".join(f"  {s}: {list(p)}" for s, p in panels.items())
        raise ValueError(
            "All samples must be profiled for the same set of marks for a joint fit.\n"
            f"Found:\n{detail}"
        )
    dupes = frame.duplicated(subset=["sample", "mark"])
    if dupes.any():
        bad = frame.loc[dupes, ["sample", "mark"]].to_dict("records")
        raise ValueError(f"Duplicate (sample, mark) entries in manifest: {bad}")
    missing = [e.path for e in entries if not os.path.exists(e.path)]
    missing += [e.control for e in entries if e.control and not os.path.exists(e.control)]
    if missing:
        raise FileNotFoundError("Manifest references missing files:\n  " + "\n  ".join(missing))
