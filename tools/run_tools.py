#!/usr/bin/env python
"""Entry point for the tool comparison harness.

Runs without installing anything: it puts ``src/`` and ``tools/`` on the path itself.

    python tools/run_tools.py check
    python tools/run_tools.py metrics
    python tools/run_tools.py run -o compare_out --simulate 4000
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (os.path.join(ROOT, "src"), HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from toolbench.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
