#!/usr/bin/env python
"""Entry point for the runtime/memory scaling benchmark (see toolbench/scaling.py).

    python tools/scaling.py -o scaling_out --sizes 6250,25000,100000
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (os.path.join(ROOT, "src"), HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from toolbench.scaling import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
