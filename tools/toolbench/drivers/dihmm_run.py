#!/usr/bin/env python
"""Train and apply diHMM (ykai16/diHMM C++ reimplementation) on ChromHMM-style binary files.

Upstream ships ``train.py`` and ``annotation.py`` with hard-coded paths that you are
meant to edit in place; this driver does the same two calls -- ``run_dihmm`` on every
input file jointly, then ``annotate`` per file -- with the paths and sizes as arguments,
so the harness can run it as a subprocess and time it like any other tool.

It deliberately imports nothing from this repository: it runs under whichever Python the
``dihmm_ext`` extension was built for (``DIHMM_PYTHON``), which need not be the one
running the harness.

Output, one file per input ``<sample>_<chrom>_binary.txt``::

    <outdir>/<sample>_<chrom>_dihmm.tsv   # header "bin_state\tdomain_state", one row per bin

Rows follow the input file's bin order. diHMM crops each chromosome to a whole number of
domains (``n_bins - n_bins % domain_size``), so the last few bins of a chromosome can be
missing; the harness leaves those unassigned rather than guessing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--build-dir", required=True, help="directory containing dihmm_ext.so")
    p.add_argument("--input-dir", required=True, help="ChromHMM-format *_binary.txt files")
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-bin-states", type=int, required=True)
    p.add_argument("--n-domain-states", type=int, required=True)
    p.add_argument("--domain-size", type=int, default=20,
                   help="bins per domain (paper: 20 x 200 bp = 4 kb)")
    p.add_argument("--max-iter", type=int, default=500, help="upstream train.py default")
    p.add_argument("--tolerance", type=float, default=1e-6, help="upstream train.py default")
    args = p.parse_args(argv)

    sys.path.insert(0, os.path.abspath(args.build_dir))
    import dihmm_ext  # noqa: E402  (built extension; path known only at runtime)

    files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith("_binary.txt")
    )
    if not files:
        print(f"no *_binary.txt in {args.input_dir}", file=sys.stderr)
        return 2
    os.makedirs(args.outdir, exist_ok=True)
    files = [os.path.abspath(f) for f in files]
    args.outdir = os.path.abspath(args.outdir)
    # The trainer checkpoints to ./partially_trained<iter>/ every 40 iterations; keep
    # those inside the output directory instead of wherever the caller happened to be.
    os.chdir(args.outdir)

    t0 = time.time()
    model = dihmm_ext.run_dihmm(args.n_bin_states, args.n_domain_states, args.domain_size,
                                args.max_iter, args.tolerance, files)
    model_dir = os.path.join(args.outdir, "model") + os.sep
    os.makedirs(model_dir, exist_ok=True)
    dihmm_ext.save_model(model, model_dir)
    print(f"[dihmm_run] trained in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    annotations = dihmm_ext.annotate(model, files)
    for path, ann in zip(files, annotations):
        pairs = ann.annotations  # (n_bins_cropped, 2): bin state, domain state
        name = os.path.basename(path)[: -len("_binary.txt")]
        with open(os.path.join(args.outdir, f"{name}_dihmm.tsv"), "w") as fh:
            fh.write("bin_state\tdomain_state\n")
            for row in pairs:
                fh.write(f"{int(row[0])}\t{int(row[1])}\n")
    print(f"[dihmm_run] annotated {len(files)} files in {time.time() - t1:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
