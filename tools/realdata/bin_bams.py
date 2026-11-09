"""Bin ENCODE ChIP-seq BAMs into 200 bp read counts for ChromState (pysam + numpy only).

    # one BAM -> <out>/per_bam/<ENCFF>.npz  (counts per chromosome)
    python tools/realdata/bin_bams.py count --bam X.bam -o <out>
    # all BAMs of a manifest -> <out>/<chrom>.npz  (X: samples x bins x marks, C: control)
    python tools/realdata/bin_bams.py merge --manifest manifest.tsv -o <out>

Reads are single-end (36-51 bp), already filtered and deduplicated by the ENCODE
pipeline; reads below MAPQ 30, duplicates and secondary/supplementary/QC-fail reads are
dropped again for safety. Each read is shifted 100 bp toward its 3' end (ChromHMM's
default) to approximate the fragment centre. The same convention as
``chromstate.io.counts.count_bam``, reimplemented here so it runs in an env without the
package. Input control counts sit in ``C`` with the mark axis broadcast, matching the
shape ChromStateDataset expects.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def count(bam_path: str, out: str, bin_size: int, shift: int, min_mapq: int) -> None:
    import pysam

    t0 = time.time()
    acc = os.path.basename(bam_path).split(".")[0]
    dest = os.path.join(out, "per_bam", f"{acc}.npz")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    stats = {"total": 0, "kept": 0}
    arrays = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        sizes = dict(zip(bam.references, bam.lengths))
        for c in CHROMS:
            if c not in sizes:
                raise SystemExit(f"{bam_path} lacks {c}")
            arrays[c] = np.zeros(sizes[c] // bin_size, dtype=np.int32)
        buf = {c: [] for c in CHROMS}
        for r in bam.fetch(until_eof=True):
            stats["total"] += 1
            if r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_qcfail or r.is_duplicate:
                continue
            if r.mapping_quality < min_mapq:
                continue
            b = buf.get(r.reference_name)
            if b is None:
                continue
            b.append(r.reference_end - shift if r.is_reverse else r.reference_start + shift)
            stats["kept"] += 1
            if len(b) >= 2_000_000:
                _tally(arrays[r.reference_name], b, bin_size); b.clear()
        for c, b in buf.items():
            _tally(arrays[c], b, bin_size)
    np.savez_compressed(dest, **arrays, total=stats["total"], kept=stats["kept"], bin_size=bin_size)
    print(f"{acc}: {stats['kept']:,}/{stats['total']:,} reads kept ({time.time() - t0:.0f}s) -> {dest}", flush=True)


def _tally(arr: np.ndarray, pos, bin_size: int) -> None:
    if not pos:
        return
    idx = np.asarray(pos, dtype=np.int64) // bin_size
    idx = idx[(idx >= 0) & (idx < len(arr))]
    np.add.at(arr, idx, 1)


def merge(manifest: str, out: str, marks) -> None:
    rows = [r for r in csv.DictReader(open(manifest), delimiter="\t") if r["kind"] == "chip"]
    cells = list(dict.fromkeys(r["cell"] for r in rows))
    reps = sorted({r["rep"] for r in rows})
    samples = [f"{c}_R{rp}" for c in cells for rp in reps]
    acc = {(r["cell"], r["rep"], r["mark"]): r["accession"] for r in rows}
    per = {}
    for s in samples:
        cell, rp = s.rsplit("_R", 1)
        for m in list(marks) + ["Input"]:
            a = acc.get((cell, rp, m))
            if a is None:
                raise SystemExit(f"manifest lacks {cell} rep{rp} {m}")
            per[(s, m)] = np.load(os.path.join(out, "per_bam", f"{a}.npz"))
    depth = []
    for (s, m), z in per.items():
        depth.append({"sample": s, "mark": m, "reads_total": int(z["total"]), "reads_kept": int(z["kept"])})
    with open(os.path.join(out, "depth.tsv"), "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(depth[0]), delimiter="\t"); w.writeheader(); w.writerows(depth)
    for c in CHROMS:
        n = len(per[(samples[0], marks[0])][c])
        X = np.stack([np.stack([per[(s, m)][c][:n] for m in marks], axis=-1) for s in samples]).astype(np.float32)
        C = np.stack([per[(s, "Input")][c][:n] for s in samples]).astype(np.float32)
        np.savez_compressed(os.path.join(out, f"{c}.npz"), X=X, C=C, samples=np.array(samples),
                            marks=np.array(marks), bin_size=int(per[(samples[0], marks[0])]["bin_size"]))
        print(f"{c}: {X.shape}", flush=True)
    with open(os.path.join(out, "samples.tsv"), "w") as fh:
        fh.write("sample\tcell\trep\n" + "".join(f"{s}\t{s.rsplit('_R', 1)[0]}\t{s.rsplit('_R', 1)[1]}\n" for s in samples))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("count"); pc.add_argument("--bam", required=True); pc.add_argument("-o", "--out", required=True)
    pc.add_argument("--bin-size", type=int, default=200); pc.add_argument("--shift", type=int, default=100)
    pc.add_argument("--min-mapq", type=int, default=30)
    pm = sub.add_parser("merge"); pm.add_argument("--manifest", required=True); pm.add_argument("-o", "--out", required=True)
    pm.add_argument("--marks", default="H3K4me1,H3K4me3,H3K27ac,H3K27me3,H3K36me3,H3K9me3")
    a = p.parse_args()
    if a.cmd == "count":
        count(a.bam, a.out, a.bin_size, a.shift, a.min_mapq)
    else:
        merge(a.manifest, a.out, a.marks.split(","))


if __name__ == "__main__":
    main()
