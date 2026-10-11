"""Input readers: BAM, BED reads, peak BED, and ChromHMM-style manifests."""

from __future__ import annotations

from .counts import (
    bin_bigwig,
    count_bam,
    count_bed_reads,
    infer_format,
    is_count_format,
    load_signal,
    read_bed_regions,
    score_peaks,
)
from .manifest import MarkFile, manifest_to_frame, read_manifest, validate_manifest

__all__ = [
    "MarkFile",
    "bin_bigwig",
    "count_bam",
    "count_bed_reads",
    "infer_format",
    "is_count_format",
    "load_signal",
    "manifest_to_frame",
    "read_bed_regions",
    "read_manifest",
    "score_peaks",
    "validate_manifest",
]
