"""Shared paths and constants for the Sci Rep 2025 CD8 T-cell analysis.

Data: Huang, Baxter et al., Sci Rep 2025 (GSE285245 CUT&RUN, GSE285248 RNA-seq).
P14 CD8 T cells: naive (TN), memory (LCMV Armstrong d30), exhausted (LCMV clone 13 d32).

Set ``CHROMSTATE_SCIREP_DATA`` to the processed-data folder if it is not at the default.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA = Path(os.environ.get(
    "CHROMSTATE_SCIREP_DATA",
    "/Users/hhua/Desktop/Berger_Wherry_Lab/0_Projects/Berger_Zhen.Zhang/CutRun_mm10",
))
OUT = Path(__file__).resolve().parent / "out"

POOLED = DATA / "Wig" / "pool_samples"
RNA_COUNTS = DATA / "RNA" / "Data" / "ReadCounts" / "RawRC_RNA.featureCounts_s2.txt"
CHROMHMM_DIR = DATA / "ChromHMM" / "UseConcatenate_noCTCF_15S"
CHROMHMM_RELABEL = DATA / "ChromHMM" / "relabel_map.txt"
POISED_DIR = DATA / "Bivalent" / "Use_K4me3_K27me3"
# ATAC-seq union peaks with mean normalised counts per condition (Mean_Naive, Mean_Arm,
# Mean_Cl13). Memory and exhausted ATAC are day 27, CUT&RUN day 30/32.
ATAC_PEAKS = DATA / "ATAC" / "Meta_ATAC_AllPeak_wMeanRC.wAnno.mm10.txt"

CONDITIONS = ["Naive", "Arm", "Cl13"]
CHROMHMM_PREFIX = {"Naive": "1_Naive", "Arm": "2_Arm", "Cl13": "3_Cl13"}
MARKS = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K9me3"]

# mm10 primary assembly; chrY and chrM are left out (female/mixed recipients, no states).
CHROMS = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
BIN_SIZE = 200

# Chromosomes the model is fitted on. Decoding then runs genome-wide with the fitted
# parameters, so every chromosome is scored but the fit stays laptop-sized. Gene-level
# prediction is cross-validated by chromosome independently of this choice.
TRAIN_CHROMS = ["chr11", "chr17", "chr19"]


def pooled_track(mark: str, condition: str) -> Path:
    return POOLED / f"pool_{mark}_{condition}.bw"


def ensure_out(*parts: str) -> Path:
    path = OUT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(chroms, conditions=None):
    """Joint (condition x bin) dataset over ``chroms`` from the step-1 binned tracks.

    The pooled tracks are DESeq2-scaled coverage, not read counts. They are rounded to
    integers so the negative-binomial model can be used, and the dataset records that in
    ``meta['input']`` -- read it as an approximation to counts, not as counts.
    """
    import numpy as np

    from chromstate.dataset import dataset_from_arrays

    conditions = list(conditions or CONDITIONS)
    blocks, samples, chrom_labels = [], [], []
    for cond in conditions:
        for chrom in chroms:
            z = np.load(OUT / "binned" / f"{chrom}.npz")
            ci = list(z["conditions"]).index(cond)
            x = z["X"][ci]
            blocks.append(x)
            samples += [cond] * len(x)
            chrom_labels += [chrom] * len(x)
    X = np.rint(np.concatenate(blocks)).astype(np.float64)
    ds = dataset_from_arrays(X, MARKS, samples, chrom_labels, bin_size=BIN_SIZE)
    ds.meta["input"] = "pooled DESeq2-scaled bamCoverage, mean per 200 bp bin, rounded"
    return ds

# Which fitted model downstream steps read; override to test the pipeline on another fit.
MODEL_DIR = os.environ.get("CHROMSTATE_SCIREP_MODEL", "model")
