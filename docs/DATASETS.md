# Candidate public datasets for ChromState (collected 2026-11-02)

What ChromState needs from a dataset, in order of value: several marks on the same cells;
two or more biological replicates per condition (every test in `compare.py` needs them);
more than one condition, ideally a perturbation; spike-in or calibrated normalisation when
a global change is expected; matched RNA-seq; raw reads (BAM/fastq), not only tracks.
Accessions marked "unconfirmed" were not verified against GEO in this pass.

## A. Perturbation with spike-in (the reversibility / global-change claim)

| Dataset | Design | Why it matters | Access |
|---|---|---|---|
| Orlando et al. 2014, *Cell Rep* (ChIP-Rx) | Lymphoma cells, GSK126 (EZH2i) dose series, H3K27me3 and other marks, Drosophila spike-in | The canonical demonstration that depth normalisation erases a global H3K27me3 loss; direct test of `normalize(method="spikein")` | GEO, accession unconfirmed |
| Egan et al. 2016, *PLoS One* | KARPAS-422, GSK126 1 uM 5 d, H3K27me3 ChIP with spike-in | Same point, independent lab; Active Motif reference data | GEO via paper |
| GSE140894 | Mouse ESCs, EPZ-6438 (EZH2i), calibrated ChIP-seq with Drosophila chromatin | Mouse; EZH2i in a bivalency-rich cell type | GEO GSE140894 |
| GSE205107 (NAR 2022, prostate cancer) | Spike-in CUT&RUN for H3K27me3, EZH2, AR after MS177 (EZH2 degrader), plus RNA-seq | CUT&RUN rather than ChIP; matched RNA; replicate structure to check | GEO GSE205107 |

## B. Multi-mark reference epigenomes with replicates and conditions

| Dataset | Design | Why it matters | Access |
|---|---|---|---|
| ENCODE mouse reference epigenomes (ENCODE3/4) | 6 core marks (H3K4me1/3, H3K27ac, H3K27me3, H3K36me3, H3K9me3), >= 2 biological replicates, tissues x developmental time points, matched RNA-seq | Replicated, multi-condition, open BAMs: the right place to exercise `differential_report` and the time-course alluvials | encodeproject.org (open) |
| BLUEPRINT / IHEC haematopoietic epigenomes | 6-7 marks + input, RNA-seq, same donors across cell types; naive CD4 (IHECRE00000009, EGAD00001002444) and memory CD8 (IHECRE00000010, EGAD00001002283) among them | Human T-cell states with the full panel; donors as replicates | Tracks open on the IHEC portal; raw reads via EGA (controlled) |
| Huang, Baxter et al. 2025, *Sci Rep* (GSE285245 CUT&RUN, GSE285248 RNA) | Naive / memory / exhausted mouse CD8, 4 marks, RNA 4 reps per group | Already the package's Tutorial 2 and 3 data; per-replicate BAMs exist on the cluster and would unlock the replicate tests | GEO; BAMs in the Berger lab folder |
| GSE87646 | In vitro CD8 exhaustion model, global H3K27me3 increase | Exhaustion with a PRC2 angle; assay types and replicates to verify | GEO GSE87646 |

## C. Bivalency ground truth

| Dataset | Design | Why it matters | Access |
|---|---|---|---|
| Epigenetics & Chromatin 2024 (mESC reChIP) | H3K4me3->H3K27me3 and H3K27me3->H3K4me3 sequential ChIP, 8,789 bivalent regions, single ChIPs in GSE135841, embryoid-body expression over 9 days | An orthogonal truth for `bivalency_score`, `biv_within_state` vs `biv_between_state`; the poised-promoter claim | GEO (reChIP accession in the paper); GSE135841 |
| reICeChIP preprint (bioRxiv 2021.09.09.458948) | Calibrated sequential ChIP, naive mESC -> NPC | Calibrated bivalency with differentiation; authors declare an EpiCypher conflict | GSE108747, GSE183155 |

## D. Single cells (for `chromstate.sc`)

| Dataset | Design | Why it matters | Access |
|---|---|---|---|
| nano-CT, Bartosovic & Castelo-Branco | Mouse brain, ATAC + H3K27ac + H3K27me3 in the same cells | Route 1 and 2 of `sc.py` in one dataset: fit on pseudobulk histone marks, test the ATAC transfer against measured marks | GEO GSE198467 (via the uCoTarget paper's reanalysis list) |
| mTopic companion data (Zenodo 15471869) | P22 mouse brain: RNA (GSE218593) with H3K27me3 / H3K27ac (GSE205055) | RNA-coupled single-cell histone data | Zenodo h5mu + GEO |
| scCUT&Tag mouse brain (GSE163532) | H3K4me3, H3K27ac, H3K27me3, H3K36me3, unimodal | Four marks for pseudobulk fits; no RNA | GEO GSE163532 |
| uCoTarget (Sci Adv) | 5 histone marks + transcription, 19,860 cells | Most marks per cell; not nano-CT | Paper's data availability |
| K562 CUT&Tag vs ENCODE (GSE286492, Nat Commun 2025) | H3K27ac / H3K27me3 CUT&Tag against ENCODE ChIP | Cross-assay calibration of the NB emission (CUT&Tag counts vs ChIP counts on the same states) | GEO GSE286492 |

## Suggested order

1. ENCODE mouse tissue time course (replicates, open BAMs) for the differential report and Tutorial 2-style alluvials.
2. GSE140894 or Orlando ChIP-Rx for the spike-in / global-change demonstration.
3. mESC reChIP for bivalency truth.
4. nano-CT GSE198467 for the single-cell transfer, validated against its own H3K27 marks.
