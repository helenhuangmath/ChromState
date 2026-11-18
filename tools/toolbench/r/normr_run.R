# Per-mark enrichment calls with normR (Helmuth et al. 2016), on ChromState's bins.
#
#   Rscript normr_run.R <counts.tsv> <outdir> [procs] [seed] [control.tsv]
#
# <counts.tsv> is the matrix toolbench.io_formats.export_count_matrix writes:
# chrom, start, end, sample, then one integer column per mark. For every (sample, mark)
# this fits normR's two-component binomial mixture -- enrichR(treatment, control, genome),
# the integer-vector method -- and writes, per sample, <outdir>/normr_<sample>.tsv with
# four columns per mark, rows in the input's row order for that sample:
#
#   <mark>_post   posterior probability of the enriched component
#   <mark>_q      normR q-value (NA where normR filtered the bin as uninformative)
#   <mark>_class  1 if enriched at FDR <= 0.05 (getClasses(fit, fdr = 0.05)), else 0
#   <mark>_enr    normR's standardised enrichment (getEnrichment)
#
# Control. normR models ChIP vs input. If <control.tsv> is given it must have the same
# rows and mark columns as <counts.tsv> and is used as each mark's input. Otherwise a
# *Poisson pseudo-input* is drawn: for each (sample, mark), control ~ Poisson(lambda) per
# bin with lambda = the mark's mean count in that sample (floor 0.5), seeded. That is an
# ideal input track at matched depth -- uniform background, only counting noise -- which
# is exactly the background ChromState's simulator has (no copy-number, mappability or
# GC structure). A *flat* control (same integer in every bin) was tried first and fails
# inside normR: with so few distinct (ChIP, input) pairs the p-values are too discrete
# for qvalue::pi0est, whose smooth.spline gets non-finite input. The lambdas used are
# written to <outdir>/normr_controls.tsv.
suppressPackageStartupMessages({
  library(normr)
  library(GenomicRanges)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2)
  stop("usage: normr_run.R <counts.tsv> <outdir> [procs] [seed] [control.tsv]")
counts_path <- args[1]
outdir <- args[2]
procs <- if (length(args) >= 3) as.integer(args[3]) else 1L
seed <- if (length(args) >= 4) as.integer(args[4]) else 0L
control_path <- if (length(args) >= 5 && nzchar(args[5])) args[5] else NA
set.seed(seed)
fdr <- 0.05
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

read_tsv <- function(p) {
  if (requireNamespace("data.table", quietly = TRUE)) {
    as.data.frame(data.table::fread(p, sep = "\t", header = TRUE))
  } else {
    utils::read.delim(p, check.names = FALSE)
  }
}
tab <- read_tsv(counts_path)
meta <- c("chrom", "start", "end", "sample")
marks <- setdiff(colnames(tab), meta)
ctl <- if (is.na(control_path)) NULL else read_tsv(control_path)
if (!is.null(ctl) && (nrow(ctl) != nrow(tab) || !all(marks %in% colnames(ctl))))
  stop("control matrix must match the count matrix's rows and mark columns")

controls <- data.frame()
for (s in unique(tab$sample)) {
  rows <- which(tab$sample == s)
  sub <- tab[rows, ]
  # normR's GRanges is 1-based closed; the exported matrix is BED-style 0-based half-open.
  gr <- GRanges(sub$chrom, IRanges(sub$start + 1L, sub$end))
  out <- sub[, c("chrom", "start", "end")]
  for (m in marks) {
    treat <- as.integer(round(sub[[m]]))
    if (is.null(ctl)) {
      lambda <- max(0.5, mean(treat))
      control <- as.integer(stats::rpois(length(treat), lambda))
      controls <- rbind(controls, data.frame(sample = s, mark = m, lambda = lambda,
                                             kind = "poisson_pseudo_input"))
    } else {
      control <- as.integer(round(ctl[rows, m]))
      controls <- rbind(controls, data.frame(sample = s, mark = m, lambda = NA,
                                             kind = "supplied_control"))
    }
    fit <- enrichR(treatment = treat, control = control, genome = gr, procs = procs,
                   verbose = FALSE)
    post <- getPosteriors(fit)
    cls <- getClasses(fit, fdr = fdr)
    out[[paste0(m, "_post")]] <- post[, ncol(post)]
    out[[paste0(m, "_q")]] <- getQvalues(fit)
    out[[paste0(m, "_class")]] <- ifelse(is.na(cls), 0L, as.integer(cls == 1L))
    out[[paste0(m, "_enr")]] <- getEnrichment(fit)
  }
  utils::write.table(out, file.path(outdir, paste0("normr_", s, ".tsv")), sep = "\t",
                     row.names = FALSE, quote = FALSE, na = "NA")
}
utils::write.table(controls, file.path(outdir, "normr_controls.tsv"), sep = "\t",
                   row.names = FALSE, quote = FALSE)
cat("normR", as.character(packageVersion("normr")), "done:", length(marks), "marks x",
    length(unique(tab$sample)), "samples\n")
