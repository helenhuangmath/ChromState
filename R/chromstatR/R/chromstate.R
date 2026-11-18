#' Load binned signal from a manifest
#'
#' Reads a ChromHMM-style `cellmarkfiletable` (sample, mark, file, optional control,
#' optional format) and projects every file onto a shared genome tiling.
#'
#' @param manifest Path to the manifest file.
#' @param chrom_sizes Path to a two-column `chrom<TAB>size` file.
#' @param bin_size Bin width in bp. Defaults to 200, matching ChromHMM, so results can
#'   be compared bin-for-bin against an existing ChromHMM segmentation.
#' @param chroms Optional character vector restricting which chromosomes to load.
#' @param blacklist Optional path to a BED file of regions to exclude.
#' @param shift Read shift in bp applied to BED/BAM input.
#' @return A `chromstate_dataset` object (an opaque handle to the Python dataset).
#' @export
#' @examples
#' \dontrun{
#' ds <- cs_load("cellmarkfiletable.txt", "hg38.chrom.sizes")
#' }
cs_load <- function(manifest, chrom_sizes, bin_size = 200, chroms = NULL,
                    blacklist = NULL, shift = 100) {
  m <- cs_module()
  genome <- m$Genome$from_file(chrom_sizes)
  binset <- m$BinSet$from_genome(genome, bin_size = as_py_int(bin_size), chroms = chroms)
  entries <- m$read_manifest(manifest)
  bl <- if (is.null(blacklist)) NULL else m$read_bed_regions(blacklist)
  kwargs <- list(bed = list(shift = as_py_int(shift)),
                 bam = list(shift = as_py_int(shift)))
  ds <- m$build_dataset(entries, binset, blacklist = bl, reader_kwargs = kwargs)
  structure(list(py = ds, manifest = manifest, bin_size = bin_size),
            class = "chromstate_dataset")
}

#' Simulate a dataset with known ground truth
#'
#' Useful for trying the workflow, and for checking that an analysis recovers states
#' that are actually there.
#'
#' @param n_bins Bins per chromosome per sample.
#' @param n_chroms Number of chromosomes.
#' @param seed Random seed.
#' @param blur Fraction of bins whose profile is a mixture of two states, creating
#'   genuinely ambiguous regions. Set above 0 for a realistic difficulty.
#' @return A list with `dataset` (a `chromstate_dataset`) and `truth` (state labels).
#' @export
cs_simulate <- function(n_bins = 2000, n_chroms = 2, seed = 0, blur = 0) {
  m <- cs_module()
  res <- m$simulate(n_bins = as_py_int(n_bins), n_chroms = as_py_int(n_chroms),
                    seed = as_py_int(seed), blur = blur)
  ds <- structure(list(py = res[[1]], manifest = NA_character_, bin_size = 200L),
                  class = "chromstate_dataset")
  list(dataset = ds,
       truth = as.integer(reticulate::py_to_r(res[[2]]$states)) + 1L,
       truth_labels = as.character(reticulate::py_to_r(res[[2]]$labels)))
}

#' Normalise a dataset
#'
#' @param dataset A `chromstate_dataset`.
#' @param method One of `"median_ratio"`, `"depth"`, `"spikein"`, `"none"`.
#' @param spikein Named list mapping `"sample|mark"` to spike-in read counts. Required
#'   for `method = "spikein"`.
#' @param reference Sample whose size factor is fixed to 1.
#' @details
#' Use `"spikein"` whenever the experiment involves a perturbation expected to change a
#' mark genome-wide (EZH2, LSD1 or DNMT inhibitors). Depth and median-of-ratios both
#' assume most of the genome is unchanged, so they absorb a global loss into the size
#' factor and the effect disappears before any model sees it.
#' @return The dataset, invisibly (modified in place).
#' @export
cs_normalize <- function(dataset, method = "median_ratio", spikein = NULL,
                         reference = NULL) {
  stopifnot(inherits(dataset, "chromstate_dataset"))
  m <- cs_module()
  py_spike <- NULL
  if (!is.null(spikein)) {
    keys <- strsplit(names(spikein), "|", fixed = TRUE)
    py_spike <- reticulate::py_dict(
      keys = lapply(keys, function(k) reticulate::tuple(k[1], k[2])),
      values = as.numeric(unlist(spikein))
    )
  }
  m$normalize(dataset$py, method = method, spikein = py_spike, reference = reference)
  invisible(dataset)
}

#' Fit a ChromState model
#'
#' @param dataset A `chromstate_dataset`.
#' @param n_states Number of chromatin states.
#' @param emission `"auto"`, `"nb"` (counts), `"gaussian"` (peak scores), or
#'   `"bernoulli"` (ChromHMM's model, for comparison).
#' @param n_restarts Minimum number of random initialisations.
#' @param seed Random seed.
#' @return A `chromstate_segmentation` object.
#' @export
cs_fit <- function(dataset, n_states = 10, emission = "auto", n_restarts = 5, seed = 0) {
  stopifnot(inherits(dataset, "chromstate_dataset"))
  m <- cs_module()
  seg <- m$fit(dataset$py, n_states = as_py_int(n_states), emission = emission,
               n_restarts = as_py_int(n_restarts), seed = as_py_int(seed))
  structure(list(py = seg, dataset = dataset, n_states = n_states),
            class = "chromstate_segmentation")
}

#' Choose the number of states by held-out likelihood
#'
#' @param dataset A `chromstate_dataset`.
#' @param candidates Integer vector of K values to try.
#' @param seed Random seed.
#' @return A data frame with one row per candidate K. Prefer `heldout_loglik_per_bin`
#'   over BIC: adjacent bins are strongly correlated, so BIC's independence assumption
#'   over-penalises and systematically selects too few states.
#' @export
cs_select_k <- function(dataset, candidates = c(4, 6, 8, 10, 12, 15, 18), seed = 0) {
  m <- cs_module()
  as_df(m$select_n_states(dataset$py, candidates = as.integer(candidates),
                          seed = as_py_int(seed)))
}

#' Per-bin quantification
#'
#' Returns the full quantification table: soft state composition, call confidence,
#' decision margin, state entropy, bivalency scores, and per-mark attribution.
#'
#' @param segmentation A `chromstate_segmentation`.
#' @param active_mark,repressive_mark Mark names used for the bivalency score.
#' @param attribution Include per-mark attribution columns.
#' @return A data frame with one row per bin.
#' @export
cs_quantify <- function(segmentation, active_mark = "H3K4me3",
                        repressive_mark = "H3K27me3", attribution = TRUE) {
  stopifnot(inherits(segmentation, "chromstate_segmentation"))
  m <- cs_module()
  as_df(m$quantify(segmentation$py, active_mark = active_mark,
                   repressive_mark = repressive_mark,
                   include_attribution = attribution))
}

#' Emission parameters
#' @param segmentation A `chromstate_segmentation`.
#' @return A data frame of states (rows) by marks (columns).
#' @export
cs_emissions <- function(segmentation) {
  df <- as_df(segmentation$py$emission_frame())
  rownames(df) <- cs_states(segmentation)$label
  df
}

#' Transition probabilities
#' @param segmentation A `chromstate_segmentation`.
#' @return A square data frame of transition probabilities.
#' @export
cs_transitions <- function(segmentation) {
  df <- as_df(segmentation$py$transition_frame())
  rownames(df) <- cs_states(segmentation)$label
  df
}

#' Per-state summary
#' @param segmentation A `chromstate_segmentation`.
#' @return A data frame with mnemonic, colour, the rule that produced the mnemonic, and
#'   genome coverage under both hard and soft assignment.
#' @export
cs_states <- function(segmentation) {
  as_df(segmentation$py$state_table())
}

#' Continuous bivalency scores
#'
#' @param segmentation A `chromstate_segmentation`.
#' @param active_mark,repressive_mark Mark names.
#' @return A data frame with `bivalency_score` (harmonic mean of the two normalised
#'   signals), and its decomposition into `bivalency_magnitude` and `bivalency_balance`.
#' @details
#' Also returned by [cs_quantify()]. Reporting magnitude and balance separately matters:
#' a weak but perfectly balanced promoter and a strong lopsided one share a harmonic
#' mean yet mean different things.
#' @export
cs_bivalency <- function(segmentation, active_mark = "H3K4me3",
                         repressive_mark = "H3K27me3") {
  m <- cs_module()
  as_df(m$bivalency(segmentation$py, active_mark = active_mark,
                    repressive_mark = repressive_mark))
}

#' Per-mark attribution of state calls
#'
#' Decomposes the log-likelihood ratio between a bin's called state and a comparison
#' state into exact per-mark contributions.
#'
#' @param segmentation A `chromstate_segmentation`.
#' @param against Optional 1-based state index to compare against. The default compares
#'   with each bin's runner-up state, answering "why this state and not the next most
#'   likely". Passing the quiescent state instead answers "why is this not background",
#'   which is usually the question that recovers a state's namesake mark.
#' @param by_state Return the per-state mean instead of per-bin values.
#' @return A data frame.
#' @export
cs_attribution <- function(segmentation, against = NULL, by_state = FALSE) {
  m <- cs_module()
  idx <- if (is.null(against)) NULL else as_py_int(against - 1L)
  if (by_state) {
    return(as_df(m$state_mark_importance(segmentation$py, against = idx),
                 keep_rownames = TRUE))
  }
  as_df(m$attribution(segmentation$py, against = idx))
}

#' State flow between two conditions
#'
#' @param segmentation A `chromstate_segmentation`.
#' @param sample_a,sample_b Sample names.
#' @param normalize Return `P(state after | state before)` rather than the joint.
#' @return A square data frame; rows are states in `sample_a`, columns in `sample_b`.
#' @export
cs_flow <- function(segmentation, sample_a, sample_b, normalize = TRUE) {
  m <- cs_module()
  df <- as_df(m$state_flow(segmentation$py, sample_a, sample_b, normalize = normalize))
  rownames(df) <- cs_states(segmentation)$label
  df
}

#' State occupancy per sample, and change from a reference
#' @param segmentation A `chromstate_segmentation`.
#' @param reference Reference sample. Defaults to the first.
#' @return A data frame.
#' @export
cs_occupancy <- function(segmentation, reference = NULL) {
  m <- cs_module()
  as_df(m$occupancy_change(segmentation$py, reference = reference))
}

#' Differential state analysis between two groups of samples
#'
#' @param segmentation A `chromstate_segmentation`.
#' @param group_a,group_b Character vectors of sample names.
#' @param n_permutations Number of label permutations for the null.
#' @param fdr Target false discovery rate.
#' @return A list with `table` (per-bin effect sizes, and q-values when replicates
#'   allow), `has_pvalues`, and `note`.
#' @details
#' P-values require at least two replicates per group. With one sample per group there
#' is no estimate of within-group variability, so this returns effect sizes only and
#' says so, rather than reporting significance it cannot support.
#' @export
cs_differential_state <- function(segmentation, group_a, group_b,
                                  n_permutations = 200, fdr = 0.05) {
  m <- cs_module()
  res <- m$differential_state(segmentation$py, as.list(group_a), as.list(group_b),
                              n_permutations = as_py_int(n_permutations), fdr = fdr)
  list(table = as_df(res$table),
       has_pvalues = as.logical(res$has_pvalues),
       note = as.character(res$note))
}

#' Convert a quantification table to GRanges
#'
#' @param table A data frame from [cs_quantify()].
#' @param keep_metadata Include all non-coordinate columns as metadata.
#' @return A `GRanges` object.
#' @export
cs_to_granges <- function(table, keep_metadata = TRUE) {
  if (!requireNamespace("GenomicRanges", quietly = TRUE)) {
    stop("cs_to_granges() requires the GenomicRanges package.", call. = FALSE)
  }
  coord <- c("chrom", "start", "end")
  missing <- setdiff(coord, names(table))
  if (length(missing)) {
    stop("Table is missing required columns: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  gr <- GenomicRanges::GRanges(
    seqnames = table$chrom,
    # BED is half-open and 0-based; GRanges is closed and 1-based.
    ranges = IRanges::IRanges(start = table$start + 1L, end = table$end)
  )
  if (keep_metadata) {
    meta <- table[, setdiff(names(table), coord), drop = FALSE]
    if (ncol(meta)) S4Vectors::mcols(gr) <- S4Vectors::DataFrame(meta)
  }
  gr
}

#' Write a segmentation as a coloured BED track
#'
#' @param segmentation A `chromstate_segmentation`.
#' @param file Output path.
#' @param sample Sample to write. Defaults to the first.
#' @param merge Merge runs of consecutive identical states.
#' @return The file path, invisibly.
#' @export
cs_write_bed <- function(segmentation, file, sample = NULL, merge = TRUE) {
  bed <- as_df(segmentation$py$to_bed(sample = sample, merge = merge))
  utils::write.table(bed, file, sep = "\t", quote = FALSE,
                     row.names = FALSE, col.names = FALSE)
  invisible(file)
}

#' @export
print.chromstate_segmentation <- function(x, ...) {
  st <- cs_states(x)
  meta <- reticulate::py_to_r(x$py$meta)
  cat("<chromstate segmentation>\n")
  cat(sprintf("  states     : %d (%s)\n", nrow(st), paste(st$label, collapse = ", ")))
  cat(sprintf("  emission   : %s\n", meta$emission))
  cat(sprintf("  log-lik    : %.1f\n", x$py$loglik))
  cat(sprintf("  restarts   : %d run, %d reached the best optimum\n",
              as.integer(meta$n_restarts), as.integer(meta$restart_hits_at_best)))
  if (isTRUE(meta$collapsed)) {
    cat("  WARNING    : fit looks degenerate; see meta$collapse_warning\n")
  }
  invisible(x)
}

#' @export
summary.chromstate_segmentation <- function(object, ...) {
  cs_states(object)
}

#' Gene-level state composition
#'
#' Mean posterior over each gene's promoter (or body) bins, per sample -- the table that
#' joins to RNA-seq. `gtf` is a GTF/GFF path; `part` is `"promoter"` or `"body"`.
#'
#' @param segmentation A `chromstate_segmentation`.
#' @param gtf Path to a gene annotation in GTF format.
#' @param part `"promoter"` (default) or `"body"`.
#' @return A data frame with one row per gene and sample: `gene_id`, `name`, `sample`,
#'   one column per state, `entropy`, `top_state`, and `signal_<mark>` columns.
#' @export
cs_gene_states <- function(segmentation, gtf, part = "promoter") {
  stopifnot(inherits(segmentation, "chromstate_segmentation"))
  m <- cs_module()
  features <- m$FeatureSet$from_gtf(gtf)
  out <- as_df(m$gene_state_matrix(segmentation$py, features, part = part))
  attr(out, "chrom") <- as_df(features$table)[, c("gene_id", "chrom")]
  out
}

#' Which chromatin states matter for gene expression
#'
#' Fits a model of expression from the full per-gene state composition, scores it out of
#' fold on chromosome-grouped splits, and reports for every state its permutation
#' importance (loss in Spearman when that state is shuffled), drop-one importance (loss
#' when it is removed and the remaining shares re-closed), a signed standardised ridge
#' coefficient, bootstrap intervals, and a rank. The result is an association between
#' chromatin state and expression; it does not establish direction or causation.
#'
#' @param gene_states Output of [cs_gene_states()], restricted to one sample (or pass
#'   `sample`).
#' @param expression Named numeric vector of log-scale expression, names are `gene_id`.
#' @param sample Sample to select from `gene_states` (if it holds several).
#' @param model `"ridge"` (default; gives signed coefficients) or `"gbm"`.
#' @param n_permutations,n_boot Permutation repeats and bootstrap resamples.
#' @param seed Random seed.
#' @return A list with `table` (one row per state, sorted by rank), `cv_spearman`,
#'   `cv_r2`, `n_genes`, and `note`.
#' @export
cs_state_importance <- function(gene_states, expression, sample = NULL, model = "ridge",
                                n_permutations = 20, n_boot = 200, seed = 0) {
  m <- cs_module()
  chrom <- attr(gene_states, "chrom")
  if (!is.null(sample)) gene_states <- gene_states[gene_states$sample == sample, ]
  pandas <- reticulate::import("pandas", convert = FALSE)
  groups <- if (is.null(chrom)) NULL else {
    pandas$Series(as.character(chrom$chrom), index = as.character(chrom$gene_id))
  }
  expr <- pandas$Series(as.numeric(expression), index = names(expression))
  res <- m$composition_state_importance(
    gene_states, expr, groups = groups, model = model,
    n_permutations = as_py_int(n_permutations), n_boot = as_py_int(n_boot),
    seed = as_py_int(seed)
  )
  list(table = as_df(res$table), cv_spearman = res$cv_spearman, cv_r2 = res$cv_r2,
       n_genes = res$n_genes, model = res$model, note = res$note)
}
