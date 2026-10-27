# Export EpiCSeg posteriors from the R archive `epicseg.R segment --save_rdata` writes.
#
#   Rscript epicseg_posteriors.R <outdir>/rdata.Rdata <outdir>
#
# The segment CLI writes the segmentation BED but keeps posteriors only inside the
# archive, as `rdata$posteriors`: nstates x nbins for one dataset, or
# nstates x nbins x ndatasets (third dimension named by the --counts labels) for several.
# One TSV per dataset, bins as rows in count-matrix order, states as columns 1..nstates
# -- the same numbering as the BED's name field after EpiCSeg's own state reordering.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: epicseg_posteriors.R <rdata.Rdata> <outdir>")
env <- new.env()
name <- load(args[1], envir = env)
if (length(name) != 1) stop("expected one object in ", args[1])
seg <- get(name, envir = env)
P <- seg$posteriors
write_one <- function(m, path) {
  utils::write.table(t(m), path, sep = "\t", row.names = FALSE, col.names = FALSE,
                     quote = FALSE)
}
if (length(dim(P)) == 3) {
  labels <- dimnames(P)[[3]]
  for (i in seq_along(labels)) {
    write_one(P[, , i], file.path(args[2], paste0("posterior_", labels[i], ".tsv")))
  }
} else {
  write_one(P, file.path(args[2], "posterior.tsv"))
}
cat("wrote posteriors for", if (length(dim(P)) == 3) dim(P)[3] else 1, "dataset(s)\n")
