# Install normR, kfoots and EpiCSeg into the chromstate-bench env's own R library,
# then write the EpiCSeg command-line launcher to $BENCH/epicseg.R.
#
# Run with the env's Rscript and R_LIBS/R_LIBS_USER empty (setup_env.sh does this). It
# refuses to install anywhere but that env, so the user's global R library is untouched.
lib <- .libPaths()[1]
if (!grepl("chromstate-bench", lib)) stop("refusing to install outside the chromstate-bench env: ", lib)
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 8)
bench <- Sys.getenv("BENCH", path.expand("~/tools/chromstate-bench"))
src <- file.path(bench, "src")

# No osx-arm64 conda build exists for these two; Bioconductor source builds work.
for (p in c("bamsignals", "normr")) {
  if (!requireNamespace(p, quietly = TRUE))
    BiocManager::install(p, lib = lib, update = FALSE, ask = FALSE, type = "source")
}
# GitHub-only, patched for R >= 4.3 by setup_env.sh (kfoots_r43.patch, epicseg_fcone.patch).
install.packages(file.path(src, "kfoots_patched"), repos = NULL, type = "source", lib = lib)
install.packages(file.path(src, "epicseg_patched"), repos = NULL, type = "source", lib = lib)

for (p in c("bamsignals", "normr", "kfoots", "epicseg")) {
  cat(p, ":", tryCatch(as.character(packageVersion(p)), error = function(e) "NOT INSTALLED"), "\n")
}
suppressPackageStartupMessages(library(epicseg))
dest <- file.path(bench, "epicseg.R")
epicseg:::getLauncher(dest)
cat("EpiCSeg launcher written to", dest, "\n")
