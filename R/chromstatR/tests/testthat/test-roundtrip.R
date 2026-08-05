skip_if_no_backend <- function() {
  testthat::skip_if_not(chromstate_available(),
                        "ChromState Python backend not available")
}

test_that("backend is reachable and reports a version", {
  skip_if_no_backend()
  expect_type(chromstate_version(), "character")
})

test_that("simulate -> fit -> quantify round-trips through reticulate", {
  skip_if_no_backend()
  sim <- cs_simulate(n_bins = 400, n_chroms = 2, seed = 1)
  expect_s3_class(sim$dataset, "chromstate_dataset")
  expect_length(sim$truth, 1600)          # 400 bins x 2 chroms x 2 samples

  cs_normalize(sim$dataset)
  seg <- cs_fit(sim$dataset, n_states = 7, n_restarts = 2, seed = 1)
  expect_s3_class(seg, "chromstate_segmentation")

  states <- cs_states(seg)
  expect_equal(nrow(states), 7)
  expect_true(all(c("label", "color", "rule", "coverage_hard") %in% names(states)))

  em <- cs_emissions(seg)
  expect_equal(nrow(em), 7)
  expect_equal(nrow(cs_transitions(seg)), 7)
  # Transition rows are probability distributions.
  expect_equal(unname(rowSums(cs_transitions(seg))), rep(1, 7), tolerance = 1e-6)

  q <- cs_quantify(seg)
  expect_equal(nrow(q), 1600)
  expect_true(all(c("state", "confidence", "entropy", "bivalency_score") %in% names(q)))
  expect_true(all(q$confidence >= 0 & q$confidence <= 1))
  expect_true(all(q$entropy >= 0 & q$entropy <= 1))
})

test_that("bivalency score equals magnitude times balance", {
  skip_if_no_backend()
  sim <- cs_simulate(n_bins = 300, n_chroms = 1, seed = 2)
  cs_normalize(sim$dataset)
  seg <- cs_fit(sim$dataset, n_states = 6, n_restarts = 1, seed = 2)
  biv <- cs_bivalency(seg)
  expect_equal(biv$bivalency_score,
               biv$bivalency_magnitude * biv$bivalency_balance,
               tolerance = 1e-8)
})

test_that("attribution against runner-up and against background differ", {
  skip_if_no_backend()
  sim <- cs_simulate(n_bins = 400, n_chroms = 1, seed = 3)
  cs_normalize(sim$dataset)
  seg <- cs_fit(sim$dataset, n_states = 7, n_restarts = 2, seed = 3)

  vs_runner <- cs_attribution(seg, by_state = TRUE)
  quies <- which(cs_states(seg)$label == "Quies")[1]
  skip_if(is.na(quies), "no quiescent state in this fit")
  vs_bg <- cs_attribution(seg, against = quies, by_state = TRUE)

  expect_equal(nrow(vs_runner), nrow(vs_bg))
  expect_false(isTRUE(all.equal(vs_runner, vs_bg)))
})

test_that("state flow rows are conditional distributions", {
  skip_if_no_backend()
  sim <- cs_simulate(n_bins = 400, n_chroms = 1, seed = 4)
  cs_normalize(sim$dataset)
  seg <- cs_fit(sim$dataset, n_states = 6, n_restarts = 1, seed = 4)
  flow <- cs_flow(seg, "control", "treated")
  expect_equal(nrow(flow), 6)
  expect_equal(unname(rowSums(flow)), rep(1, 6), tolerance = 1e-6)
})

test_that("differential state refuses p-values without replicates", {
  skip_if_no_backend()
  sim <- cs_simulate(n_bins = 300, n_chroms = 1, seed = 5)
  cs_normalize(sim$dataset)
  seg <- cs_fit(sim$dataset, n_states = 6, n_restarts = 1, seed = 5)
  res <- cs_differential_state(seg, "control", "treated")
  expect_false(res$has_pvalues)
  expect_match(res$note, "replicates")
  expect_false("pvalue" %in% names(res$table))
})

test_that("segmentation writes a well-formed BED file", {
  skip_if_no_backend()
  sim <- cs_simulate(n_bins = 300, n_chroms = 1, seed = 6)
  cs_normalize(sim$dataset)
  seg <- cs_fit(sim$dataset, n_states = 5, n_restarts = 1, seed = 6)

  path <- tempfile(fileext = ".bed")
  cs_write_bed(seg, path, sample = "control")
  bed <- utils::read.delim(path, header = FALSE)
  expect_equal(ncol(bed), 9)
  expect_true(all(bed$V3 > bed$V2))
  expect_true(all(grepl("^\\d+,\\d+,\\d+$", bed$V9)))
})

test_that("GRanges conversion shifts BED coordinates to 1-based", {
  skip_if_no_backend()
  skip_if_not_installed("GenomicRanges")
  tbl <- data.frame(chrom = c("chr1", "chr1"), start = c(0L, 200L),
                    end = c(200L, 400L), score = c(1.5, 2.5))
  gr <- cs_to_granges(tbl)
  expect_equal(as.integer(GenomicRanges::start(gr)), c(1L, 201L))
  expect_equal(as.integer(GenomicRanges::end(gr)), c(200L, 400L))
  expect_equal(S4Vectors::mcols(gr)$score, c(1.5, 2.5))
})
