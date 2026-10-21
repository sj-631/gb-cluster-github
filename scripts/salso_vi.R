#!/usr/bin/env Rscript

# NPY bridge for actual SALSO randomized VI search. No medoid fallback.
fail <- function(...) stop(..., call. = FALSE)
emit <- function(key, value) cat("SALSO_META\t", key, "\t", value, "\n", sep = "")
integer_argument <- function(value, name, lower = 1L) {
  number <- suppressWarnings(as.numeric(value))
  if (length(number) != 1L || !is.finite(number) ||
      number != floor(number) || number < lower ||
      number > .Machine$integer.max) {
    fail("Invalid integer argument: ", name)
  }
  as.integer(number)
}

for (package in c("salso", "RcppCNPy")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    fail("Required R package is missing: ", package,
         ". Install it before training; no VI-medoid fallback is used.")
  }
}
emit_environment <- function() {
  emit("r_version", paste(R.version$major, R.version$minor, sep = "."))
  emit("salso_version", as.character(utils::packageVersion("salso")))
  emit("rcppcnpy_version", as.character(utils::packageVersion("RcppCNPy")))
}

args <- commandArgs(trailingOnly = TRUE)
if (identical(args, "--check")) {
  emit("status", "available")
  emit_environment()
  emit("default_candidate_policy", "explicit n observations, not package maxNClusters=0")
  quit(save = "no", status = 0L)
}
if (!(length(args) %in% c(6L, 7L, 8L))) {
  fail("Usage: salso_vi.R <samples.npy> <output.npy> <draw_k> <n_runs> ",
       "<n_cores> <seed> [max_clusters] [payload_md5]; or salso_vi.R --check")
}
samples_path <- args[[1L]]
output_path <- args[[2L]]
draw_k <- integer_argument(args[[3L]], "draw_k")
n_runs <- integer_argument(args[[4L]], "n_runs")
n_cores <- integer_argument(args[[5L]], "n_cores")
seed <- integer_argument(args[[6L]], "seed", lower = 0L)
if (!file.exists(samples_path)) fail("Partition samples do not exist: ", samples_path)
if (file.exists(output_path)) fail("Refusing to overwrite existing SALSO output: ", output_path)

# Numeric labels are transferred as exact double values because this
# RcppCNPy build's integer reader depends on native C long width.
samples <- RcppCNPy::npyLoad(samples_path, type = "numeric")
if (length(dim(samples)) != 2L || any(dim(samples) < 1L) ||
    anyNA(samples) || any(!is.finite(samples)) || any(samples < 0L) ||
    any(samples != floor(samples)) || any(samples > .Machine$integer.max)) {
  fail("Partition samples must be a nonempty integer matrix without missing labels.")
}
payload_file <- tempfile("salso-input-payload-")
writeBin(as.double(t(samples)), payload_file, size = 8L, endian = "little")
payload_md5 <- unname(tools::md5sum(payload_file))
unlink(payload_file)
if (length(args) == 8L && payload_md5 != args[[8L]]) {
  fail("R-loaded partition matrix does not match the complete Python input payload.")
}
storage.mode(samples) <- "integer"
if (any(apply(samples, 1L, function(row) length(unique(row))) != draw_k)) {
  fail("Every posterior draw must have exactly draw_k clusters.")
}
max_clusters <- if (length(args) >= 7L) {
  integer_argument(args[[7L]], "max_clusters", lower = 0L)
} else {
  ncol(samples)
}
if (max_clusters > ncol(samples)) fail("max_clusters cannot exceed the number of observations.")
effective_max <- if (max_clusters == 0L) draw_k else max_clusters

set.seed(seed)
representative <- salso::salso(
  samples, loss = salso::VI(), maxNClusters = max_clusters,
  nRuns = n_runs, nCores = n_cores
)
info <- attr(representative, "info")
if (!is.null(dim(representative)) || length(representative) != ncol(samples) ||
    anyNA(representative) || any(!is.finite(representative)) ||
    any(representative != floor(representative))) {
  fail("SALSO returned an invalid partition vector.")
}
if (is.null(info) || !is.finite(info$expectedLoss[[1L]]) ||
    info$maxNClusters[[1L]] != effective_max ||
    info$nRuns[[1L]] < 1L) {
  fail("SALSO returned incomplete or inconsistent search metadata.")
}
realized <- length(unique(representative))
if (realized < 1L || realized > effective_max) fail("SALSO violated the candidate cluster bound.")
expected_bits <- mean(salso::partition.loss(samples, representative, loss = salso::VI()))
if (!is.finite(expected_bits) ||
    abs(expected_bits - info$expectedLoss[[1L]]) > 1e-8 * max(1, abs(expected_bits))) {
  fail("SALSO's expected loss disagrees with the returned partition.")
}
representative <- as.integer(match(representative, unique(representative))) - 1L
RcppCNPy::npySave(output_path, representative)
emit("status", "completed")
emit_environment()
emit("draws", nrow(samples))
emit("observations", ncol(samples))
emit("draw_clusters", draw_k)
emit("requested_max_clusters", max_clusters)
emit("effective_max_clusters", info$maxNClusters[[1L]])
emit("realized_clusters", realized)
emit("requested_n_runs", n_runs)
emit("actual_n_runs", info$nRuns[[1L]])
emit("n_cores", n_cores)
emit("seed", seed)
emit("input_row_major_payload_md5", payload_md5)
emit("expected_vi_bits", sprintf("%.17g", expected_bits))
