# Portable raw-feature initialization. No reference-label input is accepted.
# R may start in the C locale even when Python supplies UTF-8 paths.
if (!isTRUE(l10n_info()[["UTF-8"]])) {
  for (candidate in c("C.UTF-8", "en_US.UTF-8", ".UTF-8")) {
    suppressWarnings(Sys.setlocale("LC_CTYPE", candidate))
    if (isTRUE(l10n_info()[["UTF-8"]])) break
  }
}
if (!isTRUE(l10n_info()[["UTF-8"]])) {
  stop("A UTF-8 locale is required for portable input/output paths; configure an available UTF-8 locale in R.")
}
args <- commandArgs(TRUE)
required <- c("RcppCNPy", "jsonlite", "digest")
for (package in required) {
  if (!requireNamespace(package, quietly = TRUE)) stop(paste("Missing R package:", package))
}
if (identical(args, "--check")) {
  cat("Hartigan--Wong dependencies available\n")
  quit(status = 0L)
}
stopifnot(length(args) == 1L)
item <- jsonlite::fromJSON(args[[1]])
stopifnot(unname(tools::md5sum(item$converted_path)) == item$converted_md5)
x <- RcppCNPy::npyLoad(item$converted_path, type = "numeric")
stopifnot(identical(dim(x), c(as.integer(item$n), as.integer(item$p))), all(is.finite(x)))
bytes <- readBin(item$converted_path, what = "raw", n = file.info(item$converted_path)$size)
payload <- tail(bytes, length(x) * 8L)
actual <- writeBin(as.double(t(x)), raw(), size = 8L, endian = "little")
input_md5 <- digest::digest(payload, algo = "md5", serialize = FALSE)
loaded_md5 <- digest::digest(actual, algo = "md5", serialize = FALSE)
stopifnot(input_md5 == loaded_md5)
set.seed(as.integer(item$seed))
warnings_seen <- character()
fit <- withCallingHandlers(
  stats::kmeans(x, centers = as.integer(item$K), nstart = as.integer(item$nstart),
                iter.max = as.integer(item$iter_max), algorithm = "Hartigan-Wong"),
  warning = function(w) {
    warnings_seen <<- c(warnings_seen, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)
if (fit$ifault != 0L || length(warnings_seen)) {
  stop(paste("Hartigan--Wong did not finish cleanly:", fit$ifault, paste(warnings_seen, collapse = "; ")))
}
labels <- as.integer(fit$cluster - 1L)
stopifnot(identical(sort(unique(labels)), seq_len(item$K) - 1L))
RcppCNPy::npySave(file.path(item$output_dir, "labels.npy"), labels)
result <- list(
  seed = item$seed, algorithm = "Hartigan-Wong", n = item$n, p = item$p, K = item$K,
  nstart = item$nstart, iter_max = item$iter_max,
  initialization = "R stats::kmeans default random distinct rows",
  selection = "Lowest within-cluster SSE across starts; no reference labels loaded",
  input_payload_md5 = input_md5, loaded_matrix_md5 = loaded_md5,
  total_withinss = fit$tot.withinss, ifault = fit$ifault, warnings = warnings_seen,
  iterations = fit$iter, cluster_sizes = fit$size,
  R_version = R.version.string, RNG_kind = RNGkind(),
  RcppCNPy_version = as.character(utils::packageVersion("RcppCNPy"))
)
jsonlite::write_json(result, file.path(item$output_dir, "result.json"),
                     auto_unbox = TRUE, pretty = TRUE, digits = 17, null = "null")
