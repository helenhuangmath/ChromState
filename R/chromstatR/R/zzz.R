#' @importFrom methods is
#' @importFrom stats setNames
#' @importFrom utils head
NULL

# Lazily-bound handle to the Python module. Populated on first use rather than at load
# time, so that library(chromstatR) never fails just because Python is not configured
# yet -- the error surfaces when a function is actually called, with instructions.
.cs <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  .cs$module <- NULL
  invisible(NULL)
}

#' Access the underlying Python module
#'
#' Imports `chromstate` on first use and caches the handle.
#'
#' @param quiet If TRUE, return NULL instead of raising when the module is unavailable.
#' @return The reticulate module handle, or NULL.
#' @keywords internal
cs_module <- function(quiet = FALSE) {
  if (!is.null(.cs$module)) {
    return(.cs$module)
  }
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    if (quiet) return(NULL)
    stop("chromstatR requires the 'reticulate' package. Install it with ",
         "install.packages('reticulate').", call. = FALSE)
  }
  ok <- reticulate::py_module_available("chromstate")
  if (!ok) {
    if (quiet) return(NULL)
    stop(
      "The Python package 'chromstate' was not found in the active Python ",
      "environment (", reticulate::py_config()$python, ").\n",
      "Install it with chromstate_install(), or point reticulate at an ",
      "environment that has it using reticulate::use_python() / use_condaenv() ",
      "before calling chromstatR functions.",
      call. = FALSE
    )
  }
  .cs$module <- reticulate::import("chromstate", delay_load = FALSE, convert = TRUE)
  .cs$module
}

#' Is the ChromState Python backend available?
#'
#' @return TRUE when the Python module can be imported.
#' @export
#' @examples
#' chromstate_available()
chromstate_available <- function() {
  !is.null(cs_module(quiet = TRUE))
}

#' Version of the ChromState Python backend
#'
#' @return Version string.
#' @export
chromstate_version <- function() {
  cs_module()$`__version__`
}

#' Install the ChromState Python backend
#'
#' Thin wrapper around `reticulate::py_install()`.
#'
#' @param envname Target environment name. Defaults to reticulate's active environment.
#' @param method Installation method passed to `reticulate::py_install()`.
#' @param source Where to install from. Use a local path (for example the repository
#'   root) during development, or "pypi" once the package is released.
#' @param ... Further arguments to `reticulate::py_install()`.
#' @return Invisibly TRUE.
#' @export
chromstate_install <- function(envname = NULL, method = "auto",
                               source = "chromstate", ...) {
  reticulate::py_install(packages = source, envname = envname, method = method, ...)
  .cs$module <- NULL
  invisible(TRUE)
}

# Coerce a Python object to an R data.frame, tolerating pandas conversion quirks.
#
# `keep_rownames` matters for the tables that pandas indexes by state label
# (attribution summaries, emissions, transitions): dropping the index there would
# silently replace the mnemonics with 1..K and make the output unreadable.
as_df <- function(x, keep_rownames = FALSE) {
  df <- reticulate::py_to_r(x)
  if (!is.data.frame(df)) df <- as.data.frame(df, stringsAsFactors = FALSE)
  if (!keep_rownames) rownames(df) <- NULL
  df
}

# Integer literals matter: reticulate maps plain R numerics to Python floats, and
# arguments like n_states must arrive as int.
as_py_int <- function(x) {
  if (is.null(x)) return(NULL)
  as.integer(x)
}
