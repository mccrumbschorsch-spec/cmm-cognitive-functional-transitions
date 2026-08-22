#!/usr/bin/env Rscript

# Narrow extractor for the KLoSA multiple-imputation files.  pandas cannot
# decode unrelated Korean StrL fields in some older Stata files even when a
# column subset is requested; haven can select the required numeric fields
# before conversion.  Only the first official imputation is retained.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: extract_klosa_cognition.R INPUT_DTA OUTPUT_CSV WAVE")
}

input <- args[[1]]
output <- args[[2]]
wave <- as.integer(args[[3]])
if (!requireNamespace("haven", quietly = TRUE)) {
  stop("The R package 'haven' is required")
}

imputation <- if (wave <= 5) "vimputation_" else "v_imputation_"
items <- sprintf("w%02dC%d", wave, 401:419)
columns <- c("pid", imputation, items)
data <- haven::read_dta(input, encoding = "latin1", col_select = tidyselect::all_of(columns))
keep <- !is.na(data[[imputation]]) & data[[imputation]] == 1
data <- data[keep, c("pid", items)]
utils::write.csv(data, output, row.names = FALSE, na = "")
