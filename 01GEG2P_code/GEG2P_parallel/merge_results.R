#!/usr/bin/env Rscript
library(optparse)

# Merge all model results into one file
# Format: results/<plant>/k<k>/<trait>.csv, first column is ID, rest are model columns
merge_model_predictions <- function(plant, k, trait, model_list) {
  output_path <- sprintf("results/%s/k%d/%s.csv", plant, k, trait)
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

  merged_df <- NULL

  for (model in model_list) {
    input_path <- sprintf("results/%s/k%d/%s_%s.csv", plant, k, trait, model)
    if (!file.exists(input_path)) {
      cat(sprintf("  Warning: Model %s result file not found: %s\n", model, input_path))
      next
    }

    model_df <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
    colnames(model_df)[2] <- model

    if (is.null(merged_df)) {
      merged_df <- model_df
    } else {
      merged_df <- merge(merged_df, model_df, by = "ID", all = TRUE, sort = FALSE)
    }
  }

  if (!is.null(merged_df)) {
    write.csv(merged_df, output_path, row.names = FALSE)
    cat(sprintf("  Merged %d model results to: %s\n", length(model_list), output_path))
  } else {
    cat(sprintf("  Warning: No model result files found\n"))
  }
}

# ========== 参数设置 ==========
option_list <- list(
  make_option("--plant",   type = "character", default = "_Mazie"),
  make_option("--traits",  type = "character"),
  make_option("--kmax",    type = "integer",   default = 10),
  make_option("--models",  type = "character",
              help = "Model name list, separated by space or comma, e.g., 'BayesA BayesB RRBLUP' or 'BayesA,BayesB,RRBLUP'")
)
opt <- parse_args(OptionParser(option_list = option_list))

plant   <- opt$plant
traits  <- unlist(strsplit(opt$traits, " "))
kmax    <- opt$kmax

# 模型列表
if (!is.null(opt$models) && nzchar(opt$models)) {
  model_list <- unlist(strsplit(opt$models, ",", fixed = TRUE))
  model_list <- trimws(model_list)
  model_list <- model_list[nzchar(model_list)]
} else {
  model_list <- c("BayesA", "BayesB", "BayesC", "BL", "BRR",
                  "RRBLUP", "LASSO", "SPLS", "RR", "BRNN")
}

cat("=== 合并模型预测结果 ===\n")
cat("Plant:", plant, "\n")
cat("Traits:", paste(traits, collapse = " "), "\n")
cat("Models:", paste(model_list, collapse = ", "), "\n")
cat("K-fold:", kmax, "\n\n")

# 合并所有性状的结果
for (trait in traits) {
  cat("Merging trait:", trait, "\n")
  for (k in 1:kmax) {
    merge_model_predictions(plant, k, trait, model_list)
  }
}

cat("\n=== 所有结果合并完成！ ===\n")
