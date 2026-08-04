library(G2P)
library(data.table)
library(readr)
library(optparse)

# Save each model to a separate temp file to avoid concurrent write conflicts
# Format: results/<plant>/k<k>/<trait>_<model>.csv
save_model_predictions <- function(plant, k, trait, model, pred_res) {
  file_path <- sprintf("results/%s/k%d/%s_%s.csv", plant, k, trait, model)
  dir.create(dirname(file_path), recursive = TRUE, showWarnings = FALSE)

  # Extract test set ID and predictions (second column is usually the model's prediction)
  ids <- rownames(pred_res)
  if (is.null(ids) || all(is.na(ids))) {
    if ("ID" %in% colnames(pred_res)) {
      ids <- pred_res[, "ID"]
    } else {
      stop("pred_res lacks rownames or ID column, cannot save.")
    }
  }
  vals  <- pred_res[, 2, drop = TRUE]

  newdf <- data.frame(ID = ids, Prediction = vals, check.names = FALSE, stringsAsFactors = FALSE)
  write.csv(newdf, file_path, row.names = FALSE)
}

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
  make_option("--snp_path", type = "character"),
  make_option("--phe_path", type = "character"),
  make_option("--cvf_path", type = "character"),
  make_option("--traits",  type = "character"),
  make_option("--kmax",    type = "integer",   default = 10),
  make_option("--model",   type = "character",
              help = "Single model name, e.g., 'BayesA'. If not provided, run all models (serial)"),
  make_option("--models",  type = "character",
              help = "Model name list, separated by space or comma, e.g., 'BayesA BayesB RRBLUP' or 'BayesA,BayesB,RRBLUP'")
)
opt <- parse_args(OptionParser(option_list = option_list))

plant   <- opt$plant
snp_path <- opt$snp_path
phe_path <- opt$phe_path
cvf_path <- opt$cvf_path
traits  <- unlist(strsplit(opt$traits, " "))
kmax    <- opt$kmax

# 判断是单模型模式还是多模型模式
single_model_mode <- !is.null(opt$model) && nzchar(opt$model)

if (single_model_mode) {
  # 单模型模式：只训练指定的模型
  model_list <- opt$model
  cat("=== 单模型并行模式 ===\n")
  cat("Model:", model_list, "\n")
} else {
  # 多模型模式：支持空格或逗号分隔；未提供则用默认
  if (!is.null(opt$models) && nzchar(opt$models)) {
    model_list <- unlist(strsplit(opt$models, "[,\\s]+"))
    model_list <- model_list[nzchar(model_list)]
  } else {
    model_list <- c("BayesA", "BayesB", "BayesC", "BL", "BRR",
                    "RRBLUP", "LASSO", "SPLS", "RR", "BRNN")
      # model_list <- c("BayesA", "BayesB", "BayesC", "BL", "BRR",
      #               "RRBLUP", "LASSO", "SPLS", "RR")
  }
  cat("=== 多模型串行模式 ===\n")
  cat("Models:", paste(model_list, collapse = ", "), "\n")
}

cat("Plant:", plant, "\n")
cat("Traits:", paste(traits, collapse = " "), "\n")
cat("K-fold:", kmax, "\n")

# ========== 读取数据 ==========
pheData <- fread(phe_path, header = TRUE, stringsAsFactors = FALSE)
df <- read.csv(cvf_path, stringsAsFactors = FALSE, check.names = FALSE)
geno_df <- as.data.frame(read_csv(snp_path, col_names = TRUE, show_col_types = FALSE))

# ========== 检查和对齐样本 ==========
cat("检查样本对应关系...\n")

# 提取各文件的样本ID
cvf_ids <- df[[1]]
phe_ids <- pheData[[1]]
geno_ids <- geno_df[[1]]

# 处理ID格式转换函数（处理 A/B 和 A_X_B 格式）
process_id <- function(id_str) {
  if (grepl("_X_", id_str)) {
    parts <- strsplit(id_str, "_X_")[[1]]
    if (length(parts) == 2) {
      return(paste(parts[2], parts[1], sep = "/"))
    }
  }
  return(id_str)
}

# 标准化所有ID，防止不同文件ID格式不同
cvf_ids_std <- sapply(cvf_ids, process_id)
phe_ids_std <- sapply(phe_ids, process_id)
geno_ids_std <- sapply(geno_ids, process_id)

# 检查样本数量
cat(sprintf("CVF文件样本数: %d\n", length(cvf_ids)))
cat(sprintf("表型文件样本数: %d\n", length(phe_ids)))
cat(sprintf("基因型文件样本数: %d\n", length(geno_ids)))

# 检查样本匹配情况 - 确保CVF中的所有样本都在phe和geno文件中存在
cat("检查CVF样本是否在其他文件中完整存在...\n")

cvf_set <- unique(cvf_ids_std)
phe_set <- unique(phe_ids_std)
geno_set <- unique(geno_ids_std)

# 检查缺失样本
phe_missing <- setdiff(cvf_set, phe_set)
geno_missing <- setdiff(cvf_set, geno_set)

# 如果基因型或表型中样本比CVF多，输出警告并且以CVF中的样本为准
if (length(phe_missing) > 0) {
  cat(sprintf("警告: 表型文件中缺少 %d 个CVF文件中的样本, 将以CVF为准\n", length(phe_missing)))
  cat("前10个缺失样本:", paste(head(phe_missing, 10), collapse = ", "), "\n")
}

if (length(geno_missing) > 0) {
  cat(sprintf("警告: 基因型文件中缺少 %d 个CVF文件中的样本, 将以CVF为准\n", length(geno_missing)))
  cat("前10个缺失样本:", paste(head(geno_missing, 10), collapse = ", "), "\n")
}

# 按照CVF顺序重新排序数据，防止样本顺序不一致
cat("按照CVF顺序重新排序数据...\n")

# 重新排序表型数据
phe_ordered_indices <- match(cvf_ids_std, phe_ids_std)
if (any(is.na(phe_ordered_indices))) {
  stop("错误: 在表型文件中找不到某些CVF样本")
}
phe_ordered <- pheData[phe_ordered_indices, , drop = FALSE]

# 重新排序基因型数据
geno_ordered_indices <- match(cvf_ids_std, geno_ids_std)
if (any(is.na(geno_ordered_indices))) {
  stop("错误: 在基因型文件中找不到某些CVF样本")
}
geno_ordered <- geno_df[geno_ordered_indices, , drop = FALSE]

# 使用排序后的数据
rownames(geno_ordered) <- geno_ordered[[1]]
geno_ordered <- geno_ordered[, -1, drop = FALSE]
Markers <- as.matrix(geno_ordered)

# 更新表型数据
pheData <- phe_ordered

# 验证对齐结果
cat("验证对齐结果...\n")
aligned_phe_ids <- sapply(pheData[[1]], process_id)
aligned_geno_ids <- sapply(rownames(Markers), process_id)

if (!identical(aligned_phe_ids, cvf_ids_std)) {
  stop("错误: 表型数据对齐失败")
}

if (!identical(aligned_geno_ids, cvf_ids_std)) {
  stop("错误: 基因型数据对齐失败")
}

cat("样本对齐完成！\n\n")

# 填充表型数据缺失值
data <- apply(pheData[, -1, drop = FALSE], 2, function(x) {
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  x
})
data <- as.data.frame(data, stringsAsFactors = FALSE, check.names = FALSE)

# ========== 主循环：性状 → 模型 → 折 ==========
for (trait in traits) {
  cat("Running trait:", trait, "\n")
  
  for (model in model_list) {
    cat("  Model:", model, "\n")
    
    # 该模型的所有折结果列表（用于评估汇总）
    results_list <- list()
    
    for (k in 1:kmax) {
      test_index  <- which(df$cv_1 == k)
      train_index <- which(df$cv_1 != k)
      
      # G2P 单模型预测
      pred_res <- G2P(
        markers      = Markers,
        data         = data,
        trait        = trait,
        modelMethods = model,
        trainIdx     = train_index,
        predIdx      = test_index,
        saveAt       = ""
      )
      pred_res[is.na(pred_res)] <- 0
      
      # 保存到独立的临时文件
      save_model_predictions(plant, k, trait, model, pred_res)
      
      # 评估
      evalres <- G2PEvaluation(
        realScores = pred_res[, 1],
        predScores = pred_res[, -1, drop = FALSE],
        evalMethod = c("pearson", "MSE", "R2"),
        topAlpha   = test_index,
        probIndex  = 1
      )
      results_list[[k]] <- evalres$corMethods
    }
    
    # 汇总该模型 k 折结果
    results_matrix <- do.call(rbind, results_list)

    # 更稳健的均值计算：按行名匹配全部 pearson/MSE/R2 行后再取列均值
    row_sel <- function(mat, key) which(tolower(rownames(mat)) == tolower(key))

    idx_p <- row_sel(results_matrix, "pearson")
    idx_m <- row_sel(results_matrix, "MSE")
    idx_r <- row_sel(results_matrix, "R2")

    if (length(idx_p) == 0 || length(idx_m) == 0 || length(idx_r) == 0) {
      stop("G2PEvaluation 返回的行名不含 pearson/MSE/R2，无法汇总。")
    }

    mean_pearson <- colMeans(results_matrix[idx_p, , drop = FALSE], na.rm = TRUE)
    mean_mse     <- colMeans(results_matrix[idx_m, , drop = FALSE], na.rm = TRUE)
    mean_r2      <- colMeans(results_matrix[idx_r, , drop = FALSE], na.rm = TRUE)

    results_matrix <- rbind(
      results_matrix,
      AVE_pearson = mean_pearson,
      AVE_MSE     = mean_mse,
      AVE_R2      = mean_r2
    )
    
    # 保存该模型的汇总评估（保持你的原有路径结构与节奏）
    eval_path <- sprintf("results/%s/summary/%s_%s_eval.csv",
                         plant, trait, model)
    dir.create(dirname(eval_path), recursive = TRUE, showWarnings = FALSE)
    write.csv(results_matrix, file = eval_path, row.names = TRUE)
  }
}

# ========== 多模型串行模式：合并所有模型结果 ==========
if (!single_model_mode) {
  cat("\n=== 合并所有模型的预测结果 ===\n")
  for (trait in traits) {
    cat("合并性状:", trait, "\n")
    for (k in 1:kmax) {
      merge_model_predictions(plant, k, trait, model_list)
    }
  }
  cat("所有模型结果合并完成！\n")
}