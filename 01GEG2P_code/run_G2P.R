library(G2P)
library(data.table)
library(readr)
library(optparse)

# To avoid slow saving after running all models, append results after each model run.
# Save to results/<plant>/k<k>/<trait>.csv, first column is ID, rest are model columns.
save_fold_trait_column <- function(plant, k, trait, model, pred_res) {
  file_path <- sprintf("results/%s/k%d/%s.csv", plant, k, trait)
  dir.create(dirname(file_path), recursive = TRUE, showWarnings = FALSE)

  # Extract test set ID and predicted values 
  ids <- rownames(pred_res)
  if (is.null(ids) || all(is.na(ids))) {
    if ("ID" %in% colnames(pred_res)) {
      ids <- pred_res[, "ID"]
    } else {
      stop("pred_res lacks rownames or ID column, cannot save.")
    }
  }
  vals  <- pred_res[, 2, drop = TRUE]

  newdf <- data.frame(ID = ids, tmp = vals, check.names = FALSE, stringsAsFactors = FALSE)
  names(newdf)[2] <- model

  if (file.exists(file_path)) {
    existing <- read.csv(file_path, stringsAsFactors = FALSE, check.names = FALSE)
    if (!"ID" %in% names(existing)) names(existing)[1] <- "ID"

    # If a column with the same model name exists, remove the old column first to avoid generating .x/.y
    if (model %in% names(existing)) {
      existing[[model]] <- NULL
    }

    merged <- merge(existing, newdf, by = "ID", all = TRUE, sort = FALSE)
    # Keep original row order priority
    order_ids <- unique(c(existing$ID, newdf$ID))
    merged <- merged[match(order_ids, merged$ID), ]
    write.csv(merged, file_path, row.names = FALSE)
  } else {
    write.csv(newdf, file_path, row.names = FALSE)
  }
}

# ========== Parameter Settings ==========
option_list <- list(
  make_option("--plant",   type = "character", default = "_Mazie"),
  make_option("--snp_path", type = "character"),
  make_option("--phe_path", type = "character"),
  make_option("--cvf_path", type = "character"),
  make_option("--traits",  type = "character"),
  make_option("--kmax",    type = "integer",   default = 10),
  make_option("--models",  type = "character",
              help = "List of model names, separated by spaces or commas, e.g.: 'BayesA BayesB RRBLUP' or 'BayesA,BayesB,RRBLUP'")
)
opt <- parse_args(OptionParser(option_list = option_list))

plant   <- opt$plant
snp_path <- opt$snp_path
phe_path <- opt$phe_path
cvf_path <- opt$cvf_path
traits  <- unlist(strsplit(opt$traits, " "))
kmax    <- opt$kmax

# models: support space or comma separation; use default if not provided
if (!is.null(opt$models) && nzchar(opt$models)) {
  model_list <- unlist(strsplit(opt$models, "[,\\s]+"))
  model_list <- model_list[nzchar(model_list)]
} else {
  model_list <- c("BayesA", "BayesB", "BayesC", "BL", "BRR",
                  "RRBLUP", "LASSO", "SPLS", "RR", "BRNN")
}

cat("Plant:", plant, "\n")
cat("Traits:", paste(traits, collapse = " "), "\n")
cat("Models:", paste(model_list, collapse = ", "), "\n")
cat("K-fold:", kmax, "\n")

# ========== Read Data ==========
pheData <- fread(phe_path, header = TRUE, stringsAsFactors = FALSE)
df <- read.csv(cvf_path, stringsAsFactors = FALSE, check.names = FALSE)
geno_df <- as.data.frame(read_csv(snp_path, col_names = TRUE, show_col_types = FALSE))

# ========== Check and Align Samples ==========
cat("Checking sample correspondence...\n")

# Extract sample IDs from each file
cvf_ids <- df[[1]]
phe_ids <- pheData[[1]]
geno_ids <- geno_df[[1]]

# ID format conversion function
process_id <- function(id_str) {
  if (grepl("_X_", id_str)) {
    parts <- strsplit(id_str, "_X_")[[1]]
    if (length(parts) == 2) {
      return(paste(parts[2], parts[1], sep = "/"))
    }
  }
  return(id_str)
}

# Standardize all IDs to prevent different ID formats in different files
cvf_ids_std <- sapply(cvf_ids, process_id)
phe_ids_std <- sapply(phe_ids, process_id)
geno_ids_std <- sapply(geno_ids, process_id)

# Check sample counts
cat(sprintf("CVF file sample count: %d\n", length(cvf_ids)))
cat(sprintf("Phenotype file sample count: %d\n", length(phe_ids)))
cat(sprintf("Genotype file sample count: %d\n", length(geno_ids)))

# Check sample matching - ensure all samples in CVF exist in phe and geno files
cat("Checking if CVF samples exist completely in other files...\n")

cvf_set <- unique(cvf_ids_std)
phe_set <- unique(phe_ids_std)
geno_set <- unique(geno_ids_std)

# Check missing samples
phe_missing <- setdiff(cvf_set, phe_set)
geno_missing <- setdiff(cvf_set, geno_set)

# If genotype or phenotype has more samples than CVF, output warning and use CVF samples as standard
if (length(phe_missing) > 0) {
  cat(sprintf("Warning: %d CVF samples missing in phenotype file, will use CVF as standard\n", length(phe_missing)))
  cat("First 10 missing samples:", paste(head(phe_missing, 10), collapse = ", "), "\n")
}

if (length(geno_missing) > 0) {
  cat(sprintf("Warning: %d CVF samples missing in genotype file, will use CVF as standard\n", length(geno_missing)))
  cat("First 10 missing samples:", paste(head(geno_missing, 10), collapse = ", "), "\n")
}

# Reorder data according to CVF order to prevent inconsistent sample order
cat("Reordering data according to CVF order...\n")

# Reorder phenotype data
phe_ordered_indices <- match(cvf_ids_std, phe_ids_std)
if (any(is.na(phe_ordered_indices))) {
  stop("Error: Some CVF samples not found in phenotype file")
}
phe_ordered <- pheData[phe_ordered_indices, , drop = FALSE]

# Reorder genotype data
geno_ordered_indices <- match(cvf_ids_std, geno_ids_std)
if (any(is.na(geno_ordered_indices))) {
  stop("Error: Some CVF samples not found in genotype file")
}
geno_ordered <- geno_df[geno_ordered_indices, , drop = FALSE]

# Use reordered data
rownames(geno_ordered) <- geno_ordered[[1]]
geno_ordered <- geno_ordered[, -1, drop = FALSE]
Markers <- as.matrix(geno_ordered)

# Update phenotype data
pheData <- phe_ordered

# Verify alignment results
cat("Verifying alignment results...\n")
aligned_phe_ids <- sapply(pheData[[1]], process_id)
aligned_geno_ids <- sapply(rownames(Markers), process_id)

if (!identical(aligned_phe_ids, cvf_ids_std)) {
  stop("Error: Phenotype data alignment failed")
}

if (!identical(aligned_geno_ids, cvf_ids_std)) {
  stop("Error: Genotype data alignment failed")
}

cat("Sample alignment completed!\n\n")

# Fill missing values in phenotype data
data <- apply(pheData[, -1, drop = FALSE], 2, function(x) {
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  x
})
data <- as.data.frame(data, stringsAsFactors = FALSE, check.names = FALSE)

# ========== Main Loop: Trait → Model → Fold ==========
for (trait in traits) {
  cat("Running trait:", trait, "\n")
  
  for (model in model_list) {
    cat("  Model:", model, "\n")
    
    # List of all fold results for this model
    results_list <- list()
    
    for (k in 1:kmax) {
      test_index  <- which(df$cv_1 == k)
      train_index <- which(df$cv_1 != k)
      
      # G2P single model prediction
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
      
      # Immediately save prediction results for this model and fold 
      save_fold_trait_column(plant, k, trait, model, pred_res)
      
      # Evaluation
      evalres <- G2PEvaluation(
        realScores = pred_res[, 1],
        predScores = pred_res[, -1, drop = FALSE],
        evalMethod = c("pearson", "MSE", "R2"),
        topAlpha   = test_index,
        probIndex  = 1
      )
      results_list[[k]] <- evalres$corMethods
    }
    
    # Summarize k-fold results for this model
    results_matrix <- do.call(rbind, results_list)

    # More robust mean calculation: match all pearson/MSE/R2 rows by row name then take column mean
    row_sel <- function(mat, key) which(tolower(rownames(mat)) == tolower(key))

    idx_p <- row_sel(results_matrix, "pearson")
    idx_m <- row_sel(results_matrix, "MSE")
    idx_r <- row_sel(results_matrix, "R2")

    if (length(idx_p) == 0 || length(idx_m) == 0 || length(idx_r) == 0) {
      stop("G2PEvaluation returned row names do not include pearson/MSE/R2, cannot summarize.")
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
    
    # Save summary evaluation for this model
    eval_path <- sprintf("results/%s/summary/%s_%s_eval.csv",
                         plant, trait, model)
    dir.create(dirname(eval_path), recursive = TRUE, showWarnings = FALSE)
    write.csv(results_matrix, file = eval_path, row.names = TRUE)
  }
}