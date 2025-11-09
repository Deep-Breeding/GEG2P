#!/bin/bash

# ---------- Activate conda environment ----------
CONDA_ENV="GEG2P"   # Replace with your environment name, change the name in GEG2P_env.yml accordingly
conda env create -f GEG2P_env.yml
echo "Activating conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# ---------- Install R packages and G2P ----------
echo "Installing R packages and G2P..."
Rscript -e 'install.packages("data.table", repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")'
Rscript -e 'install.packages("optparse", repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")'
Rscript -e 'install.packages("readr", repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/")'
Rscript -e '
options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))

# List of dependency packages
pkgs <- c(
  "ggplot2","brnn","glmnet","spls","pls","e1071","BGLR","rrBLUP","randomForest",
  "hglm","hglm.data","parallel","pROC","PRROC","STPGA","reshape","reshape2",
  "grid","pbapply","pheatmap","data.table"
)

# Install missing dependencies
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}

# Download and install qtlDesign
if (!requireNamespace("qtlDesign", quietly = TRUE)) {
  url <- "https://cran.r-project.org/src/contrib/Archive/qtlDesign/qtlDesign_0.941.tar.gz"
  dest <- "qtlDesign_0.941.tar.gz"
  if (!file.exists(dest)) download.file(url, dest, mode="wb")
  install.packages(dest, repos = NULL, type = "source")
}

# Install remotes
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")

# Install G2P
if (!requireNamespace("G2P", quietly = TRUE)) remotes::install_github("G2P-env/G2P")
'
