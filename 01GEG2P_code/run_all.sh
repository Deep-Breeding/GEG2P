#!/bin/bash
# ----------Activate conda environment----------
CONDA_ENV="GEG2P"  
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

run_name="_Maize"  # Define the save directory name for each run
SNP_PATH="demo_data/genotype.csv"
PHE_PATH="demo_data/phenotype.csv"
CVF_PATH="demo_data/CVFs.csv"


TRAITS=("trait1")

KMAX=10
SNP_NUM=200

Rscript run_G2P.R \
  --plant "$run_name" \
  --snp_path "$SNP_PATH" \
  --phe_path "$PHE_PATH" \
  --cvf_path "$CVF_PATH" \
  --traits "${TRAITS[@]}" \
  --kmax $KMAX


python demo.py \
  --plant "$run_name" \
  --snp_path "$SNP_PATH" \
  --phe_path "$PHE_PATH" \
  --cvf_path "$CVF_PATH" \
  --traits "${TRAITS[@]}" \
  --kmax $KMAX \
  --snp_num $SNP_NUM \
  --run_ML --run_DL --run_predict_ML --run_predict_DL --run_GEG2P
