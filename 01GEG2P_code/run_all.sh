#!/bin/bash
# ----------Activate conda environment----------
CONDA_ENV="GEG2P"  
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

run_name="_Maize"  # Define the save directory name for each run
SNP_PATH="data/CUBIC1404-42938-BLUP/1404_42938_geno.csv"
PHE_PATH="data/CUBIC1404-42938-BLUP/Agronomic_23Traits.csv"
CVF_PATH="data/CUBIC1404-42938-BLUP/CVF.csv"


TRAITS=("EW")

KMAX=10
SNP_NUM=42938

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