#!/bin/bash

# ==================== 启动GEG2P集成 ====================
echo ""
echo ">>> 启动 GEG2P 集成..."
echo ""

# 构建traits参数
TRAITS_ARG=$(echo "$TRAITS" | tr ' ' '\n' | awk '{printf "%s ", $0}' | sed 's/ $//')

python demo_parallel.py \
    --plant "$PLANT" \
    --snp_path "$SNP_PATH" \
    --phe_path "$PHE_PATH" \
    --cvf_path "$CVF_PATH" \
    --traits $TRAITS_ARG \
    --kmax "$KMAX" \
    --snp_num "$SNP_NUM" \
    --run_predict_ML \
    --run_predict_DL \
    --run_GEG2P

echo ""
echo "========================================"
echo "GEG2P 集成完成！"
echo "========================================"