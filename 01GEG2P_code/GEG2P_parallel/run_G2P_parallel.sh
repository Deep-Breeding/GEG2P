#!/bin/bash
# 并行训练 G2P 模型的 shell 脚本
# 每个模型作为一个独立的进程同时运行

# ========== 参数设置 ==========
# 从环境变量获取参数
PLANT="${PLANT}"
SNP_PATH="${SNP_PATH}"
PHE_PATH="${PHE_PATH}"
CVF_PATH="${CVF_PATH}"
TRAITS="${TRAITS}"  # 性状列表，空格分隔
KMAX="${KMAX}"

# 从环境变量获取模型列表
MODELS="${G2P_MODELS}"

# 从环境变量获取并行控制参数
MAX_PARALLEL="${MAX_PARALLEL}"

# R 脚本路径
R_SCRIPT="run_G2P_parallel.R"
MERGE_SCRIPT="merge_results.R"

# ========== 并行执行 ==========
echo "开始并行训练，最多同时运行 $MAX_PARALLEL 个模型..."
echo "模型列表: $MODELS"
echo "性状列表: $TRAITS"
echo ""

# 将性状转换为数组
traits_array=($TRAITS)

# 循环处理每个性状
for trait in "${traits_array[@]}"; do
    echo "========================================"
    echo "处理性状: $trait"
    echo "========================================"

    # 将模型转换为数组
    models_array=($MODELS)

    # 并行启动每个模型的训练
    for model in "${models_array[@]}"; do
        # 检查当前运行的进程数
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
            sleep 1
        done

        echo "  正在启动模型: $model"
        Rscript "$R_SCRIPT" \
            --plant "$PLANT" \
            --snp_path "$SNP_PATH" \
            --phe_path "$PHE_PATH" \
            --cvf_path "$CVF_PATH" \
            --traits "$trait" \
            --kmax $KMAX \
            --model "$model" \
            > "training_logs/SS_${trait}_${model}.out" 2> "training_logs/SS_${trait}_${model}.err" &
        echo "  启动完成: $model"

    done

    # 等待所有模型完成
    echo "  等待所有模型完成..."
    wait
    echo "  性状 $trait 的所有模型训练完成！"
done

echo ""
echo "========================================"
echo "所有模型训练完成，开始合并结果..."
echo "========================================"

# 合并所有模型的结果
MODELS_CSV=$(echo "$MODELS" | tr ' ' ',')
Rscript "$MERGE_SCRIPT" \
    --plant "$PLANT" \
    --traits "$TRAITS" \
    --kmax $KMAX \
    --models "$MODELS_CSV"

echo ""
echo "========================================"
echo "所有任务完成！"
echo "========================================"
