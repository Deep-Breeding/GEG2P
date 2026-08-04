#!/bin/bash
# 并行训练基学习器，然后启动GEG2P集成

# ==================== 配置参数 ====================
# 从环境变量获取参数
PLANT="${PLANT}"
SNP_PATH="${SNP_PATH}"
PHE_PATH="${PHE_PATH}"
CVF_PATH="${CVF_PATH}"
TRAITS="${TRAITS}"  # 多个trait用空格分隔
KMAX="${KMAX}"
SNP_NUM="${SNP_NUM}"  # 根据实际数据调整为200（201列-1列ID）
DEVICE="${DEVICE}"
MAX_CPU_CORES="${MAX_CPU_CORES:-0}"  # CPU核心限制，0表示不限制

# 从环境变量获取模型列表
ML_MODELS="${ML_MODELS}"
DL_MODELS="${DL_MODELS}"

# ==================== 检测GPU数量 ====================
echo "========================================"
echo "检测可用GPU..."
echo "========================================"

# 检测GPU数量
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "检测到 $GPU_COUNT 张GPU"
    if [ "$GPU_COUNT" -eq 0 ]; then
        echo "警告: 未检测到可用的GPU，DL模型将使用CPU"
        DEVICE="cpu"
    fi
else
    echo "警告: 未找到nvidia-smi命令，DL模型将使用CPU"
    GPU_COUNT=0
    DEVICE="cpu"
fi
echo "========================================"

# ==================== 模型列表 ====================
# 将模型字符串转换为数组
IFS=' ' read -ra ML_MODELS_ARRAY <<< "$ML_MODELS"
IFS=' ' read -ra DL_MODELS_ARRAY <<< "$DL_MODELS"

# ==================== 日志目录 ====================
LOG_DIR="training_logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "开始并行训练基学习器"
echo "========================================"
echo "Plant: $PLANT"
echo "Traits: $TRAITS"
echo "Kmax: $KMAX"
echo "SNP Num: $SNP_NUM"
echo "Device: $DEVICE"
echo "GPU Count: $GPU_COUNT"
echo "CPU核心限制: ${MAX_CPU_CORES:-不限制}"
echo "========================================"

# 转换traits为数组
TRAITS_ARRAY=($TRAITS)

# GPU分配计数器（仅用于DL模型）
GPU_INDEX=0

# ==================== 并行训练ML模型 ====================
if [ "${#ML_MODELS_ARRAY[@]}" -gt 0 ] && [ "${#TRAITS_ARRAY[@]}" -gt 0 ]; then
    echo ""
    echo ">>> 启动 ML 模型训练进程..."
    echo ""

    for trait in "${TRAITS_ARRAY[@]}"; do
        for model in "${ML_MODELS_ARRAY[@]}"; do
            LOG_FILE="$LOG_DIR/ML_${model}_${trait}.log"
            echo "  启动: $model (trait: $trait) -> 日志: $LOG_FILE"

            python train_single_model.py \
                --plant "$PLANT" \
                --trait "$trait" \
                --model_name "$model" \
                --model_type ML \
                --snp_path "$SNP_PATH" \
                --phe_path "$PHE_PATH" \
                --cvf_path "$CVF_PATH" \
                --kmax "$KMAX" \
                > "$LOG_FILE" 2>&1 &

            # 记录进程ID
            echo "$!" >> "$LOG_DIR/ml_pids.txt"
        done
    done
fi

# ==================== 并行训练DL模型 ====================
if [ "${#DL_MODELS_ARRAY[@]}" -gt 0 ] && [ "${#TRAITS_ARRAY[@]}" -gt 0 ]; then
    echo ""
    echo ">>> 启动 DL 模型训练进程..."
    echo ""

    for trait in "${TRAITS_ARRAY[@]}"; do
        for model in "${DL_MODELS_ARRAY[@]}"; do
            LOG_FILE="$LOG_DIR/DL_${model}_${trait}.log"
            
            # 计算当前任务使用的GPU ID
            if [ "$GPU_COUNT" -gt 0 ]; then
                CURRENT_GPU=$((GPU_INDEX % GPU_COUNT))
                echo "  启动: $model (trait: $trait) -> GPU: $CURRENT_GPU -> 日志: $LOG_FILE"
                
                # 设置CUDA_VISIBLE_DEVICES环境变量，只让当前进程看到指定的GPU
                CUDA_VISIBLE_DEVICES=$CURRENT_GPU python train_single_model.py \
                    --plant "$PLANT" \
                    --trait "$trait" \
                    --model_name "$model" \
                    --model_type DL \
                    --snp_path "$SNP_PATH" \
                    --phe_path "$PHE_PATH" \
                    --cvf_path "$CVF_PATH" \
                    --kmax "$KMAX" \
                    --device "$DEVICE" \
                    --num_workers 0 \
                    --snp_num "$SNP_NUM" \
                    > "$LOG_FILE" 2>&1 &
            else
                echo "  启动: $model (trait: $trait) -> CPU -> 日志: $LOG_FILE"
                
                python train_single_model.py \
                    --plant "$PLANT" \
                    --trait "$trait" \
                    --model_name "$model" \
                    --model_type DL \
                    --snp_path "$SNP_PATH" \
                    --phe_path "$PHE_PATH" \
                    --cvf_path "$CVF_PATH" \
                    --kmax "$KMAX" \
                    --device "$DEVICE" \
                    --num_workers 0 \
                    --snp_num "$SNP_NUM" \
                    > "$LOG_FILE" 2>&1 &
            fi

            # 记录进程ID
            echo "$!" >> "$LOG_DIR/dl_pids.txt"
            
            # 更新GPU索引（循环分配）
            GPU_INDEX=$((GPU_INDEX + 1))
        done
    done
fi

# ==================== 等待所有训练完成 ====================
echo ""
echo "========================================"
echo "等待所有训练进程完成..."
echo "========================================"

# 清理函数：终止所有子进程
cleanup_parallel() {
    echo ""
    echo "检测到中断信号，正在清理并行训练进程..."
    
    # 读取并终止ML子进程
    if [ -f "$LOG_DIR/ml_pids.txt" ]; then
        while read pid; do
            if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                kill $pid 2>/dev/null
                echo "  终止ML进程: $pid"
            fi
        done < "$LOG_DIR/ml_pids.txt"
    fi
    
    # 读取并终止DL子进程
    if [ -f "$LOG_DIR/dl_pids.txt" ]; then
        while read pid; do
            if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                kill $pid 2>/dev/null
                echo "  终止DL进程: $pid"
            fi
        done < "$LOG_DIR/dl_pids.txt"
    fi
    
    echo "并行训练清理完成。"
    exit 130
}

# 设置信号处理trap
trap cleanup_parallel SIGINT SIGTERM

# 等待所有后台进程
wait

# 移除trap
trap - SIGINT SIGTERM

echo ""
echo "========================================"
echo "所有基学习器训练完成！"
echo "========================================"
