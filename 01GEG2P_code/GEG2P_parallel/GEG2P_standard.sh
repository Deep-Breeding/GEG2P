#!/bin/bash

# ==================== Parameter Definition ====================
# Basic Configuration
PLANT="_Mazie_parallel_5class_PH"  # Save folder name
SNP_PATH="/home/wlg/running/data/data_jichengxuexi/5物种训练集/CUBIC1404-42938-BLUP/1404_42938_geno.csv"  # 基因型
PHE_PATH="/home/wlg/running/data/data_jichengxuexi/5物种训练集/CUBIC1404-42938-BLUP/Agronomic_23Traits.csv"  # 表型
CVF_PATH="/home/wlg/running/data/data42938_all/data42938_all/1404/1404_fold_5class/5fold_cvf.csv"  # CVF
TRAITS="PH"  # Multiple traits separated by spaces
KMAX=5
SNP_NUM=42938  # SNP count
DEVICE="cuda"

# CPU core limit, default limits each process to 4 CPU cores, set to 0 for no limit
MAX_CPU_CORES=4

# Parallel Control
MAX_PARALLEL=10  # Maximum number of statistical models running in parallel

# Model Configuration
G2P_MODELS="BayesA BayesB BayesC BL BRR RRBLUP LASSO SPLS RR BRNN"  # 统计学模型
ML_MODELS="KNN XGBoost MLP RandomForest SVR"  # 机器学习模型
DL_MODELS="LCNN gmlp DNNGP DLGWAS DeepGS"  # 深度学习模型

# Script Paths
SCRIPT_DIR="/home/wlg/running/codes/GEG2P_parallel"  # 脚本根目录
G2P_SCRIPT="$SCRIPT_DIR/run_G2P_parallel.sh"  # 统计学训练脚本，默认无需更改
PARALLEL_TRAINING_SCRIPT="$SCRIPT_DIR/run_parallel_training.sh"  # 机器和深度脚本，默认无需更改
GEG2P_SCRIPT="$SCRIPT_DIR/run_GEG2P.sh"  # 集成脚本，默认无需更改


# 检测GPU数量
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        if [ "$GPU_COUNT" -eq 0 ]; then
            DEVICE="cpu"
        fi
    else
        GPU_COUNT=0
        DEVICE="cpu"
    fi
}

# 验证文件存在性
validate_files() {
    if [ ! -f "$SNP_PATH" ]; then
        echo "SNP文件不存在: $SNP_PATH"
        exit 1
    fi
    
    if [ ! -f "$PHE_PATH" ]; then
        echo "表型文件不存在: $PHE_PATH"
        exit 1
    fi
    
    if [ ! -f "$CVF_PATH" ]; then
        echo "CVF文件不存在: $CVF_PATH"
        exit 1
    fi
    
    if [ ! -f "$G2P_SCRIPT" ]; then
        echo "G2P脚本不存在: $G2P_SCRIPT"
        exit 1
    fi
    
    if [ ! -f "$PARALLEL_TRAINING_SCRIPT" ]; then
        echo "并行训练脚本不存在: $PARALLEL_TRAINING_SCRIPT"
        exit 1
    fi
    
    if [ ! -f "$GEG2P_SCRIPT" ]; then
        echo "GEG2P脚本不存在: $GEG2P_SCRIPT"
        exit 1
    fi
}


# 检查是否有命令行参数，如果有则提示需要直接定义在脚本中
if [ $# -gt 0 ]; then
    echo "警告: 此脚本不支持命令行参数，请直接修改脚本中的参数定义"
    exit 1
fi

# 检测GPU
detect_gpus

# 验证文件
validate_files

# 设置环境变量供子脚本使用
export PLANT SNP_PATH PHE_PATH CVF_PATH TRAITS KMAX SNP_NUM DEVICE MAX_PARALLEL G2P_MODELS ML_MODELS DL_MODELS MAX_CPU_CORES

# 清理函数：终止所有子进程
cleanup() {
    echo ""
    echo "检测到中断信号，正在清理..."
    
    # 终止G2P训练子进程及其所有子进程
    if [ -n "$G2P_PID" ] && kill -0 $G2P_PID 2>/dev/null; then
        echo "终止G2P训练进程 (PID: $G2P_PID)..."
        pkill -P $G2P_PID 2>/dev/null
        kill $G2P_PID 2>/dev/null
    fi
    
    # 终止并行训练子进程及其所有子进程
    if [ -n "$PARALLEL_PID" ] && kill -0 $PARALLEL_PID 2>/dev/null; then
        echo "终止并行训练进程 (PID: $PARALLEL_PID)..."
        pkill -P $PARALLEL_PID 2>/dev/null
        kill $PARALLEL_PID 2>/dev/null
    fi
    
    # 读取并终止所有已记录的子进程PID
    if [ -f "$SCRIPT_DIR/training_logs/ml_pids.txt" ]; then
        while read pid; do
            if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                kill $pid 2>/dev/null
            fi
        done < "$SCRIPT_DIR/training_logs/ml_pids.txt"
    fi
    
    if [ -f "$SCRIPT_DIR/training_logs/dl_pids.txt" ]; then
        while read pid; do
            if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                kill $pid 2>/dev/null
            fi
        done < "$SCRIPT_DIR/training_logs/dl_pids.txt"
    fi
    
    echo "清理完成，退出。"
    exit 130
}

# 设置信号处理trap
trap cleanup SIGINT SIGTERM EXIT

# 执行训练
echo "开始执行GEG2P标准流程..."
echo "CPU核心限制: ${MAX_CPU_CORES:-不限制}"

# 1. 并行训练统计学模型
echo "启动统计学基学习器训练..."
bash "$G2P_SCRIPT" &
G2P_PID=$!

# 2. 并行训练机器和深度
echo "启动深度学习和机器学习基学习器训练..."
bash "$PARALLEL_TRAINING_SCRIPT" &
PARALLEL_PID=$!

# 3. 等待两组并行任务完成
echo "等待所有并行任务完成..."
wait $G2P_PID
wait $PARALLEL_PID

echo "并行训练任务完成！"

# 移除trap，因为正常完成后不需要再清理
trap - SIGINT SIGTERM EXIT

# 4. 启动GEG2P集成
echo "启动GEG2P集成..."
bash "$GEG2P_SCRIPT"

echo "所有任务完成！"