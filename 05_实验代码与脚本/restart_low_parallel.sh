#!/bin/bash
# 低并行度重启脚本 - 避免系统过载

LOG_DIR="/root/experiment_logs"
mkdir -p $LOG_DIR

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1

echo "========================================"
echo "低并行度实验重启"
echo "最大并行数: 3 (避免系统过载)"
echo "========================================"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查已完成的实验
echo "【检查已完成的实验】"
completed_trad=$(grep -l "CNN-LSTM.*R2=" /root/experiment_logs/trad_*.log 2>/dev/null | wc -l)
echo "  Traditional完成: $completed_trad / 12"

# 确定需要运行的实验
DATASETS="wind_cn electricity air_quality gefcom"
SEEDS="42 123 456"

echo ""
echo "【需要运行的实验】"
echo ""

# 函数：检查实验是否已完成
check_completed() {
    local type=$1
    local dataset=$2
    local seed=$3
    
    if [ "$type" == "trad" ]; then
        if grep -q "CNN-LSTM.*R2=" "/root/experiment_logs/trad_${dataset}_${seed}.log" 2>/dev/null; then
            return 0  # 已完成
        fi
    elif [ "$type" == "coa" ]; then
        if grep -q "Test Metrics: R2=" "/root/experiment_logs/coa_${dataset}_${seed}.log" 2>/dev/null; then
            return 0
        fi
    elif [ "$type" == "bwo_svr" ]; then
        if grep -q "Final R2:" "/root/experiment_logs/bwo_svr_${dataset}_${seed}.log" 2>/dev/null; then
            return 0
        fi
    elif [ "$type" == "bwo_cnn" ]; then
        if grep -q "Final R2:" "/root/experiment_logs/bwo_cnn_${dataset}_${seed}.log" 2>/dev/null; then
            return 0
        fi
    fi
    return 1  # 未完成
}

# 统计需要运行的实验
tasks_to_run=""
task_count=0

for type in coa bwo_svr bwo_cnn; do
    for dataset in $DATASETS; do
        for seed in $SEEDS; do
            if ! check_completed "$type" "$dataset" "$seed"; then
                tasks_to_run="$tasks_to_run $type:$dataset:$seed"
                ((task_count++))
            fi
        done
    done
done

echo "需要运行的实验: $task_count 个"
echo ""

# 由于实验太多，优先运行COA-BiLSTM（最重要的）
echo "【优先队列】"
echo "1. COA-BiLSTM (12个实验) - 元启发式核心"
echo "2. BWO-SVR (12个实验) - 已完成的可能不需要"
echo "3. BWO-CNN (12个实验) - 可选"
echo ""

# 只运行COA-BiLSTM，这是最核心的实验
echo "【启动COA-BiLSTM实验 (串行运行，避免系统过载)】"
echo ""

coa_count=0
for dataset in $DATASETS; do
    for seed in $SEEDS; do
        if ! check_completed "coa" "$dataset" "$seed"; then
            logfile="$LOG_DIR/coa_${dataset}_${seed}.log"
            
            # 如果已有日志且较大，可能是卡住的，先备份
            if [ -f "$logfile" ] && [ $(stat -c%s "$logfile" 2>/dev/null) -gt 5000 ]; then
                mv "$logfile" "${logfile}.bak.$(date +%s)"
                echo "备份旧日志: $logfile"
            fi
            
            echo "启动: COA-BiLSTM $dataset seed=$seed"
            
            python3 << EOF > "$logfile" 2>&1 &
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
import torch

print(f"Starting COA-BiLSTM on $dataset seed=$seed")
print(f"Time: {__import__('datetime').datetime.now()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("="*60)

try:
    from coa_bilstm_experiment_v2 import run_coa_bilstm_experiment
    result = run_coa_bilstm_experiment('$dataset', $seed)
    print(f"\n{'='*60}")
    print(f"Completed! R2={result['metrics']['R2']:.4f}")
    print(f"Time: {__import__('datetime').datetime.now()}")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
EOF
            
            PID=$!
            echo "  PID: $PID"
            ((coa_count++))
            
            # 限制并行数为3
            if [ $((coa_count % 3)) -eq 0 ]; then
                echo ""
                echo "等待前3个实验完成..."
                wait
                echo "继续下一批..."
                echo ""
            fi
        fi
    done
done

# 等待所有COA完成
echo ""
echo "等待所有COA-BiLSTM完成..."
wait
echo "COA-BiLSTM全部完成!"

echo ""
echo "========================================"
echo "实验完成!"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
