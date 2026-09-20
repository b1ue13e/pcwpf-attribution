#!/bin/bash
# GPU并行运行脚本 - 同时处理多个数据集

LOG_DIR="/root/experiment_logs"
mkdir -p $LOG_DIR

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   GPU并行实验运行器${NC}"
echo -e "${GREEN}========================================${NC}"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}GPU状态:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | while read line; do
        echo "  $line"
    done
    echo ""
fi

# 函数：运行单个数据集的传统基线
run_traditional_dataset() {
    local dataset=$1
    local logfile="$LOG_DIR/traditional_${dataset}.log"
    
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] 启动传统基线: $dataset${NC}" | tee -a $LOG_DIR/master_parallel.log
    
    python3 << EOF > $logfile 2>&1
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')

from quick_baselines_experiment import QuickBaselinesExperiment
exp = QuickBaselinesExperiment()

# 只运行指定数据集
dataset = '$dataset'
seeds = [42, 123, 456]
models = ['ARIMA', 'SVR', 'LSTM', 'CNN-LSTM']

print(f"Starting {dataset}...")

for seed in seeds:
    print(f"\n  Seed: {seed}")
    for model_name in models:
        try:
            if model_name == 'ARIMA':
                result = exp.run_arima(dataset, seed)
            elif model_name == 'SVR':
                result = exp.run_svr(dataset, seed)
            elif model_name == 'LSTM':
                result = exp.run_lstm(dataset, seed)
            elif model_name == 'CNN-LSTM':
                result = exp.run_cnn_lstm(dataset, seed)
            
            if result:
                print(f"    {model_name}: R2={result['R2']:.4f}")
        except Exception as e:
            print(f"    {model_name}: Error - {e}")

print(f"\n{dataset} completed!")
EOF
    
    echo -e "${GREEN}[$(date '+%H:%M:%S')] 完成: $dataset${NC}" | tee -a $LOG_DIR/master_parallel.log
}

# 函数：运行单个数据集的COA-BiLSTM
run_coa_dataset() {
    local dataset=$1
    local logfile="$LOG_DIR/coa_${dataset}.log"
    
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] 启动COA-BiLSTM: $dataset${NC}" | tee -a $LOG_DIR/master_parallel.log
    
    python3 << EOF > $logfile 2>&1
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')

from coa_bilstm_experiment_v2 import run_coa_bilstm_experiment

dataset = '$dataset'
seeds = [42, 123, 456]

print(f"Starting COA-BiLSTM on {dataset}...")

for seed in seeds:
    try:
        print(f"\n  Seed: {seed}")
        result = run_coa_bilstm_experiment(dataset, seed)
        print(f"    R2={result['metrics']['R2']:.4f}")
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{dataset} completed!")
EOF
    
    echo -e "${GREEN}[$(date '+%H:%M:%S')] 完成COA-BiLSTM: $dataset${NC}" | tee -a $LOG_DIR/master_parallel.log
}

# 清理之前的后台进程
echo "清理之前的实验进程..."
pkill -f "quick_baselines_experiment.py" 2>/dev/null
pkill -f "traditional_baselines_experiment.py" 2>/dev/null
sleep 2

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}阶段1: 传统基线实验 (4数据集并行)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 同时启动4个数据集的传统基线实验
run_traditional_dataset "wind_cn" &
PID1=$!
run_traditional_dataset "electricity" &
PID2=$!
run_traditional_dataset "air_quality" &
PID3=$!
run_traditional_dataset "gefcom" &
PID4=$!

echo "已启动4个并行进程:"
echo "  wind_cn: PID=$PID1"
echo "  electricity: PID=$PID2"
echo "  air_quality: PID=$PID3"
echo "  gefcom: PID=$PID4"

# 等待所有传统基线完成
echo ""
echo "等待传统基线完成..."
wait $PID1
wait $PID2
wait $PID3
wait $PID4

echo -e "${GREEN}传统基线全部完成!${NC}"

# 合并结果
echo "合并传统基线结果..."
python3 /root/add_traditional_baselines.py 2>&1 | tee -a $LOG_DIR/master_parallel.log

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}阶段2: COA-BiLSTM实验 (2数据集并行)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# COA-BiLSTM更耗资源，同时运行2个
run_coa_dataset "wind_cn" &
PID5=$!
run_coa_dataset "electricity" &
PID6=$!

echo "已启动2个并行进程:"
echo "  wind_cn: PID=$PID5"
echo "  electricity: PID=$PID6"

wait $PID5
wait $PID6

run_coa_dataset "air_quality" &
PID7=$!
run_coa_dataset "gefcom" &
PID8=$!

echo ""
echo "已启动剩余2个进程:"
echo "  air_quality: PID=$PID7"
echo "  gefcom: PID=$PID8"

wait $PID7
wait $PID8

echo -e "${GREEN}COA-BiLSTM全部完成!${NC}"

# 生成最终输出
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}阶段3: 生成最终输出${NC}"
echo -e "${BLUE}========================================${NC}"

python3 /root/generate_remaining_outputs.py 2>&1 | tee -a $LOG_DIR/master_parallel.log
python3 /root/finalize_metaheuristic_outputs.py 2>&1 | tee -a $LOG_DIR/master_parallel.log

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有实验完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 显示结果
ls -lh /root/*.csv | grep -E "(all_models|traditional|metaheuristic|robustness|dispatch)" | awk '{printf "  %-40s %8s\n", $9, $5}'
