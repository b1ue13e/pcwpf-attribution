#!/bin/bash
# 全并行GPU优化版本 - 同时运行所有类型的实验

LOG_DIR="/root/experiment_logs"
mkdir -p $LOG_DIR

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  GPU全并行实验运行器${NC}"
echo -e "${GREEN}========================================${NC}"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 启动传统基线（4数据集并行）
echo -e "${BLUE}阶段1: 启动传统基线 (4数据集并行)${NC}"
echo "────────────────────────────────────────"

for dataset in wind_cn electricity air_quality gefcom; do
    logfile="$LOG_DIR/traditional_${dataset}.log"
    
    python3 << EOF > $logfile 2>&1 &
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
from quick_baselines_experiment import QuickBaselinesExperiment
exp = QuickBaselinesExperiment()
dataset = '$dataset'
seeds = [42, 123, 456]
models = ['ARIMA', 'SVR', 'LSTM', 'CNN-LSTM']
print(f"Starting {dataset}...")
for seed in seeds:
    print(f"\nSeed: {seed}")
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
                print(f"  {model_name}: R2={result['R2']:.4f}")
        except Exception as e:
            print(f"  {model_name}: Error - {e}")
print(f"\n{dataset} completed!")
with open('$LOG_DIR/traditional_${dataset}_done', 'w') as f:
    f.write('done')
EOF
    
    PID=$!
    echo "  $dataset: PID=$PID"
done

echo ""
echo "等待传统基线完成..."

# 等待所有传统基线完成
for dataset in wind_cn electricity air_quality gefcom; do
    while [ ! -f "$LOG_DIR/traditional_${dataset}_done" ]; do
        sleep 10
    done
    echo "  $dataset 完成"
done

echo -e "${GREEN}传统基线全部完成!${NC}"
echo "合并结果..."
python3 /root/add_traditional_baselines.py 2>&1 | tee -a $LOG_DIR/parallel.log

# 阶段2: 启动COA-BiLSTM（2数据集并行，因为GPU密集型）
echo ""
echo -e "${BLUE}阶段2: 启动COA-BiLSTM (2数据集并行)${NC}"
echo "────────────────────────────────────────"

run_coa() {
    local dataset=$1
    logfile="$LOG_DIR/coa_${dataset}.log"
    
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
        print(f"\nSeed: {seed}")
        result = run_coa_bilstm_experiment(dataset, seed)
        print(f"  R2={result['metrics']['R2']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
print(f"\n{dataset} completed!")
with open('$LOG_DIR/coa_${dataset}_done', 'w') as f:
    f.write('done')
EOF
}

# 分批运行COA
run_coa wind_cn &
PID1=$!
run_coa electricity &
PID2=$!

wait $PID1
wait $PID2

echo "  wind_cn, electricity 完成"

run_coa air_quality &
PID3=$!
run_coa gefcom &
PID4=$!

wait $PID3
wait $PID4

echo "  air_quality, gefcom 完成"

echo -e "${GREEN}COA-BiLSTM全部完成!${NC}"

# 阶段3: 启动BWO-SVR和BWO-CNN（并行）
echo ""
echo -e "${BLUE}阶段3: 启动BWO优化实验${NC}"
echo "────────────────────────────────────────"

# BWO-SVR
python3 << EOF > $LOG_DIR/bwo_svr.log 2>&1 &
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
print("Starting BWO-SVR experiments...")
# 运行BWO-SVR
from bwo_svr_experiment_v2 import run_all_experiments
try:
    run_all_experiments()
    print("BWO-SVR completed!")
except Exception as e:
    print(f"BWO-SVR Error: {e}")
    import traceback
    traceback.print_exc()
with open('$LOG_DIR/bwo_svr_done', 'w') as f:
    f.write('done')
EOF
BWO_SVR_PID=$!
echo "  BWO-SVR: PID=$BWO_SVR_PID"

# BWO-CNN
python3 << EOF > $LOG_DIR/bwo_cnn.log 2>&1 &
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
print("Starting BWO-CNN experiments...")
from bwo_cnn_experiment_v2 import run_all_experiments
try:
    run_all_experiments()
    print("BWO-CNN completed!")
except Exception as e:
    print(f"BWO-CNN Error: {e}")
    import traceback
    traceback.print_exc()
with open('$LOG_DIR/bwo_cnn_done', 'w') as f:
    f.write('done')
EOF
BWO_CNN_PID=$!
echo "  BWO-CNN: PID=$BWO_CNN_PID"

echo "等待BWO实验完成..."
wait $BWO_SVR_PID
echo "  BWO-SVR 完成"
wait $BWO_CNN_PID
echo "  BWO-CNN 完成"

# 生成最终输出
echo ""
echo -e "${BLUE}阶段4: 生成最终输出${NC}"
python3 /root/generate_remaining_outputs.py 2>&1 | tee -a $LOG_DIR/parallel.log
python3 /root/finalize_metaheuristic_outputs.py 2>&1 | tee -a $LOG_DIR/parallel.log

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有实验完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 清理标记文件
rm -f $LOG_DIR/*_done

# 显示结果
echo ""
echo "生成的CSV文件:"
ls -lh /root/*.csv | grep -E "(all_models|traditional|metaheuristic|robustness|dispatch)" | awk '{printf "  %-40s %8s\n", $9, $5}'
