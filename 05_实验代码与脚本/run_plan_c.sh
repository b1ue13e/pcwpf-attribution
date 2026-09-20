#!/bin/bash
# 方案C：只运行核心数据集 (wind_cn + electricity)
# 预计时间：8-10小时

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
echo -e "${GREEN}  方案C：核心数据集快速运行${NC}"
echo -e "${GREEN}  数据集: wind_cn + electricity${NC}"
echo -e "${GREEN}  预计时间: 8-10小时${NC}"
echo -e "${GREEN}========================================${NC}"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

DATASETS="wind_cn electricity"
SEEDS="42 123 456"

# 函数：运行Traditional（GPU模型）
run_traditional() {
    local dataset=$1
    local seed=$2
    
    # 检查是否已完成
    if grep -q "CNN-LSTM.*R2=" "$LOG_DIR/trad_${dataset}_${seed}.log" 2>/dev/null; then
        echo "  trad_${dataset}_${seed}: 已完成，跳过"
        return
    fi
    
    local logfile="$LOG_DIR/trad_${dataset}_${seed}.log"
    
    python3 << EOF > "$logfile" 2>&1 &
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
from quick_baselines_experiment import QuickBaselinesExperiment
exp = QuickBaselinesExperiment()

print(f"Traditional GPU models on $dataset seed=$seed")
for model_name in ['LSTM', 'CNN-LSTM']:
    try:
        if model_name == 'LSTM':
            result = exp.run_lstm('$dataset', $seed)
        else:
            result = exp.run_cnn_lstm('$dataset', $seed)
        print(f"{model_name}: R2={result['R2']:.4f}")
    except Exception as e:
        print(f"{model_name}: Error - {e}")
print("Completed")
EOF
    echo "  trad_${dataset}_${seed}: PID=$!"
}

# 函数：运行COA-BiLSTM
run_coa() {
    local dataset=$1
    local seed=$2
    
    # 检查是否已完成
    if grep -q "Test Metrics: R2=" "$LOG_DIR/coa_${dataset}_${seed}.log" 2>/dev/null; then
        echo "  coa_${dataset}_${seed}: 已完成，跳过"
        return
    fi
    
    local logfile="$LOG_DIR/coa_${dataset}_${seed}.log"
    
    # 备份旧日志
    if [ -f "$logfile" ] && [ $(stat -c%s "$logfile" 2>/dev/null) -gt 1000 ]; then
        mv "$logfile" "${logfile}.bak.$(date +%s)"
    fi
    
    python3 << EOF > "$logfile" 2>&1 &
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
from coa_bilstm_experiment_v2 import run_coa_bilstm_experiment

try:
    result = run_coa_bilstm_experiment('$dataset', $seed)
    print(f"Completed: R2={result['metrics']['R2']:.4f}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
EOF
    echo "  coa_${dataset}_${seed}: PID=$!"
}

# 函数：运行BWO-SVR
run_bwo_svr() {
    local dataset=$1
    local seed=$2
    
    if grep -q "Final R2:" "$LOG_DIR/bwo_svr_${dataset}_${seed}.log" 2>/dev/null; then
        echo "  bwo_svr_${dataset}_${seed}: 已完成，跳过"
        return
    fi
    
    local logfile="$LOG_DIR/bwo_svr_${dataset}_${seed}.log"
    
    python3 << EOF > "$logfile" 2>&1 &
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
from bwo_svr_experiment_v2 import run_bwo_svr_single_experiment

try:
    result = run_bwo_svr_single_experiment('$dataset', $seed)
    print(f"Completed: R2={result['R2']:.4f}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
EOF
    echo "  bwo_svr_${dataset}_${seed}: PID=$!"
}

# 函数：运行BWO-CNN
run_bwo_cnn() {
    local dataset=$1
    local seed=$2
    
    if grep -q "Final R2:" "$LOG_DIR/bwo_cnn_${dataset}_${seed}.log" 2>/dev/null; then
        echo "  bwo_cnn_${dataset}_${seed}: 已完成，跳过"
        return
    fi
    
    local logfile="$LOG_DIR/bwo_cnn_${dataset}_${seed}.log"
    
    python3 << EOF > "$logfile" 2>&1 &
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
from bwo_cnn_experiment_v2 import run_bwo_cnn_single_experiment

try:
    result = run_bwo_cnn_single_experiment('$dataset', $seed)
    print(f"Completed: R2={result['R2']:.4f}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
EOF
    echo "  bwo_cnn_${dataset}_${seed}: PID=$!"
}

# 阶段1：Traditional（串行，很快）
echo -e "${BLUE}【阶段1】Traditional 基线 (LSTM + CNN-LSTM)${NC}"
echo "────────────────────────────────────────────"
for dataset in $DATASETS; do
    for seed in $SEEDS; do
        run_traditional "$dataset" "$seed"
    done
done

echo ""
echo "等待Traditional完成..."
wait
echo -e "${GREEN}Traditional 完成!${NC}"

# 阶段2：COA-BiLSTM（3个并行）
echo ""
echo -e "${BLUE}【阶段2】COA-BiLSTM (3个并行)${NC}"
echo "────────────────────────────────────────────"
coa_batch=0
for dataset in $DATASETS; do
    for seed in $SEEDS; do
        run_coa "$dataset" "$seed"
        ((coa_batch++))
        
        # 每3个等待一次
        if [ $((coa_batch % 3)) -eq 0 ]; then
            echo "等待批次 $((coa_batch / 3)) 完成..."
            wait
            echo "继续..."
        fi
    done
done
wait
echo -e "${GREEN}COA-BiLSTM 完成!${NC}"

# 阶段3：BWO-SVR（4个并行）
echo ""
echo -e "${BLUE}【阶段3】BWO-SVR (4个并行)${NC}"
echo "────────────────────────────────────────────"
bwo_batch=0
for dataset in $DATASETS; do
    for seed in $SEEDS; do
        run_bwo_svr "$dataset" "$seed"
        ((bwo_batch++))
        
        if [ $((bwo_batch % 4)) -eq 0 ]; then
            echo "等待批次完成..."
            wait
        fi
    done
done
wait
echo -e "${GREEN}BWO-SVR 完成!${NC}"

# 阶段4：BWO-CNN（4个并行）
echo ""
echo -e "${BLUE}【阶段4】BWO-CNN (4个并行)${NC}"
echo "────────────────────────────────────────────"
bwo_cnn_batch=0
for dataset in $DATASETS; do
    for seed in $SEEDS; do
        run_bwo_cnn "$dataset" "$seed"
        ((bwo_cnn_batch++))
        
        if [ $((bwo_cnn_batch % 4)) -eq 0 ]; then
            echo "等待批次完成..."
            wait
        fi
    done
done
wait
echo -e "${GREEN}BWO-CNN 完成!${NC}"

# 整合结果
echo ""
echo -e "${BLUE}【整合结果】${NC}"
python3 << 'EOF'
import pandas as pd
import os
import re

results = []

# 读取已有结果
if os.path.exists('/root/all_models_clean_80_20.csv'):
    df_existing = pd.read_csv('/root/all_models_clean_80_20.csv')
    results = df_existing.to_dict('records')

# Traditional
for dataset in ['wind_cn', 'electricity']:
    for seed in [42, 123, 456]:
        logfile = f'/root/experiment_logs/trad_{dataset}_{seed}.log'
        if os.path.exists(logfile):
            with open(logfile, 'r') as f:
                content = f.read()
                for model in ['LSTM', 'CNN-LSTM']:
                    match = re.search(rf'{model}: R2=([0-9.-]+)', content)
                    if match:
                        exists = any(r.get('dataset') == dataset and 
                                   r.get('model') == model and 
                                   r.get('seed') == seed for r in results)
                        if not exists:
                            results.append({
                                'dataset': dataset,
                                'model': model,
                                'seed': seed,
                                'R2': float(match.group(1)),
                                'source': 'traditional'
                            })

# COA
for dataset in ['wind_cn', 'electricity']:
    for seed in [42, 123, 456]:
        logfile = f'/root/experiment_logs/coa_{dataset}_{seed}.log'
        if os.path.exists(logfile):
            with open(logfile, 'r') as f:
                content = f.read()
                match = re.search(r'R2=([0-9.-]+)', content)
                if match:
                    exists = any(r.get('dataset') == dataset and 
                               r.get('model') == 'COA-BiLSTM' and 
                               r.get('seed') == seed for r in results)
                    if not exists:
                        results.append({
                            'dataset': dataset,
                            'model': 'COA-BiLSTM',
                            'seed': seed,
                            'R2': float(match.group(1)),
                            'source': 'coa'
                        })

# BWO
for model_type in ['bwo_svr', 'bwo_cnn']:
    model_name = 'BWO-SVR' if model_type == 'bwo_svr' else 'BWO-CNN'
    for dataset in ['wind_cn', 'electricity']:
        for seed in [42, 123, 456]:
            logfile = f'/root/experiment_logs/{model_type}_{dataset}_{seed}.log'
            if os.path.exists(logfile):
                with open(logfile, 'r') as f:
                    content = f.read()
                    match = re.search(r'R2=([0-9.-]+)', content)
                    if match:
                        exists = any(r.get('dataset') == dataset and 
                                   r.get('model') == model_name and 
                                   r.get('seed') == seed for r in results)
                        if not exists:
                            results.append({
                                'dataset': dataset,
                                'model': model_name,
                                'seed': seed,
                                'R2': float(match.group(1)),
                                'source': model_type
                            })

df = pd.DataFrame(results)
df.to_csv('/root/all_models_clean_80_20.csv', index=False)
print(f"✅ 已整合 {len(df)} 条记录")
print(f"\n按数据集统计:")
print(df.groupby(['dataset', 'model']).size().to_string())
EOF

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}方案C执行完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
