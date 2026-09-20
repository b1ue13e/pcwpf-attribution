#!/bin/bash
# 优先级实验运行脚本
# P0: wind_cn 全量
# P1: electricity 核心

LOG_DIR="/root/experiment_logs"
mkdir -p $LOG_DIR

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

echo "========================================"
echo "优先级实验运行"
echo "P0: wind_cn 全量 | P1: electricity 核心"
echo "========================================"
echo "启动: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 函数：运行COA-BiLSTM
run_coa() {
    local dataset=$1
    local seed=$2
    local logfile="$LOG_DIR/coa_${dataset}_${seed}.log"
    
    # 检查是否已完成
    if [ -f "$logfile" ] && grep -q "Test Metrics: R2=" "$logfile" 2>/dev/null; then
        echo "  [SKIP] COA $dataset seed=$seed: 已完成"
        return
    fi
    
    # 备份旧日志
    [ -f "$logfile" ] && mv "$logfile" "${logfile}.bak.$(date +%s)"
    
    python3 << EOF > "$logfile" 2>&1 &
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
from coa_bilstm_experiment_v2 import run_coa_bilstm_experiment
print(f"[$(__import__('datetime').datetime.now().strftime('%H:%M:%S'))] Starting COA $dataset seed=$seed")
try:
    result = run_coa_bilstm_experiment('$dataset', $seed)
    print(f"Completed! R2={result['metrics']['R2']:.4f}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
EOF
    echo "  [START] COA $dataset seed=$seed: PID=$!"
}

# 函数：运行BWO-SVR
run_bwo_svr() {
    local dataset=$1
    local seed=$2
    local logfile="$LOG_DIR/bwo_svr_${dataset}_${seed}.log"
    
    if [ -f "$logfile" ] && grep -q "Final R2:" "$logfile" 2>/dev/null; then
        echo "  [SKIP] BWO-SVR $dataset seed=$seed: 已完成"
        return
    fi
    
    [ -f "$logfile" ] && mv "$logfile" "${logfile}.bak.$(date +%s)"
    
    python3 << EOF > "$logfile" 2>&1 &
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
from bwo_svr_experiment_v2 import run_bwo_svr_single_experiment
print(f"[$(__import__('datetime').datetime.now().strftime('%H:%M:%S'))] Starting BWO-SVR $dataset seed=$seed")
try:
    result = run_bwo_svr_single_experiment('$dataset', $seed)
    print(f"Completed! R2={result['R2']:.4f}")
except Exception as e:
    print(f"Error: {e}")
EOF
    echo "  [START] BWO-SVR $dataset seed=$seed: PID=$!"
}

# 函数：运行BWO-CNN
run_bwo_cnn() {
    local dataset=$1
    local seed=$2
    local logfile="$LOG_DIR/bwo_cnn_${dataset}_${seed}.log"
    
    if [ -f "$logfile" ] && grep -q "Final R2:" "$logfile" 2>/dev/null; then
        echo "  [SKIP] BWO-CNN $dataset seed=$seed: 已完成"
        return
    fi
    
    [ -f "$logfile" ] && mv "$logfile" "${logfile}.bak.$(date +%s)"
    
    python3 << EOF > "$logfile" 2>&1 &
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
from bwo_cnn_experiment_v2 import run_bwo_cnn_single_experiment
print(f"[$(__import__('datetime').datetime.now().strftime('%H:%M:%S'))] Starting BWO-CNN $dataset seed=$seed")
try:
    result = run_bwo_cnn_single_experiment('$dataset', $seed)
    print(f"Completed! R2={result['R2']:.4f}")
except Exception as e:
    print(f"Error: {e}")
EOF
    echo "  [START] BWO-CNN $dataset seed=$seed: PID=$!"
}

# ========== P0: wind_cn 全量 ==========
echo "【P0】wind_cn 全量实验"
echo "────────────────────────────────────────────"

# 阶段1: wind_cn COA-BiLSTM (3个并行)
echo "阶段1: COA-BiLSTM (3并行)"
for seed in 42 123 456; do
    run_coa "wind_cn" "$seed"
done
echo "等待wind_cn COA完成..."
wait
echo "✅ wind_cn COA 完成"

# 阶段2: wind_cn BWO-SVR (3个并行)
echo ""
echo "阶段2: BWO-SVR (3并行)"
for seed in 42 123 456; do
    run_bwo_svr "wind_cn" "$seed"
done
echo "等待wind_cn BWO-SVR完成..."
wait
echo "✅ wind_cn BWO-SVR 完成"

# 阶段3: wind_cn BWO-CNN (3个并行)
echo ""
echo "阶段3: BWO-CNN (3并行)"
for seed in 42 123 456; do
    run_bwo_cnn "wind_cn" "$seed"
done
echo "等待wind_cn BWO-CNN完成..."
wait
echo "✅ wind_cn BWO-CNN 完成"

echo ""
echo "✅✅✅ P0 wind_cn 全量完成! ✅✅✅"
echo ""

# ========== P1: electricity 核心 ==========
echo "【P1】electricity 核心实验"
echo "────────────────────────────────────────────"

# electricity COA-BiLSTM (3个并行)
echo "阶段4: COA-BiLSTM (3并行)"
for seed in 42 123 456; do
    run_coa "electricity" "$seed"
done
echo "等待electricity COA完成..."
wait
echo "✅ electricity COA 完成"

echo ""
echo "✅✅✅ P1 electricity 核心完成! ✅✅✅"
echo ""

# ========== 生成最终结果 ==========
echo "【生成最终结果】"
echo "────────────────────────────────────────────"

python3 << 'EOF'
import pandas as pd
import os
import re
import json

results = []

# 收集所有实验结果
for dataset in ['wind_cn', 'electricity']:
    # Traditional
    for seed in [42, 123, 456]:
        logfile = f'/root/experiment_logs/trad_{dataset}_{seed}.log'
        if os.path.exists(logfile):
            with open(logfile, 'r') as f:
                content = f.read()
                for model in ['LSTM', 'CNN-LSTM']:
                    match = re.search(rf'{model}: R2=([0-9.-]+)', content)
                    if match:
                        results.append({
                            'dataset': dataset,
                            'model': model,
                            'seed': seed,
                            'R2': float(match.group(1))
                        })
    
    # COA-BiLSTM
    for seed in [42, 123, 456]:
        logfile = f'/root/experiment_logs/coa_{dataset}_{seed}.log'
        if os.path.exists(logfile):
            with open(logfile, 'r') as f:
                content = f.read()
                match = re.search(r'R2=([0-9.-]+)', content)
                if match:
                    results.append({
                        'dataset': dataset,
                        'model': 'COA-BiLSTM',
                        'seed': seed,
                        'R2': float(match.group(1))
                    })
    
    # BWO-SVR
    for seed in [42, 123, 456]:
        logfile = f'/root/experiment_logs/bwo_svr_{dataset}_{seed}.log'
        if os.path.exists(logfile):
            with open(logfile, 'r') as f:
                content = f.read()
                match = re.search(r'R2=([0-9.-]+)', content)
                if match:
                    results.append({
                        'dataset': dataset,
                        'model': 'BWO-SVR',
                        'seed': seed,
                        'R2': float(match.group(1))
                    })
    
    # BWO-CNN (仅wind_cn)
    if dataset == 'wind_cn':
        for seed in [42, 123, 456]:
            logfile = f'/root/experiment_logs/bwo_cnn_{dataset}_{seed}.log'
            if os.path.exists(logfile):
                with open(logfile, 'r') as f:
                    content = f.read()
                    match = re.search(r'R2=([0-9.-]+)', content)
                    if match:
                        results.append({
                            'dataset': dataset,
                            'model': 'BWO-CNN',
                            'seed': seed,
                            'R2': float(match.group(1))
                        })

df = pd.DataFrame(results)
df.to_csv('/root/all_models_clean_80_20.csv', index=False)
print(f"✅ 已保存 {len(df)} 条记录到 all_models_clean_80_20.csv")
print("\n按数据集统计:")
print(df.groupby(['dataset', 'model']).size().to_string())
EOF

echo ""
echo "========================================"
echo "优先级实验全部完成!"
echo "结束: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
