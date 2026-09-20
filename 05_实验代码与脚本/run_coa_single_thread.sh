#!/bin/bash
# 单线程运行COA-BiLSTM实验

LOG_DIR="/root/experiment_logs"
mkdir -p $LOG_DIR

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

echo "========================================"
echo "单线程运行COA-BiLSTM实验"
echo "========================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

DATASETS="wind_cn electricity air_quality gefcom"
SEEDS="42 123 456"

completed=0
total=0

for dataset in $DATASETS; do
    for seed in $SEEDS; do
        ((total++))
        
        # 检查是否已完成
        logfile="$LOG_DIR/coa_${dataset}_${seed}.log"
        
        # 如果日志存在且包含完成标志，跳过
        if [ -f "$logfile" ]; then
            if grep -q "Test Metrics: R2=" "$logfile" 2>/dev/null; then
                echo "[$total/12] $dataset seed=$seed: 已完成，跳过"
                ((completed++))
                continue
            fi
        fi
        
        # 备份旧日志
        if [ -f "$logfile" ]; then
            mv "$logfile" "${logfile}.bak.$(date +%s)"
        fi
        
        echo ""
        echo "[$total/12] 运行: COA-BiLSTM $dataset seed=$seed"
        echo "开始: $(date '+%H:%M:%S')"
        echo "─────────────────────────────────────────"
        
        # 单线程运行
        python3 << EOF
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
import torch
from datetime import datetime

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting COA-BiLSTM on $dataset seed=$seed")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("")

try:
    from coa_bilstm_experiment_v2 import run_coa_bilstm_experiment
    result = run_coa_bilstm_experiment('$dataset', $seed)
    
    print("")
    print("="*60)
    print(f"✅ 完成!")
    print(f"R2: {result['metrics']['R2']:.4f}")
    print(f"RMSE: {result['metrics']['RMSE']:.4f}")
    print(f"MAE: {result['metrics']['MAE']:.4f}")
    print(f"结束: {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    # 保存结果到标记文件
    with open('/root/experiment_logs/coa_${dataset}_${seed}_done', 'w') as f:
        f.write(f"R2={result['metrics']['R2']:.4f}\n")
        
except Exception as e:
    print("")
    print("="*60)
    print(f"❌ 错误: {e}")
    print("="*60)
    import traceback
    traceback.print_exc()
EOF
        
        echo "结束: $(date '+%H:%M:%S')"
        ((completed++))
        
        # 每完成一个，更新结果
        echo ""
        echo "更新结果到CSV..."
        python3 << EOF
import pandas as pd
import os
import re

results = []

# 读取已有结果
if os.path.exists('/root/all_models_clean_80_20.csv'):
    df_existing = pd.read_csv('/root/all_models_clean_80_20.csv')
    results = df_existing.to_dict('records')

# 检查新完成的COA实验
for dataset in ['wind_cn', 'electricity', 'air_quality', 'gefcom']:
    for seed in [42, 123, 456]:
        done_file = f'/root/experiment_logs/coa_{dataset}_{seed}_done'
        if os.path.exists(done_file):
            # 检查是否已存在
            exists = any(r.get('dataset') == dataset and 
                        r.get('model') == 'COA-BiLSTM' and 
                        r.get('seed') == seed for r in results)
            if not exists:
                with open(done_file, 'r') as f:
                    content = f.read()
                    match = re.search(r'R2=([0-9.-]+)', content)
                    if match:
                        results.append({
                            'dataset': dataset,
                            'model': 'COA-BiLSTM',
                            'seed': seed,
                            'R2': float(match.group(1)),
                            'source': 'coa_single_thread'
                        })
                        print(f"  添加: {dataset} seed={seed} R2={match.group(1)}")

df = pd.DataFrame(results)
df.to_csv('/root/all_models_clean_80_20.csv', index=False)
print(f"总计: {len(df)} 条记录")
EOF
        
        echo "─────────────────────────────────────────"
        
    done
done

echo ""
echo "========================================"
echo "所有COA-BiLSTM实验完成!"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
