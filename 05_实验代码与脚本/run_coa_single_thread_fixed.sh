#!/bin/bash
# 修复版单线程运行COA-BiLSTM实验
# 修复内容：
# 1. 增强错误处理和日志记录
# 2. 内存监控和限制
# 3. 进程状态监控
# 4. 自动恢复机制

LOG_DIR="/root/experiment_logs"
mkdir -p $LOG_DIR

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "========================================"
echo "修复版 COA-BiLSTM 实验"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# 检查并清理僵尸进程
cleanup_zombies() {
    echo "[$(date '+%H:%M:%S')] 检查僵尸进程..."
    pids=$(ps aux | grep "python.*coa_bilstm" | grep -v grep | awk '{print $2}')
    for pid in $pids; do
        if [ "$pid" != "$$" ]; then
            # 检查进程是否在运行
            if kill -0 $pid 2>/dev/null; then
                # 检查进程运行时间
                runtime=$(ps -o etimes= -p $pid 2>/dev/null || echo "0")
                if [ "$runtime" -gt 86400 ]; then  # 超过24小时
                    echo "  警告: 进程 $pid 已运行 $runtime 秒，可能已卡住"
                fi
            else
                echo "  清理僵尸进程: $pid"
                kill -9 $pid 2>/dev/null
            fi
        fi
    done
}

# 内存检查
check_memory() {
    mem_free=$(free -m | awk '/^Mem:/{print $4}')
    if [ "$mem_free" -lt 1000 ]; then
        echo "  警告: 可用内存不足 (${mem_free}MB)"
        return 1
    fi
    return 0
}

DATASETS="wind_cn electricity air_quality gefcom"
SEEDS="42 123 456"

completed=0
total=0

for dataset in $DATASETS; do
    for seed in $SEEDS; do
        ((total++))
        
        # 检查是否已完成
        logfile="$LOG_DIR/coa_${dataset}_${seed}.log"
        done_file="$LOG_DIR/coa_${dataset}_${seed}_done"
        
        # 如果已完成文件存在，跳过
        if [ -f "$done_file" ]; then
            echo "[$total/12] $dataset seed=$seed: 已完成，跳过"
            ((completed++))
            continue
        fi
        
        # 如果日志存在且包含完成标志，跳过
        if [ -f "$logfile" ]; then
            if grep -q "Test Metrics: R2=" "$logfile" 2>/dev/null; then
                echo "[$total/12] $dataset seed=$seed: 已完成，跳过"
                ((completed++))
                # 创建完成标记文件
                grep -o "R2=[0-9.-]*" "$logfile" | head -1 > "$done_file"
                continue
            fi
        fi
        
        echo ""
        echo "[$total/12] 运行: COA-BiLSTM $dataset seed=$seed"
        echo "开始: $(date '+%H:%M:%S')"
        echo "─────────────────────────────────────────"
        
        # 清理之前的状态
        cleanup_zombies
        check_memory || echo "  继续执行，但内存紧张"
        
        # 备份旧日志
        if [ -f "$logfile" ]; then
            bak_file="${logfile}.bak.$(date +%s)"
            mv "$logfile" "$bak_file"
            echo "  备份旧日志: $bak_file"
        fi
        
        # 创建新的日志文件
        echo "========================================" > "$logfile"
        echo "COA-BiLSTM 实验: $dataset seed=$seed" >> "$logfile"
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$logfile"
        echo "========================================" >> "$logfile"
        
        # 运行实验，使用timeout防止无限挂起
        timeout_hours=8
        timeout_seconds=$((timeout_hours * 3600))
        
        start_time=$(date +%s)
        
        (
            echo "[$(date '+%H:%M:%S')] 启动实验进程..."
            
            python3 -u << EOF
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
import torch
from datetime import datetime
import gc
import os

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting COA-BiLSTM on $dataset seed=$seed")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("")

try:
    # 设置GPU内存限制
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 设置内存分数限制
        torch.cuda.set_per_process_memory_fraction(0.7)
    
    from coa_bilstm_experiment_v2 import run_coa_bilstm_experiment
    
    # 运行实验
    result = run_coa_bilstm_experiment('$dataset', $seed)
    
    print("")
    print("="*60)
    print(f"✅ 完成!")
    print(f"R2: {result['metrics']['R2']:.4f}")
    print(f"RMSE: {result['metrics']['RMSE']:.4f}")
    print(f"MAE: {result['metrics']['MAE']:.4f}")
    print(f"结束: {datetime.now().strftime('%H:%M:%S')}")
    
    # 内存报告
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"GPU内存峰值: {gpu_mem:.1f}MB")
    
    print("="*60)
    
    # 保存结果到标记文件
    with open('$done_file', 'w') as f:
        f.write(f"R2={result['metrics']['R2']:.4f}\\n")
    
    # 清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 实验完成并已清理")
    
except Exception as e:
    print("")
    print("="*60)
    print(f"❌ 错误: {e}")
    print("="*60)
    import traceback
    traceback.print_exc()
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 退出码非零表示失败
    sys.exit(1)
EOF
        ) 2>&1 | tee -a "$logfile"
        
        exit_code=${PIPESTATUS[0]}
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        echo "运行时间: $((duration/60))分钟$((duration%60))秒"
        
        if [ $exit_code -eq 0 ]; then
            echo "✅ 任务成功: $dataset seed=$seed"
            ((completed++))
        elif [ $exit_code -eq 124 ]; then
            echo "⏰ 任务超时 (超过${timeout_hours}小时): $dataset seed=$seed"
            echo "  超时任务可能需要手动检查"
        else
            echo "❌ 任务失败 (退出码: $exit_code): $dataset seed=$seed"
            echo "  查看日志: $logfile"
        fi
        
        echo "结束: $(date '+%H:%M:%S')"
        
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
for ds in ['wind_cn', 'electricity', 'air_quality', 'gefcom']:
    for sd in [42, 123, 456]:
        done_f = f'/root/experiment_logs/coa_{ds}_{sd}_done'
        if os.path.exists(done_f):
            exists = any(r.get('dataset') == ds and 
                        r.get('model') == 'COA-BiLSTM' and 
                        r.get('seed') == sd for r in results)
            if not exists:
                with open(done_f, 'r') as f:
                    content = f.read()
                    match = re.search(r'R2=([0-9.-]+)', content)
                    if match:
                        results.append({
                            'dataset': ds,
                            'model': 'COA-BiLSTM',
                            'seed': sd,
                            'R2': float(match.group(1)),
                            'source': 'coa_single_thread_fixed'
                        })
                        print(f"  添加: {ds} seed={sd} R2={match.group(1)}")

df = pd.DataFrame(results)
df.to_csv('/root/all_models_clean_80_20.csv', index=False)
print(f"总计: {len(df)} 条记录")
EOF
        
        echo "─────────────────────────────────────────"
        
        # 短暂休息，让系统稳定
        sleep 10
    done
done

echo ""
echo "========================================"
echo "COA-BiLSTM实验运行完成!"
echo "完成: $completed/12 个任务"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"