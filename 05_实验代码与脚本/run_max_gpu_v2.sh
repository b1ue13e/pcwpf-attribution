#!/bin/bash
# GPU利用率最大化运行脚本 v2

LOG_DIR="/root/experiment_logs"
mkdir -p $LOG_DIR

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1

echo "========================================"
echo "GPU利用率最大化运行器 v2"
echo "目标：GPU利用率>90%，显存>20GB"
echo "========================================"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查GPU
echo "【初始GPU状态】"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""

# 创建工作目录
mkdir -p /tmp/gpu_workers

# 函数：启动COA-BiLSTM实验
run_coa() {
    local dataset=$1
    local seed=$2
    local logfile="$LOG_DIR/coa_${dataset}_${seed}.log"
    
    python3 -c "
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')

from coa_bilstm_experiment_v2 import run_coa_bilstm_experiment

print(f'Starting COA-BiLSTM on $dataset seed=$seed')
result = run_coa_bilstm_experiment('$dataset', $seed)
print(f'Completed: R2={result[\"metrics\"][\"R2\"]:.4f}')
" > "$logfile" 2>&1 &
    
    echo $!
}

# 函数：启动BWO-CNN实验
run_bwo_cnn() {
    local dataset=$1
    local seed=$2
    local logfile="$LOG_DIR/bwo_cnn_${dataset}_${seed}.log"
    
    python3 -c "
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')

from bwo_cnn_experiment_v2 import run_bwo_cnn_single_experiment

print(f'Starting BWO-CNN on $dataset seed=$seed')
result = run_bwo_cnn_single_experiment('$dataset', $seed)
print(f'Completed: R2={result[\"R2\"]:.4f}')
" > "$logfile" 2>&1 &
    
    echo $!
}

# 函数：启动BWO-SVR实验
run_bwo_svr() {
    local dataset=$1
    local seed=$2
    local logfile="$LOG_DIR/bwo_svr_${dataset}_${seed}.log"
    
    python3 -c "
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')

from bwo_svr_experiment_v2 import run_bwo_svr_single_experiment

print(f'Starting BWO-SVR on $dataset seed=$seed')
result = run_bwo_svr_single_experiment('$dataset', $seed)
print(f'Completed: R2={result[\"R2\"]:.4f}')
" > "$logfile" 2>&1 &
    
    echo $!
}

# 函数：启动传统基线（GPU密集型）
run_traditional() {
    local dataset=$1
    local seed=$2
    local logfile="$LOG_DIR/trad_${dataset}_${seed}.log"
    
    python3 -c "
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')

from quick_baselines_experiment import QuickBaselinesExperiment
exp = QuickBaselinesExperiment()

print(f'Starting Traditional GPU models on $dataset seed=$seed')

for model_name in ['LSTM', 'CNN-LSTM']:
    try:
        if model_name == 'LSTM':
            result = exp.run_lstm('$dataset', $seed)
        else:
            result = exp.run_cnn_lstm('$dataset', $seed)
        print(f'{model_name}: R2={result[\"R2\"]:.4f}')
    except Exception as e:
        print(f'{model_name}: Error - {e}')

print('Completed')
" > "$logfile" 2>&1 &
    
    echo $!
}

# 启动所有任务
echo "【阶段1】启动所有COA-BiLSTM实验 (12任务)"
echo "────────────────────────────────────────────"
all_pids=""
count=0
for dataset in wind_cn electricity air_quality gefcom; do
    for seed in 42 123 456; do
        pid=$(run_coa "$dataset" "$seed")
        all_pids="$all_pids $pid"
        echo "  COA $dataset seed=$seed: PID=$pid"
        ((count++))
        sleep 2  # 错开启动时间，避免资源竞争
    done
done

echo ""
echo "【阶段2】启动所有BWO-CNN实验 (12任务)"
echo "────────────────────────────────────────────"
for dataset in wind_cn electricity air_quality gefcom; do
    for seed in 42 123 456; do
        pid=$(run_bwo_cnn "$dataset" "$seed")
        all_pids="$all_pids $pid"
        echo "  BWO-CNN $dataset seed=$seed: PID=$pid"
        ((count++))
        sleep 2
    done
done

echo ""
echo "【阶段3】启动所有BWO-SVR实验 (12任务)"
echo "────────────────────────────────────────────"
for dataset in wind_cn electricity air_quality gefcom; do
    for seed in 42 123 456; do
        pid=$(run_bwo_svr "$dataset" "$seed")
        all_pids="$all_pids $pid"
        echo "  BWO-SVR $dataset seed=$seed: PID=$pid"
        ((count++))
        sleep 2
    done
done

echo ""
echo "【阶段4】启动GPU密集型传统基线 (12任务)"
echo "────────────────────────────────────────────"
for dataset in wind_cn electricity air_quality gefcom; do
    for seed in 42 123 456; do
        pid=$(run_traditional "$dataset" "$seed")
        all_pids="$all_pids $pid"
        echo "  Traditional $dataset seed=$seed: PID=$pid"
        ((count++))
        sleep 2
    done
done

echo ""
echo "========================================"
echo "所有任务已启动！"
echo "总任务数: $count"
echo "========================================"

# 保存PID列表
echo "$all_pids" > /tmp/gpu_experiment_pids.txt
echo "PID列表已保存到: /tmp/gpu_experiment_pids.txt"

# 监控循环
echo ""
echo "开始监控GPU利用率..."
echo "按 Ctrl+C 停止监控（实验将继续在后台运行）"
echo ""

monitor_loop() {
    local counter=0
    while true; do
        # 检查还有多少个进程在运行
        running=0
        for pid in $all_pids; do
            if kill -0 $pid 2>/dev/null; then
                ((running++))
            fi
        done
        
        if [ $running -eq 0 ]; then
            echo ""
            echo "========================================"
            echo "所有实验已完成！"
            echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "========================================"
            break
        fi
        
        # 每5次循环显示一次详细状态
        if [ $((counter % 5)) -eq 0 ]; then
            clear
            echo "╔════════════════════════════════════════════════════════════╗"
            echo "║         GPU最大化并行运行监控                              ║"
            echo "║         $(date '+%Y-%m-%d %H:%M:%S')                      ║"
            echo "╚════════════════════════════════════════════════════════════╝"
            echo ""
            
            # GPU状态
            echo "【GPU状态】"
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null | while IFS=',' read -r util mem_used mem_total power temp; do
                printf "  利用率:%3s%%  显存:%6s/%6s MB  功耗:%7s W  温度:%3s°C\n" \
                    $(echo $util | tr -d ' ') \
                    $(echo $mem_used | tr -d ' ') \
                    $(echo $mem_total | tr -d ' ') \
                    $(echo $power | tr -d ' ') \
                    $(echo $temp | tr -d ' ')
            done
            echo ""
            
            # 进程统计
            completed=$((count - running))
            echo "【任务状态】"
            echo "  总任务: $count"
            echo "  已完成: $completed"
            echo "  运行中: $running"
            echo ""
            
            # 最近完成的实验
            echo "【最近完成的实验】"
            ls -lt $LOG_DIR/*.log 2>/dev/null | head -5 | while read line; do
                logfile=$(echo $line | awk '{print $9}')
                size=$(echo $line | awk '{print $5}')
                if [ "$size" -gt 100 ] 2>/dev/null; then
                    last_line=$(tail -1 "$logfile" 2>/dev/null)
                    if echo "$last_line" | grep -q "Completed\|R2="; then
                        name=$(basename "$logfile" .log)
                        echo "  ✓ $name"
                    fi
                fi
            done
            echo ""
            
            echo "按 Ctrl+C 停止监控，实验将继续在后台运行"
            echo ""
        fi
        
        ((counter++))
        sleep 3
    done
}

# 运行监控
monitor_loop
