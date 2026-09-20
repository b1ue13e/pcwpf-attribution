#!/bin/bash
# 最大化GPU利用率运行脚本
# 同时启动尽可能多的GPU密集型任务

LOG_DIR="/root/experiment_logs"
mkdir -p $LOG_DIR

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         🚀 GPU利用率最大化运行器                            ║${NC}"
echo -e "${GREEN}║         目标：GPU利用率>90%，显存>20GB                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查GPU
echo -e "${BLUE}【GPU状态】${NC}"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r name util mem_used mem_total temp; do
    printf "  %-25s 利用率:%3s%%  显存:%5s/%5sMB  温度:%3s°C\n" \
        "$(echo $name | tr -d ' ')" \
        $(echo $util | tr -d ' ') \
        $(echo $mem_used | tr -d ' ') \
        $(echo $mem_total | tr -d ' ') \
        $(echo $temp | tr -d ' ')
done
echo ""

# 清理函数
cleanup() {
    echo -e "\n${YELLOW}正在清理...${NC}"
    pkill -f "max_gpu_worker.py" 2>/dev/null
    exit 0
trap cleanup SIGINT SIGTERM

# 创建工作进程脚本
cat > /tmp/max_gpu_worker.py << 'WORKER_EOF'
import sys
import os
sys.path.append('/root')

exp_type = sys.argv[1]
dataset = sys.argv[2]
seed = int(sys.argv[3])
log_file = sys.argv[4]

import warnings
warnings.filterwarnings('ignore')

with open(log_file, 'w') as f:
    import torch
    f.write(f"Starting {exp_type} on {dataset} seed={seed}\n")
    f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
    f.write(f"Time: {__import__('datetime').datetime.now()}\n\n")
    f.flush()
    
    try:
        if exp_type == 'coa':
            from coa_bilstm_experiment_v2 import run_coa_bilstm_experiment
            result = run_coa_bilstm_experiment(dataset, seed)
            f.write(f"Completed: R2={result['metrics']['R2']:.4f}\n")
            
        elif exp_type == 'bwo_svr':
            from bwo_svr_experiment_v2 import run_bwo_svr_single_experiment
            result = run_bwo_svr_single_experiment(dataset, seed)
            f.write(f"Completed: R2={result['R2']:.4f}\n")
            
        elif exp_type == 'bwo_cnn':
            from bwo_cnn_experiment_v2 import run_bwo_cnn_single_experiment
            result = run_bwo_cnn_single_experiment(dataset, seed)
            f.write(f"Completed: R2={result['R2']:.4f}\n")
            
        elif exp_type == 'traditional':
            from quick_baselines_experiment import QuickBaselinesExperiment
            exp = QuickBaselinesExperiment()
            # 只运行GPU密集型模型
            for model_name in ['LSTM', 'CNN-LSTM']:
                try:
                    if model_name == 'LSTM':
                        result = exp.run_lstm(dataset, seed)
                    else:
                        result = exp.run_cnn_lstm(dataset, seed)
                    f.write(f"{model_name}: R2={result['R2']:.4f}\n")
                    f.flush()
                except Exception as e:
                    f.write(f"{model_name}: Error - {e}\n")
                    f.flush()
        
        f.write(f"\nAll done at {__import__('datetime').datetime.now()}\n")
        
    except Exception as e:
        f.write(f"\nError: {e}\n")
        import traceback
        f.write(traceback.format_exc())

WORKER_EOF

# 函数：启动工作进程
start_worker() {
    local exp_type=$1
    local dataset=$2
    local seed=$3
    local logfile="$LOG_DIR/${exp_type}_${dataset}_${seed}.log"
    
    python3 /tmp/max_gpu_worker.py "$exp_type" "$dataset" "$seed" "$logfile" &
    echo $!
}

# 计算可以并行的任务数
# RTX 4090 24GB，每个COA-BiLSTM约需2-3GB，可以同时运行8-10个
echo -e "${BLUE}【并行策略】${NC}"
echo "  基于GPU显存24GB，同时运行8-12个GPU密集型任务"
echo "  COA-BiLSTM: 高GPU占用 (~3GB/任务)"
echo "  BWO-CNN: 高GPU占用 (~2.5GB/任务)"
echo "  LSTM/CNN-LSTM: 中等GPU占用 (~1.5GB/任务)"
echo ""

# 同时启动所有任务
echo -e "${CYAN}【阶段1】同时启动所有COA-BiLSTM实验 (12任务)${NC}"
echo "────────────────────────────────────────────────────────────"
coa_pids=()
for dataset in wind_cn electricity air_quality gefcom; do
    for seed in 42 123 456; do
        pid=$(start_worker "coa" "$dataset" "$seed")
        coa_pids+=($pid)
        echo "  COA $dataset seed=$seed: PID=$pid"
        sleep 3  # 错开启动时间
    done
done

echo ""
echo -e "${CYAN}【阶段2】同时启动所有BWO-CNN实验 (12任务)${NC}"
echo "────────────────────────────────────────────────────────────"
bwo_cnn_pids=()
for dataset in wind_cn electricity air_quality gefcom; do
    for seed in 42 123 456; do
        pid=$(start_worker "bwo_cnn" "$dataset" "$seed")
        bwo_cnn_pids+=($pid)
        echo "  BWO-CNN $dataset seed=$seed: PID=$pid"
        sleep 3
    done
done

echo ""
echo -e "${CYAN}【阶段3】启动GPU密集型传统基线 (LSTM/CNN-LSTM)${NC}"
echo "────────────────────────────────────────────────────────────"
trad_pids=()
for dataset in wind_cn electricity air_quality gefcom; do
    for seed in 42 123 456; do
        pid=$(start_worker "traditional" "$dataset" "$seed")
        trad_pids+=($pid)
        echo "  Traditional $dataset seed=$seed: PID=$pid"
        sleep 2
    done
done

echo ""
echo -e "${GREEN}所有任务已启动！${NC}"
echo ""

# 监控循环
echo -e "${YELLOW}【实时监控】按Ctrl+C停止监控，实验将继续在后台运行${NC}"
echo ""

monitor_count=0
while true; do
    # 检查是否还有运行的进程
    running_count=0
    for pid in "${coa_pids[@]}" "${bwo_cnn_pids[@]}" "${trad_pids[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            ((running_count++))
        fi
    done
    
    if [ $running_count -eq 0 ]; then
        echo -e "\n${GREEN}所有实验已完成！${NC}"
        break
    fi
    
    # 每10次显示一次状态
    if [ $((monitor_count % 10)) -eq 0 ]; then
        clear
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║         GPU最大化并行运行监控                              ║${NC}"
        echo -e "${GREEN}║         $(date '+%H:%M:%S')                                    ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        # GPU状态
        echo -e "${BLUE}【GPU状态】${NC}"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r util mem_used mem_total power temp; do
            printf "  利用率:%3s%%  显存:%5s/%5sMB  功耗:%6sW  温度:%3s°C\n" \
                $(echo $util | tr -d ' ') \
                $(echo $mem_used | tr -d ' ') \
                $(echo $mem_total | tr -d ' ') \
                $(echo $power | tr -d ' ') \
                $(echo $temp | tr -d ' ')
        done
        echo ""
        
        # 进程统计
        echo -e "${BLUE}【运行状态】${NC}"
        total_tasks=$((${#coa_pids[@]} + ${#bwo_cnn_pids[@]} + ${#trad_pids[@]}))
        completed=$((total_tasks - running_count))
        echo "  总任务: $total_tasks | 已完成: $completed | 运行中: $running_count"
        echo ""
        
        # 最近完成的日志
        echo -e "${BLUE}【最近完成的实验】${NC}"
        ls -lt $LOG_DIR/*.log 2>/dev/null | head -5 | while read line; do
            size=$(echo $line | awk '{print $5}')
            name=$(echo $line | awk '{print $9}' | xargs basename)
            if [ "$size" -gt 100 ]; then
                last_line=$(tail -1 $(echo $line | awk '{print $9}') 2>/dev/null)
                if echo "$last_line" | grep -q "Completed\|R2="; then
                    echo "  ✓ $name"
                fi
            fi
        done
        echo ""
        
        echo -e "${YELLOW}按 Ctrl+C 停止监控，实验将继续在后台运行${NC}"
    fi
    
    ((monitor_count++))
    sleep 3
done

# 最终汇总
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有GPU最大化并行实验完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "日志文件:"
ls -lh $LOG_DIR/*.log | wc -l | xargs -I {} echo "  共 {} 个日志文件"
