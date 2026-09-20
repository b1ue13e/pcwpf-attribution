#!/bin/bash
# 自动继续脚本 - 在COA完成后自动启动后续实验

LOG_DIR="/root/experiment_logs"

echo "========================================"
echo "自动继续脚本"
echo "启动: $(date '+%H:%M:%S')"
echo "========================================"
echo ""

# 函数：检查wind_cn COA是否完成
check_wind_coa_done() {
    local count=0
    for seed in 42 123 456; do
        if [ -f "$LOG_DIR/coa_wind_cn_${seed}.log" ]; then
            if grep -q "R2=" "$LOG_DIR/coa_wind_cn_${seed}.log" 2>/dev/null; then
                ((count++))
            fi
        fi
    done
    echo $count
}

# 函数：启动BWO-SVR
launch_bwo_svr() {
    echo "【启动 wind_cn BWO-SVR (3并行)】"
    for seed in 42 123 456; do
        # 检查是否已完成
        if [ -f "$LOG_DIR/bwo_svr_wind_cn_${seed}.log" ] && grep -q "R2=" "$LOG_DIR/bwo_svr_wind_cn_${seed}.log" 2>/dev/null; then
            echo "  bwo_svr_wind_cn_${seed}: 已完成，跳过"
            continue
        fi
        
        python3 -c "
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
from bwo_svr_experiment_v2 import run_bwo_svr_single_experiment
result = run_bwo_svr_single_experiment('wind_cn', $seed)
print(f'Completed: R2={result[\"R2\"]:.4f}')
" > "$LOG_DIR/bwo_svr_wind_cn_${seed}.log" 2>&1 &
        echo "  bwo_svr_wind_cn_${seed}: PID=$!"
    done
    echo "等待BWO-SVR完成..."
    wait
    echo "✅ wind_cn BWO-SVR 完成"
}

# 函数：启动BWO-CNN
launch_bwo_cnn() {
    echo "【启动 wind_cn BWO-CNN (3并行)】"
    for seed in 42 123 456; do
        if [ -f "$LOG_DIR/bwo_cnn_wind_cn_${seed}.log" ] && grep -q "R2=" "$LOG_DIR/bwo_cnn_wind_cn_${seed}.log" 2>/dev/null; then
            echo "  bwo_cnn_wind_cn_${seed}: 已完成，跳过"
            continue
        fi
        
        python3 -c "
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
from bwo_cnn_experiment_v2 import run_bwo_cnn_single_experiment
result = run_bwo_cnn_single_experiment('wind_cn', $seed)
print(f'Completed: R2={result[\"R2\"]:.4f}')
" > "$LOG_DIR/bwo_cnn_wind_cn_${seed}.log" 2>&1 &
        echo "  bwo_cnn_wind_cn_${seed}: PID=$!"
    done
    echo "等待BWO-CNN完成..."
    wait
    echo "✅ wind_cn BWO-CNN 完成"
}

# 函数：启动electricity COA
launch_elec_coa() {
    echo "【启动 electricity COA-BiLSTM (3并行)】"
    for seed in 42 123 456; do
        if [ -f "$LOG_DIR/coa_electricity_${seed}.log" ] && grep -q "R2=" "$LOG_DIR/coa_electricity_${seed}.log" 2>/dev/null; then
            echo "  coa_electricity_${seed}: 已完成，跳过"
            continue
        fi
        
        python3 -c "
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
from coa_bilstm_experiment_v2 import run_coa_bilstm_experiment
result = run_coa_bilstm_experiment('electricity', $seed)
print(f'Completed: R2={result[\"metrics\"][\"R2\"]:.4f}')
" > "$LOG_DIR/coa_electricity_${seed}.log" 2>&1 &
        echo "  coa_electricity_${seed}: PID=$!"
    done
    echo "等待COA完成..."
    wait
    echo "✅ electricity COA 完成"
}

# 主循环
echo "等待 wind_cn COA 完成..."
echo ""

while true; do
    coa_done=$(check_wind_coa_done)
    echo "[$(date '+%H:%M:%S')] wind_cn COA进度: $coa_done / 3"
    
    if [ "$coa_done" -eq 3 ]; then
        echo ""
        echo "✅ wind_cn COA 全部完成！"
        echo ""
        break
    fi
    
    sleep 60  # 每分钟检查一次
done

# 启动后续实验
echo "【继续后续实验】"
echo ""

# 阶段2: BWO-SVR
launch_bwo_svr

# 阶段3: BWO-CNN  
launch_bwo_cnn

# 阶段4: electricity COA
launch_elec_coa

echo ""
echo "========================================"
echo "✅ 所有优先级实验完成！"
echo "结束: $(date '%H:%M:%S')"
echo "========================================"
