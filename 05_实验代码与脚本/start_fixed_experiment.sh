#!/bin/bash
# 启动修复版COA实验脚本

echo "========================================"
echo "启动修复版COA-BiLSTM实验 (后台运行)"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 检查是否已有相同的实验在运行
existing_pids=$(ps aux | grep "run_coa_single_thread_fixed.sh" | grep -v grep | awk '{print $2}')
if [ -n "$existing_pids" ]; then
    echo "警告: 修复脚本已在运行 (PID: $existing_pids)"
    echo "是否要停止现有进程并重新启动? [y/N]"
    read -t 5 answer
    if [ "$answer" = "y" ]; then
        kill $existing_pids 2>/dev/null
        sleep 2
    else
        echo "保持现有进程运行"
        exit 0
    fi
fi

# 创建日志目录
mkdir -p /root/experiment_logs

# 在后台运行修复脚本，并输出到日志
LOG_FILE="/root/experiment_logs/coa_fixed_experiment_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOG_FILE"

nohup ./run_coa_single_thread_fixed.sh > "$LOG_FILE" 2>&1 &
FIXED_PID=$!

echo "修复脚本已启动 (PID: $FIXED_PID)"
echo "使用以下命令监控进度:"
echo "  tail -f $LOG_FILE"
echo "  ps aux | grep 'python.*coa'"
echo "  nvidia-smi"

# 保存PID到文件
echo $FIXED_PID > /root/experiment_logs/fixed_experiment.pid

echo ""
echo "实验已在后台运行。可以使用以下命令查看状态:"
echo "  cat /root/experiment_logs/fixed_experiment.pid"
echo "  tail -50 $LOG_FILE"
