#!/bin/bash
echo "========================================"
echo "实验状态监控 $(date '+%H:%M:%S')"
echo "========================================"
echo ""
echo "【运行的Python进程】"
ps aux | grep python | grep -E "(quick_baselines|coa_bilstm|bwo_svr|bwo_cnn|traditional)" | grep -v grep | awk '{printf "PID: %6s | CPU: %4s%% | MEM: %4s%% | %s\n", $2, $3, $4, $11}'
echo ""
echo "【日志文件】"
ls -lh /root/experiment_logs/*.log 2>/dev/null | awk '{printf "%-40s %8s\n", $9, $5}'
echo ""
echo "【生成的CSV文件】"
ls -lh /root/*.csv 2>/dev/null | grep -E "(all_models|metaheuristic|robustness|dispatch|traditional)" | awk '{printf "%-40s %8s\n", $9, $5}'
echo ""
echo "【快速查看】"
if [ -f /root/experiment_logs/01_baselines.log ]; then
    echo "--- Baselines最新输出 ---"
    tail -3 /root/experiment_logs/01_baselines.log 2>/dev/null
fi
