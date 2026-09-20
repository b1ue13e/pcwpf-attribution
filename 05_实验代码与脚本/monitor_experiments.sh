#!/bin/bash
# Experiment monitor script

LOG_DIR="/root/experiment_logs"
mkdir -p $LOG_DIR

clear
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          80/20统一实验协议 - 后台运行监控面板                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo "更新时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check running processes
echo "📊 【运行中的实验进程】"
echo "────────────────────────────────────────────────────────────────"
ps aux | grep python | grep -E "(quick_baselines|coa_bilstm|bwo_svr|bwo_cnn|traditional)" | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    cmd=$(echo $line | awk '{print $11}' | xargs basename)
    time=$(echo $line | awk '{print $10}')
    printf "  PID: %-6s | CPU: %5s%% | MEM: %5s%% | TIME: %8s | %s\n" "$pid" "$cpu" "$mem" "$time" "$cmd"
done
if [ $? -ne 0 ]; then
    echo "  ⚠️  没有正在运行的实验进程"
fi
echo ""

# Check log files
echo "📁 【日志文件状态】"
echo "────────────────────────────────────────────────────────────────"
for log in $LOG_DIR/*.log; do
    if [ -f "$log" ]; then
        name=$(basename $log)
        size=$(ls -lh $log | awk '{print $5}')
        lines=$(wc -l < $log)
        last_update=$(stat -c %Y $log)
        current=$(date +%s)
        ago=$((current - last_update))
        printf "  %-30s %6s %6s行 更新于%3s秒前\n" "$name" "$size" "$lines" "$ago"
    fi
done
echo ""

# Check output files
echo "📈 【输出文件状态】"
echo "────────────────────────────────────────────────────────────────"
csv_files=(
    "all_models_clean_80_20.csv"
    "traditional_baselines_80_20.csv"
    "metaheuristic_search_trace.csv"
    "metaheuristic_best_configs.csv"
    "metaheuristic_metrics_all_datasets.csv"
    "metaheuristic_latency_energy.csv"
    "robustness_80_20.csv"
    "dispatch_rolling_80_20.csv"
    "dispatch_rts24_80_20.csv"
)

for csv in "${csv_files[@]}"; do
    if [ -f "/root/$csv" ]; then
        size=$(ls -lh /root/$csv | awk '{print $5}')
        rows=$(wc -l < /root/$csv)
        printf "  ✅ %-35s %6s %5s行\n" "$csv" "$size" "$rows"
    else
        printf "  ❌ %-35s (未生成)\n" "$csv"
    fi
done
echo ""

# Recent log output
echo "📝 【最新日志输出 (最近3行)】"
echo "────────────────────────────────────────────────────────────────"
for log in $LOG_DIR/01_baselines.log $LOG_DIR/02_coa_bilstm.log; do
    if [ -f "$log" ]; then
        name=$(basename $log)
        echo "  --- $name ---"
        tail -3 $log | sed 's/^/    /'
        echo ""
    fi
done

echo ""
echo "💡 【操作提示】"
echo "────────────────────────────────────────────────────────────────"
echo "  查看详细日志: tail -f /root/experiment_logs/*.log"
echo "  停止所有实验: pkill -f 'python3.*experiment'"
echo "  查看此面板:   /root/monitor_experiments.sh"
echo ""
