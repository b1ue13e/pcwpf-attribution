#!/bin/bash
# 实验进度检查脚本

echo "═══════════════════════════════════════════════════════════════"
echo "           🚀 80/20统一实验协议 - 后台运行状态"
echo "═══════════════════════════════════════════════════════════════"
echo "检查时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. 运行中的进程
echo "📊 【运行中的实验进程】"
echo "───────────────────────────────────────────────────────────────"
ps aux | grep python | grep -E "(quick_baselines|coa_bilstm|bwo_svr|bwo_cnn|run_all)" | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    cmd=$(echo $line | awk '{print $11}' | xargs basename)
    time=$(echo $line | awk '{print $10}')
    printf "  PID: %-7s | CPU: %6s%% | MEM: %5s%% | 运行: %8s | %s\n" "$pid" "$cpu" "$mem" "$time" "$cmd"
done

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "  ⚠️  当前没有正在运行的实验进程"
fi
echo ""

# 2. 日志文件状态
echo "📁 【日志文件】"
echo "───────────────────────────────────────────────────────────────"
for log in /root/experiment_logs/*.log; do
    if [ -f "$log" ]; then
        name=$(basename $log)
        size=$(ls -lh $log | awk '{print $5}')
        lines=$(wc -l < $log)
        printf "  %-45s %6s %6s行\n" "$name" "$size" "$lines"
    fi
done
echo ""

# 3. 输出CSV文件
echo "📈 【生成的CSV文件】"
echo "───────────────────────────────────────────────────────────────"
CSV_FILES=(
    "all_models_clean_80_20.csv:所有模型结果"
    "traditional_baselines_80_20.csv:传统基线结果"
    "metaheuristic_search_trace.csv:搜索痕迹"
    "metaheuristic_best_configs.csv:最佳配置"
    "metaheuristic_metrics_all_datasets.csv:元启发式指标"
    "metaheuristic_latency_energy.csv:延迟能耗"
    "robustness_80_20.csv:鲁棒性测试"
    "dispatch_rolling_80_20.csv:滚动调度"
    "dispatch_rts24_80_20.csv:RTS-24调度"
)

for item in "${CSV_FILES[@]}"; do
    IFS=':' read -r file desc <<< "$item"
    if [ -f "/root/$file" ]; then
        size=$(ls -lh /root/$file | awk '{print $5}')
        rows=$(wc -l < /root/$file)
        printf "  ✅ %-40s %6s %6s行  [%s]\n" "$file" "$size" "$rows" "$desc"
    else
        printf "  ❌ %-40s (未生成) [%s]\n" "$file" "$desc"
    fi
done
echo ""

# 4. 最新进度（从日志中解析）
echo "📝 【最新进度】"
echo "───────────────────────────────────────────────────────────────"
if [ -f "/root/experiment_logs/01_traditional_baselines.log" ]; then
    # 提取最近的数据集和模型
    last_dataset=$(grep "Dataset:" /root/experiment_logs/01_traditional_baselines.log | tail -1 | sed 's/.*Dataset: //')
    last_model=$(grep -E "(ARIMA|SVR|LSTM|CNN-LSTM) on" /root/experiment_logs/01_traditional_baselines.log | tail -1 | sed 's/.*  //')
    last_result=$(grep "R2=" /root/experiment_logs/01_traditional_baselines.log | tail -1 | sed 's/.*    //')
    
    echo "  传统基线当前进度:"
    [ -n "$last_dataset" ] && echo "    数据集: $last_dataset"
    [ -n "$last_model" ] && echo "    模型: $last_model"
    [ -n "$last_result" ] && echo "    结果: $last_result"
fi

if [ -f "/root/experiment_logs/master.log" ]; then
    last_event=$(tail -5 /root/experiment_logs/master.log | grep -E "(启动|完成|生成)" | tail -1)
    [ -n "$last_event" ] && echo "  总控状态: $last_event"
fi
echo ""

# 5. 预计完成时间
echo "⏱️ 【预计时间】"
echo "───────────────────────────────────────────────────────────────"
echo "  传统基线: ~30-60分钟"
echo "  COA-BiLSTM: ~2-4小时/数据集 (共4个)"
echo "  BWO-SVR: ~1-2小时/数据集 (共4个)"
echo "  BWO-CNN: ~2-3小时/数据集 (共4个)"
echo ""

# 6. 操作提示
echo "💡 【操作命令】"
echo "───────────────────────────────────────────────────────────────"
echo "  查看此状态:     /root/check_progress.sh"
echo "  查看详细日志:   tail -f /root/experiment_logs/*.log"
echo "  查看传统基线:   tail -f /root/experiment_logs/01_traditional_baselines.log"
echo "  停止所有实验:   pkill -f 'python3.*experiment'"
echo "  查看CSV内容:    head -20 /root/all_models_clean_80_20.csv"
echo ""

echo "═══════════════════════════════════════════════════════════════"
