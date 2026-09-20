#!/bin/bash
# GPU并行实验状态报告

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           🚀 GPU全并行实验监控报告                                    ║"
echo "║           $(date '+%Y-%m-%d %H:%M:%S')                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# GPU状态
echo "📊 【GPU状态】"
echo "───────────────────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r name util mem_used mem_total temp; do
    printf "  %-20s 利用率:%3s%%  显存:%5s/%5s MB  温度:%3s°C\n" \
        "$(echo $name | tr -d ' ')" \
        $(echo $util | tr -d ' ') \
        $(echo $mem_used | tr -d ' ') \
        $(echo $mem_total | tr -d ' ') \
        $(echo $temp | tr -d ' ')
done
echo ""

# 进程统计
echo "🔢 【进程统计】"
echo "───────────────────────────────────────────────────────────────────────"
total_procs=$(ps aux | grep python | grep -v grep | grep -v "vscode\|jupyter\|tensorboard" | wc -l)
exp_procs=$((total_procs - 1))  # 减去原始6276
echo "  总Python进程数: $total_procs"
echo "  实验相关进程:   $exp_procs"
echo ""

# 各类型实验进程数
coa_count=$(ps aux | grep -E "coa_bilstm|408611" | grep -v grep | wc -l)
bwo_svr_count=$(ps aux | grep -E "bwo_svr|408613" | grep -v grep | wc -l)
bwo_cnn_count=$(ps aux | grep -E "bwo_cnn|408615" | grep -v grep | wc -l)
trad_count=$(ps aux | grep python | grep -v grep | grep -E "484323|484324|484325|484326" | wc -l)

echo "  COA-BiLSTM进程:  $coa_count"
echo "  BWO-SVR进程:     $bwo_svr_count"
echo "  BWO-CNN进程:     $bwo_cnn_count"
echo "  传统基线进程:    $trad_count (4数据集并行)"
echo ""

# 查看传统基线各数据集进度
echo "📈 【传统基线进度】"
echo "───────────────────────────────────────────────────────────────────────"
for dataset in wind_cn electricity air_quality gefcom; do
    logfile="/root/experiment_logs/traditional_${dataset}.log"
    if [ -f "$logfile" ]; then
        # 计算完成度
        completed=$(grep -c "R2=" "$logfile" 2>/dev/null || echo "0")
        percent=$((completed * 100 / 12))
        
        # 获取最后运行的模型和结果
        last_line=$(grep "R2=" "$logfile" | tail -1)
        if [ -n "$last_line" ]; then
            model=$(echo "$last_line" | grep -oE "(ARIMA|SVR|LSTM|CNN-LSTM)")
            r2=$(echo "$last_line" | grep -oE "R2=[0-9.-]+" | cut -d= -f2)
            printf "  %-12s [%3s%%] %-10s R2=%.4f\n" "$dataset" "$percent" "$model" "$r2"
        else
            printf "  %-12s [  0%%] 启动中...\n" "$dataset"
        fi
    else
        printf "  %-12s 等待启动\n" "$dataset"
    fi
done
echo ""

# 查看元启发式实验进度
echo "🧬 【元启发式实验进度】"
echo "───────────────────────────────────────────────────────────────────────"

# COA-BiLSTM
if [ -f "/root/experiment_logs/02_coa_bilstm.log" ]; then
    last_dataset=$(grep "Dataset:" /root/experiment_logs/02_coa_bilstm.log | tail -1 | awk '{print $2}')
    echo "  COA-BiLSTM: 正在处理 $last_dataset"
fi

# BWO-SVR
if [ -f "/root/experiment_logs/03_bwo_svr.log" ]; then
    last_dataset=$(grep "Dataset:" /root/experiment_logs/03_bwo_svr.log | tail -1 | awk '{print $2}')
    echo "  BWO-SVR:    正在处理 $last_dataset"
fi

# BWO-CNN
if [ -f "/root/experiment_logs/04_bwo_cnn.log" ]; then
    last_dataset=$(grep "Dataset:" /root/experiment_logs/04_bwo_cnn.log | tail -1 | awk '{print $2}')
    echo "  BWO-CNN:    正在处理 $last_dataset"
fi
echo ""

# 输出文件状态
echo "📁 【输出文件状态】"
echo "───────────────────────────────────────────────────────────────────────"
files=(
    "all_models_clean_80_20.csv:主结果表"
    "traditional_baselines_80_20.csv:传统基线"
    "metaheuristic_search_trace.csv:搜索痕迹"
    "metaheuristic_best_configs.csv:最佳配置"
    "metaheuristic_metrics_all_datasets.csv:元启发式指标"
    "robustness_80_20.csv:鲁棒性测试"
)

for item in "${files[@]}"; do
    IFS=':' read -r file desc <<< "$item"
    if [ -f "/root/$file" ]; then
        size=$(ls -lh "/root/$file" 2>/dev/null | awk '{print $5}')
        rows=$(wc -l < "/root/$file" 2>/dev/null)
        printf "  ✅ %-40s %6s %5s行  %s\n" "$file" "$size" "$rows" "$desc"
    else
        printf "  ⏳ %-40s (生成中)       %s\n" "$file" "$desc"
    fi
done
echo ""

# 预计完成时间
echo "⏱️ 【预计完成时间】"
echo "───────────────────────────────────────────────────────────────────────"
echo "  传统基线:     30-60分钟 (4数据集并行中)"
echo "  COA-BiLSTM:   8-16小时 (4数据集×3种子)"
echo "  BWO-SVR:      4-8小时 (4数据集×3种子)"
echo "  BWO-CNN:      8-12小时 (4数据集×3种子)"
echo ""
echo "  并行优化后预计总时间: 12-24小时"
echo ""

# 提示
echo "💡 【操作提示】"
echo "───────────────────────────────────────────────────────────────────────"
echo "  刷新状态: /root/gpu_parallel_status.sh"
echo "  查看日志: tail -f /root/experiment_logs/*.log"
echo "  查看特定: tail -f /root/experiment_logs/traditional_wind_cn.log"
echo "  GPU监控:  watch -n 1 nvidia-smi"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
