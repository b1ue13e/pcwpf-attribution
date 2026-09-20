#!/bin/bash
# GPU并行实验监控面板

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║           🚀 GPU并行实验监控面板 - $(date '+%H:%M:%S')                  ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # GPU状态
    echo "📊 【GPU状态】"
    echo "───────────────────────────────────────────────────────────────────────"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r util mem_used mem_total power temp; do
        printf "  利用率: %3s%%  |  显存: %5s/%5s MiB  |  功耗: %5s W  |  温度: %3s°C\n" \
            $(echo $util | tr -d ' ') \
            $(echo $mem_used | tr -d ' ') \
            $(echo $mem_total | tr -d ' ') \
            $(echo $power | tr -d ' ') \
            $(echo $temp | tr -d ' ')
    done
    
    # 如果nvidia-smi失败，显示CPU信息
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
        printf "  CPU: %5.1f%%  |  内存: %5.1f%%  (GPU监控不可用)\n" $cpu_usage $mem_usage
    fi
    echo ""
    
    # 运行中的进程
    echo "🔄 【运行中的实验进程】"
    echo "───────────────────────────────────────────────────────────────────────"
    echo "  PID      CPU%   MEM%   时间     实验类型          数据集"
    echo "  ─────────────────────────────────────────────────────────────────────"
    
    ps aux | grep python | grep -E "(traditional|coa_bilstm|bwo_svr|bwo_cnn)" | grep -v grep | while read line; do
        pid=$(echo $line | awk '{print $2}')
        cpu=$(echo $line | awk '{print $3}')
        mem=$(echo $line | awk '{print $4}')
        time=$(echo $line | awk '{print $10}')
        cmd=$(echo $line | awk '{print $11}')
        
        # 确定实验类型和数据集
        if echo "$cmd" | grep -q "traditional"; then
            exp_type="传统基线"
            # 从日志文件名推断数据集
            log_file=$(ls -t /root/experiment_logs/traditional_*.log 2>/dev/null | head -1)
            dataset=$(basename "$log_file" .log 2>/dev/null | sed 's/traditional_//')
        elif echo "$cmd" | grep -q "coa_bilstm"; then
            exp_type="COA-BiLSTM"
            dataset="--"
        elif echo "$cmd" | grep -q "bwo_svr"; then
            exp_type="BWO-SVR"
            dataset="--"
        elif echo "$cmd" | grep -q "bwo_cnn"; then
            exp_type="BWO-CNN"
            dataset="--"
        else
            exp_type="未知"
            dataset="--"
        fi
        
        printf "  %-8s %5s  %5s  %-8s %-15s %s\n" "$pid" "$cpu" "$mem" "$time" "$exp_type" "$dataset"
    done
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "  (当前没有正在运行的实验进程)"
    fi
    echo ""
    
    # 各数据集进度
    echo "📈 【各数据集实验进度】"
    echo "───────────────────────────────────────────────────────────────────────"
    
    for dataset in wind_cn electricity air_quality gefcom; do
        logfile="/root/experiment_logs/traditional_${dataset}.log"
        if [ -f "$logfile" ]; then
            # 获取最后运行的模型
            last_model=$(grep -E "(ARIMA|SVR|LSTM|CNN-LSTM) on" "$logfile" | tail -1 | sed 's/.*  //' | sed "s/ on $dataset.*//")
            last_result=$(grep "R2=" "$logfile" | tail -1 | sed 's/.*    //')
            progress=$(grep -c "completed" "$logfile" 2>/dev/null || echo "0")
            
            # 计算进度 (4 models × 3 seeds = 12)
            completed=$(grep -c "R2=" "$logfile" 2>/dev/null || echo "0")
            percent=$((completed * 100 / 12))
            
            printf "  %-12s [%3s%%] %-10s | %s\n" "$dataset" "$percent" "$last_model" "$last_result"
        else
            printf "  %-12s [等待中]\n" "$dataset"
        fi
    done
    echo ""
    
    # 输出文件状态
    echo "📁 【输出文件】"
    echo "───────────────────────────────────────────────────────────────────────"
    
    files=(
        "traditional_baselines_80_20.csv:传统基线"
        "all_models_clean_80_20.csv:所有模型"
        "metaheuristic_search_trace.csv:搜索痕迹"
    )
    
    for item in "${files[@]}"; do
        IFS=':' read -r file desc <<< "$item"
        if [ -f "/root/$file" ]; then
            size=$(ls -lh "/root/$file" | awk '{print $5}')
            rows=$(wc -l < "/root/$file")
            printf "  ✅ %-40s %6s %5s行  [%s]\n" "$file" "$size" "$rows" "$desc"
        else
            printf "  ⏳ %-40s (生成中)       [%s]\n" "$file" "$desc"
        fi
    done
    echo ""
    
    # 快捷键提示
    echo "💡 【按 Ctrl+C 退出监控 | 实验在后台继续运行】"
    echo ""
    
    sleep 5
done
