#!/bin/bash
# 方案C监控脚本

clear
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         方案C监控 - 核心数据集 $(date '+%H:%M:%S')                      ║"
echo "║         wind_cn + electricity                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

DATASETS="wind_cn electricity"
MODELS="LSTM CNN-LSTM COA-BiLSTM BWO-SVR BWO-CNN"

echo "【实验进度】"
echo "─────────────────────────────────────────────────────────────────"

for dataset in $DATASETS; do
    echo ""
    echo "数据集: $dataset"
    for model in $MODELS; do
        # 确定模型类型和日志模式
        case $model in
            LSTM|CNN-LSTM)
                prefix="trad"
                check_pattern="${model}: R2="
                ;;
            COA-BiLSTM)
                prefix="coa"
                check_pattern="R2="
                ;;
            BWO-SVR)
                prefix="bwo_svr"
                check_pattern="R2="
                ;;
            BWO-CNN)
                prefix="bwo_cnn"
                check_pattern="R2="
                ;;
        esac
        
        completed=0
        for seed in 42 123 456; do
            logfile="/root/experiment_logs/${prefix}_${dataset}_${seed}.log"
            if [ -f "$logfile" ]; then
                if grep -q "$check_pattern" "$logfile" 2>/dev/null; then
                    ((completed++))
                fi
            fi
        done
        
        # 显示进度条
        printf "  %-12s [" "$model"
        for i in 1 2 3; do
            if [ $i -le $completed ]; then
                printf "█"
            else
                printf "░"
            fi
        done
        printf "] %d/3\n" $completed
    done
done

echo ""
echo "【总体统计】"
echo "─────────────────────────────────────────────────────────────────"

# 统计各模型完成数
trad_done=$(grep -l "R2=" /root/experiment_logs/trad_*.log 2>/dev/null | wc -l)
coa_done=$(grep -l "R2=" /root/experiment_logs/coa_*.log 2>/dev/null | wc -l)
bwo_svr_done=$(grep -l "R2=" /root/experiment_logs/bwo_svr_*.log 2>/dev/null | wc -l)
bwo_cnn_done=$(grep -l "R2=" /root/experiment_logs/bwo_cnn_*.log 2>/dev/null | wc -l)

echo "Traditional: $trad_done / 12 ($(expr $trad_done \* 100 / 12 2>/dev/null || echo 0)%)"
echo "COA-BiLSTM: $coa_done / 6 ($(expr $coa_done \* 100 / 6 2>/dev/null || echo 0)%)"
echo "BWO-SVR: $bwo_svr_done / 6 ($(expr $bwo_svr_done \* 100 / 6 2>/dev/null || echo 0)%)"
echo "BWO-CNN: $bwo_cnn_done / 6 ($(expr $bwo_cnn_done \* 100 / 6 2>/dev/null || echo 0)%)"

total_done=$((trad_done + coa_done + bwo_svr_done + bwo_cnn_done))
total_needed=30  # 12 + 6 + 6 + 6

echo ""
printf "总计: %d / %d (%d%%)\n" $total_done $total_needed $(expr $total_done \* 100 / $total_needed 2>/dev/null || echo 0)

echo ""
echo "【系统状态】"
echo "─────────────────────────────────────────────────────────────────"
echo "GPU利用率: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')%"
echo "显存使用: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') MB"
echo "Python进程: $(ps aux | grep python | grep -v grep | grep -v "vscode\|jupyter\|tensorboard" | wc -l) 个"

echo ""
echo "刷新: $(date '+%H:%M:%S') | Ctrl+C 退出"
