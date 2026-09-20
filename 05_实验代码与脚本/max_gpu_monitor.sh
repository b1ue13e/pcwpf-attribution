#!/bin/bash
# 最大化GPU利用率监控面板

clear
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║           🚀 GPU最大化并行实验 - 实时监控面板                           ║"
echo "║           $(date '+%Y-%m-%d %H:%M:%S')                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# GPU状态
echo "📊 【GPU状态】"
echo "───────────────────────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm --format=csv,noheader 2>/dev/null | while IFS=',' read -r name util mem_used mem_total power temp sm_clock; do
    mem_pct=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc 2>/dev/null || echo "N/A")
    printf "  GPU: %-20s\n" "$(echo $name | tr -d ' ')"
    printf "  ├─ 利用率:  %3s%%    (目标: >90%%)\n" $(echo $util | tr -d ' ')
    printf "  ├─ 显存:    %5s / %5s MB (%s%%)\n" $(echo $mem_used | tr -d ' ') $(echo $mem_total | tr -d ' ') "$mem_pct"
    printf "  ├─ 功耗:    %6s W   (TDP: 425W)\n" $(echo $power | tr -d ' ')
    printf "  ├─ 温度:    %3s°C    (安全: <85°C)\n" $(echo $temp | tr -d ' ')
    printf "  └─ SM频率:  %5s MHz\n" $(echo $sm_clock | tr -d ' ')
done
echo ""

# 进程统计
echo "🔢 【进程统计】"
echo "───────────────────────────────────────────────────────────────────────────"
total_python=$(ps aux | grep python | grep -v grep | wc -l)
experiment_procs=$(ps aux | grep python | grep -v grep | grep -v "vscode\|jupyter\|tensorboard\|nvidia" | wc -l)
coa_procs=$(ps aux | grep python | grep -v grep | grep "coa" | wc -l)
bwo_cnn_procs=$(ps aux | grep python | grep -v grep | grep "bwo_cnn" | wc -l)
bwo_svr_procs=$(ps aux | grep python | grep -v grep | grep "bwo_svr" | wc -l)
trad_procs=$(ps aux | grep python | grep -v grep | grep "trad" | wc -l)

printf "  总Python进程:     %3d\n" $total_python
printf "  实验相关进程:     %3d\n" $experiment_procs
printf "  ├─ COA-BiLSTM:   %3d\n" $coa_procs
printf "  ├─ BWO-CNN:      %3d\n" $bwo_cnn_procs
printf "  ├─ BWO-SVR:      %3d\n" $bwo_svr_procs
printf "  └─ Traditional:  %3d\n" $trad_procs
echo ""

# 日志文件统计
echo "📁 【日志文件统计】"
echo "───────────────────────────────────────────────────────────────────────────"
total_logs=$(ls /root/experiment_logs/*.log 2>/dev/null | wc -l)
coa_logs=$(ls /root/experiment_logs/coa_*.log 2>/dev/null | wc -l)
bwo_cnn_logs=$(ls /root/experiment_logs/bwo_cnn_*.log 2>/dev/null | wc -l)
bwo_svr_logs=$(ls /root/experiment_logs/bwo_svr_*.log 2>/dev/null | wc -l)
trad_logs=$(ls /root/experiment_logs/trad_*.log 2>/dev/null | wc -l)

printf "  总日志文件:  %3d / 48\n" $total_logs
printf "  ├─ COA:      %3d / 12\n" $coa_logs
printf "  ├─ BWO-CNN:  %3d / 12\n" $bwo_cnn_logs
printf "  ├─ BWO-SVR:  %3d / 12\n" $bwo_svr_logs
printf "  └─ Trad:     %3d / 12\n" $trad_logs
echo ""

# 检查已完成的实验
echo "✅ 【已完成实验】"
echo "───────────────────────────────────────────────────────────────────────────"
completed=0
for log in /root/experiment_logs/*.log; do
    if [ -f "$log" ]; then
        if grep -q "Completed\|R2=" "$log" 2>/dev/null; then
            ((completed++))
        fi
    fi
done
printf "  已完成: %3d / 48\n" $completed
printf "  进度:   %3d%%\n" $((completed * 100 / 48))
echo ""

# 最近活跃的实验
echo "🔄 【最近活跃的实验】"
echo "───────────────────────────────────────────────────────────────────────────"
ls -lt /root/experiment_logs/*.log 2>/dev/null | head -8 | awk '{
    size = $5
    name = $9
    time = $8
    gsub("/root/experiment_logs/", "", name)
    if (size > 1000) {
        size_mb = size / 1024
        printf "  %-40s %6.1f KB  %s\n", name, size/1024, time
    } else {
        printf "  %-40s %6d B   %s\n", name, size, time
    }
}'
echo ""

# 性能提示
echo "💡 【性能提示】"
echo "───────────────────────────────────────────────────────────────────────────"
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
mem_pct=$(echo "scale=0; $mem_used * 100 / $mem_total" | bc 2>/dev/null || echo "0")

if [ "$gpu_util" -lt 50 ]; then
    echo "  ⚠️  GPU利用率偏低，实验可能还在数据加载阶段"
elif [ "$gpu_util" -lt 80 ]; then
    echo "  🟡 GPU利用率良好，还有优化空间"
else
    echo "  ✅ GPU利用率优秀！"
fi

if [ "$mem_pct" -lt 50 ]; then
    echo "  ⚠️  显存使用偏低，可以增加并行任务数"
elif [ "$mem_pct" -lt 80 ]; then
    echo "  🟡 显存使用良好"
else
    echo "  ✅ 显存使用接近满载！"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "刷新时间: $(date '+%H:%M:%S') | 刷新间隔: 5秒 | 按 Ctrl+C 退出"
echo "═══════════════════════════════════════════════════════════════════════════"
