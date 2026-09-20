#!/bin/bash
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         COA单线程运行监控 - $(date '+%H:%M:%S')                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 检查运行状态
echo "【运行状态】"
echo "─────────────────────────────────────────────────────────────────"

# 检查主进程
if ps -p $1 > /dev/null 2>&1; then
    echo "主控脚本: 运行中 (PID: $1)"
else
    echo "主控脚本: 已结束"
fi

# 检查COA实验进程
coa_pids=$(ps aux | grep python | grep -v grep | grep -v "vscode\|jupyter\|tensorboard\|nvidia\|6276" | awk '{print $2}')
coa_count=$(echo "$coa_pids" | grep -v "^$" | wc -l)
echo "COA实验进程: $coa_count 个"

# 显示当前运行的COA
for pid in $coa_pids; do
    if [ -n "$pid" ]; then
        log=$(ls -la /proc/$pid/fd/ 2>/dev/null | grep "coa_" | head -1 | awk '{print $11}')
        if [ -n "$log" ]; then
            name=$(basename "$log" .log)
            echo "  └─ $name (PID: $pid)"
        fi
    fi
done

echo ""
echo "【当前日志进度】"
echo "─────────────────────────────────────────────────────────────────"

# 显示最近的日志更新
ls -lt /root/experiment_logs/coa_*.log 2>/dev/null | grep -v "bak" | head -5 | while read line; do
    name=$(echo "$line" | awk '{print $9}' | xargs basename)
    size=$(echo "$line" | awk '{print $5}')
    lines=$(wc -l < "/root/experiment_logs/$name" 2>/dev/null)
    
    # 检查是否完成
    if grep -q "Test Metrics: R2=" "/root/experiment_logs/$name" 2>/dev/null; then
        r2=$(grep "Test Metrics: R2=" "/root/experiment_logs/$name" | tail -1 | grep -oE "R2=[0-9.-]+" | head -1)
        echo "✅ $name: $size ($lines 行) - $r2"
    elif [ -f "/root/experiment_logs/${name%.log}_done" ]; then
        echo "✅ $name: 已完成"
    else
        echo "🔄 $name: $size ($lines 行) - 运行中"
    fi
done

echo ""
echo "【完成进度】"
echo "─────────────────────────────────────────────────────────────────"
completed=$(ls /root/experiment_logs/coa_*_done 2>/dev/null | wc -l)
total=12
progress=$((completed * 100 / total))
echo "已完成: $completed / $total ($(progress)%)"
echo ""

# 进度条
printf "["
for i in $(seq 1 $((completed * 50 / total))); do printf "="; done
for i in $(seq 1 $(((total - completed) * 50 / total))); do printf " "; done
printf "] %d%%\n" $progress

echo ""
echo "【GPU状态】"
echo "─────────────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | while IFS=',' read -r util mem_used mem_total; do
    printf "利用率: %3s%%  显存: %6s / %6s MB\n" $(echo $util | tr -d ' ') $(echo $mem_used | tr -d ' ') $(echo $mem_total | tr -d ' ')
done

echo ""
echo "刷新: $(date '+%H:%M:%S') | Ctrl+C 退出"
