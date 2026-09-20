#!/bin/bash
echo "检查实验进度..."
LOG_FILE=$(ls -t logs/latest_sota_*.log | head -1)
if [ -f "$LOG_FILE" ]; then
    echo "日志文件: $LOG_FILE"
    echo "最后10行:"
    tail -10 "$LOG_FILE"
else
    echo "未找到日志文件"
fi

echo ""
echo "检查结果文件..."
if [ -f "checkpoints_ts/latest_sota_summary.json" ]; then
    echo "✅ 找到汇总结果文件"
    python -c "import json; data=json.load(open('checkpoints_ts/latest_sota_summary.json')); print('模型对比:'); [print(f'  {k}: R²={v[\"R2\"]:.4f}, RMSE={v[\"RMSE\"]:.2f}') for k,v in data.items()]"
else
    echo "结果文件尚未生成，实验可能仍在运行"
fi

# 检查进程状态
PID_FILE="logs/latest_sota_pid.txt"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo "✅ 实验仍在运行 (PID: $PID)"
    else
        echo "✅ 实验已完成"
    fi
fi
