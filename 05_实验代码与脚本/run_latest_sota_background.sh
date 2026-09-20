#!/bin/bash

# 最新SOTA模型后台运行脚本
# 使用RTX 4090D (24GB)优化配置
# 运行PatchTST (2023), TimeMixer (2024), iTransformer (2024)对比实验

set -e  # 遇到错误退出

echo "================================================"
echo "🚀 启动最新SOTA模型对比实验 (后台运行)"
echo "   设备: RTX 4090D 24GB, 18 vCPU AMD EPYC, 60GB内存"
echo "   模型: PatchTST (2023), TimeMixer (2024), iTransformer (2024)"
echo "================================================"

# 创建日志目录
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/latest_sota_${TIMESTAMP}.log"

# 检查CUDA可用性
echo "检查CUDA设备..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

# 检查数据文件
DATA_FILE="wind_final.csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "警告: 数据文件 $DATA_FILE 不存在"
    echo "将使用合成数据进行测试..."
    SYNTHETIC_FLAG="--synthetic"
else
    echo "✅ 找到数据文件: $DATA_FILE"
    SYNTHETIC_FLAG=""
fi

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "开始训练最新SOTA模型..."
echo "日志文件: $LOG_FILE"
echo "预计运行时间: 2-4小时 (取决于模型复杂度和epochs)"
echo ""

# 运行脚本 (后台)
nohup python run_latest_sota.py \
    --model all \
    --data "$DATA_FILE" \
    --target power \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3 \
    --patience 20 \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --gpu_optimized \
    --seed 42 \
    $SYNTHETIC_FLAG > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!
echo "✅ 实验已在后台启动 (PID: $PID)"
echo "   查看进度: tail -f $LOG_FILE"
echo "   停止实验: kill $PID"
echo ""

# 保存PID到文件
echo $PID > logs/latest_sota_pid.txt

# 显示监控命令
echo "📊 监控命令:"
echo "   1. 实时查看日志: tail -f $LOG_FILE"
echo "   2. 查看GPU使用: watch -n 1 nvidia-smi"
echo "   3. 查看进程状态: ps aux | grep run_latest_sota"
echo "   4. 查看检查点: ls -la checkpoints_ts/"
echo ""

# 检查是否成功启动
sleep 3
if ps -p $PID > /dev/null; then
    echo "✅ 实验运行正常"
    echo "   实验将在完成后自动保存结果到:"
    echo "     - checkpoints_ts/PatchTST_latest_results.json"
    echo "     - checkpoints_ts/TimeMixer_latest_results.json"
    echo "     - checkpoints_ts/iTransformer_latest_results.json"
    echo "     - checkpoints_ts/latest_sota_summary.json"
else
    echo "❌ 实验启动失败，请查看日志: $LOG_FILE"
    exit 1
fi

echo ""
echo "================================================"
echo "🎯 实验目标:"
echo "   1. 填补审稿人指出的SOTA模型时效性缺口"
echo "   2. 提供2023-2024年最新模型的对比数据"
echo "   3. 验证Ultra-LSNT与最新SOTA的性能对比"
echo "================================================"

# 创建结果验证脚本
cat > check_results.sh << 'EOF'
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
EOF

chmod +x check_results.sh

echo "✅ 已创建结果检查脚本: ./check_results.sh"
echo "   运行: ./check_results.sh 查看实验进度"
echo ""
echo "🏁 后台实验已成功启动！"
echo "   请定期检查日志文件或运行 ./check_results.sh"
echo "================================================"