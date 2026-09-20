#!/bin/bash
# 后台运行所有80/20统一实验
# 包括：传统基线、COA-BiLSTM、BWO-SVR、BWO-CNN

LOG_DIR="/root/experiment_logs"
mkdir -p $LOG_DIR

# 记录启动时间
echo "========================================" > $LOG_DIR/master.log
echo "80/20统一实验协议 - 后台运行" >> $LOG_DIR/master.log
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')" >> $LOG_DIR/master.log
echo "========================================" >> $LOG_DIR/master.log

# 设置环境变量
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1

echo "环境变量已设置" >> $LOG_DIR/master.log

# 函数：启动实验并记录PID
start_experiment() {
    local name=$1
    local script=$2
    local logfile=$3
    
    echo "" >> $LOG_DIR/master.log
    echo "[$(date '+%H:%M:%S')] 启动: $name" >> $LOG_DIR/master.log
    
    nohup python3 $script > $logfile 2>&1 &
    local pid=$!
    
    echo "PID: $pid" >> $LOG_DIR/master.log
    echo "日志: $logfile" >> $LOG_DIR/master.log
    
    echo $pid > $LOG_DIR/${name}.pid
    return $pid
}

# 1. 传统基线实验
echo ""
echo "【1/4】启动传统基线实验 (ARIMA, SVR, LSTM, CNN-LSTM)..."
start_experiment "traditional_baselines" "/root/quick_baselines_experiment.py" "$LOG_DIR/01_traditional_baselines.log"
BASELINES_PID=$!

# 等待传统基线完成（约30-60分钟）
echo "等待传统基线实验完成..."
wait $BASELINES_PID
BASELINES_STATUS=$?
echo "[$(date '+%H:%M:%S')] 传统基线完成，状态: $BASELINES_STATUS" >> $LOG_DIR/master.log

# 如果成功，合并结果
if [ $BASELINES_STATUS -eq 0 ] && [ -f "/root/traditional_baselines_80_20.csv" ]; then
    echo "合并传统基线结果..." >> $LOG_DIR/master.log
    python3 /root/add_traditional_baselines.py >> $LOG_DIR/master.log 2>&1
fi

# 2. COA-BiLSTM 实验
echo ""
echo "【2/4】启动 COA-BiLSTM 实验..."
start_experiment "coa_bilstm" "/root/coa_bilstm_experiment_v2.py" "$LOG_DIR/02_coa_bilstm.log"
COA_PID=$!

# 3. BWO-SVR 实验 (并行)
echo ""
echo "【3/4】启动 BWO-SVR 实验..."
start_experiment "bwo_svr" "/root/bwo_svr_experiment_v2.py" "$LOG_DIR/03_bwo_svr.log"
BWO_SVR_PID=$!

# 4. BWO-CNN 实验 (并行)
echo ""
echo "【4/4】启动 BWO-CNN 实验..."
start_experiment "bwo_cnn" "/root/bwo_cnn_experiment_v2.py" "$LOG_DIR/04_bwo_cnn.log"
BWO_CNN_PID=$!

# 保存所有PID
cat > $LOG_DIR/all_pids.txt << EOF
启动时间: $(date '+%Y-%m-%d %H:%M:%S')
传统基线: $BASELINES_PID (已完成, 状态: $BASELINES_STATUS)
COA-BiLSTM: $COA_PID
BWO-SVR: $BWO_SVR_PID
BWO-CNN: $BWO_CNN_PID
EOF

echo "" >> $LOG_DIR/master.log
echo "所有实验已启动!" >> $LOG_DIR/master.log
echo "COA-BiLSTM PID: $COA_PID" >> $LOG_DIR/master.log
echo "BWO-SVR PID: $BWO_SVR_PID" >> $LOG_DIR/master.log
echo "BWO-CNN PID: $BWO_CNN_PID" >> $LOG_DIR/master.log

# 等待所有并行实验完成
echo ""
echo "等待所有实验完成（这可能需要几小时）..."
echo "[$(date '+%H:%M:%S')] 等待所有并行实验完成..." >> $LOG_DIR/master.log

wait $COA_PID
COA_STATUS=$?
echo "[$(date '+%H:%M:%S')] COA-BiLSTM 完成，状态: $COA_STATUS" >> $LOG_DIR/master.log

wait $BWO_SVR_PID
BWO_SVR_STATUS=$?
echo "[$(date '+%H:%M:%S')] BWO-SVR 完成，状态: $BWO_SVR_STATUS" >> $LOG_DIR/master.log

wait $BWO_CNN_PID
BWO_CNN_STATUS=$?
echo "[$(date '+%H:%M:%S')] BWO-CNN 完成，状态: $BWO_CNN_STATUS" >> $LOG_DIR/master.log

# 生成最终输出
echo "" >> $LOG_DIR/master.log
echo "[$(date '+%H:%M:%S')] 生成最终输出文件..." >> $LOG_DIR/master.log

python3 /root/generate_remaining_outputs.py >> $LOG_DIR/master.log 2>&1
python3 /root/finalize_metaheuristic_outputs.py >> $LOG_DIR/master.log 2>&1

# 完成报告
echo "" >> $LOG_DIR/master.log
echo "========================================" >> $LOG_DIR/master.log
echo "所有实验完成!" >> $LOG_DIR/master.log
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> $LOG_DIR/master.log
echo "========================================" >> $LOG_DIR/master.log

echo ""
echo "========================================"
echo "所有实验已完成!"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""
echo "查看结果:"
echo "  ls -lh /root/*.csv"
echo ""
echo "查看完整日志:"
echo "  cat $LOG_DIR/master.log"
