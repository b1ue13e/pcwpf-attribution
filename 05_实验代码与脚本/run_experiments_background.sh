#!/bin/bash
# Background experiment runner for 80/20 unified protocol

LOG_DIR="/root/experiment_logs"
mkdir -p $LOG_DIR

echo "========================================"
echo "Starting Background Experiments"
echo "========================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

# 1. Traditional Baselines
echo "[1/4] Starting Traditional Baselines (ARIMA, SVR, LSTM, CNN-LSTM)..."
nohup python3 /root/quick_baselines_experiment.py > $LOG_DIR/01_baselines.log 2>&1 &
BASELINES_PID=$!
echo "  PID: $BASELINES_PID"
echo "  Log: $LOG_DIR/01_baselines.log"

# Wait for baselines to complete
wait $BASELINES_PID
BASELINES_STATUS=$?
echo "  Exit status: $BASELINES_STATUS"
echo ""

# 2. COA-BiLSTM
echo "[2/4] Starting COA-BiLSTM experiments..."
nohup python3 /root/coa_bilstm_experiment_v2.py > $LOG_DIR/02_coa_bilstm.log 2>&1 &
COA_PID=$!
echo "  PID: $COA_PID"
echo "  Log: $LOG_DIR/02_coa_bilstm.log"

# Run in parallel with others later

# 3. BWO-SVR
echo "[3/4] Starting BWO-SVR experiments..."
nohup python3 /root/bwo_svr_experiment_v2.py > $LOG_DIR/03_bwo_svr.log 2>&1 &
BWO_SVR_PID=$!
echo "  PID: $BWO_SVR_PID"
echo "  Log: $LOG_DIR/03_bwo_svr.log"

# 4. BWO-CNN
echo "[4/4] Starting BWO-CNN experiments..."
nohup python3 /root/bwo_cnn_experiment_v2.py > $LOG_DIR/04_bwo_cnn.log 2>&1 &
BWO_CNN_PID=$!
echo "  PID: $BWO_CNN_PID"
echo "  Log: $LOG_DIR/04_bwo_cnn.log"

echo ""
echo "========================================"
echo "All experiments started in background!"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Check status:"
echo "  ps aux | grep -E '(python|experiment)' | grep -v grep"
echo ""
echo "PIDs:"
echo "  Baselines: $BASELINES_PID (completed: $BASELINES_STATUS)"
echo "  COA-BiLSTM: $COA_PID"
echo "  BWO-SVR: $BWO_SVR_PID"
echo "  BWO-CNN: $BWO_CNN_PID"

# Save PIDs to file
cat > $LOG_DIR/pids.txt << EOF
BASELINES_PID=$BASELINES_PID
COA_PID=$COA_PID
BWO_SVR_PID=$BWO_SVR_PID
BWO_CNN_PID=$BWO_CNN_PID
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
EOF

echo ""
echo "PIDs saved to: $LOG_DIR/pids.txt"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
