#!/bin/bash
LOG_DIR="/root/experiment_logs"

echo "========================================"
echo "自动继续脚本"
echo "启动: $(date '+%H:%M:%S')"
echo "========================================"
echo ""

# 等待wind_cn COA完成
echo "等待 wind_cn COA 完成..."
while true; do
    done_count=0
    for seed in 42 123 456; do
        if [ -f "$LOG_DIR/coa_wind_cn_${seed}.log" ]; then
            if grep -q "\[DONE\]" "$LOG_DIR/coa_wind_cn_${seed}.log" 2>/dev/null; then
                ((done_count++))
            fi
        fi
    done
    
    echo "[$(date '+%H:%M:%S')] COA进度: $done_count/3"
    
    if [ "$done_count" -eq 3 ]; then
        echo ""
        echo "✅ wind_cn COA 全部完成！"
        break
    fi
    
    sleep 60
done

# 启动BWO-SVR
echo ""
echo "【启动 wind_cn BWO-SVR】"
python3 -u -c "
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
print('[START] BWO-SVR wind_cn seed=42', flush=True)
from bwo_svr_experiment_v2 import run_bwo_svr_single_experiment
try:
    result = run_bwo_svr_single_experiment('wind_cn', 42)
    print(f'[DONE] R2={result[\"R2\"]:.4f}', flush=True)
except Exception as e:
    print(f'[ERROR] {e}', flush=True)
" > "$LOG_DIR/bwo_svr_wind_cn_42.log" 2>&1 &
echo "  BWO-SVR seed=42: PID=$!"

sleep 3

python3 -u -c "
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
print('[START] BWO-SVR wind_cn seed=123', flush=True)
from bwo_svr_experiment_v2 import run_bwo_svr_single_experiment
try:
    result = run_bwo_svr_single_experiment('wind_cn', 123)
    print(f'[DONE] R2={result[\"R2\"]:.4f}', flush=True)
except Exception as e:
    print(f'[ERROR] {e}', flush=True)
" > "$LOG_DIR/bwo_svr_wind_cn_123.log" 2>&1 &
echo "  BWO-SVR seed=123: PID=$!"

sleep 3

python3 -u -c "
import sys
sys.path.append('/root')
import warnings
warnings.filterwarnings('ignore')
print('[START] BWO-SVR wind_cn seed=456', flush=True)
from bwo_svr_experiment_v2 import run_bwo_svr_single_experiment
try:
    result = run_bwo_svr_single_experiment('wind_cn', 456)
    print(f'[DONE] R2={result[\"R2\"]:.4f}', flush=True)
except Exception as e:
    print(f'[ERROR] {e}', flush=True)
" > "$LOG_DIR/bwo_svr_wind_cn_456.log" 2>&1 &
echo "  BWO-SVR seed=456: PID=$!"

echo "等待BWO-SVR完成..."
wait
echo "✅ BWO-SVR 完成"

echo ""
echo "========================================"
echo "后续实验完成！"
echo "========================================"
