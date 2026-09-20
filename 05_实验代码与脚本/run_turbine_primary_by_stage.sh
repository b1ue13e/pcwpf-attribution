#!/bin/bash
# ============================================================================
# 风电机组数据 - 分阶段执行脚本
# 用法: bash run_turbine_primary_by_stage.sh [stage_number]
# 
# 阶段说明:
#   1 - SOTA搜索 (核心预测实验)
#   2 - FG-MoE训练 (安全护栏)
#   3 - Clip-Only对比
#   4 - Safety Deployment
#   5 - Fairness Closure
#   6 - Observed Fault Bank
# ============================================================================

set -e

# 配置
DATASET="wind_turbine_primary"
TARGET="power"
DATASET_NAME="TurbinePrimary"
OUTPUT_TAG="turbine_primary"
DEVICE="cuda"
NUM_WORKERS=4
BASE_OUT_DIR="results/turbine_primary_fullrun"

mkdir -p "$BASE_OUT_DIR"

# 获取阶段参数
STAGE=${1:-all}

run_stage_1() {
    echo "========================================"
    echo "阶段 1: SOTA搜索"
    echo "========================================"
    python run_ultra_lsnt_sota_search_4090.py \
        --train-data "$DATASET" \
        --target "$TARGET" \
        --train-name "$DATASET_NAME" \
        --presets "balanced,wide,deep,aggressive" \
        --seeds "42,52,62" \
        --epochs 80 \
        --pretrain-epochs 30 \
        --transfer-epochs 12 \
        --batch-size 128 \
        --device "$DEVICE" \
        --num-workers "$NUM_WORKERS" \
        --export-best-edge-ready \
        --out-dir "$BASE_OUT_DIR/sota_search"
    echo "✓ 阶段1完成"
}

run_stage_2() {
    echo "========================================"
    echo "阶段 2: FG-MoE训练"
    echo "========================================"
    python run_pi_moe_full_training_evidence.py \
        --data "$DATASET" \
        --target "$TARGET" \
        --dataset-name "$DATASET_NAME" \
        --output-tag "$OUTPUT_TAG" \
        --out-dir "$BASE_OUT_DIR/fgmoe" \
        --epochs 30 \
        --batch-size 128 \
        --device "$DEVICE" \
        --num-workers "$NUM_WORKERS" \
        --force-train
    echo "✓ 阶段2完成"
}

run_stage_3() {
    echo "========================================"
    echo "阶段 3: Clip-Only对比"
    echo "========================================"
    python run_clip_only_guardrail_control.py \
        --data "$DATASET" \
        --target "$TARGET" \
        --dataset-name "$DATASET_NAME" \
        --output-tag "$OUTPUT_TAG" \
        --out-dir "$BASE_OUT_DIR/guardrail_compare" \
        --device "$DEVICE"
    echo "✓ 阶段3完成"
}

run_stage_4() {
    echo "========================================"
    echo "阶段 4: Safety Deployment"
    echo "========================================"
    FG_CHECKPOINT="$BASE_OUT_DIR/fgmoe/pi_moe_full_checkpoint_${OUTPUT_TAG}.pth"
    if [ -f "$FG_CHECKPOINT" ]; then
        python run_safety_deployment.py \
            --mode paper \
            --data "$DATASET" \
            --target "$TARGET" \
            --dataset-name "$DATASET_NAME" \
            --output-tag "$OUTPUT_TAG" \
            --fg-checkpoint "$FG_CHECKPOINT" \
            --outdir "$BASE_OUT_DIR/safety_deployment" \
            --batch-size 32 \
            --device "$DEVICE"
        echo "✓ 阶段4完成"
    else
        echo "⚠ 跳过阶段4: FG检查点不存在"
    fi
}

run_stage_5() {
    echo "========================================"
    echo "阶段 5: Fairness Closure"
    echo "========================================"
    python run_fairness_closure_fullfault_4090.py \
        --data "$DATASET" \
        --target "$TARGET" \
        --dataset-name "$DATASET_NAME" \
        --output-tag "$OUTPUT_TAG" \
        --out-dir "$BASE_OUT_DIR/fairness_closure" \
        --models "dlinear,timemixer,itransformer,ultra" \
        --seeds "42,123,2025" \
        --faults "gaussian,drift,quantization,missingness" \
        --epochs 15 \
        --batch-size 128 \
        --device "$DEVICE" \
        --num-workers "$NUM_WORKERS"
    echo "✓ 阶段5完成"
}

run_stage_6() {
    echo "========================================"
    echo "阶段 6: Observed Fault Bank"
    echo "========================================"
    python -c "
import pandas as pd
import numpy as np
import json
from pathlib import Path

df = pd.read_csv('prepared_priority_datasets/wind_turbine_primary.csv')
fault_bank = []

# Flatline detection
for col in ['windspeed', 'power']:
    same_val = (df[col].diff() == 0).astype(int)
    count = 0
    start_idx = None
    for i, val in enumerate(same_val):
        if val == 1:
            if count == 0:
                start_idx = i
            count += 1
        else:
            if count >= 5:
                fault_bank.append({
                    'type': 'flatline',
                    'column': col,
                    'start_idx': start_idx,
                    'length': count,
                    'start_time': str(df['datetime'].iloc[start_idx]),
                    'severity': 'observed'
                })
            count = 0

# Spike detection
for col in ['windspeed', 'power']:
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    spike_indices = df.index[z_scores > 3].tolist()
    for idx in spike_indices[:50]:
        fault_bank.append({
            'type': 'spike',
            'column': col,
            'index': idx,
            'time': str(df['datetime'].iloc[idx]),
            'value': float(df[col].iloc[idx]),
            'z_score': float(z_scores.iloc[idx]),
            'severity': 'observed'
        })

out_dir = Path('$BASE_OUT_DIR/observed_fault_bank')
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / 'observed_fault_bank.json', 'w') as f:
    json.dump(fault_bank, f, indent=2)
print(f'提取故障片段: {len(fault_bank)}')
"
    echo "✓ 阶段6完成"
}

# 执行逻辑
case "$STAGE" in
    1)
        run_stage_1
        ;;
    2)
        run_stage_2
        ;;
    3)
        run_stage_3
        ;;
    4)
        run_stage_4
        ;;
    5)
        run_stage_5
        ;;
    6)
        run_stage_6
        ;;
    all|"")
        run_stage_1
        run_stage_2
        run_stage_3
        run_stage_4
        run_stage_5
        run_stage_6
        ;;
    *)
        echo "用法: bash $0 [stage_number]"
        echo ""
        echo "阶段说明:"
        echo "  1 - SOTA搜索 (核心预测实验, 3-4小时)"
        echo "  2 - FG-MoE训练 (安全护栏, 1-2小时)"
        echo "  3 - Clip-Only对比 (30分钟)"
        echo "  4 - Safety Deployment (20分钟)"
        echo "  5 - Fairness Closure (2-3小时)"
        echo "  6 - Observed Fault Bank (即时)"
        echo "  all - 全部阶段 (默认)"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "指定阶段执行完成！"
echo "========================================"
