#!/bin/bash
# ============================================================================
# 风电机组运行数据集 (wind_turbine_primary) 完整实验套件
# 用于远端 4090 Linux 系统
# ============================================================================

set -e  # 遇到错误立即退出

echo "========================================"
echo "风电机组数据完整实验套件 - RTX 4090"
echo "========================================"
echo ""

# 配置参数
DATASET="wind_turbine_primary"
TARGET="power"
DATASET_NAME="TurbinePrimary"
OUTPUT_TAG="turbine_primary"
DEVICE="cuda"
NUM_WORKERS=4  # Linux下可以用多进程

# 输出目录
BASE_OUT_DIR="results/turbine_primary_fullrun"
mkdir -p "$BASE_OUT_DIR"

# ============================================================================
# 阶段 1: Ultra-LSNT SOTA 搜索 (核心预测实验)
# ============================================================================
echo ""
echo "========================================"
echo "阶段 1: SOTA 搜索 (多配置多种子)"
echo "========================================"
echo "预计时间: 3-4小时 (RTX 4090)"
echo ""

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
    --skip-threshold 0.5 \
    --target-skip-rate 0.12 \
    --skip-budget-weight 1.0 \
    --slow-path-penalty-weight 0.15 \
    --easy-alignment-weight 0.2 \
    --jump-gate-threshold-grid "0.20,0.22,0.24,0.25,0.26,0.27,0.28,0.30,0.32,0.36,0.42,0.48" \
    --jump-gate-min-skip 0.02 \
    --jump-gate-score-penalty 0.05 \
    --export-best-edge-ready \
    --out-dir "$BASE_OUT_DIR/sota_search"

echo ""
echo "✓ 阶段 1 完成: SOTA 搜索完成"
echo ""

# ============================================================================
# 阶段 2: FG-MoE 完整训练 (安全护栏)
# ============================================================================
echo ""
echo "========================================"
echo "阶段 2: FG-MoE 训练 (Physics-Informed)"
echo "========================================"
echo "预计时间: 1-2小时"
echo ""

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
    --force-train \
    --base-config checkpoints_ts/main/model_config.json \
    --base-checkpoint checkpoints_ts/main/best_model.pth

echo ""
echo "✓ 阶段 2 完成: FG-MoE 训练完成"
echo ""

# ============================================================================
# 阶段 3: Clip-Only 控制对比
# ============================================================================
echo ""
echo "========================================"
echo "阶段 3: Clip-Only vs FG-MoE 对比"
echo "========================================"
echo "预计时间: 30分钟"
echo ""

python run_clip_only_guardrail_control.py \
    --data "$DATASET" \
    --target "$TARGET" \
    --dataset-name "$DATASET_NAME" \
    --output-tag "$OUTPUT_TAG" \
    --out-dir "$BASE_OUT_DIR/guardrail_compare" \
    --device "$DEVICE"

echo ""
echo "✓ 阶段 3 完成: Guardrail 对比完成"
echo ""

# ============================================================================
# 阶段 4: 安全部署诊断 (Safety Deployment)
# ============================================================================
echo ""
echo "========================================"
echo "阶段 4: Safety Deployment 耦合诊断"
echo "========================================"
echo "预计时间: 20分钟"
echo ""

# 检查FG-MoE检查点是否存在
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
    
    echo ""
    echo "✓ 阶段 4 完成: Safety Deployment 完成"
else
    echo "⚠ FG-MoE 检查点未找到，跳过阶段 4"
fi

echo ""

# ============================================================================
# 阶段 5: 完整故障公平性闭合 (Full-Fault Fairness Closure)
# ============================================================================
echo ""
echo "========================================"
echo "阶段 5: Full-Fault Fairness Closure"
echo "========================================"
echo "对比模型: DLinear, TimeMixer, iTransformer, Ultra-LSNT"
echo "故障类型: Gaussian, Drift, Quantization, Missingness"
echo "预计时间: 2-3小时"
echo ""

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

echo ""
echo "✓ 阶段 5 完成: Fairness Closure 完成"
echo ""

# ============================================================================
# 阶段 6: Observed Fault Bank 构建 (真实故障回放)
# ============================================================================
echo ""
echo "========================================"
echo "阶段 6: Observed Fault Bank 提取"
echo "========================================"
echo ""

python -c "
import pandas as pd
import numpy as np
import json
from pathlib import Path

# 读取数据
df = pd.read_csv('prepared_priority_datasets/wind_turbine_primary.csv')
print(f'总样本数: {len(df)}')

# 检测各类真实故障
fault_bank = []

# 1. Flatline (传感器卡死)
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
                    'start_time': df['datetime'].iloc[start_idx],
                    'severity': 'observed'
                })
            count = 0

# 2. Spike (异常尖峰)
for col in ['windspeed', 'power']:
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    spike_indices = df.index[z_scores > 3].tolist()
    for idx in spike_indices[:50]:  # 限制数量
        fault_bank.append({
            'type': 'spike',
            'column': col,
            'index': idx,
            'time': df['datetime'].iloc[idx],
            'value': float(df[col].iloc[idx]),
            'z_score': float(z_scores.iloc[idx]),
            'severity': 'observed'
        })

# 保存
out_dir = Path('$BASE_OUT_DIR/observed_fault_bank')
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / 'observed_fault_bank.json', 'w') as f:
    json.dump(fault_bank, f, indent=2)

print(f'提取故障片段: {len(fault_bank)}')
print(f'  - Flatline: {len([f for f in fault_bank if f[\"type\"]==\"flatline\"])}')
print(f'  - Spike: {len([f for f in fault_bank if f[\"type\"]==\"spike\"])}')
print(f'结果保存至: {out_dir}')
"

echo ""
echo "✓ 阶段 6 完成: Observed Fault Bank 构建完成"
echo ""

# ============================================================================
# 生成实验摘要
# ============================================================================
echo ""
echo "========================================"
echo "所有阶段完成！"
echo "========================================"
echo ""
echo "输出目录结构:"
echo "  $BASE_OUT_DIR/"
echo "    ├── sota_search/           # SOTA搜索主结果"
echo "    │   ├── leaderboard.csv"
echo "    │   └── best_run/"
echo "    ├── fgmoe/                 # FG-MoE训练结果"
echo "    │   └── pi_moe_full_checkpoint_${OUTPUT_TAG}.pth"
echo "    ├── guardrail_compare/     # Clip-Only对比"
echo "    │   └── fgmoe_clip_only_control_${OUTPUT_TAG}.csv"
echo "    ├── safety_deployment/     # 安全部署诊断"
echo "    │   └── table_safety_coupling_${OUTPUT_TAG}.csv"
echo "    ├── fairness_closure/      # 完整故障公平性闭合"
echo "    │   ├── fairness_closure_fullfault_raw_${OUTPUT_TAG}.csv"
echo "    │   └── fairness_closure_fullfault_summary_${OUTPUT_TAG}.csv"
echo "    └── observed_fault_bank/   # 真实故障库"
echo "        └── observed_fault_bank.json"
echo ""
echo "核心验证指标:"
echo "  - Ultra-LSNT R2 (clean): 查看 sota_search/leaderboard.csv"
echo "  - FG-MoE vs Ultra-LSNT: 查看 fgmoe/ 目录对比"
echo "  - 多模型鲁棒性对比: 查看 fairness_closure/ 目录"
echo ""
echo "========================================"
