#!/bin/bash
# ============================================================================
# 风电机组数据快速冒烟测试 (Quick Smoke Test)
# 用于远端 4090 Linux 系统快速验证
# ============================================================================

set -e

echo "========================================"
echo "风电机组数据 - 快速冒烟测试"
echo "========================================"
echo ""

DATASET="wind_turbine_primary"
TARGET="power"
DATASET_NAME="TurbinePrimary"
OUTPUT_TAG="turbine_smoke"
DEVICE="cuda"

BASE_OUT_DIR="results/turbine_primary_smoke"
mkdir -p "$BASE_OUT_DIR"

echo "配置:"
echo "  数据集: $DATASET"
echo "  输出标签: $OUTPUT_TAG"
echo "  设备: $DEVICE"
echo ""

# 快速SOTA搜索（单配置单种子）
echo ">>> 运行快速SOTA搜索..."
python run_ultra_lsnt_sota_search_4090.py \
    --train-data "$DATASET" \
    --target "$TARGET" \
    --train-name "$DATASET_NAME" \
    --presets "balanced" \
    --seeds "42" \
    --epochs 5 \
    --pretrain-epochs 0 \
    --transfer-epochs 0 \
    --batch-size 128 \
    --device "$DEVICE" \
    --num-workers 4 \
    --max-train-batches 50 \
    --max-eval-batches 20 \
    --out-dir "$BASE_OUT_DIR/sota_search"

echo ""
echo "✓ 快速冒烟测试完成！"
echo ""
echo "检查结果:"
echo "  $BASE_OUT_DIR/sota_search/leaderboard.csv"
echo ""

# 如果成功，提示可以进行完整训练
if [ -f "$BASE_OUT_DIR/sota_search/leaderboard.csv" ]; then
    echo "测试通过！可以运行完整实验:"
    echo "  bash run_turbine_primary_full_suite_4090.sh"
fi
