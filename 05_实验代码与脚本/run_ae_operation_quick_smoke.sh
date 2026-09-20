#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
SMOKE_GPU="${SMOKE_GPU:-0}"
EXPORT_ROOT="${EXPORT_ROOT:-results/ae_operation_exports/smoke}"
DISPATCH_ROOT="${DISPATCH_ROOT:-results/ae_operation_dispatch/smoke}"
SUMMARY_ROOT="${SUMMARY_ROOT:-results/ae_operation_summary/smoke}"
TRAIN_ROOT="${TRAIN_ROOT:-results/ae_operation_guardrails/smoke}"
LOG_ROOT="${LOG_ROOT:-logs_ae_operation_smoke}"

mkdir -p "$EXPORT_ROOT" "$DISPATCH_ROOT" "$SUMMARY_ROOT" "$TRAIN_ROOT" "$LOG_ROOT"

echo "[smoke] checking CLIs"
"$PYTHON_EXECUTABLE" export_guardrail_prediction_archives.py --help >"$LOG_ROOT/export_help.log" 2>&1
"$PYTHON_EXECUTABLE" run_dispatch_closure_mapping_decision_4090.py --help >"$LOG_ROOT/dispatch_help.log" 2>&1
"$PYTHON_EXECUTABLE" build_decision_aware_dispatch_evidence.py --help >"$LOG_ROOT/build_help.log" 2>&1
"$PYTHON_EXECUTABLE" aggregate_ae_operation_results.py --help >"$LOG_ROOT/aggregate_help.log" 2>&1

ULTRA_TRAIN_DIR="$TRAIN_ROOT/windcn_ultra_seed42"
ULTRA_DIR="$EXPORT_ROOT/windcn_ultra_seed42"
mkdir -p "$ULTRA_DIR"
mkdir -p "$ULTRA_TRAIN_DIR"

echo "[smoke] training revised Ultra-LSNT guardrail"
CUDA_VISIBLE_DEVICES="$SMOKE_GPU" "$PYTHON_EXECUTABLE" run_pi_moe_full_training_evidence.py \
  --dataset wind_final.csv \
  --target power \
  --dataset-name Wind_CN \
  --backbone ultra \
  --base-config checkpoints_ts/main/model_config.json \
  --base-checkpoint checkpoints_ts/main/best_model.pth \
  --seed 42 \
  --out-dir "$ULTRA_TRAIN_DIR" \
  --output-tag windcn_ultra_seed42_smoke \
  --force-train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  >"$LOG_ROOT/train_ultra_smoke.log" 2>&1

ULTRA_FG_CKPT="$ULTRA_TRAIN_DIR/pi_moe_full_checkpoint_windcn_ultra_seed42_smoke.pth"

echo "[smoke] exporting Ultra-LSNT guardrail variants"
CUDA_VISIBLE_DEVICES="$SMOKE_GPU" "$PYTHON_EXECUTABLE" export_guardrail_prediction_archives.py \
  --dataset wind_final.csv \
  --target power \
  --dataset-name Wind_CN \
  --backbone ultra \
  --variants raw,clip_only,full_guardrail \
  --base-config checkpoints_ts/main/model_config.json \
  --base-checkpoint checkpoints_ts/main/best_model.pth \
  --fg-checkpoint "$ULTRA_FG_CKPT" \
  --seed 42 \
  --output-tag windcn_ultra_seed42_smoke \
  --out-dir "$ULTRA_DIR" \
  --max-eval-batches 2 \
  >"$LOG_ROOT/export_ultra_smoke.log" 2>&1

MANIFEST_JSON="$DISPATCH_ROOT/model_manifest_smoke.json"
export MANIFEST_JSON ULTRA_DIR
python - <<'PY'
import json
import os

manifest_path = os.environ["MANIFEST_JSON"]
ultra_dir = os.environ["ULTRA_DIR"].replace("\\", "/")
models = [
    {"name": "DLinear", "path": "results/supplementary_evidence/dlinear_predictions_full.npz", "order": 0},
    {"name": "TimeMixer", "path": "checkpoints_ts/TimeMixer_preds.npy", "order": 1},
    {"name": "iTransformer raw", "path": "checkpoints_ts/iTransformer_preds.npy", "order": 2},
    {"name": "Ultra-LSNT raw", "path": f"{ultra_dir}/prediction_archive_raw_windcn_ultra_seed42_smoke.npz", "order": 3},
    {"name": "Ultra-LSNT + clip-only", "path": f"{ultra_dir}/prediction_archive_clip_only_windcn_ultra_seed42_smoke.npz", "order": 4},
    {"name": "Ultra-LSNT + FG-MoE", "path": f"{ultra_dir}/prediction_archive_full_guardrail_windcn_ultra_seed42_smoke.npz", "order": 5},
]
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump({"models": models}, f, indent=2, ensure_ascii=False)
PY

echo "[smoke] running affine smoke dispatch"
"$PYTHON_EXECUTABLE" run_dispatch_closure_mapping_decision_4090.py \
  --days 2 \
  --mapping-methods affine \
  --risk-subset-mode shared \
  --model-manifest "$MANIFEST_JSON" \
  --out-dir "$DISPATCH_ROOT" \
  --output-prefix dispatch_operation_smoke_affine \
  >"$LOG_ROOT/dispatch_smoke.log" 2>&1

echo "[smoke] building operation summary"
"$PYTHON_EXECUTABLE" build_decision_aware_dispatch_evidence.py \
  --mode operation \
  --dispatch-agg-csv "$DISPATCH_ROOT/dispatch_operation_smoke_affine_aggregate.csv" \
  --dispatch-meta-json "$DISPATCH_ROOT/dispatch_operation_smoke_affine.meta.json" \
  --focus-mapping affine \
  --focus-subset high_risk_shared \
  --out-summary-csv "$SUMMARY_ROOT/operation_smoke_summary.csv" \
  --out-figure-prefix "$SUMMARY_ROOT/fig_operation_smoke" \
  --out-meta "$SUMMARY_ROOT/operation_smoke_summary.meta.json" \
  >"$LOG_ROOT/build_smoke.log" 2>&1

echo "[smoke] aggregating smoke outputs"
"$PYTHON_EXECUTABLE" aggregate_ae_operation_results.py \
  --dispatch-root "$DISPATCH_ROOT" \
  --focus-mapping affine \
  --focus-subset high_risk_shared \
  --out-dir "$SUMMARY_ROOT" \
  >"$LOG_ROOT/aggregate_smoke.log" 2>&1

echo "[smoke] completed"
