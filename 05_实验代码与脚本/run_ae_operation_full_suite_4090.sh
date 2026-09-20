#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
EXPORT_ROOT="${EXPORT_ROOT:-results/ae_operation_exports}"
DISPATCH_ROOT="${DISPATCH_ROOT:-results/ae_operation_dispatch}"
SUMMARY_ROOT="${SUMMARY_ROOT:-results/ae_operation_summary}"
TRAIN_ROOT="${TRAIN_ROOT:-results/ae_operation_guardrails}"
LOG_ROOT="${LOG_ROOT:-logs_ae_operation}"
ULTRA_BASE_CKPT="${ULTRA_BASE_CKPT:-checkpoints_ts/main/best_model.pth}"
ULTRA_BASE_CFG="${ULTRA_BASE_CFG:-checkpoints_ts/main/model_config.json}"
ITR_BASE_CKPT="${ITR_BASE_CKPT:-checkpoints_ts/iTransformer_best.pth}"
ULTRA_TRAIN_EPOCHS="${ULTRA_TRAIN_EPOCHS:-8}"
ULTRA_TRAIN_LR="${ULTRA_TRAIN_LR:-2e-5}"
ULTRA_WEIGHT_DECAY="${ULTRA_WEIGHT_DECAY:-0.005}"
ULTRA_AUX_WEIGHT="${ULTRA_AUX_WEIGHT:-0.005}"
ULTRA_LAMBDA2="${ULTRA_LAMBDA2:-0.03}"
ULTRA_CORRUPTION_WEIGHT="${ULTRA_CORRUPTION_WEIGHT:-0.20}"
ULTRA_TAIL_WEIGHT="${ULTRA_TAIL_WEIGHT:-0.005}"
ULTRA_RAMP_WEIGHT="${ULTRA_RAMP_WEIGHT:-0.002}"
ULTRA_CONSISTENCY_WEIGHT="${ULTRA_CONSISTENCY_WEIGHT:-0.005}"
ULTRA_TAIL_QUANTILE="${ULTRA_TAIL_QUANTILE:-0.95}"
ULTRA_CORRUPTION_SIGMA_MIN="${ULTRA_CORRUPTION_SIGMA_MIN:-0.04}"
ULTRA_CORRUPTION_SIGMA_MAX="${ULTRA_CORRUPTION_SIGMA_MAX:-0.18}"
ULTRA_STRUCTURED_FAULT_PROB="${ULTRA_STRUCTURED_FAULT_PROB:-0.35}"
ULTRA_DRIFT_PROB="${ULTRA_DRIFT_PROB:-0.25}"
ULTRA_QUANTIZATION_PROB="${ULTRA_QUANTIZATION_PROB:-0.20}"

mkdir -p "$EXPORT_ROOT" "$DISPATCH_ROOT" "$SUMMARY_ROOT" "$TRAIN_ROOT" "$LOG_ROOT"

echo "[suite] quick smoke first"
PYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
  EXPORT_ROOT="$EXPORT_ROOT/smoke" \
  DISPATCH_ROOT="$DISPATCH_ROOT/smoke" \
  SUMMARY_ROOT="$SUMMARY_ROOT/smoke" \
  TRAIN_ROOT="$TRAIN_ROOT/smoke" \
  LOG_ROOT="$LOG_ROOT/smoke" \
  SMOKE_GPU=0 \
  bash "$ROOT_DIR/run_ae_operation_quick_smoke.sh"

PIDS=()
NAMES=()

spawn_train_export_ultra() {
  local gpu="$1"
  local seed="$2"
  local train_dir="$TRAIN_ROOT/windcn_ultra/seed_${seed}"
  local out_dir="$EXPORT_ROOT/windcn_ultra/seed_${seed}"
  local log_file="$LOG_ROOT/train_export_ultra_seed${seed}.log"
  mkdir -p "$train_dir" "$out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_EXECUTABLE" run_pi_moe_full_training_evidence.py \
      --dataset wind_final.csv \
      --target power \
      --dataset-name Wind_CN \
      --backbone ultra \
      --base-config "$ULTRA_BASE_CFG" \
      --base-checkpoint "$ULTRA_BASE_CKPT" \
      --seed "$seed" \
      --out-dir "$train_dir" \
      --output-tag "windcn_ultra_seed${seed}" \
      --force-train \
      --epochs "$ULTRA_TRAIN_EPOCHS" \
      --lr "$ULTRA_TRAIN_LR" \
      --weight-decay "$ULTRA_WEIGHT_DECAY" \
      --aux-weight "$ULTRA_AUX_WEIGHT" \
      --lambda2 "$ULTRA_LAMBDA2" \
      --corruption-weight "$ULTRA_CORRUPTION_WEIGHT" \
      --tail-weight "$ULTRA_TAIL_WEIGHT" \
      --ramp-weight "$ULTRA_RAMP_WEIGHT" \
      --consistency-weight "$ULTRA_CONSISTENCY_WEIGHT" \
      --tail-quantile "$ULTRA_TAIL_QUANTILE" \
      --corruption-sigma-min "$ULTRA_CORRUPTION_SIGMA_MIN" \
      --corruption-sigma-max "$ULTRA_CORRUPTION_SIGMA_MAX" \
      --structured-fault-prob "$ULTRA_STRUCTURED_FAULT_PROB" \
      --drift-prob "$ULTRA_DRIFT_PROB" \
      --quantization-prob "$ULTRA_QUANTIZATION_PROB"
    "$PYTHON_EXECUTABLE" export_guardrail_prediction_archives.py \
      --dataset wind_final.csv \
      --target power \
      --dataset-name Wind_CN \
      --backbone ultra \
      --variants raw,clip_only,full_guardrail \
      --base-config "$ULTRA_BASE_CFG" \
      --base-checkpoint "$ULTRA_BASE_CKPT" \
      --fg-checkpoint "$train_dir/pi_moe_full_checkpoint_windcn_ultra_seed${seed}.pth" \
      --seed "$seed" \
      --output-tag "windcn_ultra_seed${seed}" \
      --out-dir "$out_dir"
  ) >"$log_file" 2>&1 &
  PIDS+=("$!")
  NAMES+=("train_export_ultra_seed${seed}")
}

spawn_export_itransformer() {
  local gpu="$1"
  local seed="$2"
  local fg_ckpt="pretrained_guardrails/windcn_itransformer_fg_seed${seed}.pth"
  local out_dir="$EXPORT_ROOT/windcn_itransformer/seed_${seed}"
  local log_file="$LOG_ROOT/export_itransformer_seed${seed}.log"
  mkdir -p "$out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_EXECUTABLE" export_guardrail_prediction_archives.py \
      --dataset wind_final.csv \
      --target power \
      --dataset-name Wind_CN \
      --backbone itransformer \
      --variants raw,full_guardrail \
      --base-checkpoint "$ITR_BASE_CKPT" \
      --fg-checkpoint "$fg_ckpt" \
      --seed "$seed" \
      --output-tag "windcn_itransformer_seed${seed}" \
      --out-dir "$out_dir"
  ) >"$log_file" 2>&1 &
  PIDS+=("$!")
  NAMES+=("export_itransformer_seed${seed}")
}

echo "[suite] launching prediction exports"
spawn_train_export_ultra 1 42
spawn_train_export_ultra 2 123
spawn_train_export_ultra 3 2025

spawn_export_itransformer 4 42
spawn_export_itransformer 5 123
spawn_export_itransformer 6 2025

for idx in "${!PIDS[@]}"; do
  wait "${PIDS[$idx]}"
  echo "[suite] completed ${NAMES[$idx]}"
done

write_manifest() {
  local manifest_path="$1"
  local seed="$2"
  local include_itr_fg="$3"
  export MANIFEST_PATH="$manifest_path"
  export SEED_TAG="$seed"
  export EXPORT_ROOT_ABS="$EXPORT_ROOT"
  export INCLUDE_ITR_FG="$include_itr_fg"
  python - <<'PY'
import json
import os

manifest_path = os.environ["MANIFEST_PATH"]
seed = os.environ["SEED_TAG"]
export_root = os.environ["EXPORT_ROOT_ABS"].replace("\\", "/")
include_itr_fg = os.environ["INCLUDE_ITR_FG"] == "1"

models = [
    {"name": "DLinear", "path": "results/supplementary_evidence/dlinear_predictions_full.npz", "order": 0},
    {"name": "TimeMixer", "path": "checkpoints_ts/TimeMixer_preds.npy", "order": 1},
    {"name": "iTransformer raw", "path": f"{export_root}/windcn_itransformer/seed_{seed}/prediction_archive_raw_windcn_itransformer_seed{seed}.npz", "order": 2},
]
if include_itr_fg:
    models.append(
        {"name": "iTransformer + FG-MoE", "path": f"{export_root}/windcn_itransformer/seed_{seed}/prediction_archive_full_guardrail_windcn_itransformer_seed{seed}.npz", "order": 3}
    )
    base_order = 4
else:
    base_order = 3
models.extend(
    [
        {"name": "Ultra-LSNT raw", "path": f"{export_root}/windcn_ultra/seed_{seed}/prediction_archive_raw_windcn_ultra_seed{seed}.npz", "order": base_order},
        {"name": "Ultra-LSNT + clip-only", "path": f"{export_root}/windcn_ultra/seed_{seed}/prediction_archive_clip_only_windcn_ultra_seed{seed}.npz", "order": base_order + 1},
        {"name": "Ultra-LSNT + FG-MoE", "path": f"{export_root}/windcn_ultra/seed_{seed}/prediction_archive_full_guardrail_windcn_ultra_seed{seed}.npz", "order": base_order + 2},
    ]
)
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump({"models": models}, f, indent=2, ensure_ascii=False)
PY
}

SEED42_MAIN_MANIFEST="$DISPATCH_ROOT/seed_42/main_model_manifest.json"
SEED42_DECOUPLING_MANIFEST="$DISPATCH_ROOT/seed_42/decoupling_model_manifest.json"
SEED123_MAIN_MANIFEST="$DISPATCH_ROOT/seed_123/main_model_manifest.json"
SEED2025_MAIN_MANIFEST="$DISPATCH_ROOT/seed_2025/main_model_manifest.json"

mkdir -p "$DISPATCH_ROOT/seed_42" "$DISPATCH_ROOT/seed_123" "$DISPATCH_ROOT/seed_2025"
write_manifest "$SEED42_MAIN_MANIFEST" 42 0
write_manifest "$SEED42_DECOUPLING_MANIFEST" 42 1
write_manifest "$SEED123_MAIN_MANIFEST" 123 0
write_manifest "$SEED2025_MAIN_MANIFEST" 2025 0

run_dispatch_job() {
  local gpu="$1"
  local mapping="$2"
  local seed="$3"
  local manifest="$4"
  local tag="$5"
  local decision_mode="${6:-point}"
  local out_dir="$DISPATCH_ROOT/seed_${seed}/${tag}"
  local log_file="$LOG_ROOT/dispatch_seed${seed}_${tag}.log"
  mkdir -p "$out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    extra_args=()
    if [[ "$decision_mode" == "uq_holdout" ]]; then
      extra_args+=(
        --decision-mode uq_holdout
        --uq-alpha 0.10
        --uq-regime-mode shared_risk
        --uq-margin-mode rolling_bias_scaled
        --uq-score-floor-quantile 0.25
        --uq-score-floor-min-mw 5
        --uq-horizon-smooth-window 3
        --uq-max-relative-margin 0.75
        --uq-rolling-window-days 14
        --uq-min-history-samples 6
        --uq-hour-band-radius 1
        --uq-bias-quantile 0.50
        --uq-bias-shrink 0.75
        --holdout-start-day 80
      )
    fi
    "$PYTHON_EXECUTABLE" run_dispatch_closure_mapping_decision_4090.py \
      --days 120 \
      --mapping-methods "$mapping" \
      --risk-subset-mode shared \
      --model-manifest "$manifest" \
      --out-dir "$out_dir" \
      --output-prefix "dispatch_operation_seed${seed}_${tag}" \
      "${extra_args[@]}"
  ) >"$log_file" 2>&1 &
  PIDS+=("$!")
  NAMES+=("dispatch_seed${seed}_${tag}")
}

PIDS=()
NAMES=()

echo "[suite] launching dispatch closures"
run_dispatch_job 7 risk_conditional 42 "$SEED42_MAIN_MANIFEST" risk_conditional
run_dispatch_job 8 quantile 42 "$SEED42_MAIN_MANIFEST" quantile
run_dispatch_job 9 affine 42 "$SEED42_MAIN_MANIFEST" affine

for idx in "${!PIDS[@]}"; do
  wait "${PIDS[$idx]}"
  echo "[suite] completed ${NAMES[$idx]}"
done

PIDS=()
NAMES=()
run_dispatch_job 7 risk_conditional 42 "$SEED42_DECOUPLING_MANIFEST" risk_conditional_decoupling
run_dispatch_job 8 risk_conditional 123 "$SEED123_MAIN_MANIFEST" risk_conditional
run_dispatch_job 9 risk_conditional 2025 "$SEED2025_MAIN_MANIFEST" risk_conditional

for idx in "${!PIDS[@]}"; do
  wait "${PIDS[$idx]}"
  echo "[suite] completed ${NAMES[$idx]}"
done

PIDS=()
NAMES=()
echo "[suite] launching calibrated UQ holdout dispatch"
run_dispatch_job 7 risk_conditional 42 "$SEED42_MAIN_MANIFEST" risk_conditional_uq_holdout uq_holdout

for idx in "${!PIDS[@]}"; do
  wait "${PIDS[$idx]}"
  echo "[suite] completed ${NAMES[$idx]}"
done

echo "[suite] building main operation-aware summary"
"$PYTHON_EXECUTABLE" build_decision_aware_dispatch_evidence.py \
  --mode operation \
  --dispatch-agg-csv "$DISPATCH_ROOT/seed_42/risk_conditional/dispatch_operation_seed42_risk_conditional_aggregate.csv" \
  --dispatch-meta-json "$DISPATCH_ROOT/seed_42/risk_conditional/dispatch_operation_seed42_risk_conditional.meta.json" \
  --focus-mapping risk_conditional \
  --focus-subset high_risk_shared \
  --out-summary-csv "$SUMMARY_ROOT/operation_high_risk_summary_seed42.csv" \
  --out-figure-prefix "$SUMMARY_ROOT/fig_operation_high_risk_seed42" \
  --out-meta "$SUMMARY_ROOT/operation_high_risk_summary_seed42.meta.json" \
  >"$LOG_ROOT/build_operation_main.log" 2>&1

echo "[suite] building decoupling supplementary summary"
"$PYTHON_EXECUTABLE" build_decision_aware_dispatch_evidence.py \
  --mode operation \
  --dispatch-agg-csv "$DISPATCH_ROOT/seed_42/risk_conditional_decoupling/dispatch_operation_seed42_risk_conditional_decoupling_aggregate.csv" \
  --dispatch-meta-json "$DISPATCH_ROOT/seed_42/risk_conditional_decoupling/dispatch_operation_seed42_risk_conditional_decoupling.meta.json" \
  --focus-mapping risk_conditional \
  --focus-subset high_risk_shared \
  --model-order "iTransformer raw,iTransformer + FG-MoE,Ultra-LSNT raw,Ultra-LSNT + FG-MoE" \
  --out-summary-csv "$SUMMARY_ROOT/operation_decoupling_summary_seed42.csv" \
  --out-figure-prefix "$SUMMARY_ROOT/fig_operation_decoupling_seed42" \
  --out-meta "$SUMMARY_ROOT/operation_decoupling_summary_seed42.meta.json" \
  >"$LOG_ROOT/build_operation_decoupling.log" 2>&1

echo "[suite] building calibrated UQ holdout summary"
"$PYTHON_EXECUTABLE" build_decision_aware_dispatch_evidence.py \
  --mode operation_uq \
  --dispatch-agg-csv "$DISPATCH_ROOT/seed_42/risk_conditional_uq_holdout/dispatch_operation_seed42_risk_conditional_uq_holdout_aggregate.csv" \
  --dispatch-meta-json "$DISPATCH_ROOT/seed_42/risk_conditional_uq_holdout/dispatch_operation_seed42_risk_conditional_uq_holdout.meta.json" \
  --focus-mapping risk_conditional \
  --focus-subset all_holdout \
  --out-summary-csv "$SUMMARY_ROOT/operation_uq_holdout_summary_seed42.csv" \
  --out-figure-prefix "$SUMMARY_ROOT/fig_operation_uq_holdout_seed42" \
  --out-meta "$SUMMARY_ROOT/operation_uq_holdout_summary_seed42.meta.json" \
  --out-latex-table "$SUMMARY_ROOT/operation_uq_holdout_summary_seed42_table.tex" \
  >"$LOG_ROOT/build_operation_uq_holdout.log" 2>&1

echo "[suite] aggregating seed stability"
"$PYTHON_EXECUTABLE" aggregate_ae_operation_results.py \
  --dispatch-root "$DISPATCH_ROOT" \
  --focus-mapping risk_conditional \
  --focus-subset high_risk_shared \
  --out-dir "$SUMMARY_ROOT" \
  >"$LOG_ROOT/aggregate_operation.log" 2>&1

echo "[suite] completed"
