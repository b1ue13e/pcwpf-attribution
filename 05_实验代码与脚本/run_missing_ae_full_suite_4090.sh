#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
RESULT_ROOT="${RESULT_ROOT:-results/missing_ae_guardrails}"
DEPLOY_ROOT="${DEPLOY_ROOT:-results/deployment_evidence_missing_ae}"
LOG_ROOT="${LOG_ROOT:-logs_missing_ae}"
GUARDRAIL_EPOCHS="${GUARDRAIL_EPOCHS:-5}"
BASE_TRAIN_EPOCHS="${BASE_TRAIN_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"

mkdir -p "$RESULT_ROOT" "$DEPLOY_ROOT" "$LOG_ROOT"

echo "[suite] quick smoke first"
PYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" OUT_ROOT="$RESULT_ROOT/smoke" DEPLOY_ROOT="$DEPLOY_ROOT/smoke" SMOKE_GPU=0 \
  bash "$ROOT_DIR/run_missing_ae_quick_smoke.sh"

PIDS=()
NAMES=()
ULTRA42_PID=""

spawn_windcn_itransformer() {
  local gpu="$1"
  local seed="$2"
  local out_dir="$RESULT_ROOT/windcn_itransformer/seed_${seed}"
  local output_tag="windcn_itransformer_seed${seed}"
  local log_file="$LOG_ROOT/windcn_itransformer_seed${seed}.log"
  mkdir -p "$out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_EXECUTABLE" run_pi_moe_full_training_evidence.py \
      --backbone itransformer \
      --data wind_final.csv \
      --target power \
      --dataset-name Wind_CN \
      --output-tag "$output_tag" \
      --out-dir "$out_dir" \
      --seed "$seed" \
      --base-checkpoint checkpoints_ts/iTransformer_best.pth \
      --epochs "$GUARDRAIL_EPOCHS" \
      --batch-size "$BATCH_SIZE"

    "$PYTHON_EXECUTABLE" run_clip_only_guardrail_control.py \
      --backbone itransformer \
      --data wind_final.csv \
      --target power \
      --dataset-name Wind_CN \
      --output-tag "$output_tag" \
      --out-dir "$out_dir" \
      --seed "$seed" \
      --base-checkpoint checkpoints_ts/iTransformer_best.pth
  ) >"$log_file" 2>&1 &
  PIDS+=("$!")
  NAMES+=("windcn_itransformer_seed${seed}")
}

spawn_windus_itransformer() {
  local gpu="$1"
  local seed="$2"
  local out_dir="$RESULT_ROOT/windus_itransformer/seed_${seed}"
  local output_tag="windus_itransformer_seed${seed}"
  local base_ckpt="checkpoints_ts/exp_wind_us_itransformer/seed_${seed}/best_model.pth"
  local log_file="$LOG_ROOT/windus_itransformer_seed${seed}.log"
  mkdir -p "$out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_EXECUTABLE" run_pi_moe_full_training_evidence.py \
      --backbone itransformer \
      --data wind_us.csv \
      --target "power (MW)" \
      --dataset-name Wind_US \
      --output-tag "$output_tag" \
      --out-dir "$out_dir" \
      --seed "$seed" \
      --base-checkpoint "$base_ckpt" \
      --save-base-checkpoint "$base_ckpt" \
      --epochs "$GUARDRAIL_EPOCHS" \
      --base-train-epochs "$BASE_TRAIN_EPOCHS" \
      --batch-size "$BATCH_SIZE"

    "$PYTHON_EXECUTABLE" run_clip_only_guardrail_control.py \
      --backbone itransformer \
      --data wind_us.csv \
      --target "power (MW)" \
      --dataset-name Wind_US \
      --output-tag "$output_tag" \
      --out-dir "$out_dir" \
      --seed "$seed" \
      --base-checkpoint "$base_ckpt"
  ) >"$log_file" 2>&1 &
  PIDS+=("$!")
  NAMES+=("windus_itransformer_seed${seed}")
}

spawn_windus_ultra() {
  local gpu="$1"
  local seed="$2"
  local out_dir="$RESULT_ROOT/windus_ultra/seed_${seed}"
  local output_tag="windus_ultra_seed${seed}"
  local log_file="$LOG_ROOT/windus_ultra_seed${seed}.log"
  mkdir -p "$out_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_EXECUTABLE" run_pi_moe_full_training_evidence.py \
      --backbone ultra \
      --data wind_us.csv \
      --target "power (MW)" \
      --dataset-name Wind_US \
      --output-tag "$output_tag" \
      --out-dir "$out_dir" \
      --seed "$seed" \
      --base-config checkpoints_ts/exp_wind_us/model_config.json \
      --base-checkpoint checkpoints_ts/exp_wind_us/best_model.pth \
      --epochs "$GUARDRAIL_EPOCHS" \
      --batch-size "$BATCH_SIZE"

    "$PYTHON_EXECUTABLE" run_clip_only_guardrail_control.py \
      --backbone ultra \
      --data wind_us.csv \
      --target "power (MW)" \
      --dataset-name Wind_US \
      --output-tag "$output_tag" \
      --out-dir "$out_dir" \
      --seed "$seed" \
      --base-config checkpoints_ts/exp_wind_us/model_config.json \
      --base-checkpoint checkpoints_ts/exp_wind_us/best_model.pth
  ) >"$log_file" 2>&1 &
  local pid="$!"
  PIDS+=("$pid")
  NAMES+=("windus_ultra_seed${seed}")
  if [[ "$seed" == "42" ]]; then
    ULTRA42_PID="$pid"
  fi
}

echo "[suite] launching guardrail jobs"
spawn_windcn_itransformer 1 42
spawn_windcn_itransformer 2 123
spawn_windcn_itransformer 3 2025

spawn_windus_itransformer 4 42
spawn_windus_itransformer 5 123
spawn_windus_itransformer 6 2025

spawn_windus_ultra 7 42
spawn_windus_ultra 8 123
spawn_windus_ultra 9 2025

if [[ -z "$ULTRA42_PID" ]]; then
  echo "[suite] internal error: missing Ultra-LSNT seed42 PID" >&2
  exit 1
fi

echo "[suite] waiting for windus_ultra_seed42 before safety audit"
wait "$ULTRA42_PID"

echo "[suite] running safety deployment on GPU0"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_EXECUTABLE" run_safety_deployment.py \
  --mode paper \
  --data wind_us.csv \
  --target "power (MW)" \
  --dataset-name Wind_US \
  --output-tag windus_ultra_seed42 \
  --checkpoint checkpoints_ts/exp_wind_us/best_model.pth \
  --fg-checkpoint "$RESULT_ROOT/windus_ultra/seed_42/pi_moe_full_checkpoint_windus_ultra_seed42.pth" \
  --outdir "$DEPLOY_ROOT" \
  --batch-size 32 \
  --base-config checkpoints_ts/exp_wind_us/model_config.json \
  >"$LOG_ROOT/windus_ultra_seed42_safety.log" 2>&1

echo "[suite] waiting for remaining jobs"
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  name="${NAMES[$idx]}"
  if [[ "$pid" == "$ULTRA42_PID" ]]; then
    continue
  fi
  wait "$pid"
  echo "[suite] completed $name"
done

echo "[suite] aggregate all outputs"
"$PYTHON_EXECUTABLE" aggregate_missing_ae_results.py \
  --results-root "$RESULT_ROOT" \
  --deployment-root "$DEPLOY_ROOT" \
  --out-dir "$RESULT_ROOT"

echo "[suite] completed"
