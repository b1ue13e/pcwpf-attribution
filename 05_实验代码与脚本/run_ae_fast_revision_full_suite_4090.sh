#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ae_proxy_sensitivity}"
SUMMARY_ROOT="${SUMMARY_ROOT:-results/ae_revision_summary}"
DISPATCH_PARAM_ROOT="${DISPATCH_PARAM_ROOT:-results/ae_dispatch_param_sensitivity}"
LOG_ROOT="${LOG_ROOT:-logs_ae_fast_revision}"
SMOKE_GPU="${SMOKE_GPU:-0}"

mkdir -p "$RESULTS_ROOT" "$SUMMARY_ROOT" "$DISPATCH_PARAM_ROOT" "$LOG_ROOT"

echo "[suite] quick smoke first"
PYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
RESULTS_ROOT="$RESULTS_ROOT/smoke" \
SUMMARY_ROOT="$SUMMARY_ROOT/smoke" \
DISPATCH_PARAM_ROOT="$DISPATCH_PARAM_ROOT/smoke" \
SMOKE_GPU="$SMOKE_GPU" \
bash "$ROOT_DIR/run_ae_fast_revision_quick_smoke.sh"

CASES=(baseline_fixed sv_low sv_high delta_low delta_high omega_low omega_high)
PREP_GPUS=(0 1 2 3 4 5 6)

PIDS=()
NAMES=()

spawn_prep_case() {
  local gpu="$1"
  local case_id="$2"
  local log_file="$LOG_ROOT/prep_${case_id}.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_EXECUTABLE" run_ae_proxy_sensitivity_case.py \
      --case-id "$case_id" \
      --phase prep \
      --results-root "$RESULTS_ROOT" \
      --summary-root "$SUMMARY_ROOT"
  ) >"$log_file" 2>&1 &
  PIDS+=("$!")
  NAMES+=("prep_${case_id}")
}

echo "[suite] launching proxy prep jobs"
for idx in "${!CASES[@]}"; do
  spawn_prep_case "${PREP_GPUS[$idx]}" "${CASES[$idx]}"
done

for idx in "${!PIDS[@]}"; do
  wait "${PIDS[$idx]}"
  echo "[suite] completed ${NAMES[$idx]}"
done

run_dispatch_queue() {
  local gpu="$1"
  shift
  local queue=("$@")
  local log_file="$LOG_ROOT/dispatch_queue_gpu${gpu}.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    for case_id in "${queue[@]}"; do
      "$PYTHON_EXECUTABLE" run_ae_proxy_sensitivity_case.py \
        --case-id "$case_id" \
        --phase dispatch \
        --results-root "$RESULTS_ROOT" \
        --summary-root "$SUMMARY_ROOT"
    done
  ) >"$log_file" 2>&1 &
  PIDS+=("$!")
  NAMES+=("dispatch_queue_gpu${gpu}")
}

PIDS=()
NAMES=()

echo "[suite] launching high-risk dispatch readouts"
run_dispatch_queue 7 baseline_fixed sv_high delta_high omega_high
run_dispatch_queue 8 sv_low delta_low omega_low

(
  export CUDA_VISIBLE_DEVICES=9
  "$PYTHON_EXECUTABLE" run_ae_dispatch_param_sensitivity.py \
    --out-dir "$DISPATCH_PARAM_ROOT"
) >"$LOG_ROOT/dispatch_param_sensitivity.log" 2>&1 &
PIDS+=("$!")
NAMES+=("dispatch_param_sensitivity")

for idx in "${!PIDS[@]}"; do
  wait "${PIDS[$idx]}"
  echo "[suite] completed ${NAMES[$idx]}"
done

echo "[suite] aggregating proxy sensitivity summary"
"$PYTHON_EXECUTABLE" aggregate_ae_proxy_sensitivity_results.py \
  --results-root "$RESULTS_ROOT" \
  --summary-root "$SUMMARY_ROOT"

echo "[suite] done"
