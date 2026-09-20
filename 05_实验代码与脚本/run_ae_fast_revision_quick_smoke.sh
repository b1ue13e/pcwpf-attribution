#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ae_proxy_sensitivity}"
SUMMARY_ROOT="${SUMMARY_ROOT:-results/ae_revision_summary}"
DISPATCH_PARAM_ROOT="${DISPATCH_PARAM_ROOT:-results/ae_dispatch_param_sensitivity/smoke}"
SMOKE_GPU="${SMOKE_GPU:-0}"

mkdir -p "$RESULTS_ROOT" "$SUMMARY_ROOT" "$DISPATCH_PARAM_ROOT"

echo "[smoke] baseline proxy case"
CUDA_VISIBLE_DEVICES="$SMOKE_GPU" \
"$PYTHON_EXECUTABLE" run_ae_proxy_sensitivity_case.py \
  --case-id baseline_fixed \
  --phase all \
  --results-root "$RESULTS_ROOT" \
  --summary-root "$SUMMARY_ROOT" \
  --max-train-batches 1 \
  --max-eval-batches 1 \
  --dispatch-days 3 \
  --force-train

echo "[smoke] dispatch parameter sensitivity"
"$PYTHON_EXECUTABLE" run_ae_dispatch_param_sensitivity.py \
  --days 4 \
  --calibration-days 2 \
  --reserve-cost-grid 0.2 \
  --storage-capacity-grid 50000 \
  --out-dir "$DISPATCH_PARAM_ROOT"

echo "[smoke] aggregate proxy sensitivity"
"$PYTHON_EXECUTABLE" aggregate_ae_proxy_sensitivity_results.py \
  --results-root "$RESULTS_ROOT" \
  --summary-root "$SUMMARY_ROOT"

echo "[smoke] done"
