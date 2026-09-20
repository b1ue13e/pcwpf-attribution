#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
SMOKE_GPU="${SMOKE_GPU:-0}"
OUT_ROOT="${OUT_ROOT:-results/smoke_missing_ae_guardrails}"
DEPLOY_ROOT="${DEPLOY_ROOT:-results/deployment_evidence_missing_ae_smoke}"

mkdir -p "$OUT_ROOT" "$DEPLOY_ROOT"

echo "[smoke] checking CLI help"
"$PYTHON_EXECUTABLE" run_pi_moe_full_training_evidence.py --help >/dev/null
"$PYTHON_EXECUTABLE" run_clip_only_guardrail_control.py --help >/dev/null
"$PYTHON_EXECUTABLE" run_safety_deployment.py --help >/dev/null
"$PYTHON_EXECUTABLE" aggregate_missing_ae_results.py --help >/dev/null

echo "[smoke] Wind_CN + iTransformer"
CUDA_VISIBLE_DEVICES="$SMOKE_GPU" "$PYTHON_EXECUTABLE" run_pi_moe_full_training_evidence.py \
  --backbone itransformer \
  --data wind_final.csv \
  --target power \
  --dataset-name Wind_CN \
  --output-tag windcn_itransformer_smoke \
  --out-dir "$OUT_ROOT/windcn_itransformer/seed_42" \
  --seed 42 \
  --base-checkpoint checkpoints_ts/iTransformer_best.pth \
  --epochs 1 \
  --base-train-epochs 1 \
  --batch-size 64 \
  --max-train-batches 2 \
  --max-eval-batches 2

CUDA_VISIBLE_DEVICES="$SMOKE_GPU" "$PYTHON_EXECUTABLE" run_clip_only_guardrail_control.py \
  --backbone itransformer \
  --data wind_final.csv \
  --target power \
  --dataset-name Wind_CN \
  --output-tag windcn_itransformer_smoke \
  --out-dir "$OUT_ROOT/windcn_itransformer/seed_42" \
  --seed 42 \
  --base-checkpoint checkpoints_ts/iTransformer_best.pth \
  --max-eval-batches 2

echo "[smoke] Wind_US + Ultra-LSNT"
CUDA_VISIBLE_DEVICES="$SMOKE_GPU" "$PYTHON_EXECUTABLE" run_pi_moe_full_training_evidence.py \
  --backbone ultra \
  --data wind_us.csv \
  --target "power (MW)" \
  --dataset-name Wind_US \
  --output-tag windus_ultra_smoke \
  --out-dir "$OUT_ROOT/windus_ultra/seed_42" \
  --seed 42 \
  --base-config checkpoints_ts/exp_wind_us/model_config.json \
  --base-checkpoint checkpoints_ts/exp_wind_us/best_model.pth \
  --epochs 1 \
  --batch-size 64 \
  --max-train-batches 2 \
  --max-eval-batches 2

CUDA_VISIBLE_DEVICES="$SMOKE_GPU" "$PYTHON_EXECUTABLE" run_safety_deployment.py \
  --mode smoke \
  --data wind_us.csv \
  --target "power (MW)" \
  --dataset-name Wind_US \
  --output-tag windus_ultra_smoke \
  --checkpoint checkpoints_ts/exp_wind_us/best_model.pth \
  --fg-checkpoint "$OUT_ROOT/windus_ultra/seed_42/pi_moe_full_checkpoint_windus_ultra_smoke.pth" \
  --outdir "$DEPLOY_ROOT" \
  --warmup 2 \
  --measure 5 \
  --batch-size 16 \
  --max-eval-batches 2

echo "[smoke] aggregate"
"$PYTHON_EXECUTABLE" aggregate_missing_ae_results.py \
  --results-root "$OUT_ROOT" \
  --deployment-root "$DEPLOY_ROOT" \
  --out-dir "$OUT_ROOT"

echo "[smoke] completed"
