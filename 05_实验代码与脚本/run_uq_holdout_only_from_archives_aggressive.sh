#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
OUT_DIR="${OUT_DIR:-results/ae_operation_uq_holdout_seed42_aggressive}"
PREFIX="${PREFIX:-dispatch_operation_seed42_uq_holdout_aggressive}"

mkdir -p "$OUT_DIR"

"$PYTHON_EXECUTABLE" run_dispatch_closure_mapping_decision_4090.py \
  --days 120 \
  --mapping-methods risk_conditional \
  --risk-subset-mode shared \
  --model-manifest seed_42_main_model_manifest.json \
  --out-dir "$OUT_DIR" \
  --output-prefix "$PREFIX" \
  --decision-mode uq_holdout \
  --uq-alpha 0.10 \
  --uq-regime-mode shared_risk \
  --uq-margin-mode rolling_bias_scaled \
  --uq-score-floor-quantile 0.10 \
  --uq-score-floor-min-mw 2 \
  --uq-horizon-smooth-window 5 \
  --uq-max-relative-margin 0.45 \
  --uq-rolling-window-days 21 \
  --uq-min-history-samples 10 \
  --uq-hour-band-radius 2 \
  --uq-bias-quantile 0.70 \
  --uq-bias-shrink 1.10 \
  --holdout-start-day 80

"$PYTHON_EXECUTABLE" build_decision_aware_dispatch_evidence.py \
  --mode operation_uq \
  --dispatch-agg-csv "$OUT_DIR/${PREFIX}_aggregate.csv" \
  --dispatch-meta-json "$OUT_DIR/${PREFIX}.meta.json" \
  --focus-mapping risk_conditional \
  --focus-subset all_holdout \
  --out-summary-csv "$OUT_DIR/operation_uq_holdout_summary_seed42.csv" \
  --out-figure-prefix "$OUT_DIR/fig_operation_uq_holdout_seed42" \
  --out-meta "$OUT_DIR/operation_uq_holdout_summary_seed42.meta.json" \
  --out-latex-table "$OUT_DIR/operation_uq_holdout_summary_seed42_table.tex"

echo "[done] aggressive rolling_bias_scaled UQ holdout outputs written to $OUT_DIR"
