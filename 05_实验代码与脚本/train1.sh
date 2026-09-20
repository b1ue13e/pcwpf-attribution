#!/usr/bin/env bash
set -Eeuo pipefail

# train1.sh
# Single-command full pipeline for cloud server.
# Run:
#   chmod +x train1.sh
#   ./train1.sh
#
# Background run:
#   nohup ./train1.sh > logs_train1/nohup_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Optional env vars (no arg needed):
#   PYTHON_BIN=python3
#   LOG_DIR=logs_train1
#   MD_BASELINE2=timemixer|patchtst
#   MD_BASELINE_EPOCHS=10
#   MD_BASELINE_BATCH=512            # 适配RTX 4090大显存
#   USE_AMP=1                        # 启用混合精度训练（RTX 4090推荐）
#   DRY_RUN=0|1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${LOG_DIR:-logs_train1}"
DRY_RUN="${DRY_RUN:-0}"

MD_BASELINE2="${MD_BASELINE2:-timemixer}"
MD_BASELINE_EPOCHS="${MD_BASELINE_EPOCHS:-10}"
MD_BASELINE_BATCH="${MD_BASELINE_BATCH:-512}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "[FATAL] python3/python not found."
    exit 127
  fi
fi

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/train1_${TS}.log"

run_cmd() {
  local name="$1"
  shift
  local cmd=("$@")
  echo "" | tee -a "$MASTER_LOG"
  echo "[$(date '+%F %T')] [STEP] $name" | tee -a "$MASTER_LOG"
  echo "[$(date '+%F %T')] [CMD ] ${cmd[*]}" | tee -a "$MASTER_LOG"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$(date '+%F %T')] [SKIP] DRY_RUN=1" | tee -a "$MASTER_LOG"
    return 0
  fi
  "${cmd[@]}" 2>&1 | tee -a "$MASTER_LOG"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "[$(date '+%F %T')] [FAIL] $name (exit=$rc)" | tee -a "$MASTER_LOG"
    exit $rc
  fi
  echo "[$(date '+%F %T')] [ OK ] $name" | tee -a "$MASTER_LOG"
}

echo "================ train1 full pipeline ================" | tee -a "$MASTER_LOG"
echo "Start time: $(date '+%F %T')" | tee -a "$MASTER_LOG"
echo "Root dir  : $ROOT_DIR" | tee -a "$MASTER_LOG"
echo "Python    : $PYTHON_BIN" | tee -a "$MASTER_LOG"
echo "Log dir   : $LOG_DIR" | tee -a "$MASTER_LOG"
echo "DRY_RUN   : $DRY_RUN" | tee -a "$MASTER_LOG"
echo "Baseline2 : $MD_BASELINE2" | tee -a "$MASTER_LOG"
echo "======================================================" | tee -a "$MASTER_LOG"

# 1) Multi-domain Ultra-LSNT summary
run_cmd "multi_domain_ultra" \
  "$PYTHON_BIN" evaluate_multi_domain.py

# 2) Multi-domain baselines (>=2 baselines per dataset)
run_cmd "multi_domain_baselines" \
  "$PYTHON_BIN" run_multi_domain_baselines.py \
  --model2 "$MD_BASELINE2" \
  --epochs "$MD_BASELINE_EPOCHS" \
  --batch-size "$MD_BASELINE_BATCH"

# 3) LightGBM baseline (energy domain must-have)
run_cmd "lightgbm_baseline" \
  "$PYTHON_BIN" run_gbdt.py

# 4) PatchTST specific comparison (ICLR 2023 SOTA)
run_cmd "patchtst_comparison" \
  "$PYTHON_BIN" run_dm_patchtst_fixed.py

# 5) Ultra-LSNT-Lite optimized performance
run_cmd "ultra_lsnt_lite_optimized" \
  "$PYTHON_BIN" test_ultra_lsnt_lite_optimized.py

# 6) Air Quality kernel ablation (k=3/7/15 included in output csv)
run_cmd "kernel_ablation" \
  "$PYTHON_BIN" run_universal_ablation.py

# 7) noise_20 comparison artifacts
run_cmd "noise20" \
  "$PYTHON_BIN" test_dlinear_noise_robustness.py

# 8) DM extended (PatchTST + TimeMixer/iTransformer + DLinear)
run_cmd "dm_extended" \
  "$PYTHON_BIN" run_background_suite.py \
  --tasks dm_extended \
  --log-dir "$LOG_DIR"

# 9) Lite suite comparison
run_cmd "lite_suite" \
  "$PYTHON_BIN" run_background_suite.py \
  --tasks lite_suite \
  --log-dir "$LOG_DIR"

# 10) Paper digest (collects the 5 requested items)
run_cmd "paper_digest" \
  "$PYTHON_BIN" run_background_suite.py \
  --tasks paper_digest \
  --log-dir "$LOG_DIR"

echo "" | tee -a "$MASTER_LOG"
echo "End time : $(date '+%F %T')" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "Digest md: $(ls -1t results/paper_supplement_digest_*.md 2>/dev/null | head -1)" | tee -a "$MASTER_LOG"
echo "Digest json: $(ls -1t results/paper_supplement_digest_*.json 2>/dev/null | head -1)" | tee -a "$MASTER_LOG"
