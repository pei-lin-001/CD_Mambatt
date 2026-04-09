#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/shelterpl/cd_mambatt"
LOG_DIR="$ROOT/runs/logs"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/overnight_architecture_search_20260408.log"
MASTER_SUMMARY="$LOG_DIR/overnight_architecture_search_20260408.summary.txt"

run_case() {
  local name="$1"
  shift
  local out_dir="$ROOT/runs/$name"
  local run_log="$LOG_DIR/${name}.log"
  local diag_json="$LOG_DIR/${name}.conditioning.json"

  echo "============================================================" | tee -a "$MASTER_LOG"
  echo "[$(date '+%F %T')] START $name" | tee -a "$MASTER_LOG"
  echo "output_dir=$out_dir" | tee -a "$MASTER_LOG"

  conda run -n cd_mamba python -u train_cd_mambatt_v3.py \
    --task FD001_TO_FD003 \
    --mamba-block-mode dd_spd \
    --target-lr 1.5e-3 \
    --target-lr-scheduler cosine \
    --target-lr-min 1e-5 \
    --lambda-mmd 0.1 \
    --lambda-source-stage 1.0 \
    --lambda-pseudo 0.5 \
    --lambda-monotonic 0.05 \
    --pseudo-start-quantile 0.50 \
    --pseudo-end-quantile 0.90 \
    --inv-alignment-mode mmd \
    --domain-feature-tap frontend_mean \
    --lambda-inv-mmd 0.1 \
    --lambda-spec-domain 0.1 \
    --seeds 42,43,44 \
    --output-dir "$out_dir" \
    --resample-few-shot-per-seed \
    --source-val-all-windows \
    --target-val-all-windows \
    --num-workers 0 \
    --device cuda \
    "$@" 2>&1 | tee "$run_log"

  echo "[$(date '+%F %T')] FINISH $name" | tee -a "$MASTER_LOG"
  python - <<PY | tee -a "$MASTER_SUMMARY"
import json
from pathlib import Path
summary = json.loads(Path("$out_dir/FD001_TO_FD003/summary.json").read_text())["summary"]
print(json.dumps({
    "name": "$name",
    "frontend_adapter_mode": summary.get("frontend_adapter_mode"),
    "domain_conditioned_gate": summary.get("domain_conditioned_gate"),
    "mean_cd_test_rmse": summary["mean_cd_test_rmse"],
    "std_cd_test_rmse": summary["std_cd_test_rmse"],
    "best_cd_test_rmse": summary["best_cd_test_rmse"],
    "mean_direct_target_rmse": summary["mean_direct_target_rmse"],
}, ensure_ascii=False))
PY

  PYTHONPATH="$ROOT" conda run -n cd_mamba python scripts/analyze_conditioning_effect.py \
    --run-root "$out_dir" \
    --task FD001_TO_FD003 \
    --device cuda > "$diag_json"
  echo "conditioning_diag=$diag_json" | tee -a "$MASTER_LOG"
}

cd "$ROOT"

run_case "cd_mambatt_v3_frontaffine_20260408" \
  --frontend-adapter-mode target_affine

run_case "cd_mambatt_v3_frontresidual_20260408" \
  --frontend-adapter-mode target_residual

run_case "cd_mambatt_v3_frontresidual_dualstate_20260408" \
  --frontend-adapter-mode target_residual \
  --spd-scan-mode dual_state

echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$MASTER_LOG" "$MASTER_SUMMARY"
