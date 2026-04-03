#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/shelterpl/miniconda3/envs/cd_mamba/bin/python"
PROJECT_ROOT="/home/shelterpl/cd_mambatt"
RUNNER="$PROJECT_ROOT/train_cd_mambatt_v2.py"

COMMON_ARGS=(
  --device cuda
  --seeds 42,43,44,45,46
  --source-epochs 50
  --target-epochs 20
  --target-shots 5
  --target-val-units 10
  --resample-few-shot-per-seed
  --lambda-mmd 0.1
  --lambda-source-stage 1.0
  --lambda-pseudo 0.5
  --pseudo-start-quantile 0.5
  --pseudo-end-quantile 0.9
  --lambda-monotonic 0.05
  --monotonic-margin 0.0
  --monotonic-pair-gap 1
  --monotonic-pair-stride 5
  --target-lr 5e-4
  --dim-feedforward 84
  --transformer-norm-mode pre
  --source-val-all-windows
  --target-val-all-windows
)

run_task() {
  local task_name="$1"
  local output_dir="$2"
  echo "============================================================"
  echo "[$(date '+%F %T')] START ${task_name}"
  echo "output_dir=${output_dir}"
  "$PYTHON_BIN" "$RUNNER" \
    --task "$task_name" \
    "${COMMON_ARGS[@]}" \
    --output-dir "$output_dir"
  echo "[$(date '+%F %T')] DONE ${task_name}"
}

run_task \
  "FD002_TO_FD004" \
  "$PROJECT_ROOT/runs/cd_mambatt_bestcfg_fd002_to_fd004_5seeds_fixcond"

run_task \
  "FD001_TO_FD004" \
  "$PROJECT_ROOT/runs/cd_mambatt_bestcfg_fd001_to_fd004_5seeds_fixcond"

