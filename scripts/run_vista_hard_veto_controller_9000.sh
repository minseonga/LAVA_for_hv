#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PY_BIN="${PY_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"

PER_CASE_CSV="${PER_CASE_CSV:-$CAL_ROOT/experiments/pope_full_9000/vista_method_9000/taxonomy/per_case_compare.csv}"
FEATURES_CSV="${FEATURES_CSV:-$CAL_ROOT/experiments/pope_feature_screen_v1_full9000/features_unified_table.csv}"
OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/pope_full_9000/vista_hard_veto_controller_9000}"

C_COL="${C_COL:-faithful_minus_global_attn}"
E_COL="${E_COL:-guidance_mismatch_score}"
CALIB_RATIO="${CALIB_RATIO:-0.3}"
SEED="${SEED:-42}"
LAMBDA_D1="${LAMBDA_D1:-1.0}"
MAX_D1_WRONG_RATE="${MAX_D1_WRONG_RATE:-0.35}"
Q_GRID="${Q_GRID:-0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95}"
FALLBACK_ACTION="${FALLBACK_ACTION:-vga}"

mkdir -p "$OUT_DIR"
cd "$CAL_ROOT"

"$PY_BIN" scripts/run_vga_hard_veto_controller.py \
  --per_case_csv "$PER_CASE_CSV" \
  --features_csv "$FEATURES_CSV" \
  --out_dir "$OUT_DIR" \
  --c_col "$C_COL" \
  --e_col "$E_COL" \
  --calib_ratio "$CALIB_RATIO" \
  --seed "$SEED" \
  --lambda_d1 "$LAMBDA_D1" \
  --max_d1_wrong_rate "$MAX_D1_WRONG_RATE" \
  --q_grid "$Q_GRID" \
  --fallback_when_missing_feature "$FALLBACK_ACTION"

echo "[done] $OUT_DIR"
