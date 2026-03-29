#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PY_BIN="${PY_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"

PER_CASE_CSV="${PER_CASE_CSV:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/vga/taxonomy/per_case_compare.csv}"
BASE_FEATURES_CSV="${BASE_FEATURES_CSV:-$CAL_ROOT/experiments/pope_feature_screen_v1_full9000/features_unified_table.csv}"
PER_HEAD_CSV="${PER_HEAD_CSV:-$CAL_ROOT/experiments/pope_full_9000/full9000_feature_extract/traces_baseline_9000/per_head_yes_trace.csv}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/pope_full_9000/vga_new_frg_hard_veto_9000}"

LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
TOP_RATIO="${TOP_RATIO:-0.2}"

CALIB_RATIO="${CALIB_RATIO:-0.3}"
SEED="${SEED:-42}"
LAMBDA_D1="${LAMBDA_D1:-1.0}"
MAX_D1_WRONG_RATE="${MAX_D1_WRONG_RATE:-0.35}"
Q_GRID="${Q_GRID:-0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95}"
FALLBACK_ACTION="${FALLBACK_ACTION:-vga}"

FRG_CSV="$OUT_ROOT/new_frg_features.csv"
MERGED_CSV="$OUT_ROOT/features_with_new_frg.csv"
FRG_ONLY_DIR="$OUT_ROOT/frg_only"
FRG_PLUS_GMI_DIR="$OUT_ROOT/frg_plus_gmi"

mkdir -p "$OUT_ROOT"
cd "$CAL_ROOT"

"$PY_BIN" scripts/build_shared_probe_frg.py \
  --per_head_csv "$PER_HEAD_CSV" \
  --out_csv "$FRG_CSV" \
  --late_start "$LATE_START" \
  --late_end "$LATE_END" \
  --top_ratio "$TOP_RATIO"

"$PY_BIN" - <<PY
import pandas as pd
base = pd.read_csv("$BASE_FEATURES_CSV")
new = pd.read_csv("$FRG_CSV")
merged = base.merge(new, on="id", how="left")
merged.to_csv("$MERGED_CSV", index=False)
print("[saved]", "$MERGED_CSV")
PY

"$PY_BIN" scripts/run_vga_hard_veto_controller.py \
  --per_case_csv "$PER_CASE_CSV" \
  --features_csv "$MERGED_CSV" \
  --out_dir "$FRG_ONLY_DIR" \
  --c_col "new_frg_shared_topk_gap" \
  --e_col "guidance_mismatch_score" \
  --use_c 1 \
  --use_e 0 \
  --calib_ratio "$CALIB_RATIO" \
  --seed "$SEED" \
  --lambda_d1 "$LAMBDA_D1" \
  --max_d1_wrong_rate "$MAX_D1_WRONG_RATE" \
  --q_grid "$Q_GRID" \
  --fallback_when_missing_feature "$FALLBACK_ACTION"

"$PY_BIN" scripts/run_vga_hard_veto_controller.py \
  --per_case_csv "$PER_CASE_CSV" \
  --features_csv "$MERGED_CSV" \
  --out_dir "$FRG_PLUS_GMI_DIR" \
  --c_col "new_frg_shared_topk_gap" \
  --e_col "guidance_mismatch_score" \
  --use_c 1 \
  --use_e 1 \
  --calib_ratio "$CALIB_RATIO" \
  --seed "$SEED" \
  --lambda_d1 "$LAMBDA_D1" \
  --max_d1_wrong_rate "$MAX_D1_WRONG_RATE" \
  --q_grid "$Q_GRID" \
  --fallback_when_missing_feature "$FALLBACK_ACTION"

echo "[done] $OUT_ROOT"
