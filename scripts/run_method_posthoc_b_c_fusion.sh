#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"

SCORES_CSV="${SCORES_CSV:-}"
FEATURES_CSV="${FEATURES_CSV:-}"
OUT_ROOT="${OUT_ROOT:-}"

B_FEATURE_COLS="${B_FEATURE_COLS:-stage_b_score}"
C_FEATURE_COLS="${C_FEATURE_COLS:-cheap_lp_content_min,cheap_lp_content_tail_gap,cheap_lp_content_tail_z,cheap_lp_content_q10,cheap_lp_content_min_len_corr,cheap_target_gap_content_min,cheap_lp_content_std,cheap_entropy_content_mean,cheap_margin_content_mean,cheap_target_gap_content_std,cheap_conflict_lp_minus_entropy}"
MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
TOP_K_C="${TOP_K_C:-3}"
WEIGHT_GRID="${WEIGHT_GRID:-0.25,0.5,0.75,1.0,1.5,2.0,3.0}"
TAU_OBJECTIVE="${TAU_OBJECTIVE:-final_acc}"
MIN_BASELINE_RATE="${MIN_BASELINE_RATE:-0.0}"
MAX_BASELINE_RATE="${MAX_BASELINE_RATE:-1.0}"
MIN_SELECTED_COUNT="${MIN_SELECTED_COUNT:-0}"

if [[ -z "$SCORES_CSV" || ! -f "$SCORES_CSV" ]]; then
  echo "[error] missing SCORES_CSV: $SCORES_CSV" >&2
  exit 1
fi
if [[ -z "$FEATURES_CSV" || ! -f "$FEATURES_CSV" ]]; then
  echo "[error] missing FEATURES_CSV: $FEATURES_CSV" >&2
  exit 1
fi
if [[ -z "$OUT_ROOT" ]]; then
  echo "[error] OUT_ROOT is required" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

cd "$CAL_ROOT"
"$CAL_PYTHON_BIN" scripts/build_posthoc_b_c_fusion_controller.py \
  --scores_csv "$SCORES_CSV" \
  --features_csv "$FEATURES_CSV" \
  --out_dir "$OUT_ROOT" \
  --b_feature_cols "$B_FEATURE_COLS" \
  --c_feature_cols "$C_FEATURE_COLS" \
  --min_feature_auroc "$MIN_FEATURE_AUROC" \
  --top_k_c "$TOP_K_C" \
  --weight_grid "$WEIGHT_GRID" \
  --tau_objective "$TAU_OBJECTIVE" \
  --min_baseline_rate "$MIN_BASELINE_RATE" \
  --max_baseline_rate "$MAX_BASELINE_RATE" \
  --min_selected_count "$MIN_SELECTED_COUNT"

echo "[done] $OUT_ROOT/summary.json"
