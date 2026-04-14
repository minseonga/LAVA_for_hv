#!/usr/bin/env bash
set -euo pipefail

# Conditional target/non-target analysis after a benefit candidate gate.
#
# This is CPU-only. It takes v72 yes/no joined features, gates to candidate
# subsets such as base_only_object_count >= 1, then asks which cost/net-benefit
# features separate fallback-safe/gain targets from non-targets inside the gate.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PY_BIN="${PY_BIN:-python}"
export PYTHONDONTWRITEBYTECODE=1

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
V72_OUT="${V72_OUT:-$CAL_ROOT/experiments/coco_chair_v72_caption_object_yesno_support_smoke100}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v73_conditional_cost_analysis}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-100}"
MAX_OBJECTS="${MAX_OBJECTS:-12}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_ci_unique_noworse}"

FEATURES_CSV="${FEATURES_CSV:-$V72_OUT/features/${SPLIT}_caption_object_yesno_joined_limit${LIMIT}_max${MAX_OBJECTS}.csv}"
ORACLE_ROWS="${ORACLE_ROWS:-$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg/unique_safe_oracle_rows.csv}"

MIN_SELECTED="${MIN_SELECTED:-3}"
MAX_SELECTED="${MAX_SELECTED:-80}"
MIN_VALID_FRAC="${MIN_VALID_FRAC:-0.8}"
QUANTILES="${QUANTILES:-0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30}"

mkdir -p "$OUT_ROOT"

echo "[settings] out=$OUT_ROOT"
echo "[settings] features=$FEATURES_CSV"
echo "[settings] oracle=$ORACLE_ROWS"
echo "[settings] target=$TARGET_COL limit=$LIMIT max_objects=$MAX_OBJECTS"

(
  cd "$CAL_ROOT"
  PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/analyze_conditional_fallback_cost_features.py \
    --features_csv "$FEATURES_CSV" \
    --oracle_rows_csv "$ORACLE_ROWS" \
    --target_col "$TARGET_COL" \
    --out_dir "$OUT_ROOT" \
    --min_selected "$MIN_SELECTED" \
    --max_selected "$MAX_SELECTED" \
    --min_valid_frac "$MIN_VALID_FRAC" \
    --quantiles "$QUANTILES"
)

echo "[done] $OUT_ROOT"
