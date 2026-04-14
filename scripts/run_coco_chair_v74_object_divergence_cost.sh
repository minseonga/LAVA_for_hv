#!/usr/bin/env bash
set -euo pipefail

# Object-divergence + caption cost analysis without image yes/no.
#
# Input:
#   normal baseline/intervention captions
#   v70 caption-conditioned object lists
#
# Output:
#   deployable pairwise object/caption shape features
#   conditional cost analysis inside benefit gates such as jaccard_gap_q70

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PY_BIN="${PY_BIN:-python}"
export PYTHONDONTWRITEBYTECODE=1

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
CANDIDATE_OUT="${CANDIDATE_OUT:-$CAL_ROOT/experiments/coco_chair_v70_caption_conditioned_object_extraction_smoke100}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v74_object_divergence_cost_smoke100}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-100}"
SOURCE_LIMIT="${SOURCE_LIMIT:-500}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_ci_unique_noworse}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

Q_FILE="${Q_FILE:-$SOURCE_OUT/splits/${SPLIT}_caption_q_limited${SOURCE_LIMIT}.jsonl}"
BASE_PRED="${BASE_PRED:-$SOURCE_OUT/$SPLIT/pred_baseline_caption.jsonl}"
INT_PRED="${INT_PRED:-$SOURCE_OUT/$SPLIT/pred_origin_entropy_simg_caption.jsonl}"
BASE_OBJ="${BASE_OBJ:-$CANDIDATE_OUT/$SPLIT/pred_base_caption_objects.jsonl}"
INT_OBJ="${INT_OBJ:-$CANDIDATE_OUT/$SPLIT/pred_int_caption_objects.jsonl}"
ORACLE_ROWS="${ORACLE_ROWS:-$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg/unique_safe_oracle_rows.csv}"

MIN_SELECTED="${MIN_SELECTED:-3}"
MAX_SELECTED="${MAX_SELECTED:-80}"
MIN_VALID_FRAC="${MIN_VALID_FRAC:-0.8}"
QUANTILES="${QUANTILES:-0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30}"

mkdir -p "$OUT_ROOT/features" "$OUT_ROOT/conditional_cost"

FEATURES="$OUT_ROOT/features/${SPLIT}_object_divergence_cost_features_limit${LIMIT}.csv"
FEATURES_SUMMARY="$OUT_ROOT/features/${SPLIT}_object_divergence_cost_features_limit${LIMIT}.summary.json"

echo "[settings] out=$OUT_ROOT source=$SOURCE_OUT candidates=$CANDIDATE_OUT split=$SPLIT limit=$LIMIT"
echo "[settings] q_file=$Q_FILE"
echo "[settings] base_pred=$BASE_PRED"
echo "[settings] int_pred=$INT_PRED"
echo "[settings] base_obj=$BASE_OBJ"
echo "[settings] int_obj=$INT_OBJ"
echo "[settings] target=$TARGET_COL"

echo "[1/2] extract object-divergence and caption cost features"
(
  cd "$CAL_ROOT"
  PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_caption_object_divergence_cost_features.py \
    --question_file "$Q_FILE" \
    --baseline_pred_jsonl "$BASE_PRED" \
    --intervention_pred_jsonl "$INT_PRED" \
    --baseline_object_pred_jsonl "$BASE_OBJ" \
    --intervention_object_pred_jsonl "$INT_OBJ" \
    --oracle_rows_csv "$ORACLE_ROWS" \
    --target_col "$TARGET_COL" \
    --out_csv "$FEATURES" \
    --out_summary_json "$FEATURES_SUMMARY" \
    --limit "$LIMIT" \
    --reuse_if_exists "$REUSE_IF_EXISTS"
)

echo "[2/2] conditional target/non-target cost analysis"
(
  cd "$CAL_ROOT"
  PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/analyze_conditional_fallback_cost_features.py \
    --features_csv "$FEATURES" \
    --oracle_rows_csv "$ORACLE_ROWS" \
    --target_col "$TARGET_COL" \
    --out_dir "$OUT_ROOT/conditional_cost" \
    --min_selected "$MIN_SELECTED" \
    --max_selected "$MAX_SELECTED" \
    --min_valid_frac "$MIN_VALID_FRAC" \
    --quantiles "$QUANTILES"
)

echo "[done] $OUT_ROOT"
