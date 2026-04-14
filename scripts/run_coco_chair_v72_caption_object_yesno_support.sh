#!/usr/bin/env bash
set -euo pipefail

# Verify caption-extracted object-list deltas with image-grounded yes/no replay.
#
# Intended default input is v70:
#   pred_base_caption_objects.jsonl
#   pred_int_caption_objects.jsonl
#
# This tests whether v70's improved candidate extraction becomes a deployable
# fallback-safe/gain router once support and baseline-risk signals are attached.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PY_BIN="${PY_BIN:-python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
export PYTHONDONTWRITEBYTECODE=1

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
CANDIDATE_OUT="${CANDIDATE_OUT:-$CAL_ROOT/experiments/coco_chair_v70_caption_conditioned_object_extraction}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v72_caption_object_yesno_support}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-500}"
SOURCE_LIMIT="${SOURCE_LIMIT:-500}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
SCORE_MODE="${SCORE_MODE:-yesno}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_ci_unique_noworse}"
MAX_OBJECTS="${MAX_OBJECTS:-12}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

Q_FILE="${Q_FILE:-$SOURCE_OUT/splits/${SPLIT}_caption_q_limited${SOURCE_LIMIT}.jsonl}"
BASE_OBJ="${BASE_OBJ:-$CANDIDATE_OUT/$SPLIT/pred_base_caption_objects.jsonl}"
INT_OBJ="${INT_OBJ:-$CANDIDATE_OUT/$SPLIT/pred_int_caption_objects.jsonl}"
BASE_CHAIR="${BASE_CHAIR:-$SOURCE_OUT/$SPLIT/chair_baseline.json}"
INT_CHAIR="${INT_CHAIR:-$SOURCE_OUT/$SPLIT/chair_origin_entropy_simg.json}"
ORACLE_DIR="${ORACLE_DIR:-$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg}"
ORACLE_ROWS="${ORACLE_ROWS:-$ORACLE_DIR/unique_safe_oracle_rows.csv}"

mkdir -p "$OUT_ROOT/features"

echo "[settings] out=$OUT_ROOT source=$SOURCE_OUT candidates=$CANDIDATE_OUT split=$SPLIT limit=$LIMIT"
echo "[settings] q_file=$Q_FILE"
echo "[settings] base_obj=$BASE_OBJ"
echo "[settings] int_obj=$INT_OBJ"
echo "[settings] target=$TARGET_COL score_mode=$SCORE_MODE max_objects=$MAX_OBJECTS gpu=$GPU"

if [[ ! -f "$ORACLE_ROWS" ]]; then
  echo "[prep] build missing unique-safe oracle rows: $ORACLE_ROWS"
  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/analyze_chair_unique_safe_oracle.py \
      --baseline_chair_json "$BASE_CHAIR" \
      --intervention_chair_json "$INT_CHAIR" \
      --out_dir "$ORACLE_DIR" \
      --main_oracle_col "$TARGET_COL"
  )
fi

FEATURES="$OUT_ROOT/features/${SPLIT}_caption_object_delta_yesno_features_limit${LIMIT}_max${MAX_OBJECTS}.csv"
FEATURES_SUMMARY="$OUT_ROOT/features/${SPLIT}_caption_object_delta_yesno_features_limit${LIMIT}_max${MAX_OBJECTS}.summary.json"
JOINED="$OUT_ROOT/features/${SPLIT}_caption_object_yesno_joined_limit${LIMIT}_max${MAX_OBJECTS}.csv"
METRICS="$OUT_ROOT/features/${SPLIT}_caption_object_yesno_feature_metrics_limit${LIMIT}_max${MAX_OBJECTS}.csv"
SUMMARY="$OUT_ROOT/features/${SPLIT}_caption_object_yesno_summary_limit${LIMIT}_max${MAX_OBJECTS}.json"

echo "[1/2] extract yes/no support over caption-extracted object deltas"
(
  cd "$CAL_ROOT"
  PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_caption_object_list_delta_yesno_features.py \
    --question_file "$Q_FILE" \
    --image_folder "$IMAGE_FOLDER" \
    --baseline_object_pred_jsonl "$BASE_OBJ" \
    --intervention_object_pred_jsonl "$INT_OBJ" \
    --out_csv "$FEATURES" \
    --out_summary_json "$FEATURES_SUMMARY" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --max_objects "$MAX_OBJECTS" \
    --question_template "Is there a {object} in the image? Answer yes or no." \
    --score_mode "$SCORE_MODE" \
    --reuse_if_exists "$REUSE_IF_EXISTS"
)

echo "[2/2] evaluate yes/no features against fallback-safe/gain oracle"
(
  cd "$CAL_ROOT"
  PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/analyze_generative_yesno_support_proxy.py \
    --yesno_features_csv "$FEATURES" \
    --oracle_rows_csv "$ORACLE_ROWS" \
    --target_col "$TARGET_COL" \
    --out_csv "$JOINED" \
    --out_feature_metrics_csv "$METRICS" \
    --out_summary_json "$SUMMARY"
)

echo "[done] $OUT_ROOT"
