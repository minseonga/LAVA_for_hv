#!/usr/bin/env bash
set -euo pipefail

# Candidate-grounded yes/no support verifier for the true VGA-origin caption run.
#
# This runs two candidate sources against the same unique-safe fallback oracle:
#   1. chair_object: CHAIR/COCO generated-object deltas. This is benchmark-aware
#      and should be read as an upper-bound / diagnostic candidate source.
#   2. semantic_unit: caption-pair baseline-only semantic units. This avoids
#      CHAIR parser/GT at feature time, but its candidate extractor is noisier.
#
# Both use image-grounded yes/no replay:
#   Is there a {candidate} in the image? Answer yes or no.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PY_BIN="${PY_BIN:-python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
export PYTHONDONTWRITEBYTECODE=1

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v69_candidate_grounded_yesno}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-500}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
SCORE_MODE="${SCORE_MODE:-yesno}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_ci_unique_noworse}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

RUN_CHAIR_OBJECT="${RUN_CHAIR_OBJECT:-true}"
RUN_SEMANTIC_UNIT="${RUN_SEMANTIC_UNIT:-true}"
MAX_UNITS="${MAX_UNITS:-4}"
CANDIDATE_MODE="${CANDIDATE_MODE:-semantic_units}"
SPACY_MODEL="${SPACY_MODEL:-en_core_web_sm}"

Q_FILE="$SOURCE_OUT/splits/${SPLIT}_caption_q_limited500.jsonl"
BASE_PRED="${BASE_PRED:-$SOURCE_OUT/$SPLIT/pred_baseline_caption.jsonl}"
INT_PRED="$SOURCE_OUT/$SPLIT/pred_origin_entropy_simg_caption.jsonl"
BASE_CHAIR="$SOURCE_OUT/$SPLIT/chair_baseline.json"
INT_CHAIR="$SOURCE_OUT/$SPLIT/chair_origin_entropy_simg.json"
ORACLE_DIR="$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg"
ORACLE_ROWS="$ORACLE_DIR/unique_safe_oracle_rows.csv"

mkdir -p "$OUT_ROOT/features"

echo "[settings] out=$OUT_ROOT source=$SOURCE_OUT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] target=$TARGET_COL score_mode=$SCORE_MODE run_chair_object=$RUN_CHAIR_OBJECT run_semantic_unit=$RUN_SEMANTIC_UNIT"
echo "[settings] candidate_mode=$CANDIDATE_MODE max_units=$MAX_UNITS spacy_model=$SPACY_MODEL"

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

if [[ "$RUN_CHAIR_OBJECT" == "true" ]]; then
  echo "[1/2][chair_object] extract object-wise yes/no features"
  CHAIR_FEATURES="$OUT_ROOT/features/${SPLIT}_chair_object_delta_yesno_features_limit${LIMIT}.csv"
  CHAIR_FEATURES_SUMMARY="$OUT_ROOT/features/${SPLIT}_chair_object_delta_yesno_features_limit${LIMIT}.summary.json"
  CHAIR_JOINED="$OUT_ROOT/features/${SPLIT}_chair_object_yesno_joined_limit${LIMIT}.csv"
  CHAIR_METRICS="$OUT_ROOT/features/${SPLIT}_chair_object_yesno_feature_metrics_limit${LIMIT}.csv"
  CHAIR_SUMMARY="$OUT_ROOT/features/${SPLIT}_chair_object_yesno_summary_limit${LIMIT}.json"
  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_chair_object_delta_yesno_features.py \
      --question_file "$Q_FILE" \
      --image_folder "$IMAGE_FOLDER" \
      --baseline_chair_json "$BASE_CHAIR" \
      --intervention_chair_json "$INT_CHAIR" \
      --out_csv "$CHAIR_FEATURES" \
      --out_summary_json "$CHAIR_FEATURES_SUMMARY" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --conv_mode "$CONV_MODE" \
      --device "$DEVICE" \
      --limit "$LIMIT" \
      --score_mode "$SCORE_MODE" \
      --reuse_if_exists "$REUSE_IF_EXISTS"

    PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/analyze_generative_yesno_support_proxy.py \
      --yesno_features_csv "$CHAIR_FEATURES" \
      --oracle_rows_csv "$ORACLE_ROWS" \
      --target_col "$TARGET_COL" \
      --out_csv "$CHAIR_JOINED" \
      --out_feature_metrics_csv "$CHAIR_METRICS" \
      --out_summary_json "$CHAIR_SUMMARY"
  )
fi

if [[ "$RUN_SEMANTIC_UNIT" == "true" ]]; then
  echo "[2/2][semantic_unit] extract caption-pair candidate yes/no features"
  SEM_FEATURES="$OUT_ROOT/features/${SPLIT}_${CANDIDATE_MODE}_yesno_features_limit${LIMIT}_max${MAX_UNITS}.csv"
  SEM_FEATURES_SUMMARY="$OUT_ROOT/features/${SPLIT}_${CANDIDATE_MODE}_yesno_features_limit${LIMIT}_max${MAX_UNITS}.summary.json"
  SEM_JOINED="$OUT_ROOT/features/${SPLIT}_${CANDIDATE_MODE}_yesno_joined_limit${LIMIT}_max${MAX_UNITS}.csv"
  SEM_METRICS="$OUT_ROOT/features/${SPLIT}_${CANDIDATE_MODE}_yesno_feature_metrics_limit${LIMIT}_max${MAX_UNITS}.csv"
  SEM_SUMMARY="$OUT_ROOT/features/${SPLIT}_${CANDIDATE_MODE}_yesno_summary_limit${LIMIT}_max${MAX_UNITS}.json"
  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_generative_semantic_unit_yesno_features.py \
      --question_file "$Q_FILE" \
      --image_folder "$IMAGE_FOLDER" \
      --baseline_pred_jsonl "$BASE_PRED" \
      --intervention_pred_jsonl "$INT_PRED" \
      --out_csv "$SEM_FEATURES" \
      --out_summary_json "$SEM_FEATURES_SUMMARY" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --conv_mode "$CONV_MODE" \
      --device "$DEVICE" \
      --limit "$LIMIT" \
      --max_units "$MAX_UNITS" \
      --candidate_mode "$CANDIDATE_MODE" \
      --spacy_model "$SPACY_MODEL" \
      --question_template "Is there a {unit} in the image? Answer yes or no." \
      --score_mode "$SCORE_MODE" \
      --reuse_if_exists "$REUSE_IF_EXISTS"

    PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/analyze_generative_yesno_support_proxy.py \
      --yesno_features_csv "$SEM_FEATURES" \
      --oracle_rows_csv "$ORACLE_ROWS" \
      --target_col "$TARGET_COL" \
      --out_csv "$SEM_JOINED" \
      --out_feature_metrics_csv "$SEM_METRICS" \
      --out_summary_json "$SEM_SUMMARY"
  )
fi

echo "[done] $OUT_ROOT"
