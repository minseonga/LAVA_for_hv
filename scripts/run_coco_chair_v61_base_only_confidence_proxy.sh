#!/usr/bin/env bash
set -euo pipefail

# Baseline-only content confidence proxy.
#
# Reuses v59 baseline/intervention captions and unique-Ci-safe oracle labels.
# It teacher-forces the baseline caption once, then measures logprob/margin/
# entropy only on baseline caption content that is absent from the intervention
# caption.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v61_base_only_confidence_proxy}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-500}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_ci_unique_noworse}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

mkdir -p "$OUT_ROOT/features"

reuse_file() {
  local path="$1"
  [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]
}

Q_FILE="$SOURCE_OUT/splits/${SPLIT}_caption_q_limited${LIMIT}.jsonl"
BASE_PRED="$SOURCE_OUT/$SPLIT/pred_baseline_caption.jsonl"
INT_PRED="$SOURCE_OUT/$SPLIT/pred_origin_entropy_simg_caption.jsonl"
ORACLE_DIR="$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg"
ORACLE_ROWS="$ORACLE_DIR/unique_safe_oracle_rows.csv"
BASE_CHAIR="$SOURCE_OUT/$SPLIT/chair_baseline.json"
INT_CHAIR="$SOURCE_OUT/$SPLIT/chair_origin_entropy_simg.json"
BASE_TRACE="$OUT_ROOT/features/${SPLIT}_baseline_teacher_forced_trace.csv"
BASE_TRACE_SUMMARY="$OUT_ROOT/features/${SPLIT}_baseline_teacher_forced_trace.summary.json"

echo "[settings] out=$OUT_ROOT source=$SOURCE_OUT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] target=$TARGET_COL"

if [[ ! -f "$ORACLE_ROWS" ]]; then
  echo "[prep] build missing unique-safe oracle rows: $ORACLE_ROWS"
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/analyze_chair_unique_safe_oracle.py \
      --baseline_chair_json "$BASE_CHAIR" \
      --intervention_chair_json "$INT_CHAIR" \
      --out_dir "$ORACLE_DIR" \
      --main_oracle_col "$TARGET_COL"
  )
fi

if ! reuse_file "$BASE_TRACE"; then
  echo "[1/2] teacher-force baseline captions"
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/extract_vga_generative_mention_features.py \
      --question_file "$Q_FILE" \
      --image_folder "$IMAGE_FOLDER" \
      --baseline_pred_jsonl "$BASE_PRED" \
      --out_csv "$BASE_TRACE" \
      --out_summary_json "$BASE_TRACE_SUMMARY" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --conv_mode "$CONV_MODE" \
      --device "$DEVICE" \
      --limit "$LIMIT" \
      --pred_text_key auto \
      --reuse_if_exists "$REUSE_IF_EXISTS" \
      --max_mentions 16 \
      --visual_trace false
  )
else
  echo "[reuse] $BASE_TRACE"
fi

echo "[2/2] analyze baseline-only confidence features"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/analyze_generative_base_only_confidence_proxy.py \
    --baseline_pred_jsonl "$BASE_PRED" \
    --intervention_pred_jsonl "$INT_PRED" \
    --baseline_trace_csv "$BASE_TRACE" \
    --oracle_rows_csv "$ORACLE_ROWS" \
    --target_col "$TARGET_COL" \
    --out_csv "$OUT_ROOT/features/${SPLIT}_base_only_confidence_features.csv" \
    --out_feature_metrics_csv "$OUT_ROOT/features/${SPLIT}_base_only_confidence_feature_metrics.csv" \
    --out_summary_json "$OUT_ROOT/features/${SPLIT}_base_only_confidence_summary.json"
)

echo "[done] $OUT_ROOT"
