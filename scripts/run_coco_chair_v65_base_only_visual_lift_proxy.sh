#!/usr/bin/env bash
set -euo pipefail

# No-GT/no-CHAIR feature probe:
# baseline-only caption content visual lift =
#   log p(token | real image, baseline prefix)
# - log p(token | ablated image, baseline prefix)
#
# Labels are only used for offline AUC diagnostics.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v65_base_only_visual_lift_proxy}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-50}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_ci_unique_noworse}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
ABLATION_MODE="${ABLATION_MODE:-blur}"
BLUR_RADIUS="${BLUR_RADIUS:-8.0}"

mkdir -p "$OUT_ROOT/features"

Q_FILE="$SOURCE_OUT/splits/${SPLIT}_caption_q_limited500.jsonl"
BASE_PRED="$SOURCE_OUT/$SPLIT/pred_baseline_caption.jsonl"
INT_PRED="$SOURCE_OUT/$SPLIT/pred_origin_entropy_simg_caption.jsonl"
ORACLE_DIR="$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg"
ORACLE_ROWS="$ORACLE_DIR/unique_safe_oracle_rows.csv"

if [[ "$LIMIT" != "0" && "$LIMIT" != "500" ]]; then
  Q_LIMITED="$OUT_ROOT/splits/${SPLIT}_caption_q_limited${LIMIT}.jsonl"
  mkdir -p "$(dirname "$Q_LIMITED")"
  if [[ "$REUSE_IF_EXISTS" != "true" || ! -f "$Q_LIMITED" ]]; then
    head -n "$LIMIT" "$Q_FILE" > "$Q_LIMITED"
  fi
  Q_USE="$Q_LIMITED"
else
  Q_USE="$Q_FILE"
fi

echo "[settings] out=$OUT_ROOT source=$SOURCE_OUT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] ablation_mode=$ABLATION_MODE blur_radius=$BLUR_RADIUS target=$TARGET_COL"

(
  cd "$CAL_ROOT"
  PYTHONDONTWRITEBYTECODE=1 "$CAL_PYTHON_BIN" scripts/extract_generative_base_only_visual_lift_features.py \
    --question_file "$Q_USE" \
    --image_folder "$IMAGE_FOLDER" \
    --baseline_pred_jsonl "$BASE_PRED" \
    --intervention_pred_jsonl "$INT_PRED" \
    --oracle_rows_csv "$ORACLE_ROWS" \
    --target_col "$TARGET_COL" \
    --out_csv "$OUT_ROOT/features/${SPLIT}_base_only_visual_lift_features.csv" \
    --out_feature_metrics_csv "$OUT_ROOT/features/${SPLIT}_base_only_visual_lift_feature_metrics.csv" \
    --out_summary_json "$OUT_ROOT/features/${SPLIT}_base_only_visual_lift_summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --pred_text_key auto \
    --reuse_if_exists "$REUSE_IF_EXISTS" \
    --ablation_mode "$ABLATION_MODE" \
    --blur_radius "$BLUR_RADIUS"
)

echo "[done] $OUT_ROOT"
