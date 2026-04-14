#!/usr/bin/env bash
set -euo pipefail

# Caption-conditioned object extraction diagnostic.
#
# This tests the current bottleneck directly: can a prompt-based extractor recover
# baseline-only concrete object mentions better than generic semantic units/spaCy?
# It does not change the evaluated captions. The image is still passed through
# the LLaVA wrapper, but the prompt explicitly asks for objects mentioned in the
# provided caption only.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v70_caption_conditioned_object_extraction}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-500}"
SOURCE_LIMIT="${SOURCE_LIMIT:-500}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VGA_CONV_MODE="${VGA_CONV_MODE:-llava_v1}"
MAX_GEN_LEN="${MAX_GEN_LEN:-96}"
SEED="${SEED:-17}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_ci_unique_noworse}"

Q_SRC="$SOURCE_OUT/splits/${SPLIT}_caption_q_limited${SOURCE_LIMIT}.jsonl"
BASE_PRED="$SOURCE_OUT/$SPLIT/pred_baseline_caption.jsonl"
INT_PRED="$SOURCE_OUT/$SPLIT/pred_origin_entropy_simg_caption.jsonl"
ORACLE_ROWS="$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg/unique_safe_oracle_rows.csv"

Q_BASE="$OUT_ROOT/splits/${SPLIT}_base_caption_object_extract_q_limited${LIMIT}.jsonl"
Q_INT="$OUT_ROOT/splits/${SPLIT}_int_caption_object_extract_q_limited${LIMIT}.jsonl"
BASE_OBJ="$OUT_ROOT/$SPLIT/pred_base_caption_objects.jsonl"
INT_OBJ="$OUT_ROOT/$SPLIT/pred_int_caption_objects.jsonl"

mkdir -p "$OUT_ROOT/splits" "$OUT_ROOT/$SPLIT" "$OUT_ROOT/features"

reuse_file() {
  local path="$1"
  [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]
}

run_extractor() {
  local name="$1"
  local q_file="$2"
  local pred_file="$3"

  if reuse_file "$pred_file"; then
    echo "[reuse] $pred_file"
    return
  fi

  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" scripts/run_vga_origin_llava_caption_compat.py \
      --vga-root "$VGA_ROOT" \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$q_file" \
      --answers-file "$pred_file" \
      --conv-mode "$VGA_CONV_MODE" \
      --max_gen_len "$MAX_GEN_LEN" \
      --use_add false \
      --attn_coef 0.2 \
      --cd_alpha 0.02 \
      --start_layer 2 \
      --end_layer 15 \
      --head_balancing simg \
      --sampling false \
      --seed "$SEED"
  )
  echo "[saved][$name] $pred_file"
}

echo "[settings] out=$OUT_ROOT source=$SOURCE_OUT split=$SPLIT limit=$LIMIT source_limit=$SOURCE_LIMIT gpu=$GPU"
echo "[settings] target=$TARGET_COL max_gen_len=$MAX_GEN_LEN"

if ! reuse_file "$Q_BASE"; then
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/build_caption_conditioned_object_extraction_questions.py" \
    --question_file "$Q_SRC" \
    --pred_jsonl "$BASE_PRED" \
    --out_jsonl "$Q_BASE" \
    --out_summary_json "$OUT_ROOT/splits/${SPLIT}_base_caption_object_extract_q.summary.json" \
    --limit "$LIMIT" \
    --category "baseline_caption_object_extraction"
else
  echo "[reuse] $Q_BASE"
fi

if ! reuse_file "$Q_INT"; then
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/build_caption_conditioned_object_extraction_questions.py" \
    --question_file "$Q_SRC" \
    --pred_jsonl "$INT_PRED" \
    --out_jsonl "$Q_INT" \
    --out_summary_json "$OUT_ROOT/splits/${SPLIT}_int_caption_object_extract_q.summary.json" \
    --limit "$LIMIT" \
    --category "intervention_caption_object_extraction"
else
  echo "[reuse] $Q_INT"
fi

echo "[1/3] extract objects from baseline captions"
run_extractor "base_caption" "$Q_BASE" "$BASE_OBJ"

echo "[2/3] extract objects from intervention captions"
run_extractor "int_caption" "$Q_INT" "$INT_OBJ"

echo "[3/3] analyze caption-conditioned object extraction proxy"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/analyze_caption_conditioned_object_extraction_proxy.py \
    --baseline_object_pred_jsonl "$BASE_OBJ" \
    --intervention_object_pred_jsonl "$INT_OBJ" \
    --oracle_rows_csv "$ORACLE_ROWS" \
    --target_col "$TARGET_COL" \
    --out_csv "$OUT_ROOT/features/${SPLIT}_caption_conditioned_object_proxy_features.csv" \
    --out_feature_metrics_csv "$OUT_ROOT/features/${SPLIT}_caption_conditioned_object_proxy_feature_metrics.csv" \
    --out_summary_json "$OUT_ROOT/features/${SPLIT}_caption_conditioned_object_proxy_summary.json"
)

echo "[done] $OUT_ROOT"
