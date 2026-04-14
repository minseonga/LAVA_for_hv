#!/usr/bin/env bash
set -euo pipefail

# Object-list confidence proxy.
#
# 1. Generate baseline and intervention object lists for the same image.
# 2. Teacher-force the baseline object-list output to get token lp/gap/entropy.
# 3. Analyze baseline_obj_list - intervention_obj_list units with baseline trace.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v63_objlist_confidence_proxy}"
OBJECT_LIST_REUSE_OUT="${OBJECT_LIST_REUSE_OUT:-$CAL_ROOT/experiments/coco_chair_v60_self_inventory_proxy}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-500}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"

VGA_CONV_MODE="${VGA_CONV_MODE:-llava_v1}"
VGA_MAX_GEN_LEN="${VGA_MAX_GEN_LEN:-96}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
SEED="${SEED:-17}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

OBJECT_LIST_PROMPT="${OBJECT_LIST_PROMPT:-List the salient visible objects and entities in this image. Answer only with a comma-separated list of nouns or short noun phrases. Do not write a sentence.}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_ci_unique_noworse}"

mkdir -p "$OUT_ROOT/splits" "$OUT_ROOT/$SPLIT" "$OUT_ROOT/features"

reuse_file() {
  local path="$1"
  [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]
}

run_inventory_probe() {
  local name="$1"
  local use_add="$2"
  local q_file="$3"
  local pred="$4"

  if reuse_file "$pred"; then
    echo "[reuse] $pred"
    return
  fi

  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" scripts/run_vga_origin_llava_caption_compat.py \
      --vga-root "$VGA_ROOT" \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$q_file" \
      --answers-file "$pred" \
      --conv-mode "$VGA_CONV_MODE" \
      --max_gen_len "$VGA_MAX_GEN_LEN" \
      --use_add "$use_add" \
      --attn_coef "$VGA_ATTN_COEF" \
      --cd_alpha "$VGA_CD_ALPHA" \
      --start_layer "$VGA_START_LAYER" \
      --end_layer "$VGA_END_LAYER" \
      --head_balancing simg \
      --sampling "$VGA_SAMPLING" \
      --seed "$SEED"
  )
}

Q_SRC="$SOURCE_OUT/splits/${SPLIT}_caption_q_limited${LIMIT}.jsonl"
Q_OBJ="$OUT_ROOT/splits/${SPLIT}_object_list_q_limited${LIMIT}.jsonl"
BASE_OBJ_LOCAL="$OUT_ROOT/$SPLIT/pred_baseline_object_list.jsonl"
INT_OBJ_LOCAL="$OUT_ROOT/$SPLIT/pred_origin_entropy_simg_object_list.jsonl"
BASE_OBJ="$BASE_OBJ_LOCAL"
INT_OBJ="$INT_OBJ_LOCAL"
BASE_CHAIR="$SOURCE_OUT/$SPLIT/chair_baseline.json"
INT_CHAIR="$SOURCE_OUT/$SPLIT/chair_origin_entropy_simg.json"
ORACLE_DIR="$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg"
ORACLE_ROWS="$ORACLE_DIR/unique_safe_oracle_rows.csv"
BASE_OBJ_TRACE="$OUT_ROOT/features/${SPLIT}_baseline_objlist_teacher_forced_trace.csv"
BASE_OBJ_TRACE_SUMMARY="$OUT_ROOT/features/${SPLIT}_baseline_objlist_teacher_forced_trace.summary.json"

echo "[settings] out=$OUT_ROOT source=$SOURCE_OUT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] target=$TARGET_COL object_list_reuse_out=$OBJECT_LIST_REUSE_OUT"

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

if ! reuse_file "$Q_OBJ"; then
  echo "[1/5] build object-list questions"
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/build_coco_chair_object_list_questions.py" \
    --in_jsonl "$Q_SRC" \
    --out_jsonl "$Q_OBJ" \
    --out_summary_json "$OUT_ROOT/splits/${SPLIT}_object_list_q.summary.json" \
    --prompt "$OBJECT_LIST_PROMPT"
else
  echo "[reuse] $Q_OBJ"
fi

echo "[2/5] baseline-mode object-list probe"
if [[ -n "$OBJECT_LIST_REUSE_OUT" && -f "$OBJECT_LIST_REUSE_OUT/$SPLIT/pred_baseline_object_list.jsonl" ]]; then
  BASE_OBJ="$OBJECT_LIST_REUSE_OUT/$SPLIT/pred_baseline_object_list.jsonl"
  echo "[reuse external] $BASE_OBJ"
else
  run_inventory_probe "baseline" false "$Q_OBJ" "$BASE_OBJ_LOCAL"
fi

echo "[3/5] intervention-mode object-list probe"
if [[ -n "$OBJECT_LIST_REUSE_OUT" && -f "$OBJECT_LIST_REUSE_OUT/$SPLIT/pred_origin_entropy_simg_object_list.jsonl" ]]; then
  INT_OBJ="$OBJECT_LIST_REUSE_OUT/$SPLIT/pred_origin_entropy_simg_object_list.jsonl"
  echo "[reuse external] $INT_OBJ"
else
  run_inventory_probe "origin_entropy_simg" true "$Q_OBJ" "$INT_OBJ_LOCAL"
fi

if ! reuse_file "$BASE_OBJ_TRACE"; then
  echo "[4/5] teacher-force baseline object-list outputs"
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/extract_vga_generative_mention_features.py \
      --question_file "$Q_OBJ" \
      --image_folder "$IMAGE_FOLDER" \
      --baseline_pred_jsonl "$BASE_OBJ" \
      --out_csv "$BASE_OBJ_TRACE" \
      --out_summary_json "$BASE_OBJ_TRACE_SUMMARY" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --conv_mode "$CONV_MODE" \
      --device "$DEVICE" \
      --limit "$LIMIT" \
      --pred_text_key auto \
      --reuse_if_exists "$REUSE_IF_EXISTS" \
      --max_mentions 32 \
      --visual_trace false
  )
else
  echo "[reuse] $BASE_OBJ_TRACE"
fi

echo "[5/5] analyze obj-list base-only confidence features"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/analyze_generative_objlist_base_only_confidence_proxy.py \
    --baseline_objlist_jsonl "$BASE_OBJ" \
    --intervention_objlist_jsonl "$INT_OBJ" \
    --baseline_objlist_trace_csv "$BASE_OBJ_TRACE" \
    --oracle_rows_csv "$ORACLE_ROWS" \
    --target_col "$TARGET_COL" \
    --out_csv "$OUT_ROOT/features/${SPLIT}_objlist_base_only_confidence_features.csv" \
    --out_feature_metrics_csv "$OUT_ROOT/features/${SPLIT}_objlist_base_only_confidence_feature_metrics.csv" \
    --out_summary_json "$OUT_ROOT/features/${SPLIT}_objlist_base_only_confidence_summary.json"
)

echo "[done] $OUT_ROOT"
