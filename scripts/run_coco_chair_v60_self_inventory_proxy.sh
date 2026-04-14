#!/usr/bin/env bash
set -euo pipefail

# Auxiliary self-inventory proxy for generative fallback.
#
# This keeps the evaluated captioning protocol fixed. It reuses v59 baseline and
# VGA/PVG captions, then runs a cheap image-level object-list probe as auxiliary
# metadata. The first-pass diagnostic uses only the baseline-mode object list to
# test whether baseline-only caption content is likely image-supported without
# CHAIR/GT at inference.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v60_self_inventory_proxy}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-500}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VGA_CONV_MODE="${VGA_CONV_MODE:-llava_v1}"
VGA_MAX_GEN_LEN="${VGA_MAX_GEN_LEN:-96}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
SEED="${SEED:-17}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
RUN_INTERVENTION_LIST="${RUN_INTERVENTION_LIST:-false}"

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
  local pred="$OUT_ROOT/$SPLIT/pred_${name}_object_list.jsonl"

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
BASE_PRED="$SOURCE_OUT/$SPLIT/pred_baseline_caption.jsonl"
INT_PRED="$SOURCE_OUT/$SPLIT/pred_origin_entropy_simg_caption.jsonl"
ORACLE_ROWS="$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg/unique_safe_oracle_rows.csv"
BASE_INV="$OUT_ROOT/$SPLIT/pred_baseline_object_list.jsonl"
INT_INV="$OUT_ROOT/$SPLIT/pred_origin_entropy_simg_object_list.jsonl"

echo "[settings] out=$OUT_ROOT source=$SOURCE_OUT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] run_intervention_list=$RUN_INTERVENTION_LIST target=$TARGET_COL"

if ! reuse_file "$Q_OBJ"; then
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/build_coco_chair_object_list_questions.py" \
    --in_jsonl "$Q_SRC" \
    --out_jsonl "$Q_OBJ" \
    --out_summary_json "$OUT_ROOT/splits/${SPLIT}_object_list_q.summary.json" \
    --prompt "$OBJECT_LIST_PROMPT"
else
  echo "[reuse] $Q_OBJ"
fi

echo "[1/3] baseline-mode object-list probe"
run_inventory_probe "baseline" false "$Q_OBJ"

analysis_args=(
  --baseline_pred_jsonl "$BASE_PRED"
  --intervention_pred_jsonl "$INT_PRED"
  --baseline_inventory_pred_jsonl "$BASE_INV"
  --oracle_rows_csv "$ORACLE_ROWS"
  --target_col "$TARGET_COL"
  --out_csv "$OUT_ROOT/features/${SPLIT}_self_inventory_proxy_features.csv"
  --out_feature_metrics_csv "$OUT_ROOT/features/${SPLIT}_self_inventory_proxy_feature_metrics.csv"
  --out_summary_json "$OUT_ROOT/features/${SPLIT}_self_inventory_proxy_summary.json"
)

if [[ "$RUN_INTERVENTION_LIST" == "true" ]]; then
  echo "[2/3] intervention-mode object-list probe"
  run_inventory_probe "origin_entropy_simg" true "$Q_OBJ"
  analysis_args+=(--intervention_inventory_pred_jsonl "$INT_INV")
else
  echo "[2/3] skip intervention-mode object-list probe"
fi

echo "[3/3] analyze self-inventory proxy features"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/analyze_generative_self_inventory_proxy.py "${analysis_args[@]}"
)

echo "[done] $OUT_ROOT"
