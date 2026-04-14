#!/usr/bin/env bash
set -euo pipefail

# Fused caption+object-list generation diagnostic.
#
# One generation produces:
#   Caption: ...
#   Objects: ...
#
# Only the parsed Caption is evaluated with CHAIR. Parsed Objects are used as a
# router-feature diagnostic against the fused-caption safe fallback oracle.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
BASE_GPU="${BASE_GPU:-$GPU}"
INT_GPU="${INT_GPU:-$GPU}"
RUN_PARALLEL="${RUN_PARALLEL:-false}"

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v71_fused_generation_object_list}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-500}"
SOURCE_LIMIT="${SOURCE_LIMIT:-500}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
COCO_ANN_ROOT="${COCO_ANN_ROOT:-/home/kms/data/images/mscoco/annotations}"
CHAIR_CACHE="${CHAIR_CACHE:-$SOURCE_OUT/chair_cache.pkl}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VGA_CONV_MODE="${VGA_CONV_MODE:-llava_v1}"
MAX_GEN_LEN="${MAX_GEN_LEN:-640}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"
VGA_HEAD_BALANCING="${VGA_HEAD_BALANCING:-simg}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
SEED="${SEED:-17}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_ci_unique_noworse}"

mkdir -p "$OUT_ROOT/splits" "$OUT_ROOT/$SPLIT" "$OUT_ROOT/summary" "$OUT_ROOT/features"

reuse_file() {
  local path="$1"
  [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]
}

resolve_question_source() {
  local candidate="$SOURCE_OUT/splits/${SPLIT}_caption_q_limited${SOURCE_LIMIT}.jsonl"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return
  fi
  candidate="$SOURCE_OUT/splits/${SPLIT}_caption_q.jsonl"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return
  fi
  echo "$SOURCE_OUT/splits/${SPLIT}_caption_q_limited${SOURCE_LIMIT}.jsonl"
}

run_origin_fused() {
  local name="$1"
  local use_add="$2"
  local gpu_id="$3"
  local pred_file="$4"

  if reuse_file "$pred_file"; then
    echo "[reuse][$name] $pred_file"
    return
  fi

  echo "[generate][$name] gpu=$gpu_id use_add=$use_add"
  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$VGA_PYTHON_BIN" scripts/run_vga_origin_llava_caption_compat.py \
      --vga-root "$VGA_ROOT" \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$Q_FUSED" \
      --answers-file "$pred_file" \
      --conv-mode "$VGA_CONV_MODE" \
      --max_gen_len "$MAX_GEN_LEN" \
      --use_add "$use_add" \
      --attn_coef "$VGA_ATTN_COEF" \
      --cd_alpha "$VGA_CD_ALPHA" \
      --start_layer "$VGA_START_LAYER" \
      --end_layer "$VGA_END_LAYER" \
      --head_balancing "$VGA_HEAD_BALANCING" \
      --sampling "$VGA_SAMPLING" \
      --seed "$SEED"
  )
  echo "[saved][$name] $pred_file"
}

parse_fused() {
  local name="$1"
  local raw_file="$2"
  local caption_file="$3"
  local objects_file="$4"
  local summary_file="$5"

  if reuse_file "$caption_file" && reuse_file "$objects_file" && reuse_file "$summary_file"; then
    echo "[reuse][$name] parsed outputs"
    return
  fi

  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/parse_fused_caption_object_outputs.py" \
    --in_jsonl "$raw_file" \
    --out_caption_jsonl "$caption_file" \
    --out_objects_jsonl "$objects_file" \
    --out_summary_json "$summary_file"
}

run_chair_eval() {
  local cap_file="$1"
  local save_path="$2"
  local prepared_cap_file="$3"

  if reuse_file "$save_path"; then
    echo "[reuse] $save_path"
    return
  fi

  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/prepare_chair_caption_jsonl.py \
      --in_file "$cap_file" \
      --out_file "$prepared_cap_file" \
      --image_id_key image_id \
      --image_key image \
      --drop_missing
    PYTHONPATH="$EAZY_ROOT:${PYTHONPATH:-}" "$EAZY_PYTHON_BIN" "$EAZY_ROOT/eval_script/chair.py" \
      --cap_file "$prepared_cap_file" \
      --image_id_key image_id \
      --caption_key output \
      --coco_path "$COCO_ANN_ROOT" \
      --cache "$CHAIR_CACHE" \
      --save_path "$save_path"
  )
}

Q_SRC="$(resolve_question_source)"
Q_FUSED="$OUT_ROOT/splits/${SPLIT}_fused_caption_object_q_limited${LIMIT}.jsonl"

RAW_BASE="$OUT_ROOT/$SPLIT/pred_baseline_fused_raw.jsonl"
RAW_INT="$OUT_ROOT/$SPLIT/pred_origin_entropy_simg_fused_raw.jsonl"

BASE_CAP="$OUT_ROOT/$SPLIT/pred_baseline_fused_caption.jsonl"
INT_CAP="$OUT_ROOT/$SPLIT/pred_origin_entropy_simg_fused_caption.jsonl"
BASE_OBJ="$OUT_ROOT/$SPLIT/pred_baseline_fused_objects.jsonl"
INT_OBJ="$OUT_ROOT/$SPLIT/pred_origin_entropy_simg_fused_objects.jsonl"

BASE_CHAIR="$OUT_ROOT/$SPLIT/chair_fused_baseline.json"
INT_CHAIR="$OUT_ROOT/$SPLIT/chair_fused_origin_entropy_simg.json"
FUSED_ORACLE_DIR="$OUT_ROOT/unique_safe_oracle_${SPLIT}_fused_origin_entropy_simg"

echo "[settings] out=$OUT_ROOT source=$SOURCE_OUT split=$SPLIT limit=$LIMIT source_limit=$SOURCE_LIMIT"
echo "[settings] q_src=$Q_SRC"
echo "[settings] model=$MODEL_PATH max_gen_len=$MAX_GEN_LEN target=$TARGET_COL"
echo "[settings] parallel=$RUN_PARALLEL base_gpu=$BASE_GPU int_gpu=$INT_GPU"

if ! reuse_file "$Q_FUSED"; then
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/build_fused_caption_object_questions.py" \
    --in_jsonl "$Q_SRC" \
    --out_jsonl "$Q_FUSED" \
    --out_summary_json "$OUT_ROOT/splits/${SPLIT}_fused_caption_object_q.summary.json" \
    --limit "$LIMIT"
else
  echo "[reuse] $Q_FUSED"
fi

if [[ "$RUN_PARALLEL" == "true" ]]; then
  echo "[1/6] fused baseline/intervention generation in parallel"
  run_origin_fused "baseline_fused" false "$BASE_GPU" "$RAW_BASE" &
  base_pid=$!
  run_origin_fused "origin_entropy_simg_fused" true "$INT_GPU" "$RAW_INT" &
  int_pid=$!
  wait "$base_pid"
  wait "$int_pid"
else
  echo "[1/6] fused baseline generation"
  run_origin_fused "baseline_fused" false "$GPU" "$RAW_BASE"
  echo "[2/6] fused intervention generation"
  run_origin_fused "origin_entropy_simg_fused" true "$GPU" "$RAW_INT"
fi

echo "[3/6] parse fused outputs into Caption and Objects"
parse_fused "baseline_fused" "$RAW_BASE" "$BASE_CAP" "$BASE_OBJ" "$OUT_ROOT/$SPLIT/parse_baseline_fused_summary.json"
parse_fused "origin_entropy_simg_fused" "$RAW_INT" "$INT_CAP" "$INT_OBJ" "$OUT_ROOT/$SPLIT/parse_origin_entropy_simg_fused_summary.json"

echo "[4/6] CHAIR on parsed Caption only"
run_chair_eval "$BASE_CAP" "$BASE_CHAIR" "$OUT_ROOT/$SPLIT/chair_input_fused_baseline.jsonl"
run_chair_eval "$INT_CAP" "$INT_CHAIR" "$OUT_ROOT/$SPLIT/chair_input_fused_origin_entropy_simg.jsonl"

echo "[5/6] summarize normal prompt vs fused prompt caption metrics"
(
  cd "$CAL_ROOT"
  summary_args=(
    --out_csv "$OUT_ROOT/summary/chair_v71_fused_generation_object_list.csv"
    --out_json "$OUT_ROOT/summary/chair_v71_fused_generation_object_list.json"
  )
  normal_baseline_exists=false
  if [[ -f "$SOURCE_OUT/$SPLIT/chair_baseline.json" ]]; then
    normal_baseline_exists=true
    summary_args+=(--entry "baseline::${SPLIT}::$SOURCE_OUT/$SPLIT/chair_baseline.json")
  else
    summary_args+=(--entry "baseline::${SPLIT}::$BASE_CHAIR")
  fi
  if [[ -f "$SOURCE_OUT/$SPLIT/chair_origin_entropy_simg.json" ]]; then
    summary_args+=(--entry "normal_origin_entropy_simg::${SPLIT}::$SOURCE_OUT/$SPLIT/chair_origin_entropy_simg.json")
  fi
  if [[ "$normal_baseline_exists" == "true" ]]; then
    summary_args+=(--entry "fused_baseline::${SPLIT}::$BASE_CHAIR")
  fi
  summary_args+=(--entry "fused_origin_entropy_simg::${SPLIT}::$INT_CHAIR")
  "$CAL_PYTHON_BIN" scripts/summarize_chair_main_table.py "${summary_args[@]}"

  shift_args=(
    --out_csv "$OUT_ROOT/summary/chair_v71_fused_vs_normal_shift.csv"
    --out_json "$OUT_ROOT/summary/chair_v71_fused_vs_normal_shift.json"
  )
  if [[ -f "$SOURCE_OUT/$SPLIT/chair_baseline.json" ]]; then
    shift_args+=(--pair "baseline::$SOURCE_OUT/$SPLIT/chair_baseline.json::$BASE_CHAIR")
  fi
  if [[ -f "$SOURCE_OUT/$SPLIT/chair_origin_entropy_simg.json" ]]; then
    shift_args+=(--pair "origin_entropy_simg::$SOURCE_OUT/$SPLIT/chair_origin_entropy_simg.json::$INT_CHAIR")
  fi
  if [[ "${#shift_args[@]}" -gt 4 ]]; then
    "$CAL_PYTHON_BIN" scripts/compare_chair_metric_shift.py "${shift_args[@]}"
  fi
)

echo "[6/6] fused oracle and object-list router-feature diagnostics"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/analyze_chair_unique_safe_oracle.py \
    --baseline_chair_json "$BASE_CHAIR" \
    --intervention_chair_json "$INT_CHAIR" \
    --out_dir "$FUSED_ORACLE_DIR" \
    --main_oracle_col "$TARGET_COL"

  "$CAL_PYTHON_BIN" scripts/analyze_caption_conditioned_object_extraction_proxy.py \
    --baseline_object_pred_jsonl "$BASE_OBJ" \
    --intervention_object_pred_jsonl "$INT_OBJ" \
    --oracle_rows_csv "$FUSED_ORACLE_DIR/unique_safe_oracle_rows.csv" \
    --target_col "$TARGET_COL" \
    --out_csv "$OUT_ROOT/features/${SPLIT}_fused_object_proxy_features.csv" \
    --out_feature_metrics_csv "$OUT_ROOT/features/${SPLIT}_fused_object_proxy_feature_metrics.csv" \
    --out_summary_json "$OUT_ROOT/features/${SPLIT}_fused_object_proxy_summary.json"
)

echo "[done] $OUT_ROOT"
