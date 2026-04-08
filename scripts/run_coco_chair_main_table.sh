#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
PAI_ROOT="${PAI_ROOT:-/home/kms/PAI}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
PAI_PYTHON_BIN="${PAI_PYTHON_BIN:-/home/kms/miniconda3/envs/pai_base/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
COCO_ANN_ROOT="${COCO_ANN_ROOT:-/home/kms/data/COCO/annotations_trainval2014/annotations}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_main_table_v1}"

SEED="${SEED:-42}"
N_VAL="${N_VAL:-500}"
N_TEST="${N_TEST:-500}"
PROMPT="${PROMPT:-Please describe this image in detail.}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_VGA="${RUN_VGA:-1}"
RUN_PAI="${RUN_PAI:-1}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VGA_MODEL_PATH="${VGA_MODEL_PATH:-$MODEL_PATH}"
PAI_MODEL_PATH="${PAI_MODEL_PATH:-$MODEL_PATH}"

VGA_CONV_MODE="${VGA_CONV_MODE:-llava_v1}"
VGA_MAX_GEN_LEN="${VGA_MAX_GEN_LEN:-128}"
VGA_USE_ADD="${VGA_USE_ADD:-true}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_HEAD_BALANCING="${VGA_HEAD_BALANCING:-simg}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"

PAI_MODEL="${PAI_MODEL:-llava-1.5}"
PAI_USE_ATTN="${PAI_USE_ATTN:-1}"
PAI_USE_CFG="${PAI_USE_CFG:-1}"
PAI_BEAM="${PAI_BEAM:-1}"
PAI_SAMPLE="${PAI_SAMPLE:-0}"
PAI_ALPHA="${PAI_ALPHA:-0.2}"
PAI_GAMMA="${PAI_GAMMA:-1.1}"
PAI_START_LAYER="${PAI_START_LAYER:-2}"
PAI_END_LAYER="${PAI_END_LAYER:-32}"
PAI_MAX_NEW_TOKENS="${PAI_MAX_NEW_TOKENS:-128}"

chair_ann_ready() {
  local root="$1"
  [[ -f "$root/instances_val2014.json" ]] && \
  [[ -f "$root/instances_train2014.json" ]] && \
  [[ -f "$root/captions_val2014.json" ]] && \
  [[ -f "$root/captions_train2014.json" ]]
}

resolve_coco_ann_root() {
  local raw="$1"
  local image_parent=""
  local image_grandparent=""
  image_parent="$(dirname "$IMAGE_FOLDER")"
  image_grandparent="$(dirname "$image_parent")"
  local candidates=(
    "$raw"
    "$raw/annotations"
    "$raw/annotations_trainval2014"
    "$raw/annotations_trainval2014/annotations"
    "$image_parent/annotations"
    "$image_parent/annotations_trainval2014/annotations"
    "$image_grandparent/annotations"
    "$image_grandparent/annotations_trainval2014/annotations"
    "/home/kms/data/COCO/annotations"
    "/home/kms/data/COCO/annotations_trainval2014/annotations"
  )
  local cand=""
  for cand in "${candidates[@]}"; do
    if [[ -n "$cand" ]] && chair_ann_ready "$cand"; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

reuse_file() {
  local path="$1"
  if [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]; then
    return 0
  fi
  return 1
}

run_chair_eval() {
  local cap_file="$1"
  local image_id_key="$2"
  local caption_key="$3"
  local save_path="$4"
  local prepared_cap_file="$5"
  if reuse_file "$save_path"; then
    echo "[reuse] $save_path"
    return
  fi
  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/prepare_chair_caption_jsonl.py \
      --in_file "$cap_file" \
      --out_file "$prepared_cap_file" \
      --image_id_key "$image_id_key" \
      --image_key image \
      --drop_missing
    PYTHONPATH="$EAZY_ROOT:${PYTHONPATH:-}" "$EAZY_PYTHON_BIN" "$EAZY_ROOT/eval_script/chair.py" \
      --cap_file "$prepared_cap_file" \
      --image_id_key "$image_id_key" \
      --caption_key "$caption_key" \
      --coco_path "$COCO_ANN_ROOT" \
      --cache "$CHAIR_CACHE" \
      --save_path "$save_path"
  )
}

mkdir -p "$OUT_ROOT"

COCO_ANN_ROOT="$(resolve_coco_ann_root "$COCO_ANN_ROOT" || true)"
if [[ -z "$COCO_ANN_ROOT" ]]; then
  echo "[error] could not locate COCO annotation root for CHAIR eval" >&2
  echo "[hint] set COCO_ANN_ROOT to the directory that directly contains instances_val2014.json and captions_val2014.json" >&2
  exit 1
fi

SPLIT_DIR="$OUT_ROOT/splits"
VAL_DIR="$OUT_ROOT/val"
TEST_DIR="$OUT_ROOT/test"
SUMMARY_DIR="$OUT_ROOT/summary"
mkdir -p "$SPLIT_DIR" "$VAL_DIR" "$TEST_DIR" "$SUMMARY_DIR"

VAL_Q="$SPLIT_DIR/val_caption_q.jsonl"
TEST_Q="$SPLIT_DIR/test_caption_q.jsonl"
SPLIT_SUMMARY="$SPLIT_DIR/summary.json"

CHAIR_CACHE="$OUT_ROOT/chair_cache.pkl"

echo "[assets] image_folder=$IMAGE_FOLDER"
echo "[assets] coco_ann_root=$COCO_ANN_ROOT"

echo "[1/6] build random COCO CHAIR splits"
if ! reuse_file "$SPLIT_SUMMARY"; then
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/build_coco_chair_splits.py \
      --image_folder "$IMAGE_FOLDER" \
      --out_dir "$SPLIT_DIR" \
      --n_val "$N_VAL" \
      --n_test "$N_TEST" \
      --seed "$SEED" \
      --prompt "$PROMPT"
  )
else
  echo "[reuse] $SPLIT_SUMMARY"
fi

run_split() {
  local split_name="$1"
  local q_jsonl="$2"
  local split_dir="$3"

  local baseline_jsonl="$split_dir/pred_vanilla_caption.jsonl"
  local vga_jsonl="$split_dir/pred_vga_caption.jsonl"
  local pai_jsonl="$split_dir/pred_pai_caption.jsonl"

  local baseline_chair_input="$split_dir/chair_input_baseline.jsonl"
  local vga_chair_input="$split_dir/chair_input_vga.jsonl"
  local pai_chair_input="$split_dir/chair_input_pai.jsonl"

  local baseline_chair_json="$split_dir/chair_baseline.json"
  local vga_chair_json="$split_dir/chair_vga.json"
  local pai_chair_json="$split_dir/chair_pai.json"

  echo "[2/6][$split_name] baseline caption"
  if [[ "$RUN_BASELINE" == "1" ]] && ! reuse_file "$baseline_jsonl"; then
    (
      cd "$CAL_ROOT"
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --image-folder "$IMAGE_FOLDER" \
        --question-file "$q_jsonl" \
        --answers-file "$baseline_jsonl" \
        --conv-mode llava_v1 \
        --temperature 0 \
        --num_beams 1 \
        --max_new_tokens 128
    )
  else
    echo "[reuse] $baseline_jsonl"
  fi

  echo "[3/6][$split_name] VGA caption"
  if [[ "$RUN_VGA" == "1" ]] && ! reuse_file "$vga_jsonl"; then
    (
      cd "$VGA_ROOT"
      CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" "$VGA_ROOT/eval/object_hallucination_vqa_llava.py" \
        --model-path "$VGA_MODEL_PATH" \
        --image-folder "$IMAGE_FOLDER" \
        --question-file "$q_jsonl" \
        --answers-file "$vga_jsonl" \
        --conv-mode "$VGA_CONV_MODE" \
        --max_gen_len "$VGA_MAX_GEN_LEN" \
        --use_add "$VGA_USE_ADD" \
        --attn_coef "$VGA_ATTN_COEF" \
        --head_balancing "$VGA_HEAD_BALANCING" \
        --sampling "$VGA_SAMPLING" \
        --cd_alpha "$VGA_CD_ALPHA" \
        --seed "$SEED" \
        --start_layer "$VGA_START_LAYER" \
        --end_layer "$VGA_END_LAYER"
    )
  else
    echo "[reuse] $vga_jsonl"
  fi

  echo "[4/6][$split_name] PAI caption"
  local pai_flags=()
  if [[ "$PAI_USE_ATTN" == "1" ]]; then
    pai_flags+=(--use_attn)
  fi
  if [[ "$PAI_USE_CFG" == "1" ]]; then
    pai_flags+=(--use_cfg)
  fi
  if [[ "$PAI_SAMPLE" == "1" ]]; then
    pai_flags+=(--sample)
  fi
  if [[ "$RUN_PAI" == "1" ]] && ! reuse_file "$pai_jsonl"; then
    (
      cd "$CAL_ROOT"
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$PAI_PYTHON_BIN" "$CAL_ROOT/scripts/run_pai_question_subset.py" \
        --pai_root "$PAI_ROOT" \
        --question_file "$q_jsonl" \
        --image_folder "$IMAGE_FOLDER" \
        --answers_file "$pai_jsonl" \
        --model "$PAI_MODEL" \
        --model_path "$PAI_MODEL_PATH" \
        --gpu_id 0 \
        --beam "$PAI_BEAM" \
        --alpha "$PAI_ALPHA" \
        --gamma "$PAI_GAMMA" \
        --start_layer "$PAI_START_LAYER" \
        --end_layer "$PAI_END_LAYER" \
        --max_new_tokens "$PAI_MAX_NEW_TOKENS" \
        "${pai_flags[@]}" \
        --seed "$SEED"
    )
  else
    echo "[reuse] $pai_jsonl"
  fi

  echo "[5/6][$split_name] CHAIR eval"
  if [[ "$RUN_BASELINE" == "1" ]]; then
    run_chair_eval "$baseline_jsonl" image_id text "$baseline_chair_json" "$baseline_chair_input"
  fi
  if [[ "$RUN_VGA" == "1" ]]; then
    run_chair_eval "$vga_jsonl" image_id output "$vga_chair_json" "$vga_chair_input"
  fi
  if [[ "$RUN_PAI" == "1" ]]; then
    run_chair_eval "$pai_jsonl" image_id output "$pai_chair_json" "$pai_chair_input"
  fi
}

run_split "val" "$VAL_Q" "$VAL_DIR"
run_split "test" "$TEST_Q" "$TEST_DIR"

echo "[6/6] summarize CHAIR main table"
SUMMARY_CSV="$SUMMARY_DIR/chair_main_table.csv"
SUMMARY_JSON="$SUMMARY_DIR/chair_main_table.json"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/summarize_chair_main_table.py \
    --entry "baseline::val::$VAL_DIR/chair_baseline.json" \
    --entry "vga::val::$VAL_DIR/chair_vga.json" \
    --entry "pai::val::$VAL_DIR/chair_pai.json" \
    --entry "baseline::test::$TEST_DIR/chair_baseline.json" \
    --entry "vga::test::$TEST_DIR/chair_vga.json" \
    --entry "pai::test::$TEST_DIR/chair_pai.json" \
    --out_csv "$SUMMARY_CSV" \
    --out_json "$SUMMARY_JSON"
)

echo "[done] $OUT_ROOT"
