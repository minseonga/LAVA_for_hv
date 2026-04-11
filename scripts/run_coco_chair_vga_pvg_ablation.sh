#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
COCO_ANN_ROOT="${COCO_ANN_ROOT:-/home/kms/data/COCO/annotations_trainval2014/annotations}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_len512_seed17}"

SEED="${SEED:-17}"
N_VAL="${N_VAL:-500}"
N_TEST="${N_TEST:-500}"
PROMPT="${PROMPT:-Please describe this image in detail.}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_FULL_PVG="${RUN_FULL_PVG:-1}"
RUN_NO_PVG="${RUN_NO_PVG:-1}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VGA_MODEL_PATH="${VGA_MODEL_PATH:-$MODEL_PATH}"

BASELINE_MAX_NEW_TOKENS="${BASELINE_MAX_NEW_TOKENS:-512}"
VGA_CONV_MODE="${VGA_CONV_MODE:-llava_v1}"
VGA_MAX_GEN_LEN="${VGA_MAX_GEN_LEN:-512}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_HEAD_BALANCING="${VGA_HEAD_BALANCING:-simg}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"

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
    "/home/kms/data/images/mscoco/annotations"
    "/home/kms/data/images/mscoco/annotations_trainval2014/annotations"
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

run_vga_variant() {
  local split_name="$1"
  local q_jsonl="$2"
  local split_dir="$3"
  local variant="$4"
  local use_add="$5"

  local pred_jsonl="$split_dir/pred_${variant}_caption.jsonl"
  local chair_input="$split_dir/chair_input_${variant}.jsonl"
  local chair_json="$split_dir/chair_${variant}.json"

  echo "[vga][$split_name][$variant] caption use_add=$use_add"
  if ! reuse_file "$pred_jsonl"; then
    (
      cd "$VGA_ROOT"
      CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" "$VGA_ROOT/eval/object_hallucination_vqa_llava.py" \
        --model-path "$VGA_MODEL_PATH" \
        --image-folder "$IMAGE_FOLDER" \
        --question-file "$q_jsonl" \
        --answers-file "$pred_jsonl" \
        --conv-mode "$VGA_CONV_MODE" \
        --max_gen_len "$VGA_MAX_GEN_LEN" \
        --use_add "$use_add" \
        --attn_coef "$VGA_ATTN_COEF" \
        --head_balancing "$VGA_HEAD_BALANCING" \
        --sampling "$VGA_SAMPLING" \
        --cd_alpha "$VGA_CD_ALPHA" \
        --seed "$SEED" \
        --start_layer "$VGA_START_LAYER" \
        --end_layer "$VGA_END_LAYER"
    )
  else
    echo "[reuse] $pred_jsonl"
  fi

  echo "[vga][$split_name][$variant] CHAIR"
  run_chair_eval "$pred_jsonl" image_id output "$chair_json" "$chair_input"
}

run_split() {
  local split_name="$1"
  local q_jsonl="$2"
  local split_dir="$3"
  mkdir -p "$split_dir"

  local baseline_jsonl="$split_dir/pred_vanilla_caption.jsonl"
  local baseline_chair_input="$split_dir/chair_input_baseline.jsonl"
  local baseline_chair_json="$split_dir/chair_baseline.json"

  echo "[baseline][$split_name] caption"
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
        --max_new_tokens "$BASELINE_MAX_NEW_TOKENS"
    )
  else
    echo "[reuse] $baseline_jsonl"
  fi

  if [[ "$RUN_BASELINE" == "1" ]]; then
    echo "[baseline][$split_name] CHAIR"
    run_chair_eval "$baseline_jsonl" image_id text "$baseline_chair_json" "$baseline_chair_input"
  fi
  if [[ "$RUN_FULL_PVG" == "1" ]]; then
    run_vga_variant "$split_name" "$q_jsonl" "$split_dir" "vga_full_pvg" "true"
  fi
  if [[ "$RUN_NO_PVG" == "1" ]]; then
    run_vga_variant "$split_name" "$q_jsonl" "$split_dir" "vga_no_pvg" "false"
  fi
}

mkdir -p "$OUT_ROOT"
COCO_ANN_ROOT="$(resolve_coco_ann_root "$COCO_ANN_ROOT" || true)"
if [[ -z "$COCO_ANN_ROOT" ]]; then
  echo "[error] could not locate COCO annotation root for CHAIR eval" >&2
  echo "[hint] set COCO_ANN_ROOT to the directory that directly contains COCO 2014 instances/captions annotations" >&2
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
echo "[assets] out_root=$OUT_ROOT"
echo "[settings] seed=$SEED max_len=$VGA_MAX_GEN_LEN gamma=$VGA_ATTN_COEF lambda=$VGA_CD_ALPHA layers=$VGA_START_LAYER-$VGA_END_LAYER head=$VGA_HEAD_BALANCING sampling=$VGA_SAMPLING"

echo "[1/4] build/reuse random COCO CHAIR splits"
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

echo "[2/4] run val/test captions and CHAIR"
run_split "val" "$VAL_Q" "$VAL_DIR"
run_split "test" "$TEST_Q" "$TEST_DIR"

echo "[3/4] summarize main metrics"
(
  cd "$CAL_ROOT"
  summarize_args=(
    --out_csv "$SUMMARY_DIR/chair_pvg_ablation.csv"
    --out_json "$SUMMARY_DIR/chair_pvg_ablation.json"
  )
  if [[ -f "$VAL_DIR/chair_baseline.json" ]]; then
    summarize_args+=(--entry "baseline::val::$VAL_DIR/chair_baseline.json")
  fi
  if [[ -f "$TEST_DIR/chair_baseline.json" ]]; then
    summarize_args+=(--entry "baseline::test::$TEST_DIR/chair_baseline.json")
  fi
  if [[ -f "$VAL_DIR/chair_vga_full_pvg.json" ]]; then
    summarize_args+=(--entry "vga_full_pvg::val::$VAL_DIR/chair_vga_full_pvg.json")
  fi
  if [[ -f "$TEST_DIR/chair_vga_full_pvg.json" ]]; then
    summarize_args+=(--entry "vga_full_pvg::test::$TEST_DIR/chair_vga_full_pvg.json")
  fi
  if [[ -f "$VAL_DIR/chair_vga_no_pvg.json" ]]; then
    summarize_args+=(--entry "vga_no_pvg::val::$VAL_DIR/chair_vga_no_pvg.json")
  fi
  if [[ -f "$TEST_DIR/chair_vga_no_pvg.json" ]]; then
    summarize_args+=(--entry "vga_no_pvg::test::$TEST_DIR/chair_vga_no_pvg.json")
  fi
  "$CAL_PYTHON_BIN" scripts/summarize_chair_main_table.py "${summarize_args[@]}"
)

echo "[4/4] audit object counts on test split"
(
  cd "$CAL_ROOT"
  audit_paths=()
  if [[ -f "$TEST_DIR/chair_baseline.json" ]]; then
    audit_paths+=("$TEST_DIR/chair_baseline.json")
  fi
  if [[ -f "$TEST_DIR/chair_vga_full_pvg.json" ]]; then
    audit_paths+=("$TEST_DIR/chair_vga_full_pvg.json")
  fi
  if [[ -f "$TEST_DIR/chair_vga_no_pvg.json" ]]; then
    audit_paths+=("$TEST_DIR/chair_vga_no_pvg.json")
  fi
  if [[ "${#audit_paths[@]}" -gt 0 ]]; then
    "$CAL_PYTHON_BIN" scripts/audit_chair_object_metrics.py "${audit_paths[@]}" \
      --out_json "$SUMMARY_DIR/chair_object_audit_test.json"
  fi
)

echo "[done] $OUT_ROOT"
