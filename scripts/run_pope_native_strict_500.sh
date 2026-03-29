#!/usr/bin/env bash
set -euo pipefail

# Strict mode:
# - 500 subset (125 x 4 groups) from balanced-1000
# - Native execution in VISTA/VHR/EAZY repos
# - Standard CSV adapters for cross-repo comparison

ROOT="${ROOT:-/home/kms/LLaVA_calibration}"
OUT_DIR="${OUT_DIR:-$ROOT/experiments/pope_native_strict_500}"
IMAGE_DIR="${IMAGE_DIR:-/home/kms/data/pope/val2014}"
SUBSET_GT_CSV="${SUBSET_GT_CSV:-/home/kms/LLaVA_calibration/experiments/pope_full_9000/vcs_vga_balanced_1000/subset_gt.csv}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"
VISTA_PYTHON_BIN="${VISTA_PYTHON_BIN:-/home/kms/miniconda3/envs/vista/bin/python}"
VHR_PYTHON_BIN="${VHR_PYTHON_BIN:-/home/kms/miniconda3/envs/vhr/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy/bin/python}"

VISTA_REPO="${VISTA_REPO:-/home/kms/VISTA}"
VHR_REPO="${VHR_REPO:-/home/kms/VHR}"
EAZY_REPO="${EAZY_REPO:-/home/kms/EAZY}"

GPU_ID="${GPU_ID:-6}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"

# Methods/config (native-style defaults or paper method)
NOW_TAG="$(date +%Y%m%d_%H%M%S)"
VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-strict_native_500_${NOW_TAG}}"
VISTA_MODEL_PATH="${VISTA_MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VHR_METHOD="${VHR_METHOD:-vhr}"
VHR_MODEL_PATH="${VHR_MODEL_PATH:-liuhaotian/llava-v1.5-7b-hf}"
EAZY_K="${EAZY_K:-3}"
EAZY_BEAM="${EAZY_BEAM:-1}"

require_file() {
  local p="$1"
  if [[ ! -e "$p" ]]; then
    echo "[error] missing: $p" >&2
    exit 1
  fi
}

require_file "$CAL_PYTHON_BIN"
require_file "$VISTA_PYTHON_BIN"
require_file "$VHR_PYTHON_BIN"
require_file "$EAZY_PYTHON_BIN"
require_file "$SUBSET_GT_CSV"
require_file "$IMAGE_DIR"

mkdir -p "$OUT_DIR"

SUBSET_DIR="$OUT_DIR/subset_inputs"
VISTA_NATIVE_DIR="$OUT_DIR/vista_native"
VHR_NATIVE_DIR="$OUT_DIR/vhr_native"
EAZY_NATIVE_DIR="$OUT_DIR/eazy_native"
ADAPT_DIR="$OUT_DIR/adapted_csv"
mkdir -p "$SUBSET_DIR" "$VISTA_NATIVE_DIR" "$VHR_NATIVE_DIR" "$EAZY_NATIVE_DIR" "$ADAPT_DIR"

echo "[1/6] build strict subset (500)"
"$CAL_PYTHON_BIN" "$ROOT/scripts/build_pope_strict_subset_500.py" \
  --subset_gt_csv "$SUBSET_GT_CSV" \
  --out_dir "$SUBSET_DIR" \
  --per_group 125 \
  --seed "$SEED"

RANDOM_JSON="$SUBSET_DIR/coco_pope_random_strict500.json"
POPULAR_JSON="$SUBSET_DIR/coco_pope_popular_strict500.json"
ADVER_JSON="$SUBSET_DIR/coco_pope_adversarial_strict500.json"

backup_and_copy() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" && ! -f "${dst}.bak_strict500" ]]; then
    cp "$dst" "${dst}.bak_strict500"
  fi
  cp "$src" "$dst"
}

echo "[2/6] inject strict subset json into native repo expected paths"
backup_and_copy "$RANDOM_JSON" "$VISTA_REPO/pope_coco/coco_pope_random.json"
backup_and_copy "$POPULAR_JSON" "$VISTA_REPO/pope_coco/coco_pope_popular.json"
backup_and_copy "$ADVER_JSON" "$VISTA_REPO/pope_coco/coco_pope_adversarial.json"

backup_and_copy "$RANDOM_JSON" "$VHR_REPO/data/pope/coco/coco_pope_random.json"
backup_and_copy "$POPULAR_JSON" "$VHR_REPO/data/pope/coco/coco_pope_popular.json"
backup_and_copy "$ADVER_JSON" "$VHR_REPO/data/pope/coco/coco_pope_adversarial.json"
mkdir -p "$VHR_REPO/data/coco"
if [[ ! -e "$VHR_REPO/data/coco/val2014" ]]; then
  ln -s "$IMAGE_DIR" "$VHR_REPO/data/coco/val2014"
fi

backup_and_copy "$RANDOM_JSON" "$EAZY_REPO/pope_coco/coco_pope_random.json"
backup_and_copy "$POPULAR_JSON" "$EAZY_REPO/pope_coco/coco_pope_popular.json"
backup_and_copy "$ADVER_JSON" "$EAZY_REPO/pope_coco/coco_pope_adversarial.json"

echo "[3/6] run VISTA native (3 categories)"
for t in random popular adversarial; do
  (cd "$VISTA_REPO" && CUDA_VISIBLE_DEVICES="$GPU_ID" LLAVA_MODEL_PATH="$VISTA_MODEL_PATH" "$VISTA_PYTHON_BIN" pope_eval.py \
    --model llava-1.5 \
    --pope-type "$t" \
    --data-path "$IMAGE_DIR" \
    --batch-size 1 \
    --subset-size -1 \
    --num-workers 1 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --num-beams 1 \
    --temperature 0.0 \
    --exp_folder "$VISTA_EXP_FOLDER" \
    --seed "$SEED")

  VISTA_PRED=$(ls -t "$VISTA_REPO/exp_results/$VISTA_EXP_FOLDER/llava-1.5"/*_"$t"_*.jsonl | head -n 1)
  cp "$VISTA_PRED" "$VISTA_NATIVE_DIR/vista_${t}.jsonl"
done

echo "[4/6] run VHR native (POPE only)"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$VHR_PYTHON_BIN" "$ROOT/scripts/run_vhr_pope_only.py" \
  --vhr_repo "$VHR_REPO" \
  --model_path "$VHR_MODEL_PATH" \
  --device "cuda:0" \
  --method "$VHR_METHOD" \
  --vhr_aug_ratio 2.0 \
  --vhr_layers 14 \
  --vhr_layer1 \
  --vhr_filter \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --seed "$SEED" \
  --output_dir "$VHR_NATIVE_DIR/results" \
  --num_chunks 1 \
  --chunk_idx 0

for t in random popular adversarial; do
  VHR_PRED=$(find "$VHR_NATIVE_DIR/results" -type f -name "*pope_${t}_0-${VHR_METHOD}-${MAX_NEW_TOKENS}.json" | head -n 1)
  cp "$VHR_PRED" "$VHR_NATIVE_DIR/vhr_${t}.jsonl"
done

echo "[5/6] run EAZY native one-pass (3 categories)"
for t in random popular adversarial; do
  EAZY_PRED="$EAZY_NATIVE_DIR/eazy_${t}.jsonl"
  (cd "$EAZY_REPO" && CUDA_VISIBLE_DEVICES="$GPU_ID" "$EAZY_PYTHON_BIN" eval_script/pope_eval_eazy_onepass.py \
    --model llava-1.5 \
    --pope-type "$t" \
    --gpu-id 0 \
    --data_path "$IMAGE_DIR" \
    --batch_size 1 \
    --num_workers 1 \
    --beam "$EAZY_BEAM" \
    --k "$EAZY_K" \
    --save-jsonl "$EAZY_PRED")
done

echo "[6/6] adapt to standard CSV"
for t in random popular adversarial; do
  IN_JSON="$SUBSET_DIR/coco_pope_${t}_strict500.json"

  "$CAL_PYTHON_BIN" "$ROOT/scripts/adapt_native_pope_outputs_to_standard_csv.py" \
    --source vista \
    --pred_jsonl "$VISTA_NATIVE_DIR/vista_${t}.jsonl" \
    --input_jsonl "$IN_JSON" \
    --out_csv "$ADAPT_DIR/vista_${t}.csv"

  "$CAL_PYTHON_BIN" "$ROOT/scripts/adapt_native_pope_outputs_to_standard_csv.py" \
    --source vhr \
    --pred_jsonl "$VHR_NATIVE_DIR/vhr_${t}.jsonl" \
    --input_jsonl "$IN_JSON" \
    --out_csv "$ADAPT_DIR/vhr_${t}.csv"

  "$CAL_PYTHON_BIN" "$ROOT/scripts/adapt_native_pope_outputs_to_standard_csv.py" \
    --source eazy \
    --pred_jsonl "$EAZY_NATIVE_DIR/eazy_${t}.jsonl" \
    --input_jsonl "$IN_JSON" \
    --out_csv "$ADAPT_DIR/eazy_${t}.csv"
done

echo "[done] $OUT_DIR"
