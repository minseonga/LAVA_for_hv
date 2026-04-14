#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_first_next_len512}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v58_vss_head_ablation_smoke}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-50}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
COCO_ANN_ROOT="${COCO_ANN_ROOT:-/home/kms/data/images/mscoco/annotations}"
CHAIR_CACHE="${CHAIR_CACHE:-$SOURCE_OUT/chair_cache.pkl}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VGA_CONV_MODE="${VGA_CONV_MODE:-llava_v1}"
VGA_MAX_GEN_LEN="${VGA_MAX_GEN_LEN:-512}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
SEED="${SEED:-17}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

mkdir -p "$OUT_ROOT/$SPLIT" "$OUT_ROOT/summary"

reuse_file() {
  local path="$1"
  [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]
}

run_chair_eval() {
  local cap_file="$1"
  local caption_key="$2"
  local save_path="$3"
  local prepared_cap_file="$4"
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
      --caption_key "$caption_key" \
      --coco_path "$COCO_ANN_ROOT" \
      --cache "$CHAIR_CACHE" \
      --save_path "$save_path"
  )
}

make_limited_jsonl() {
  local in_file="$1"
  local out_file="$2"
  if reuse_file "$out_file"; then
    echo "[reuse] $out_file"
    return
  fi
  "$CAL_PYTHON_BIN" - "$in_file" "$out_file" "$LIMIT" <<'PY'
import json
import os
import sys

src, dst, limit_s = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(limit_s)
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
n = 0
with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
    for line in f:
        if not line.strip():
            continue
        try:
            json.loads(line)
        except Exception:
            continue
        g.write(line)
        n += 1
        if limit > 0 and n >= limit:
            break
print(f"[saved] {dst} n={n}")
PY
}

run_variant() {
  local name="$1"
  local vss_mode="$2"
  local head="$3"
  local pred="$OUT_ROOT/$SPLIT/pred_${name}_caption.jsonl"
  local chair_input="$OUT_ROOT/$SPLIT/chair_input_${name}.jsonl"
  local chair_json="$OUT_ROOT/$SPLIT/chair_${name}.json"

  echo "[variant][$name] vss=$vss_mode head=$head"
  if ! reuse_file "$pred"; then
    (
      cd "$CAL_ROOT"
      CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" scripts/run_vga_llava_caption_vss_variant.py \
        --vga-root "$VGA_ROOT" \
        --model-path "$MODEL_PATH" \
        --image-folder "$IMAGE_FOLDER" \
        --question-file "$SOURCE_OUT/splits/${SPLIT}_caption_q.jsonl" \
        --answers-file "$pred" \
        --conv-mode "$VGA_CONV_MODE" \
        --max_gen_len "$VGA_MAX_GEN_LEN" \
        --limit "$LIMIT" \
        --use_add true \
        --attn_coef "$VGA_ATTN_COEF" \
        --cd_alpha "$VGA_CD_ALPHA" \
        --start_layer "$VGA_START_LAYER" \
        --end_layer "$VGA_END_LAYER" \
        --head_balancing "$head" \
        --sampling "$VGA_SAMPLING" \
        --vss_mode "$vss_mode"
    )
  else
    echo "[reuse] $pred"
  fi

  run_chair_eval "$pred" output "$chair_json" "$chair_input"
}

echo "[settings] out=$OUT_ROOT source=$SOURCE_OUT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] model=$MODEL_PATH gamma=$VGA_ATTN_COEF lambda=$VGA_CD_ALPHA layers=$VGA_START_LAYER-$VGA_END_LAYER"

BASE_SRC="$SOURCE_OUT/$SPLIT/pred_vanilla_caption.jsonl"
BASE_LIMITED="$OUT_ROOT/$SPLIT/pred_baseline_caption.jsonl"
if [[ -f "$BASE_SRC" ]]; then
  make_limited_jsonl "$BASE_SRC" "$BASE_LIMITED"
  run_chair_eval "$BASE_LIMITED" text "$OUT_ROOT/$SPLIT/chair_baseline.json" "$OUT_ROOT/$SPLIT/chair_input_baseline.jsonl"
else
  echo "[warn] missing baseline source: $BASE_SRC" >&2
fi

run_variant "vga_entropy_simg" "entropy" "simg"
run_variant "vga_entropy_none" "entropy" "none"
run_variant "vga_nll_simg" "nll" "simg"
run_variant "vga_nll_none" "nll" "none"

(
  cd "$CAL_ROOT"
  summary_args=(
    --out_csv "$OUT_ROOT/summary/chair_v58_vss_head_smoke.csv"
    --out_json "$OUT_ROOT/summary/chair_v58_vss_head_smoke.json"
  )
  for method in baseline vga_entropy_simg vga_entropy_none vga_nll_simg vga_nll_none; do
    if [[ -f "$OUT_ROOT/$SPLIT/chair_${method}.json" ]]; then
      summary_args+=(--entry "${method}::${SPLIT}::$OUT_ROOT/$SPLIT/chair_${method}.json")
    fi
  done
  "$CAL_PYTHON_BIN" scripts/summarize_chair_main_table.py "${summary_args[@]}"

  audit_paths=()
  for method in baseline vga_entropy_simg vga_entropy_none vga_nll_simg vga_nll_none; do
    if [[ -f "$OUT_ROOT/$SPLIT/chair_${method}.json" ]]; then
      audit_paths+=("$OUT_ROOT/$SPLIT/chair_${method}.json")
    fi
  done
  if [[ "${#audit_paths[@]}" -gt 0 ]]; then
    "$CAL_PYTHON_BIN" scripts/audit_chair_object_metrics.py \
      --chair_json "${audit_paths[@]}" \
      --out_json "$OUT_ROOT/summary/chair_object_audit_${SPLIT}.json"
  fi
)

echo "[done] $OUT_ROOT"
