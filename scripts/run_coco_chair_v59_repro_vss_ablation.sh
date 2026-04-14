#!/usr/bin/env bash
set -euo pipefail

# Reproduction-first CHAIR captioning ablation.
#
# The anchor is the original VGA runner:
#   VGA_origin/eval/object_hallucination_vqa_llava.py
#
# The goal is not to find the best-looking variant. It is to first establish a
# current-code control, then run only interpretable ablations against that
# control. NLL VSS is kept as a diagnostic because it requires a separate runner.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_first_next_len512}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation}"
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
RUN_NLL_DIAG="${RUN_NLL_DIAG:-true}"

mkdir -p "$OUT_ROOT/splits" "$OUT_ROOT/$SPLIT" "$OUT_ROOT/summary"

reuse_file() {
  local path="$1"
  [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]
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
        json.loads(line)
        g.write(line)
        n += 1
        if limit > 0 and n >= limit:
            break
print(f"[saved] {dst} n={n}")
PY
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

run_origin_variant() {
  local name="$1"
  local head="$2"
  local q_file="$3"
  local pred="$OUT_ROOT/$SPLIT/pred_${name}_caption.jsonl"
  local chair_input="$OUT_ROOT/$SPLIT/chair_input_${name}.jsonl"
  local chair_json="$OUT_ROOT/$SPLIT/chair_${name}.json"

  echo "[origin][$name] vss=entropy head=$head"
  if ! reuse_file "$pred"; then
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
        --use_add true \
        --attn_coef "$VGA_ATTN_COEF" \
        --cd_alpha "$VGA_CD_ALPHA" \
        --start_layer "$VGA_START_LAYER" \
        --end_layer "$VGA_END_LAYER" \
        --head_balancing "$head" \
        --sampling "$VGA_SAMPLING" \
        --seed "$SEED"
    )
  else
    echo "[reuse] $pred"
  fi

  run_chair_eval "$pred" output "$chair_json" "$chair_input"
}

run_diag_vss_variant() {
  local name="$1"
  local vss_mode="$2"
  local head="$3"
  local q_file="$4"
  local pred="$OUT_ROOT/$SPLIT/pred_${name}_caption.jsonl"
  local chair_input="$OUT_ROOT/$SPLIT/chair_input_${name}.jsonl"
  local chair_json="$OUT_ROOT/$SPLIT/chair_${name}.json"

  echo "[diagnostic][$name] vss=$vss_mode head=$head"
  if ! reuse_file "$pred"; then
    (
      cd "$CAL_ROOT"
      CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" scripts/run_vga_llava_caption_vss_variant.py \
        --vga-root "$VGA_ROOT" \
        --model-path "$MODEL_PATH" \
        --image-folder "$IMAGE_FOLDER" \
        --question-file "$q_file" \
        --answers-file "$pred" \
        --conv-mode "$VGA_CONV_MODE" \
        --max_gen_len "$VGA_MAX_GEN_LEN" \
        --limit 0 \
        --use_add true \
        --attn_coef "$VGA_ATTN_COEF" \
        --cd_alpha "$VGA_CD_ALPHA" \
        --start_layer "$VGA_START_LAYER" \
        --end_layer "$VGA_END_LAYER" \
        --head_balancing "$head" \
        --sampling "$VGA_SAMPLING" \
        --seed "$SEED" \
        --vss_mode "$vss_mode"
    )
  else
    echo "[reuse] $pred"
  fi

  run_chair_eval "$pred" output "$chair_json" "$chair_input"
}

compare_caption_outputs() {
  local ref_file="$1"
  local cand_file="$2"
  local out_json="$3"
  "$CAL_PYTHON_BIN" - "$ref_file" "$cand_file" "$out_json" <<'PY'
import json
import os
import sys

ref_path, cand_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "output" in obj:
                rows.append(obj)
    return rows

ref = load(ref_path)
cand = load(cand_path)
common = min(len(ref), len(cand))
same_id = 0
same_output = 0
first_diffs = []
for a, b in zip(ref[:common], cand[:common]):
    if str(a.get("question_id")) == str(b.get("question_id")):
        same_id += 1
    if a.get("output", "") == b.get("output", ""):
        same_output += 1
    elif len(first_diffs) < 5:
        first_diffs.append(
            {
                "question_id_ref": str(a.get("question_id")),
                "question_id_cand": str(b.get("question_id")),
                "ref_output": a.get("output", "")[:500],
                "cand_output": b.get("output", "")[:500],
            }
        )

summary = {
    "ref_file": os.path.abspath(ref_path),
    "cand_file": os.path.abspath(cand_path),
    "n_ref": len(ref),
    "n_cand": len(cand),
    "n_common": common,
    "same_id": same_id,
    "same_output": same_output,
    "same_output_rate": same_output / common if common else 0.0,
    "first_diffs": first_diffs,
}
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

Q_SRC="$SOURCE_OUT/splits/${SPLIT}_caption_q.jsonl"
Q_LIMITED="$OUT_ROOT/splits/${SPLIT}_caption_q_limited${LIMIT}.jsonl"

echo "[settings] out=$OUT_ROOT source=$SOURCE_OUT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] model=$MODEL_PATH gamma=$VGA_ATTN_COEF lambda=$VGA_CD_ALPHA layers=$VGA_START_LAYER-$VGA_END_LAYER sampling=$VGA_SAMPLING"
make_limited_jsonl "$Q_SRC" "$Q_LIMITED"

BASE_SRC="$SOURCE_OUT/$SPLIT/pred_vanilla_caption.jsonl"
BASE_LIMITED="$OUT_ROOT/$SPLIT/pred_baseline_caption.jsonl"
if [[ -f "$BASE_SRC" ]]; then
  make_limited_jsonl "$BASE_SRC" "$BASE_LIMITED"
  run_chair_eval "$BASE_LIMITED" text "$OUT_ROOT/$SPLIT/chair_baseline.json" "$OUT_ROOT/$SPLIT/chair_input_baseline.jsonl"
else
  echo "[warn] missing baseline source: $BASE_SRC" >&2
fi

run_origin_variant "origin_entropy_simg" "simg" "$Q_LIMITED"
run_origin_variant "origin_entropy_none" "none" "$Q_LIMITED"

if [[ "$RUN_NLL_DIAG" == "true" ]]; then
  run_diag_vss_variant "diag_entropy_simg" "entropy" "simg" "$Q_LIMITED"
  compare_caption_outputs \
    "$OUT_ROOT/$SPLIT/pred_origin_entropy_simg_caption.jsonl" \
    "$OUT_ROOT/$SPLIT/pred_diag_entropy_simg_caption.jsonl" \
    "$OUT_ROOT/summary/origin_vs_diag_entropy_simg_compare.json"
  run_diag_vss_variant "diag_nll_simg" "nll" "simg" "$Q_LIMITED"
fi

(
  cd "$CAL_ROOT"
  summary_args=(
    --out_csv "$OUT_ROOT/summary/chair_v59_repro_vss_ablation.csv"
    --out_json "$OUT_ROOT/summary/chair_v59_repro_vss_ablation.json"
  )
  for method in baseline origin_entropy_simg origin_entropy_none diag_entropy_simg diag_nll_simg; do
    if [[ -f "$OUT_ROOT/$SPLIT/chair_${method}.json" ]]; then
      summary_args+=(--entry "${method}::${SPLIT}::$OUT_ROOT/$SPLIT/chair_${method}.json")
    fi
  done
  "$CAL_PYTHON_BIN" scripts/summarize_chair_main_table.py "${summary_args[@]}"

  audit_paths=()
  for method in baseline origin_entropy_simg origin_entropy_none diag_entropy_simg diag_nll_simg; do
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
