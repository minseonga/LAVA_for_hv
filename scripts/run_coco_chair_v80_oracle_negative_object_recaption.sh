#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-1}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v80_oracle_negative_object_recaption}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-100}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
COCO_ANN_ROOT="${COCO_ANN_ROOT:-/home/kms/data/images/mscoco/annotations}"
CHAIR_CACHE="${CHAIR_CACHE:-$SOURCE_OUT/chair_cache.pkl}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
CONV_MODE="${CONV_MODE:-llava_v1}"
MAX_GEN_LEN="${MAX_GEN_LEN:-160}"
SEED="${SEED:-17}"
MAX_OBJECTS="${MAX_OBJECTS:-4}"
OBJECT_COL="${OBJECT_COL:-int_only_hallucinated_unique}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
PROMPT_MODE="${PROMPT_MODE:-coverage_preserve}"
RECAP_USE_ADD="${RECAP_USE_ADD:-false}"
RECAP_ATTN_COEF="${RECAP_ATTN_COEF:-0.2}"
RECAP_CD_ALPHA="${RECAP_CD_ALPHA:-0.02}"
RECAP_START_LAYER="${RECAP_START_LAYER:-2}"
RECAP_END_LAYER="${RECAP_END_LAYER:-15}"
RECAP_HEAD_BALANCING="${RECAP_HEAD_BALANCING:-simg}"

mkdir -p "$OUT_ROOT/splits" "$OUT_ROOT/$SPLIT" "$OUT_ROOT/summary"
export PYTHONDONTWRITEBYTECODE=1

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

Q_SRC="$SOURCE_OUT/splits/${SPLIT}_caption_q_limited500.jsonl"
if [[ ! -f "$Q_SRC" ]]; then
  Q_SRC="$SOURCE_OUT/splits/${SPLIT}_caption_q.jsonl"
fi
Q_LIMITED="$OUT_ROOT/splits/${SPLIT}_caption_q_limited${LIMIT}.jsonl"
INT_SRC="$SOURCE_OUT/$SPLIT/pred_origin_entropy_simg_caption.jsonl"
INT_LIMITED="$OUT_ROOT/$SPLIT/pred_origin_entropy_simg_caption_limited${LIMIT}.jsonl"
ORACLE_ROWS="$SOURCE_OUT/unique_safe_oracle_${SPLIT}_origin_entropy_simg/unique_safe_oracle_rows.csv"
if [[ ! -f "$ORACLE_ROWS" ]]; then
  ORACLE_ROWS="$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg/unique_safe_oracle_rows.csv"
fi

echo "[settings] source=$SOURCE_OUT"
echo "[settings] out=$OUT_ROOT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] object_col=$OBJECT_COL max_objects=$MAX_OBJECTS max_gen_len=$MAX_GEN_LEN prompt_mode=$PROMPT_MODE recap_use_add=$RECAP_USE_ADD"

make_limited_jsonl "$Q_SRC" "$Q_LIMITED"
make_limited_jsonl "$INT_SRC" "$INT_LIMITED"

NEG_Q="$OUT_ROOT/splits/${SPLIT}_negative_object_recaption_q_limited${LIMIT}.jsonl"
NEG_IDS="$OUT_ROOT/splits/${SPLIT}_negative_object_selected_ids_limited${LIMIT}.json"
if ! reuse_file "$NEG_Q"; then
  case "$PROMPT_MODE" in
    conservative)
      PROMPT_TEMPLATE='Describe the image in one accurate, concise caption.

The previous caption may have mentioned uncertain objects: {objects}.
Do not mention those objects unless they are clearly visible in the image.
Focus only on concrete physical objects that are visually grounded.'
      ;;
    coverage_preserve)
      PROMPT_TEMPLATE='Write a detailed but accurate caption for the image.

The previous caption may have mentioned uncertain objects: {objects}.
Avoid those uncertain objects unless they are clearly visible.
Do not omit other clearly visible objects.
Mention the main visible people, animals, vehicles, furniture, and objects you are confident about.'
      ;;
    revise)
      PROMPT_TEMPLATE='Revise the image caption to be accurate and visually grounded.

The previous caption may have included uncertain objects: {objects}.
Remove or avoid only those uncertain objects unless they are clearly visible.
Keep other clearly visible objects and scene details.
Write one fluent caption, not a list.'
      ;;
    detailed)
      PROMPT_TEMPLATE='Describe the image in a detailed, accurate caption.

Avoid mentioning these uncertain objects unless they are clearly visible: {objects}.
Include all salient visible objects you are confident about.
Do not make the caption overly short.'
      ;;
    balanced)
      PROMPT_TEMPLATE='Write a detailed but accurate caption for the image.

The following objects are uncertain and may be hallucinated:
{objects}

Avoid mentioning those uncertain objects unless they are clearly visible.
You may mention other concrete objects if they are clearly visible in the image.
Do not replace uncertain objects with guessed related objects.
Do not make the caption overly short; preserve visible scene details.'
      ;;
    *)
      echo "[error] unknown PROMPT_MODE=$PROMPT_MODE" >&2
      exit 2
      ;;
  esac
  "$CAL_PYTHON_BIN" scripts/build_negative_object_recaption_questions.py \
    --question_file "$Q_LIMITED" \
    --oracle_rows_csv "$ORACLE_ROWS" \
    --out_jsonl "$NEG_Q" \
    --out_selected_ids_json "$NEG_IDS" \
    --out_summary_json "$OUT_ROOT/splits/${SPLIT}_negative_object_recaption_questions_summary.json" \
    --object_col "$OBJECT_COL" \
    --template "$PROMPT_TEMPLATE" \
    --limit 0 \
    --max_objects "$MAX_OBJECTS" \
    --selected_only
else
  echo "[reuse] $NEG_Q"
fi

RECAP_PRED="$OUT_ROOT/$SPLIT/pred_oracle_negative_object_recaption_selected.jsonl"
if ! reuse_file "$RECAP_PRED"; then
  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" scripts/run_vga_origin_llava_caption_compat.py \
      --vga-root "$VGA_ROOT" \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$NEG_Q" \
      --answers-file "$RECAP_PRED" \
      --conv-mode "$CONV_MODE" \
      --max_gen_len "$MAX_GEN_LEN" \
      --use_add "$RECAP_USE_ADD" \
      --attn_coef "$RECAP_ATTN_COEF" \
      --cd_alpha "$RECAP_CD_ALPHA" \
      --start_layer "$RECAP_START_LAYER" \
      --end_layer "$RECAP_END_LAYER" \
      --head_balancing "$RECAP_HEAD_BALANCING" \
      --sampling false \
      --seed "$SEED"
  )
else
  echo "[reuse] $RECAP_PRED"
fi

MERGED_PRED="$OUT_ROOT/$SPLIT/pred_oracle_negative_object_recaption_merged.jsonl"
if ! reuse_file "$MERGED_PRED"; then
  "$CAL_PYTHON_BIN" scripts/materialize_selected_recaption_predictions.py \
    --base_pred_jsonl "$INT_LIMITED" \
    --repair_pred_jsonl "$RECAP_PRED" \
    --out_jsonl "$MERGED_PRED" \
    --out_summary_json "$OUT_ROOT/$SPLIT/pred_oracle_negative_object_recaption_merged.summary.json"
else
  echo "[reuse] $MERGED_PRED"
fi

run_chair_eval "$INT_LIMITED" output "$OUT_ROOT/$SPLIT/chair_intervention_limited${LIMIT}.json" "$OUT_ROOT/$SPLIT/chair_input_intervention_limited${LIMIT}.jsonl"
run_chair_eval "$MERGED_PRED" output "$OUT_ROOT/$SPLIT/chair_oracle_negative_object_recaption_merged.json" "$OUT_ROOT/$SPLIT/chair_input_oracle_negative_object_recaption_merged.jsonl"

"$CAL_PYTHON_BIN" scripts/summarize_chair_main_table.py \
  --entry "intervention::$SPLIT::$OUT_ROOT/$SPLIT/chair_intervention_limited${LIMIT}.json" \
  --entry "oracle_negative_recaption::$SPLIT::$OUT_ROOT/$SPLIT/chair_oracle_negative_object_recaption_merged.json" \
  --out_csv "$OUT_ROOT/summary/chair_v80_oracle_negative_object_recaption.csv" \
  --out_json "$OUT_ROOT/summary/chair_v80_oracle_negative_object_recaption.json"

echo "[done] $OUT_ROOT"
