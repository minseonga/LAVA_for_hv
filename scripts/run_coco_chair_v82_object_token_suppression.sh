#!/usr/bin/env bash
set -euo pipefail

# Two-pass deployable generative repair:
#   precomputed intervention object risk CSV
#   -> regenerate selected intervention samples with risky object-token suppression
#   -> merge into original intervention predictions
#   -> CHAIR evaluation.
#
# This script intentionally does not run object extraction or yes/no probing.
# The suppression generation call loads the LLaVA/VGA backbone once and processes
# all selected samples in one process.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-$CAL_ROOT/VGA_origin}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-1}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
export PYTHONDONTWRITEBYTECODE=1

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v82_object_token_suppression}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-100}"
SOURCE_LIMIT="${SOURCE_LIMIT:-500}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
COCO_ANN_ROOT="${COCO_ANN_ROOT:-/home/kms/data/images/mscoco/annotations}"
CHAIR_CACHE="${CHAIR_CACHE:-$SOURCE_OUT/chair_cache.pkl}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
CONV_MODE="${CONV_MODE:-llava_v1}"
SEED="${SEED:-17}"

RISK_MAX_OBJECTS="${RISK_MAX_OBJECTS:-8}"
RISK_FILTER_TO_VOCAB="${RISK_FILTER_TO_VOCAB:-true}"
RISK_MAX_YES_PROB="${RISK_MAX_YES_PROB:-0.40}"
RISK_MAX_LP_MARGIN="${RISK_MAX_LP_MARGIN:-999.0}"
RISK_MIN_SECOND_GAP="${RISK_MIN_SECOND_GAP:-0.0}"
RISK_MIN_OBJECT_COUNT="${RISK_MIN_OBJECT_COUNT:-1}"

SUPPRESS_MAX_GEN_LEN="${SUPPRESS_MAX_GEN_LEN:-512}"
SUPPRESS_USE_ADD="${SUPPRESS_USE_ADD:-true}"
SUPPRESS_ATTN_COEF="${SUPPRESS_ATTN_COEF:-0.2}"
SUPPRESS_CD_ALPHA="${SUPPRESS_CD_ALPHA:-0.02}"
SUPPRESS_START_LAYER="${SUPPRESS_START_LAYER:-2}"
SUPPRESS_END_LAYER="${SUPPRESS_END_LAYER:-15}"
SUPPRESS_HEAD_BALANCING="${SUPPRESS_HEAD_BALANCING:-simg}"
SUPPRESS_MODE="${SUPPRESS_MODE:-first_token}"
SUPPRESS_BIAS="${SUPPRESS_BIAS:--100.0}"
SUPPRESS_SKIP_WITHOUT_IDS="${SUPPRESS_SKIP_WITHOUT_IDS:-true}"

REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

mkdir -p "$OUT_ROOT/splits" "$OUT_ROOT/$SPLIT" "$OUT_ROOT/summary"

reuse_file() {
  local path="$1"
  [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]
}

remove_if_overwrite() {
  local path="$1"
  if [[ "$REUSE_IF_EXISTS" != "true" && -f "$path" ]]; then
    rm -f "$path"
  fi
}

make_limited_jsonl() {
  local in_file="$1"
  local out_file="$2"
  if reuse_file "$out_file"; then
    echo "[reuse] $out_file"
    return
  fi
  remove_if_overwrite "$out_file"
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
  remove_if_overwrite "$save_path"
  remove_if_overwrite "$prepared_cap_file"
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

Q_SRC="$SOURCE_OUT/splits/${SPLIT}_caption_q_limited${SOURCE_LIMIT}.jsonl"
if [[ ! -f "$Q_SRC" ]]; then
  Q_SRC="$SOURCE_OUT/splits/${SPLIT}_caption_q.jsonl"
fi
INT_SRC="$SOURCE_OUT/$SPLIT/pred_origin_entropy_simg_caption.jsonl"

Q_LIMITED="$OUT_ROOT/splits/${SPLIT}_caption_q_limited${LIMIT}.jsonl"
INT_LIMITED="$OUT_ROOT/$SPLIT/pred_origin_entropy_simg_caption_limited${LIMIT}.jsonl"

RISK_TAG="max${RISK_MAX_OBJECTS}"
if [[ "$RISK_FILTER_TO_VOCAB" == "true" ]]; then
  RISK_TAG="${RISK_TAG}_vocab"
fi

DEFAULT_RISK_CSV="$OUT_ROOT/features/${SPLIT}_intervention_object_yesno_risk_limit${LIMIT}_${RISK_TAG}.csv"
RISK_FEATURES_CSV="${RISK_FEATURES_CSV:-$DEFAULT_RISK_CSV}"
if [[ ! -f "$RISK_FEATURES_CSV" ]]; then
  echo "[error] missing RISK_FEATURES_CSV: $RISK_FEATURES_CSV" >&2
  echo "        Run v81 risk extraction first, or pass RISK_FEATURES_CSV=/path/to/*.csv." >&2
  exit 2
fi

ACTION_TAG="${RISK_TAG}_${SUPPRESS_MODE}_bias${SUPPRESS_BIAS}_yp${RISK_MAX_YES_PROB}"
SUPPRESS_PRED="$OUT_ROOT/$SPLIT/pred_object_token_suppression_selected_${ACTION_TAG}.jsonl"
SUPPRESS_SUMMARY="$OUT_ROOT/$SPLIT/pred_object_token_suppression_selected_${ACTION_TAG}.summary.json"
MERGED_PRED="$OUT_ROOT/$SPLIT/pred_object_token_suppression_merged_${ACTION_TAG}.jsonl"

echo "[settings] source=$SOURCE_OUT"
echo "[settings] out=$OUT_ROOT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] risk_csv=$RISK_FEATURES_CSV"
echo "[settings] risk max_yes_prob=$RISK_MAX_YES_PROB max_lp_margin=$RISK_MAX_LP_MARGIN min_second_gap=$RISK_MIN_SECOND_GAP min_object_count=$RISK_MIN_OBJECT_COUNT"
echo "[settings] suppression mode=$SUPPRESS_MODE bias=$SUPPRESS_BIAS use_add=$SUPPRESS_USE_ADD max_gen_len=$SUPPRESS_MAX_GEN_LEN"

make_limited_jsonl "$Q_SRC" "$Q_LIMITED"
make_limited_jsonl "$INT_SRC" "$INT_LIMITED"

if ! reuse_file "$SUPPRESS_PRED"; then
  remove_if_overwrite "$SUPPRESS_PRED"
  remove_if_overwrite "$SUPPRESS_SUMMARY"
  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" scripts/run_vga_caption_with_token_suppression.py \
      --vga-root "$VGA_ROOT" \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$Q_LIMITED" \
      --risk-features-csv "$RISK_FEATURES_CSV" \
      --answers-file "$SUPPRESS_PRED" \
      --out-summary-json "$SUPPRESS_SUMMARY" \
      --conv-mode "$CONV_MODE" \
      --max_gen_len "$SUPPRESS_MAX_GEN_LEN" \
      --use_add "$SUPPRESS_USE_ADD" \
      --attn_coef "$SUPPRESS_ATTN_COEF" \
      --cd_alpha "$SUPPRESS_CD_ALPHA" \
      --start_layer "$SUPPRESS_START_LAYER" \
      --end_layer "$SUPPRESS_END_LAYER" \
      --head_balancing "$SUPPRESS_HEAD_BALANCING" \
      --sampling false \
      --seed "$SEED" \
      --risk_max_yes_prob "$RISK_MAX_YES_PROB" \
      --risk_max_lp_margin "$RISK_MAX_LP_MARGIN" \
      --risk_min_second_gap "$RISK_MIN_SECOND_GAP" \
      --risk_min_object_count "$RISK_MIN_OBJECT_COUNT" \
      --suppress_mode "$SUPPRESS_MODE" \
      --suppress_bias "$SUPPRESS_BIAS" \
      --skip_without_suppress_ids "$SUPPRESS_SKIP_WITHOUT_IDS"
  )
else
  echo "[reuse] $SUPPRESS_PRED"
fi

if ! reuse_file "$MERGED_PRED"; then
  remove_if_overwrite "$MERGED_PRED"
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/materialize_selected_recaption_predictions.py" \
    --base_pred_jsonl "$INT_LIMITED" \
    --repair_pred_jsonl "$SUPPRESS_PRED" \
    --out_jsonl "$MERGED_PRED" \
    --out_summary_json "$OUT_ROOT/$SPLIT/pred_object_token_suppression_merged_${ACTION_TAG}.summary.json" \
    --repair_source_label "object_token_suppression"
else
  echo "[reuse] $MERGED_PRED"
fi

run_chair_eval "$INT_LIMITED" output "$OUT_ROOT/$SPLIT/chair_intervention_limited${LIMIT}.json" "$OUT_ROOT/$SPLIT/chair_input_intervention_limited${LIMIT}.jsonl"
run_chair_eval "$MERGED_PRED" output "$OUT_ROOT/$SPLIT/chair_object_token_suppression_merged_${ACTION_TAG}.json" "$OUT_ROOT/$SPLIT/chair_input_object_token_suppression_merged_${ACTION_TAG}.jsonl"

"$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/summarize_chair_main_table.py" \
  --entry "intervention::$SPLIT::$OUT_ROOT/$SPLIT/chair_intervention_limited${LIMIT}.json" \
  --entry "object_token_suppression::$SPLIT::$OUT_ROOT/$SPLIT/chair_object_token_suppression_merged_${ACTION_TAG}.json" \
  --out_csv "$OUT_ROOT/summary/chair_v82_object_token_suppression_${ACTION_TAG}.csv" \
  --out_json "$OUT_ROOT/summary/chair_v82_object_token_suppression_${ACTION_TAG}.json"

echo "[done] $OUT_ROOT"
