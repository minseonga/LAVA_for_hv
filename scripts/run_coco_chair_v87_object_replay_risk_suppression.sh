#!/usr/bin/env bash
set -euo pipefail

# Deployable generative repair with caption-replay risk scoring:
#   intervention caption -> caption-conditioned object extraction
#   -> one teacher-forced replay of the intervention caption
#   -> score only mentioned object-token spans
#   -> thresholded top1 risky-object token suppression
#   -> CHAIR evaluation.
#
# Compared with v81/v83, this avoids K object-wise yes/no probes. The expensive
# risk detector stage is one replay forward per sample after object extraction.

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
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v87_object_replay_risk_suppression}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-100}"
SOURCE_LIMIT="${SOURCE_LIMIT:-500}"
INTERVENTION_PRED_BASENAME="${INTERVENTION_PRED_BASENAME:-pred_origin_entropy_simg_caption.jsonl}"
INTERVENTION_PRED_JSONL="${INTERVENTION_PRED_JSONL:-}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-17}"

OBJECT_MAX_GEN_LEN="${OBJECT_MAX_GEN_LEN:-96}"
RISK_MAX_OBJECTS="${RISK_MAX_OBJECTS:-8}"
RISK_OBJECT_VOCAB="${RISK_OBJECT_VOCAB:-coco80}"
RISK_FILTER_TO_VOCAB="${RISK_FILTER_TO_VOCAB:-true}"
RISK_ORACLE_OBJECT_COL="${RISK_ORACLE_OBJECT_COL:-int_hallucinated_unique}"
REPLAY_RISK_SCORE_MODE="${REPLAY_RISK_SCORE_MODE:-gap_prob}"
STOP_AFTER_RISK="${STOP_AFTER_RISK:-false}"

RISK_MAX_YES_PROB="${RISK_MAX_YES_PROB:-0.35}"
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
SUPPRESS_MODE="${SUPPRESS_MODE:-all_tokens}"
SUPPRESS_BIAS="${SUPPRESS_BIAS:--0.5}"
SUPPRESS_SKIP_WITHOUT_IDS="${SUPPRESS_SKIP_WITHOUT_IDS:-true}"

REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

mkdir -p "$OUT_ROOT/splits" "$OUT_ROOT/$SPLIT" "$OUT_ROOT/features" "$OUT_ROOT/summary"

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

require_nonempty_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "$path" ]]; then
    echo "[error] empty or missing $label: $path" >&2
    exit 3
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

Q_SRC="$SOURCE_OUT/splits/${SPLIT}_caption_q_limited${SOURCE_LIMIT}.jsonl"
if [[ ! -f "$Q_SRC" ]]; then
  Q_SRC="$SOURCE_OUT/splits/${SPLIT}_caption_q.jsonl"
fi
INT_SRC="${INTERVENTION_PRED_JSONL:-$SOURCE_OUT/$SPLIT/$INTERVENTION_PRED_BASENAME}"
Q_LIMITED="$OUT_ROOT/splits/${SPLIT}_caption_q_limited${LIMIT}.jsonl"
INT_LIMITED="$OUT_ROOT/$SPLIT/pred_origin_entropy_simg_caption_limited${LIMIT}.jsonl"

ORACLE_ROWS="$SOURCE_OUT/unique_safe_oracle_${SPLIT}_origin_entropy_simg/unique_safe_oracle_rows.csv"
if [[ ! -f "$ORACLE_ROWS" ]]; then
  ORACLE_ROWS="$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg/unique_safe_oracle_rows.csv"
fi

OBJ_Q="$OUT_ROOT/splits/${SPLIT}_int_caption_object_extract_q_limited${LIMIT}.jsonl"
OBJ_PRED="$OUT_ROOT/$SPLIT/pred_int_caption_objects.jsonl"

RISK_TAG="max${RISK_MAX_OBJECTS}_${RISK_OBJECT_VOCAB}_${REPLAY_RISK_SCORE_MODE}"
if [[ "$RISK_FILTER_TO_VOCAB" == "true" ]]; then
  RISK_TAG="${RISK_TAG}_vocab"
fi
RISK_CSV="$OUT_ROOT/features/${SPLIT}_intervention_object_replay_risk_limit${LIMIT}_${RISK_TAG}.csv"
RISK_SUMMARY="$OUT_ROOT/features/${SPLIT}_intervention_object_replay_risk_limit${LIMIT}_${RISK_TAG}.summary.json"

echo "[settings] source=$SOURCE_OUT"
echo "[settings] out=$OUT_ROOT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] replay_risk_score_mode=$REPLAY_RISK_SCORE_MODE max_objects=$RISK_MAX_OBJECTS vocab=$RISK_OBJECT_VOCAB filter_to_vocab=$RISK_FILTER_TO_VOCAB"
echo "[settings] risk_threshold=$RISK_MAX_YES_PROB suppress_mode=$SUPPRESS_MODE suppress_bias=$SUPPRESS_BIAS"

make_limited_jsonl "$Q_SRC" "$Q_LIMITED"
make_limited_jsonl "$INT_SRC" "$INT_LIMITED"

if ! reuse_file "$OBJ_Q"; then
  remove_if_overwrite "$OBJ_Q"
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/build_caption_conditioned_object_extraction_questions.py" \
    --question_file "$Q_LIMITED" \
    --pred_jsonl "$INT_LIMITED" \
    --out_jsonl "$OBJ_Q" \
    --out_summary_json "$OUT_ROOT/splits/${SPLIT}_int_caption_object_extract_q.summary.json" \
    --limit 0 \
    --category "intervention_caption_object_extraction"
else
  echo "[reuse] $OBJ_Q"
fi

if ! reuse_file "$OBJ_PRED"; then
  remove_if_overwrite "$OBJ_PRED"
  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" scripts/run_vga_origin_llava_caption_compat.py \
      --vga-root "$VGA_ROOT" \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$OBJ_Q" \
      --answers-file "$OBJ_PRED" \
      --conv-mode "$CONV_MODE" \
      --max_gen_len "$OBJECT_MAX_GEN_LEN" \
      --use_add false \
      --attn_coef 0.2 \
      --cd_alpha 0.02 \
      --start_layer 2 \
      --end_layer 15 \
      --head_balancing simg \
      --sampling false \
      --seed "$SEED"
  )
else
  echo "[reuse] $OBJ_PRED"
fi
require_nonempty_file "$OBJ_PRED" "intervention object prediction"

if ! reuse_file "$RISK_CSV"; then
  remove_if_overwrite "$RISK_CSV"
  remove_if_overwrite "$RISK_SUMMARY"
  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/extract_intervention_object_replay_risk_features.py \
      --question_file "$Q_LIMITED" \
      --image_folder "$IMAGE_FOLDER" \
      --intervention_pred_jsonl "$INT_LIMITED" \
      --intervention_object_pred_jsonl "$OBJ_PRED" \
      --out_csv "$RISK_CSV" \
      --out_summary_json "$RISK_SUMMARY" \
      --oracle_rows_csv "$ORACLE_ROWS" \
      --oracle_object_col "$RISK_ORACLE_OBJECT_COL" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --conv_mode "$CONV_MODE" \
      --device "$DEVICE" \
      --limit 0 \
      --max_objects "$RISK_MAX_OBJECTS" \
      --object_vocab "$RISK_OBJECT_VOCAB" \
      --filter_to_vocab "$RISK_FILTER_TO_VOCAB" \
      --risk_score_mode "$REPLAY_RISK_SCORE_MODE" \
      --reuse_if_exists "$REUSE_IF_EXISTS"
  )
else
  echo "[reuse] $RISK_CSV"
fi
require_nonempty_file "$RISK_CSV" "replay risk csv"

if [[ "$STOP_AFTER_RISK" == "true" ]]; then
  echo "[done-risk] $RISK_CSV"
  exit 0
fi

SOURCE_OUT="$SOURCE_OUT" \
OUT_ROOT="$OUT_ROOT" \
SPLIT="$SPLIT" \
LIMIT="$LIMIT" \
SOURCE_LIMIT="$SOURCE_LIMIT" \
INTERVENTION_PRED_BASENAME="$INTERVENTION_PRED_BASENAME" \
INTERVENTION_PRED_JSONL="$INTERVENTION_PRED_JSONL" \
IMAGE_FOLDER="$IMAGE_FOLDER" \
MODEL_PATH="$MODEL_PATH" \
CONV_MODE="$CONV_MODE" \
SEED="$SEED" \
RISK_FEATURES_CSV="$RISK_CSV" \
RISK_MAX_OBJECTS="$RISK_MAX_OBJECTS" \
RISK_FILTER_TO_VOCAB="$RISK_FILTER_TO_VOCAB" \
RISK_MAX_YES_PROB="$RISK_MAX_YES_PROB" \
RISK_MAX_LP_MARGIN="$RISK_MAX_LP_MARGIN" \
RISK_MIN_SECOND_GAP="$RISK_MIN_SECOND_GAP" \
RISK_MIN_OBJECT_COUNT="$RISK_MIN_OBJECT_COUNT" \
SUPPRESS_MAX_GEN_LEN="$SUPPRESS_MAX_GEN_LEN" \
SUPPRESS_USE_ADD="$SUPPRESS_USE_ADD" \
SUPPRESS_ATTN_COEF="$SUPPRESS_ATTN_COEF" \
SUPPRESS_CD_ALPHA="$SUPPRESS_CD_ALPHA" \
SUPPRESS_START_LAYER="$SUPPRESS_START_LAYER" \
SUPPRESS_END_LAYER="$SUPPRESS_END_LAYER" \
SUPPRESS_HEAD_BALANCING="$SUPPRESS_HEAD_BALANCING" \
SUPPRESS_MODE="$SUPPRESS_MODE" \
SUPPRESS_BIAS="$SUPPRESS_BIAS" \
SUPPRESS_SKIP_WITHOUT_IDS="$SUPPRESS_SKIP_WITHOUT_IDS" \
REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
bash "$CAL_ROOT/scripts/run_coco_chair_v82_object_token_suppression.sh"
