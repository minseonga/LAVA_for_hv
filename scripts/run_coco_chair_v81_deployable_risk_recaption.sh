#!/usr/bin/env bash
set -euo pipefail

# Deployable generative repair pipeline:
#   intervention caption -> caption-conditioned object extraction
#   -> object-wise image yes/no risk score
#   -> thresholded top1 risky-object recaption
#   -> CHAIR evaluation.

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
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v81_deployable_risk_recaption}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-100}"
SOURCE_LIMIT="${SOURCE_LIMIT:-500}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
COCO_ANN_ROOT="${COCO_ANN_ROOT:-/home/kms/data/images/mscoco/annotations}"
CHAIR_CACHE="${CHAIR_CACHE:-$SOURCE_OUT/chair_cache.pkl}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-17}"

OBJECT_MAX_GEN_LEN="${OBJECT_MAX_GEN_LEN:-96}"
RISK_MAX_OBJECTS="${RISK_MAX_OBJECTS:-8}"
RISK_SCORE_MODE="${RISK_SCORE_MODE:-yesno}"
RISK_QUESTION_TEMPLATE="${RISK_QUESTION_TEMPLATE:-Is there a {object} in the image? Answer yes or no.}"
RISK_ORACLE_OBJECT_COL="${RISK_ORACLE_OBJECT_COL:-int_hallucinated_unique}"
RISK_MAX_YES_PROB="${RISK_MAX_YES_PROB:-0.40}"
RISK_MAX_LP_MARGIN="${RISK_MAX_LP_MARGIN:-999.0}"
RISK_MIN_SECOND_GAP="${RISK_MIN_SECOND_GAP:-0.0}"
RISK_MIN_OBJECT_COUNT="${RISK_MIN_OBJECT_COUNT:-1}"

RECAP_MAX_GEN_LEN="${RECAP_MAX_GEN_LEN:-220}"
RECAP_USE_ADD="${RECAP_USE_ADD:-false}"
RECAP_ATTN_COEF="${RECAP_ATTN_COEF:-0.2}"
RECAP_CD_ALPHA="${RECAP_CD_ALPHA:-0.02}"
RECAP_START_LAYER="${RECAP_START_LAYER:-2}"
RECAP_END_LAYER="${RECAP_END_LAYER:-15}"
RECAP_HEAD_BALANCING="${RECAP_HEAD_BALANCING:-simg}"

REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

mkdir -p "$OUT_ROOT/splits" "$OUT_ROOT/$SPLIT" "$OUT_ROOT/features" "$OUT_ROOT/summary"

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

Q_SRC="$SOURCE_OUT/splits/${SPLIT}_caption_q_limited${SOURCE_LIMIT}.jsonl"
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

OBJ_Q="$OUT_ROOT/splits/${SPLIT}_int_caption_object_extract_q_limited${LIMIT}.jsonl"
OBJ_PRED="$OUT_ROOT/$SPLIT/pred_int_caption_objects.jsonl"
RISK_CSV="$OUT_ROOT/features/${SPLIT}_intervention_object_yesno_risk_limit${LIMIT}_max${RISK_MAX_OBJECTS}.csv"
RISK_SUMMARY="$OUT_ROOT/features/${SPLIT}_intervention_object_yesno_risk_limit${LIMIT}_max${RISK_MAX_OBJECTS}.summary.json"
RECAP_Q="$OUT_ROOT/splits/${SPLIT}_risk_object_recaption_q_limited${LIMIT}_yp${RISK_MAX_YES_PROB}.jsonl"
RECAP_IDS="$OUT_ROOT/splits/${SPLIT}_risk_object_recaption_selected_ids_limited${LIMIT}_yp${RISK_MAX_YES_PROB}.json"
RECAP_PRED="$OUT_ROOT/$SPLIT/pred_risk_object_recaption_selected_yp${RISK_MAX_YES_PROB}.jsonl"
MERGED_PRED="$OUT_ROOT/$SPLIT/pred_risk_object_recaption_merged_yp${RISK_MAX_YES_PROB}.jsonl"

echo "[settings] source=$SOURCE_OUT"
echo "[settings] out=$OUT_ROOT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] risk_max_yes_prob=$RISK_MAX_YES_PROB risk_max_lp_margin=$RISK_MAX_LP_MARGIN risk_min_second_gap=$RISK_MIN_SECOND_GAP risk_max_objects=$RISK_MAX_OBJECTS"
echo "[settings] recap_use_add=$RECAP_USE_ADD recap_max_gen_len=$RECAP_MAX_GEN_LEN"

make_limited_jsonl "$Q_SRC" "$Q_LIMITED"
make_limited_jsonl "$INT_SRC" "$INT_LIMITED"

if ! reuse_file "$OBJ_Q"; then
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

if ! reuse_file "$RISK_CSV"; then
  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/extract_intervention_object_yesno_risk_features.py \
      --question_file "$Q_LIMITED" \
      --image_folder "$IMAGE_FOLDER" \
      --intervention_object_pred_jsonl "$OBJ_PRED" \
      --out_csv "$RISK_CSV" \
      --out_summary_json "$RISK_SUMMARY" \
      --oracle_rows_csv "$ORACLE_ROWS" \
      --oracle_object_col "$RISK_ORACLE_OBJECT_COL" \
      --model_path "$MODEL_PATH" \
      --conv_mode "$CONV_MODE" \
      --device "$DEVICE" \
      --limit 0 \
      --max_objects "$RISK_MAX_OBJECTS" \
      --question_template "$RISK_QUESTION_TEMPLATE" \
      --score_mode "$RISK_SCORE_MODE" \
      --reuse_if_exists "$REUSE_IF_EXISTS"
  )
else
  echo "[reuse] $RISK_CSV"
fi

if ! reuse_file "$RECAP_Q"; then
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/build_risk_object_recaption_questions.py" \
    --question_file "$Q_LIMITED" \
    --risk_features_csv "$RISK_CSV" \
    --out_jsonl "$RECAP_Q" \
    --out_selected_ids_json "$RECAP_IDS" \
    --out_summary_json "$OUT_ROOT/splits/${SPLIT}_risk_object_recaption_questions_summary_yp${RISK_MAX_YES_PROB}.json" \
    --max_yes_prob "$RISK_MAX_YES_PROB" \
    --max_lp_margin "$RISK_MAX_LP_MARGIN" \
    --min_second_gap "$RISK_MIN_SECOND_GAP" \
    --min_object_count "$RISK_MIN_OBJECT_COUNT"
else
  echo "[reuse] $RECAP_Q"
fi

if ! reuse_file "$RECAP_PRED"; then
  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" scripts/run_vga_origin_llava_caption_compat.py \
      --vga-root "$VGA_ROOT" \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$RECAP_Q" \
      --answers-file "$RECAP_PRED" \
      --conv-mode "$CONV_MODE" \
      --max_gen_len "$RECAP_MAX_GEN_LEN" \
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

if ! reuse_file "$MERGED_PRED"; then
  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/materialize_selected_recaption_predictions.py" \
    --base_pred_jsonl "$INT_LIMITED" \
    --repair_pred_jsonl "$RECAP_PRED" \
    --out_jsonl "$MERGED_PRED" \
    --out_summary_json "$OUT_ROOT/$SPLIT/pred_risk_object_recaption_merged_yp${RISK_MAX_YES_PROB}.summary.json"
else
  echo "[reuse] $MERGED_PRED"
fi

run_chair_eval "$INT_LIMITED" output "$OUT_ROOT/$SPLIT/chair_intervention_limited${LIMIT}.json" "$OUT_ROOT/$SPLIT/chair_input_intervention_limited${LIMIT}.jsonl"
run_chair_eval "$MERGED_PRED" output "$OUT_ROOT/$SPLIT/chair_risk_object_recaption_merged_yp${RISK_MAX_YES_PROB}.json" "$OUT_ROOT/$SPLIT/chair_input_risk_object_recaption_merged_yp${RISK_MAX_YES_PROB}.jsonl"

"$CAL_PYTHON_BIN" scripts/summarize_chair_main_table.py \
  --entry "intervention::$SPLIT::$OUT_ROOT/$SPLIT/chair_intervention_limited${LIMIT}.json" \
  --entry "risk_object_recaption::$SPLIT::$OUT_ROOT/$SPLIT/chair_risk_object_recaption_merged_yp${RISK_MAX_YES_PROB}.json" \
  --out_csv "$OUT_ROOT/summary/chair_v81_deployable_risk_recaption_yp${RISK_MAX_YES_PROB}.csv" \
  --out_json "$OUT_ROOT/summary/chair_v81_deployable_risk_recaption_yp${RISK_MAX_YES_PROB}.json"

echo "[done] $OUT_ROOT"
