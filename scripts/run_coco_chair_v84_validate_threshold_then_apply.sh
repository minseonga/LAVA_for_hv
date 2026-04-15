#!/usr/bin/env bash
set -Eeuo pipefail

# Validation-calibrated object-token suppression.
#
# Flow:
#   1. Build risk CSV on validation split.
#   2. Sweep yes-prob thresholds on validation split.
#   3. Select threshold from validation CHAIR metrics.
#   4. Optionally build test risk CSV and apply the selected threshold to test.
#
# This wrapper is detector-agnostic. Run it once with RISK_SCORE_MODE=yesno
# for the slower stronger detector, and once with RISK_SCORE_MODE=next_token_yesno
# for the fast detector.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-1}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
export PYTHONDONTWRITEBYTECODE=1

VAL_SOURCE_OUT="${VAL_SOURCE_OUT:-${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}}"
TEST_SOURCE_OUT="${TEST_SOURCE_OUT:-$VAL_SOURCE_OUT}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v84_validate_threshold_then_apply}"

VAL_SPLIT="${VAL_SPLIT:-val}"
TEST_SPLIT="${TEST_SPLIT:-test}"
VAL_LIMIT="${VAL_LIMIT:-500}"
TEST_LIMIT="${TEST_LIMIT:-500}"
SOURCE_LIMIT="${SOURCE_LIMIT:-500}"
RUN_TEST="${RUN_TEST:-true}"
INTERVENTION_PRED_BASENAME="${INTERVENTION_PRED_BASENAME:-pred_origin_entropy_simg_caption.jsonl}"
VAL_INTERVENTION_PRED_BASENAME="${VAL_INTERVENTION_PRED_BASENAME:-$INTERVENTION_PRED_BASENAME}"
TEST_INTERVENTION_PRED_BASENAME="${TEST_INTERVENTION_PRED_BASENAME:-$INTERVENTION_PRED_BASENAME}"

THRESHOLDS="${THRESHOLDS:-0.35 0.40 0.45 0.50 0.60}"
RISK_SCORE_MODE="${RISK_SCORE_MODE:-next_token_yesno}"
RISK_FILTER_TO_VOCAB="${RISK_FILTER_TO_VOCAB:-true}"
RISK_MAX_OBJECTS="${RISK_MAX_OBJECTS:-8}"
RISK_PROBE_BATCH_SIZE="${RISK_PROBE_BATCH_SIZE:-8}"
RISK_OBJECT_VOCAB="${RISK_OBJECT_VOCAB:-coco80}"
RISK_QUESTION_TEMPLATE="${RISK_QUESTION_TEMPLATE:-}"

SUPPRESS_MODE="${SUPPRESS_MODE:-first_token}"
SUPPRESS_BIAS="${SUPPRESS_BIAS:--1.0}"
SUPPRESS_USE_ADD="${SUPPRESS_USE_ADD:-true}"
SUPPRESS_MAX_GEN_LEN="${SUPPRESS_MAX_GEN_LEN:-512}"

SELECT_OBJECTIVE="${SELECT_OBJECTIVE:-f1_then_chair}"
SELECT_MAX_RECALL_DROP="${SELECT_MAX_RECALL_DROP:-0.002}"
SELECT_MIN_DELTA_F1="${SELECT_MIN_DELTA_F1:-0.0}"
SELECT_MAX_DELTA_CHAIR_I="${SELECT_MAX_DELTA_CHAIR_I:-0.0}"
SELECT_MAX_DELTA_CHAIR_S="${SELECT_MAX_DELTA_CHAIR_S:-0.0}"
SELECT_MIN_CHANGED="${SELECT_MIN_CHANGED:-1}"

REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

mkdir -p "$OUT_ROOT/calibration"

risk_tag="max${RISK_MAX_OBJECTS}"
if [[ "$RISK_FILTER_TO_VOCAB" == "true" ]]; then
  risk_tag="${risk_tag}_vocab"
fi
if [[ "$RISK_SCORE_MODE" != "yesno" ]]; then
  risk_tag="${risk_tag}_${RISK_SCORE_MODE}"
fi

echo "[settings] out=$OUT_ROOT"
echo "[settings] val_source=$VAL_SOURCE_OUT split=$VAL_SPLIT limit=$VAL_LIMIT"
echo "[settings] test_source=$TEST_SOURCE_OUT split=$TEST_SPLIT limit=$TEST_LIMIT run_test=$RUN_TEST"
echo "[settings] val_intervention_pred_basename=$VAL_INTERVENTION_PRED_BASENAME"
echo "[settings] test_intervention_pred_basename=$TEST_INTERVENTION_PRED_BASENAME"
echo "[settings] detector=$RISK_SCORE_MODE risk_tag=$risk_tag thresholds=$THRESHOLDS"
echo "[settings] suppression mode=$SUPPRESS_MODE bias=$SUPPRESS_BIAS"

if [[ ! -f "$VAL_SOURCE_OUT/splits/${VAL_SPLIT}_caption_q_limited${SOURCE_LIMIT}.jsonl" && ! -f "$VAL_SOURCE_OUT/splits/${VAL_SPLIT}_caption_q.jsonl" ]]; then
  echo "[error] validation question split not found under VAL_SOURCE_OUT=$VAL_SOURCE_OUT split=$VAL_SPLIT" >&2
  echo "        Expected one of:" >&2
  echo "          $VAL_SOURCE_OUT/splits/${VAL_SPLIT}_caption_q_limited${SOURCE_LIMIT}.jsonl" >&2
  echo "          $VAL_SOURCE_OUT/splits/${VAL_SPLIT}_caption_q.jsonl" >&2
  exit 2
fi
if [[ ! -f "$VAL_SOURCE_OUT/$VAL_SPLIT/$VAL_INTERVENTION_PRED_BASENAME" ]]; then
  echo "[error] missing validation intervention predictions: $VAL_SOURCE_OUT/$VAL_SPLIT/$VAL_INTERVENTION_PRED_BASENAME" >&2
  echo "        If this source uses a different filename, set VAL_INTERVENTION_PRED_BASENAME, e.g." >&2
  echo "        VAL_INTERVENTION_PRED_BASENAME=pred_vga_full_pvg_caption.jsonl" >&2
  exit 2
fi

VAL_RISK_OUT="$OUT_ROOT/val_risk_${RISK_SCORE_MODE}"
echo "[1/4] validation risk extraction -> $VAL_RISK_OUT"
SOURCE_OUT="$VAL_SOURCE_OUT" \
OUT_ROOT="$VAL_RISK_OUT" \
SPLIT="$VAL_SPLIT" \
LIMIT="$VAL_LIMIT" \
SOURCE_LIMIT="$SOURCE_LIMIT" \
INTERVENTION_PRED_BASENAME="$VAL_INTERVENTION_PRED_BASENAME" \
RISK_SCORE_MODE="$RISK_SCORE_MODE" \
RISK_FILTER_TO_VOCAB="$RISK_FILTER_TO_VOCAB" \
RISK_MAX_OBJECTS="$RISK_MAX_OBJECTS" \
RISK_PROBE_BATCH_SIZE="$RISK_PROBE_BATCH_SIZE" \
RISK_OBJECT_VOCAB="$RISK_OBJECT_VOCAB" \
RISK_QUESTION_TEMPLATE="$RISK_QUESTION_TEMPLATE" \
STOP_AFTER_RISK=true \
REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
bash "$CAL_ROOT/scripts/run_coco_chair_v81_deployable_risk_recaption.sh"

VAL_RISK_CSV="$VAL_RISK_OUT/features/${VAL_SPLIT}_intervention_object_yesno_risk_limit${VAL_LIMIT}_${risk_tag}.csv"
if [[ ! -s "$VAL_RISK_CSV" ]]; then
  echo "[error] expected validation risk csv missing: $VAL_RISK_CSV" >&2
  exit 3
fi

echo "[2/4] validation threshold sweep"
for th in $THRESHOLDS; do
  run_root="$OUT_ROOT/val_sweep_${RISK_SCORE_MODE}/yp${th}"
  SOURCE_OUT="$VAL_SOURCE_OUT" \
  RISK_FEATURES_CSV="$VAL_RISK_CSV" \
  OUT_ROOT="$run_root" \
  SPLIT="$VAL_SPLIT" \
  LIMIT="$VAL_LIMIT" \
  SOURCE_LIMIT="$SOURCE_LIMIT" \
  INTERVENTION_PRED_BASENAME="$VAL_INTERVENTION_PRED_BASENAME" \
  RISK_MAX_OBJECTS="$RISK_MAX_OBJECTS" \
  RISK_FILTER_TO_VOCAB="$RISK_FILTER_TO_VOCAB" \
  RISK_MAX_YES_PROB="$th" \
  SUPPRESS_MODE="$SUPPRESS_MODE" \
  SUPPRESS_BIAS="$SUPPRESS_BIAS" \
  SUPPRESS_USE_ADD="$SUPPRESS_USE_ADD" \
  SUPPRESS_MAX_GEN_LEN="$SUPPRESS_MAX_GEN_LEN" \
  REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
  bash "$CAL_ROOT/scripts/run_coco_chair_v82_object_token_suppression.sh"
done

SELECT_JSON="$OUT_ROOT/calibration/selected_threshold_${RISK_SCORE_MODE}.json"
echo "[3/4] select validation threshold -> $SELECT_JSON"
"$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/select_chair_threshold_from_sweep.py" \
  --root_glob "$OUT_ROOT/val_sweep_${RISK_SCORE_MODE}/yp*" \
  --out_json "$SELECT_JSON" \
  --objective "$SELECT_OBJECTIVE" \
  --max_recall_drop "$SELECT_MAX_RECALL_DROP" \
  --min_delta_f1 "$SELECT_MIN_DELTA_F1" \
  --max_delta_chair_i "$SELECT_MAX_DELTA_CHAIR_I" \
  --max_delta_chair_s "$SELECT_MAX_DELTA_CHAIR_S" \
  --min_changed "$SELECT_MIN_CHANGED"

SELECTED_TH="$("$CAL_PYTHON_BIN" - "$SELECT_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected_threshold"])
PY
)"
echo "[selected-threshold] $SELECTED_TH"

if [[ "$RUN_TEST" != "true" ]]; then
  echo "[done] validation only: $SELECT_JSON"
  exit 0
fi

echo "[4/4] apply selected threshold on test"
TEST_RISK_OUT="$OUT_ROOT/test_risk_${RISK_SCORE_MODE}"
SOURCE_OUT="$TEST_SOURCE_OUT" \
OUT_ROOT="$TEST_RISK_OUT" \
SPLIT="$TEST_SPLIT" \
LIMIT="$TEST_LIMIT" \
SOURCE_LIMIT="$SOURCE_LIMIT" \
INTERVENTION_PRED_BASENAME="$TEST_INTERVENTION_PRED_BASENAME" \
RISK_SCORE_MODE="$RISK_SCORE_MODE" \
RISK_FILTER_TO_VOCAB="$RISK_FILTER_TO_VOCAB" \
RISK_MAX_OBJECTS="$RISK_MAX_OBJECTS" \
RISK_PROBE_BATCH_SIZE="$RISK_PROBE_BATCH_SIZE" \
RISK_OBJECT_VOCAB="$RISK_OBJECT_VOCAB" \
RISK_QUESTION_TEMPLATE="$RISK_QUESTION_TEMPLATE" \
STOP_AFTER_RISK=true \
REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
bash "$CAL_ROOT/scripts/run_coco_chair_v81_deployable_risk_recaption.sh"

TEST_RISK_CSV="$TEST_RISK_OUT/features/${TEST_SPLIT}_intervention_object_yesno_risk_limit${TEST_LIMIT}_${risk_tag}.csv"
if [[ ! -s "$TEST_RISK_CSV" ]]; then
  echo "[error] expected test risk csv missing: $TEST_RISK_CSV" >&2
  exit 3
fi

TEST_APPLY_OUT="$OUT_ROOT/test_apply_${RISK_SCORE_MODE}_yp${SELECTED_TH}"
SOURCE_OUT="$TEST_SOURCE_OUT" \
RISK_FEATURES_CSV="$TEST_RISK_CSV" \
OUT_ROOT="$TEST_APPLY_OUT" \
SPLIT="$TEST_SPLIT" \
LIMIT="$TEST_LIMIT" \
SOURCE_LIMIT="$SOURCE_LIMIT" \
INTERVENTION_PRED_BASENAME="$TEST_INTERVENTION_PRED_BASENAME" \
RISK_MAX_OBJECTS="$RISK_MAX_OBJECTS" \
RISK_FILTER_TO_VOCAB="$RISK_FILTER_TO_VOCAB" \
RISK_MAX_YES_PROB="$SELECTED_TH" \
SUPPRESS_MODE="$SUPPRESS_MODE" \
SUPPRESS_BIAS="$SUPPRESS_BIAS" \
SUPPRESS_USE_ADD="$SUPPRESS_USE_ADD" \
SUPPRESS_MAX_GEN_LEN="$SUPPRESS_MAX_GEN_LEN" \
REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
bash "$CAL_ROOT/scripts/run_coco_chair_v82_object_token_suppression.sh"

echo "[done] selected=$SELECTED_TH val=$SELECT_JSON test=$TEST_APPLY_OUT"
