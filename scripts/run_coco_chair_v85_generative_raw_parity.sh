#!/usr/bin/env bash
set -Eeuo pipefail

# Rebuild the generative object-risk suppression pipeline from raw test inputs
# and compare it against a cached staged output. This is a parity check, not a
# threshold search.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-1}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
export PYTHONDONTWRITEBYTECODE=1

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
REFERENCE_OUT="${REFERENCE_OUT:-$CAL_ROOT/experiments/coco_chair_v83_fast_next_token_suppression_full500_yp0.35_fix}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v85_generative_raw_parity_yp0.35}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-500}"
SOURCE_LIMIT="${SOURCE_LIMIT:-500}"
THRESHOLD="${THRESHOLD:-0.35}"

INTERVENTION_PRED_BASENAME="${INTERVENTION_PRED_BASENAME:-pred_origin_entropy_simg_caption.jsonl}"
RISK_SCORE_MODE="${RISK_SCORE_MODE:-next_token_yesno}"
RISK_FILTER_TO_VOCAB="${RISK_FILTER_TO_VOCAB:-true}"
RISK_MAX_OBJECTS="${RISK_MAX_OBJECTS:-8}"
RISK_PROBE_BATCH_SIZE="${RISK_PROBE_BATCH_SIZE:-8}"
RISK_OBJECT_VOCAB="${RISK_OBJECT_VOCAB:-coco80}"

SUPPRESS_MODE="${SUPPRESS_MODE:-first_token}"
SUPPRESS_BIAS="${SUPPRESS_BIAS:--1.0}"
SUPPRESS_USE_ADD="${SUPPRESS_USE_ADD:-true}"
SUPPRESS_MAX_GEN_LEN="${SUPPRESS_MAX_GEN_LEN:-512}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-false}"

mkdir -p "$OUT_ROOT"

risk_tag="max${RISK_MAX_OBJECTS}"
if [[ "$RISK_FILTER_TO_VOCAB" == "true" ]]; then
  risk_tag="${risk_tag}_vocab"
fi
if [[ "$RISK_SCORE_MODE" != "yesno" ]]; then
  risk_tag="${risk_tag}_${RISK_SCORE_MODE}"
fi
suppress_tag="max${RISK_MAX_OBJECTS}"
if [[ "$RISK_FILTER_TO_VOCAB" == "true" ]]; then
  suppress_tag="${suppress_tag}_vocab"
fi
suppress_tag="${suppress_tag}_${SUPPRESS_MODE}_bias${SUPPRESS_BIAS}_yp${THRESHOLD}"

echo "[settings] source=$SOURCE_OUT"
echo "[settings] reference=$REFERENCE_OUT"
echo "[settings] out=$OUT_ROOT split=$SPLIT limit=$LIMIT threshold=$THRESHOLD"
echo "[settings] risk_score_mode=$RISK_SCORE_MODE risk_tag=$risk_tag"

echo "[1/3] rebuild raw risk features"
RISK_OUT="$OUT_ROOT/raw_risk_${RISK_SCORE_MODE}"
SOURCE_OUT="$SOURCE_OUT" \
OUT_ROOT="$RISK_OUT" \
SPLIT="$SPLIT" \
LIMIT="$LIMIT" \
SOURCE_LIMIT="$SOURCE_LIMIT" \
INTERVENTION_PRED_BASENAME="$INTERVENTION_PRED_BASENAME" \
RISK_SCORE_MODE="$RISK_SCORE_MODE" \
RISK_FILTER_TO_VOCAB="$RISK_FILTER_TO_VOCAB" \
RISK_MAX_OBJECTS="$RISK_MAX_OBJECTS" \
RISK_PROBE_BATCH_SIZE="$RISK_PROBE_BATCH_SIZE" \
RISK_OBJECT_VOCAB="$RISK_OBJECT_VOCAB" \
STOP_AFTER_RISK=true \
REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
bash "$CAL_ROOT/scripts/run_coco_chair_v81_deployable_risk_recaption.sh"

RISK_CSV="$RISK_OUT/features/${SPLIT}_intervention_object_yesno_risk_limit${LIMIT}_${risk_tag}.csv"
if [[ ! -s "$RISK_CSV" ]]; then
  echo "[error] missing rebuilt risk csv: $RISK_CSV" >&2
  exit 3
fi

echo "[2/3] rebuild suppression output"
APPLY_OUT="$OUT_ROOT/raw_apply_yp${THRESHOLD}"
SOURCE_OUT="$SOURCE_OUT" \
RISK_FEATURES_CSV="$RISK_CSV" \
OUT_ROOT="$APPLY_OUT" \
SPLIT="$SPLIT" \
LIMIT="$LIMIT" \
SOURCE_LIMIT="$SOURCE_LIMIT" \
INTERVENTION_PRED_BASENAME="$INTERVENTION_PRED_BASENAME" \
RISK_MAX_OBJECTS="$RISK_MAX_OBJECTS" \
RISK_FILTER_TO_VOCAB="$RISK_FILTER_TO_VOCAB" \
RISK_MAX_YES_PROB="$THRESHOLD" \
SUPPRESS_MODE="$SUPPRESS_MODE" \
SUPPRESS_BIAS="$SUPPRESS_BIAS" \
SUPPRESS_USE_ADD="$SUPPRESS_USE_ADD" \
SUPPRESS_MAX_GEN_LEN="$SUPPRESS_MAX_GEN_LEN" \
REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
bash "$CAL_ROOT/scripts/run_coco_chair_v82_object_token_suppression.sh"

REF_PRED="$REFERENCE_OUT/$SPLIT/pred_object_token_suppression_merged_${suppress_tag}.jsonl"
CAND_PRED="$APPLY_OUT/$SPLIT/pred_object_token_suppression_merged_${suppress_tag}.jsonl"
REF_SUMMARY="$REFERENCE_OUT/summary/chair_v82_object_token_suppression_${suppress_tag}.json"
CAND_SUMMARY="$APPLY_OUT/summary/chair_v82_object_token_suppression_${suppress_tag}.json"

echo "[3/3] parity audit"
if [[ ! -f "$REF_PRED" ]]; then
  echo "[warn] missing reference pred: $REF_PRED" >&2
fi
if [[ ! -f "$REF_SUMMARY" ]]; then
  echo "[warn] missing reference summary: $REF_SUMMARY" >&2
fi

audit_args=(
  --out_json "$OUT_ROOT/parity_audit_yp${THRESHOLD}.json"
)
if [[ -f "$REF_PRED" && -f "$CAND_PRED" ]]; then
  audit_args+=(
    --reference_pred_jsonl "$REF_PRED"
    --candidate_pred_jsonl "$CAND_PRED"
    --reference_text_key auto
    --candidate_text_key auto
  )
fi
if [[ -f "$REF_SUMMARY" && -f "$CAND_SUMMARY" ]]; then
  audit_args+=(
    --reference_chair_summary_json "$REF_SUMMARY"
    --candidate_chair_summary_json "$CAND_SUMMARY"
  )
fi

"$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/audit_samplewise_online_parity.py" "${audit_args[@]}"
echo "[done] $OUT_ROOT"

