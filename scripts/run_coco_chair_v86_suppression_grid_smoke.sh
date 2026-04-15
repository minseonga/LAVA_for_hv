#!/usr/bin/env bash
set -Eeuo pipefail

# Smoke grid for the deployable generative object-risk controller.
# Reuses a precomputed fast next-token object-risk CSV, then sweeps:
#   - risk yes-prob threshold
#   - token suppression strength
#   - suppression granularity: first_token vs all_tokens
#
# This is intended for LIMIT=100 first. Do not use it as the final full500
# calibration protocol without selecting thresholds on validation.

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-1}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

SOURCE_OUT="${SOURCE_OUT:-$CAL_ROOT/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/coco_chair_v86_suppression_grid_smoke100}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-100}"
SOURCE_LIMIT="${SOURCE_LIMIT:-500}"

RISK_FEATURES_CSV="${RISK_FEATURES_CSV:-$CAL_ROOT/experiments/coco_chair_v83_fast_next_token_risk_full500/features/test_intervention_object_yesno_risk_limit500_max8_vocab_next_token_yesno.csv}"
if [[ ! -f "$RISK_FEATURES_CSV" ]]; then
  RISK_FEATURES_CSV="$CAL_ROOT/experiments/coco_chair_v83_fast_next_token_risk_smoke100/features/test_intervention_object_yesno_risk_limit100_max8_vocab_next_token_yesno.csv"
fi

THRESHOLDS="${THRESHOLDS:-0.35 0.45 0.60 0.70}"
BIASES="${BIASES:--0.5 -1.0 -1.5}"
SUPPRESS_MODES="${SUPPRESS_MODES:-first_token all_tokens}"

SUPPRESS_USE_ADD="${SUPPRESS_USE_ADD:-true}"
SUPPRESS_MAX_GEN_LEN="${SUPPRESS_MAX_GEN_LEN:-512}"
RISK_MAX_OBJECTS="${RISK_MAX_OBJECTS:-8}"
RISK_FILTER_TO_VOCAB="${RISK_FILTER_TO_VOCAB:-true}"
RISK_MAX_LP_MARGIN="${RISK_MAX_LP_MARGIN:-999.0}"
RISK_MIN_SECOND_GAP="${RISK_MIN_SECOND_GAP:-0.0}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

mkdir -p "$OUT_ROOT/runs" "$OUT_ROOT/summary"

if [[ ! -f "$RISK_FEATURES_CSV" ]]; then
  echo "[error] missing risk csv: $RISK_FEATURES_CSV" >&2
  exit 2
fi

sanitize() {
  local x="$1"
  x="${x//-/m}"
  x="${x//./p}"
  echo "$x"
}

echo "[settings] source=$SOURCE_OUT"
echo "[settings] out=$OUT_ROOT split=$SPLIT limit=$LIMIT gpu=$GPU"
echo "[settings] risk_csv=$RISK_FEATURES_CSV"
echo "[settings] thresholds=$THRESHOLDS"
echo "[settings] biases=$BIASES"
echo "[settings] modes=$SUPPRESS_MODES"

for mode in $SUPPRESS_MODES; do
  for th in $THRESHOLDS; do
    for bias in $BIASES; do
      th_tag="$(sanitize "$th")"
      bias_tag="$(sanitize "$bias")"
      run_out="$OUT_ROOT/runs/mode_${mode}_th_${th_tag}_bias_${bias_tag}"
      echo "[grid] mode=$mode th=$th bias=$bias out=$run_out"
      SOURCE_OUT="$SOURCE_OUT" \
      OUT_ROOT="$run_out" \
      SPLIT="$SPLIT" \
      LIMIT="$LIMIT" \
      SOURCE_LIMIT="$SOURCE_LIMIT" \
      RISK_FEATURES_CSV="$RISK_FEATURES_CSV" \
      RISK_MAX_OBJECTS="$RISK_MAX_OBJECTS" \
      RISK_FILTER_TO_VOCAB="$RISK_FILTER_TO_VOCAB" \
      RISK_MAX_YES_PROB="$th" \
      RISK_MAX_LP_MARGIN="$RISK_MAX_LP_MARGIN" \
      RISK_MIN_SECOND_GAP="$RISK_MIN_SECOND_GAP" \
      SUPPRESS_MODE="$mode" \
      SUPPRESS_BIAS="$bias" \
      SUPPRESS_USE_ADD="$SUPPRESS_USE_ADD" \
      SUPPRESS_MAX_GEN_LEN="$SUPPRESS_MAX_GEN_LEN" \
      REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
      bash "$CAL_ROOT/scripts/run_coco_chair_v82_object_token_suppression.sh"
    done
  done
done

"$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/summarize_coco_chair_v86_suppression_grid.py" \
  --grid_root "$OUT_ROOT/runs" \
  --out_csv "$OUT_ROOT/summary/v86_suppression_grid.csv" \
  --out_json "$OUT_ROOT/summary/v86_suppression_grid.json"

echo "[done] $OUT_ROOT"
