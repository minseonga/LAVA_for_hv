#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-python}"
CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
METHOD_NAME="${METHOD_NAME:-method}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/prediction_audit_${METHOD_NAME}}"

HELDOUT_GT_CSV="${HELDOUT_GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
HELDOUT_BASELINE_PRED_JSONL="${HELDOUT_BASELINE_PRED_JSONL:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/baseline/pred_vanilla_9000.jsonl}"
HELDOUT_INTERVENTION_PRED_JSONL="${HELDOUT_INTERVENTION_PRED_JSONL:-}"
HELDOUT_BASELINE_PRED_TEXT_KEY="${HELDOUT_BASELINE_PRED_TEXT_KEY:-text}"
HELDOUT_INTERVENTION_PRED_TEXT_KEY="${HELDOUT_INTERVENTION_PRED_TEXT_KEY:-output}"

DISCOVERY_GT_CSV="${DISCOVERY_GT_CSV:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_gt.csv}"
DISCOVERY_BASELINE_PRED_JSONL="${DISCOVERY_BASELINE_PRED_JSONL:-$CAL_ROOT/experiments/common_pope_discovery_v3_panel_v1/discriminative/baseline/pred_vanilla_discovery.jsonl}"
DISCOVERY_INTERVENTION_PRED_JSONL="${DISCOVERY_INTERVENTION_PRED_JSONL:-}"
DISCOVERY_BASELINE_PRED_TEXT_KEY="${DISCOVERY_BASELINE_PRED_TEXT_KEY:-text}"
DISCOVERY_INTERVENTION_PRED_TEXT_KEY="${DISCOVERY_INTERVENTION_PRED_TEXT_KEY:-output}"

EXAMPLES_PER_GROUP="${EXAMPLES_PER_GROUP:-50}"

if [[ -z "$HELDOUT_INTERVENTION_PRED_JSONL" ]]; then
  echo "[error] HELDOUT_INTERVENTION_PRED_JSONL is required" >&2
  exit 1
fi
if [[ -z "$DISCOVERY_INTERVENTION_PRED_JSONL" ]]; then
  echo "[error] DISCOVERY_INTERVENTION_PRED_JSONL is required" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/audit_discriminative_prediction_artifact.py \
  --gt_csv "$HELDOUT_GT_CSV" \
  --baseline_pred_jsonl "$HELDOUT_BASELINE_PRED_JSONL" \
  --intervention_pred_jsonl "$HELDOUT_INTERVENTION_PRED_JSONL" \
  --baseline_name baseline \
  --intervention_name "$METHOD_NAME" \
  --baseline_pred_text_key "$HELDOUT_BASELINE_PRED_TEXT_KEY" \
  --intervention_pred_text_key "$HELDOUT_INTERVENTION_PRED_TEXT_KEY" \
  --discovery_gt_csv "$DISCOVERY_GT_CSV" \
  --discovery_baseline_pred_jsonl "$DISCOVERY_BASELINE_PRED_JSONL" \
  --discovery_intervention_pred_jsonl "$DISCOVERY_INTERVENTION_PRED_JSONL" \
  --discovery_baseline_pred_text_key "$DISCOVERY_BASELINE_PRED_TEXT_KEY" \
  --discovery_intervention_pred_text_key "$DISCOVERY_INTERVENTION_PRED_TEXT_KEY" \
  --examples_per_group "$EXAMPLES_PER_GROUP" \
  --out_dir "$OUT_ROOT"

echo "[done] audit -> $OUT_ROOT/summary.json"
