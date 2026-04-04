#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments/vga_susceptibility_panel}"

POPE_FEATURES_CSV="${POPE_FEATURES_CSV:-/home/kms/LLaVA_calibration/experiments/paper_main_b_c_v1_full/test/cheap_online_features.csv}"
POPE_DECISIONS_CSV="${POPE_DECISIONS_CSV:-/home/kms/LLaVA_calibration/experiments/paper_main_b_c_v1_full/test/fixed_eval/decision_rows.csv}"
POPE_GT_CSV="${POPE_GT_CSV:-/home/kms/LLaVA_calibration/experiments/pope_full_9000/pope_9000_gt.csv}"

AMBER_ROOT="${AMBER_ROOT:-/home/kms/data/AMBER}"
AMBER_FEATURES_CSV="${AMBER_FEATURES_CSV:-/home/kms/LLaVA_calibration/experiments/amber_fixed_transfer_from_pope/discriminative/cheap_online_features.csv}"
AMBER_BASELINE_PRED_JSONL="${AMBER_BASELINE_PRED_JSONL:-/home/kms/LLaVA_calibration/experiments/amber_fixed_transfer_from_pope/discriminative/pred_baseline.jsonl}"
AMBER_INTERVENTION_PRED_JSONL="${AMBER_INTERVENTION_PRED_JSONL:-/home/kms/LLaVA_calibration/experiments/amber_fixed_transfer_from_pope/discriminative/pred_vga.jsonl}"
AMBER_GT_CSV="${AMBER_GT_CSV:-}"

PANEL_CSV="${OUT_ROOT}/panel.csv"
PANEL_SUMMARY_JSON="${OUT_ROOT}/panel_summary.json"
ANALYSIS_DIR="${OUT_ROOT}/analysis"
FOCUS_FEATURE="${FOCUS_FEATURE:-cheap_lp_content_min}"

cd "${ROOT_DIR}"

echo "[1/2] build susceptibility panel"
PYTHONPATH="${ROOT_DIR}" \
"${PYTHON_BIN}" scripts/build_vga_susceptibility_panel.py \
  --pope_features_csv "${POPE_FEATURES_CSV}" \
  --pope_decisions_csv "${POPE_DECISIONS_CSV}" \
  --pope_gt_csv "${POPE_GT_CSV}" \
  --amber_features_csv "${AMBER_FEATURES_CSV}" \
  --amber_baseline_pred_jsonl "${AMBER_BASELINE_PRED_JSONL}" \
  --amber_intervention_pred_jsonl "${AMBER_INTERVENTION_PRED_JSONL}" \
  --amber_gt_csv "${AMBER_GT_CSV}" \
  --amber_root "${AMBER_ROOT}" \
  --method_name vga \
  --out_csv "${PANEL_CSV}" \
  --out_summary_json "${PANEL_SUMMARY_JSON}"

echo "[2/2] analyze susceptibility panel"
PYTHONPATH="${ROOT_DIR}" \
"${PYTHON_BIN}" scripts/analyze_vga_susceptibility.py \
  --panel_csv "${PANEL_CSV}" \
  --out_dir "${ANALYSIS_DIR}" \
  --focus_feature "${FOCUS_FEATURE}"

echo "[done] ${ANALYSIS_DIR}/summary.json"
