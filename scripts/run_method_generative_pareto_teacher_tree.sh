#!/usr/bin/env bash
set -euo pipefail

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
CLAIM_TABLE_CSV="${CLAIM_TABLE_CSV:-}"
CHAIR_TABLE_CSV="${CHAIR_TABLE_CSV:-}"
BASELINE_CHAIR_JSON="${BASELINE_CHAIR_JSON:-}"
INTERVENTION_CHAIR_JSON="${INTERVENTION_CHAIR_JSON:-}"
OUT_ROOT="${OUT_ROOT:-}"

TEACHER_MODE="${TEACHER_MODE:-strict_pareto}"
MIN_F1_GAIN="${MIN_F1_GAIN:-0.0}"
MIN_RECALL_GAIN="${MIN_RECALL_GAIN:-0.0}"
FEATURE_COLS="${FEATURE_COLS:-auto}"
MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
TOP_N_FEATURES="${TOP_N_FEATURES:-8}"
FEATURE_FAMILY_MODE="${FEATURE_FAMILY_MODE:-overall}"
TOP_N_PROBE_FEATURES="${TOP_N_PROBE_FEATURES:-8}"
TOP_N_PAIR_FEATURES="${TOP_N_PAIR_FEATURES:-8}"
MAX_DEPTH_VALUES="${MAX_DEPTH_VALUES:-1,2,3}"
MIN_LEAF_VALUES="${MIN_LEAF_VALUES:-3,5,8,10}"
SPLIT_QUANTILES="${SPLIT_QUANTILES:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}"
TAU_QUANTILES="${TAU_QUANTILES:-0.0,0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.0}"
CONSTRAINT_MODE="${CONSTRAINT_MODE:-both}"
CHAIR_EPS="${CHAIR_EPS:-0.0}"
SELECTION_OBJECTIVE="${SELECTION_OBJECTIVE:-f1}"
MIN_BASELINE_RATE="${MIN_BASELINE_RATE:-0.0}"
MAX_BASELINE_RATE="${MAX_BASELINE_RATE:-1.0}"

if [[ -z "$CLAIM_TABLE_CSV" || -z "$CHAIR_TABLE_CSV" || -z "$BASELINE_CHAIR_JSON" || -z "$INTERVENTION_CHAIR_JSON" || -z "$OUT_ROOT" ]]; then
  echo "[error] required env vars: CLAIM_TABLE_CSV, CHAIR_TABLE_CSV, BASELINE_CHAIR_JSON, INTERVENTION_CHAIR_JSON, OUT_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

PYTHONPATH="${PYTHONPATH:-.}" "$CAL_PYTHON_BIN" scripts/build_generative_pareto_teacher_tree_controller.py \
  --claim_table_csv "$CLAIM_TABLE_CSV" \
  --chair_table_csv "$CHAIR_TABLE_CSV" \
  --baseline_chair_json "$BASELINE_CHAIR_JSON" \
  --intervention_chair_json "$INTERVENTION_CHAIR_JSON" \
  --out_dir "$OUT_ROOT" \
  --teacher_mode "$TEACHER_MODE" \
  --min_f1_gain "$MIN_F1_GAIN" \
  --min_recall_gain "$MIN_RECALL_GAIN" \
  --feature_cols "$FEATURE_COLS" \
  --min_feature_auroc "$MIN_FEATURE_AUROC" \
  --top_n_features "$TOP_N_FEATURES" \
  --feature_family_mode "$FEATURE_FAMILY_MODE" \
  --top_n_probe_features "$TOP_N_PROBE_FEATURES" \
  --top_n_pair_features "$TOP_N_PAIR_FEATURES" \
  --max_depth_values "$MAX_DEPTH_VALUES" \
  --min_leaf_values "$MIN_LEAF_VALUES" \
  --split_quantiles "$SPLIT_QUANTILES" \
  --tau_quantiles "$TAU_QUANTILES" \
  --constraint_mode "$CONSTRAINT_MODE" \
  --chair_eps "$CHAIR_EPS" \
  --selection_objective "$SELECTION_OBJECTIVE" \
  --min_baseline_rate "$MIN_BASELINE_RATE" \
  --max_baseline_rate "$MAX_BASELINE_RATE"
