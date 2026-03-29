#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PY_BIN="${PY_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"

POPE_ROOT="${POPE_ROOT:-$CAL_ROOT/experiments/pope_full_9000}"
FEATURE_INPUT_ROOT="${FEATURE_INPUT_ROOT:-$POPE_ROOT/full9000_feature_extract}"

SUBSET_IDS_CSV="${SUBSET_IDS_CSV:-$FEATURE_INPUT_ROOT/subset_ids_all9000.csv}"
SUBSET_GT_CSV="${SUBSET_GT_CSV:-$POPE_ROOT/pope_9000_gt.csv}"
PER_LAYER_TRACE_CSV="${PER_LAYER_TRACE_CSV:-$FEATURE_INPUT_ROOT/traces_baseline_9000/per_layer_yes_trace.csv}"
PER_HEAD_TRACE_CSV="${PER_HEAD_TRACE_CSV:-$FEATURE_INPUT_ROOT/traces_baseline_9000/per_head_yes_trace.csv}"
VSC_CSV="${VSC_CSV:-$FEATURE_INPUT_ROOT/vsc_9000.csv}"
SAMPLES_CSV="${SAMPLES_CSV:-$FEATURE_INPUT_ROOT/samples_baseline_9000.csv}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"

FEATURE_OUT_DIR="${FEATURE_OUT_DIR:-$CAL_ROOT/experiments/pope_feature_screen_v1_full9000_discovery_headset}"
CONTROLLER_OUT_DIR="${CONTROLLER_OUT_DIR:-$POPE_ROOT/vga_discovery_headset_frg_only_offline_9000}"
PER_CASE_CSV="${PER_CASE_CSV:-$POPE_ROOT/all_models_full_strict/vga/taxonomy/per_case_compare.csv}"

C_COL="${C_COL:-faithful_minus_global_attn}"
E_COL="${E_COL:-guidance_mismatch_score}"
EARLY_START="${EARLY_START:-10}"
EARLY_END="${EARLY_END:-15}"
LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
LAYER_FOCUS="${LAYER_FOCUS:-17}"
CALIB_RATIO="${CALIB_RATIO:-0.3}"
SEED="${SEED:-42}"
LAMBDA_D1="${LAMBDA_D1:-1.0}"
MAX_D1_WRONG_RATE="${MAX_D1_WRONG_RATE:-0.35}"
Q_GRID="${Q_GRID:-0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95}"
FALLBACK_ACTION="${FALLBACK_ACTION:-vga}"

mkdir -p "$FEATURE_OUT_DIR" "$CONTROLLER_OUT_DIR"
cd "$CAL_ROOT"

echo "[1/2] build discovery-headset feature table"
"$PY_BIN" scripts/build_pope_feature_screen_v1.py \
  --subset_ids_csv "$SUBSET_IDS_CSV" \
  --subset_gt_csv "$SUBSET_GT_CSV" \
  --per_layer_trace_csv "$PER_LAYER_TRACE_CSV" \
  --per_head_trace_csv "$PER_HEAD_TRACE_CSV" \
  --headset_json "$HEADSET_JSON" \
  --vsc_csv "$VSC_CSV" \
  --samples_csv "$SAMPLES_CSV" \
  --use_split all \
  --out_dir "$FEATURE_OUT_DIR" \
  --early_start "$EARLY_START" \
  --early_end "$EARLY_END" \
  --late_start "$LATE_START" \
  --late_end "$LATE_END" \
  --layer_focus "$LAYER_FOCUS" \
  --eps 1e-6

echo "[2/2] calibrate FRG-only offline hard veto"
"$PY_BIN" scripts/run_vga_hard_veto_controller.py \
  --per_case_csv "$PER_CASE_CSV" \
  --features_csv "$FEATURE_OUT_DIR/features_unified_table.csv" \
  --out_dir "$CONTROLLER_OUT_DIR" \
  --c_col "$C_COL" \
  --e_col "$E_COL" \
  --calib_ratio "$CALIB_RATIO" \
  --seed "$SEED" \
  --lambda_d1 "$LAMBDA_D1" \
  --max_d1_wrong_rate "$MAX_D1_WRONG_RATE" \
  --q_grid "$Q_GRID" \
  --fallback_when_missing_feature "$FALLBACK_ACTION" \
  --use_c 1 \
  --use_e 0

echo "[done] feature_dir=$FEATURE_OUT_DIR"
echo "[done] controller_dir=$CONTROLLER_OUT_DIR"
echo "[saved] $FEATURE_OUT_DIR/features_unified_table.csv"
echo "[saved] $CONTROLLER_OUT_DIR/summary.json"
