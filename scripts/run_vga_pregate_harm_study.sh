#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
AMBER_ROOT="${AMBER_ROOT:-/home/kms/data/AMBER}"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_pregate_harm_v1}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
SAMPLING="${SAMPLING:-false}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
NUM_BEAMS="${NUM_BEAMS:-1}"
CD_ALPHA="${CD_ALPHA:-0.02}"
ATTN_COEF="${ATTN_COEF:-0.2}"
START_LAYER="${START_LAYER:-16}"
END_LAYER="${END_LAYER:-24}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
ATTN_NORM="${ATTN_NORM:-false}"
LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
PROBE_FEATURE_MODE="${PROBE_FEATURE_MODE:-static_headset}"
PROBE_POSITION_MODE="${PROBE_POSITION_MODE:-baseline_yesno_offline_fullseq}"
PROBE_BRANCH_SOURCE="${PROBE_BRANCH_SOURCE:-baseline_jsonl}"
PROBE_FORCE_MANUAL_FULLSEQ="${PROBE_FORCE_MANUAL_FULLSEQ:-false}"
PROBE_PREVIEW_MAX_NEW_TOKENS="${PROBE_PREVIEW_MAX_NEW_TOKENS:-3}"
PROBE_PREVIEW_REUSE_BASELINE="${PROBE_PREVIEW_REUSE_BASELINE:-true}"
PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST="${PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST:-true}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
TOP_K="${TOP_K:-3}"
MAX_BASELINE_RATE="${MAX_BASELINE_RATE:-1.0}"
AMBER_DISCOVERY_RATIO="${AMBER_DISCOVERY_RATIO:-0.2}"
FEATURE_COLS="${FEATURE_COLS:-frg,frg_shared_mean,frg_shared_topk,g_top5_mass,gmi,c_agg_cos,c_agg_ip,e_agg_combo,e_agg_js,late_head_vis_ratio_mean,late_head_vis_ratio_topkmean}"

DISCOVERY_IMAGE_FOLDER="${DISCOVERY_IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
POPE_IMAGE_FOLDER="${POPE_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
AMBER_IMAGE_FOLDER="${AMBER_IMAGE_FOLDER:-$AMBER_ROOT/image}"

POPE_DISC_Q_FILE="${POPE_DISC_Q_FILE:-$CAL_ROOT/experiments/tau_c_calibration_mix_train2014_2785/assets/discovery_q_with_object.jsonl}"
POPE_DISC_GT_CSV="${POPE_DISC_GT_CSV:-$CAL_ROOT/experiments/tau_c_calibration_mix_train2014_2785/assets/discovery_gt.csv}"
POPE_DISC_BASELINE_JSONL="${POPE_DISC_BASELINE_JSONL:-$CAL_ROOT/experiments/tau_c_calibration_mix_train2014_2785/baseline/pred_baseline.jsonl}"
POPE_DISC_INTERVENTION_JSONL="${POPE_DISC_INTERVENTION_JSONL:-$CAL_ROOT/experiments/tau_c_calibration_mix_train2014_2785/vga/pred_vga.jsonl}"

POPE_TEST_Q_FILE="${POPE_TEST_Q_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
POPE_TEST_GT_CSV="${POPE_TEST_GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
POPE_TEST_BASELINE_JSONL="${POPE_TEST_BASELINE_JSONL:-$CAL_ROOT/experiments/paper_main_b_c_v1_full/test_stageb/pred_baseline.jsonl}"
POPE_TEST_INTERVENTION_JSONL="${POPE_TEST_INTERVENTION_JSONL:-$CAL_ROOT/experiments/paper_main_b_c_v1_full/test_stageb/pred_vga.jsonl}"

AMBER_SPLIT_DIR="${AMBER_SPLIT_DIR:-$OUT_ROOT/assets/amber_discriminative_split}"
AMBER_DISC_Q_FILE="${AMBER_DISC_Q_FILE:-$AMBER_SPLIT_DIR/discovery/assets/discovery_q_with_object.jsonl}"
AMBER_DISC_GT_CSV="${AMBER_DISC_GT_CSV:-$AMBER_SPLIT_DIR/discovery/assets/discovery_gt.csv}"
AMBER_TEST_Q_FILE="${AMBER_TEST_Q_FILE:-$AMBER_SPLIT_DIR/test/assets/test_q_with_object.jsonl}"
AMBER_TEST_GT_CSV="${AMBER_TEST_GT_CSV:-$AMBER_SPLIT_DIR/test/assets/test_gt.csv}"
AMBER_BASELINE_JSONL="${AMBER_BASELINE_JSONL:-$CAL_ROOT/experiments/amber_fixed_transfer_from_pope/discriminative/pred_baseline.jsonl}"
AMBER_INTERVENTION_JSONL="${AMBER_INTERVENTION_JSONL:-$CAL_ROOT/experiments/amber_fixed_transfer_from_pope/discriminative/pred_vga.jsonl}"

HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"

POPE_DISC_FALLBACK_ROOT="$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial"

POPE_DISC_PROBE_DIR="$OUT_ROOT/discovery/pope/probe"
POPE_DISC_TABLE_DIR="$OUT_ROOT/discovery/pope/table"
AMBER_DISC_PROBE_DIR="$OUT_ROOT/discovery/amber_disc/probe"
AMBER_DISC_TABLE_DIR="$OUT_ROOT/discovery/amber_disc/table"
DISC_ANALYSIS_DIR="$OUT_ROOT/discovery/analysis"
DISC_CONTROLLER_DIR="$OUT_ROOT/discovery/unified_controller"

POPE_TEST_PROBE_DIR="$OUT_ROOT/test/pope/probe"
POPE_TEST_TABLE_DIR="$OUT_ROOT/test/pope/table"
POPE_TEST_APPLY_DIR="$OUT_ROOT/test/pope/apply"
AMBER_TEST_PROBE_DIR="$OUT_ROOT/test/amber_disc/probe"
AMBER_TEST_TABLE_DIR="$OUT_ROOT/test/amber_disc/table"
AMBER_TEST_APPLY_DIR="$OUT_ROOT/test/amber_disc/apply"

mkdir -p "$OUT_ROOT"
cd "$ROOT_DIR"

if [[ ! -f "$POPE_DISC_Q_FILE" || ! -f "$POPE_DISC_GT_CSV" || ! -f "$POPE_DISC_BASELINE_JSONL" || ! -f "$POPE_DISC_INTERVENTION_JSONL" ]]; then
  FALLBACK_Q="$POPE_DISC_FALLBACK_ROOT/assets/discovery_q_with_object.jsonl"
  FALLBACK_GT="$POPE_DISC_FALLBACK_ROOT/assets/discovery_gt.csv"
  FALLBACK_BASELINE="$POPE_DISC_FALLBACK_ROOT/baseline/pred_baseline.jsonl"
  FALLBACK_INTERVENTION="$POPE_DISC_FALLBACK_ROOT/vga/pred_vga.jsonl"
  if [[ -f "$FALLBACK_Q" && -f "$FALLBACK_GT" && -f "$FALLBACK_BASELINE" && -f "$FALLBACK_INTERVENTION" ]]; then
    echo "[config] fallback to POPE adversarial discovery assets"
    POPE_DISC_Q_FILE="$FALLBACK_Q"
    POPE_DISC_GT_CSV="$FALLBACK_GT"
    POPE_DISC_BASELINE_JSONL="$FALLBACK_BASELINE"
    POPE_DISC_INTERVENTION_JSONL="$FALLBACK_INTERVENTION"
  fi
fi

if [[ ! -f "$AMBER_DISC_Q_FILE" || ! -f "$AMBER_TEST_Q_FILE" ]]; then
  echo "[0/10] prepare AMBER discriminative split assets"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/prepare_amber_discriminative_assets.py \
    --amber_root "$AMBER_ROOT" \
    --out_dir "$AMBER_SPLIT_DIR" \
    --discovery_ratio "$AMBER_DISCOVERY_RATIO" \
    --seed "$SEED"
fi

if [[ "$REUSE_IF_EXISTS" != "true" || ! -f "$POPE_DISC_PROBE_DIR/probe_features.csv" ]]; then
  echo "[1/10] extract POPE discovery probe features"
  mkdir -p "$POPE_DISC_PROBE_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/extract_pnp_probe_features.py \
    --backend vga \
    --vga_root "$VGA_ROOT" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --image_folder "$DISCOVERY_IMAGE_FOLDER" \
    --question_file "$POPE_DISC_Q_FILE" \
    --out_dir "$POPE_DISC_PROBE_DIR" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --sampling "$SAMPLING" \
    --max_gen_len "$MAX_GEN_LEN" \
    --num_beams "$NUM_BEAMS" \
    --cd_alpha "$CD_ALPHA" \
    --attn_coef "$ATTN_COEF" \
    --start_layer "$START_LAYER" \
    --end_layer "$END_LAYER" \
    --head_balancing "$HEAD_BALANCING" \
    --attn_norm "$ATTN_NORM" \
    --late_start "$LATE_START" \
    --late_end "$LATE_END" \
    --probe_feature_mode "$PROBE_FEATURE_MODE" \
    --headset_json "$HEADSET_JSON" \
    --probe_position_mode "$PROBE_POSITION_MODE" \
    --probe_branch_source "$PROBE_BRANCH_SOURCE" \
    --branch_text_jsonl "$POPE_DISC_BASELINE_JSONL" \
    --probe_force_manual_fullseq "$PROBE_FORCE_MANUAL_FULLSEQ" \
    --probe_preview_max_new_tokens "$PROBE_PREVIEW_MAX_NEW_TOKENS" \
    --probe_preview_reuse_baseline "$PROBE_PREVIEW_REUSE_BASELINE" \
    --probe_preview_fallback_to_prompt_last "$PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST" \
    --seed "$SEED" \
    --max_samples "$MAX_SAMPLES"
else
  echo "[1/10] reuse POPE discovery probe features"
fi

echo "[2/10] build POPE discovery pregate table"
mkdir -p "$POPE_DISC_TABLE_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_table.py \
  --probe_features_csv "$POPE_DISC_PROBE_DIR/probe_features.csv" \
  --baseline_pred_jsonl "$POPE_DISC_BASELINE_JSONL" \
  --intervention_pred_jsonl "$POPE_DISC_INTERVENTION_JSONL" \
  --question_file "$POPE_DISC_Q_FILE" \
  --gt_csv "$POPE_DISC_GT_CSV" \
  --benchmark_name pope \
  --split_name discovery \
  --out_csv "$POPE_DISC_TABLE_DIR/table.csv" \
  --out_summary_json "$POPE_DISC_TABLE_DIR/summary.json"

if [[ "$REUSE_IF_EXISTS" != "true" || ! -f "$AMBER_DISC_PROBE_DIR/probe_features.csv" ]]; then
  echo "[3/10] extract AMBER-discovery probe features"
  mkdir -p "$AMBER_DISC_PROBE_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/extract_pnp_probe_features.py \
    --backend vga \
    --vga_root "$VGA_ROOT" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --image_folder "$AMBER_IMAGE_FOLDER" \
    --question_file "$AMBER_DISC_Q_FILE" \
    --out_dir "$AMBER_DISC_PROBE_DIR" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --sampling "$SAMPLING" \
    --max_gen_len "$MAX_GEN_LEN" \
    --num_beams "$NUM_BEAMS" \
    --cd_alpha "$CD_ALPHA" \
    --attn_coef "$ATTN_COEF" \
    --start_layer "$START_LAYER" \
    --end_layer "$END_LAYER" \
    --head_balancing "$HEAD_BALANCING" \
    --attn_norm "$ATTN_NORM" \
    --late_start "$LATE_START" \
    --late_end "$LATE_END" \
    --probe_feature_mode "$PROBE_FEATURE_MODE" \
    --headset_json "$HEADSET_JSON" \
    --probe_position_mode "$PROBE_POSITION_MODE" \
    --probe_branch_source "$PROBE_BRANCH_SOURCE" \
    --branch_text_jsonl "$AMBER_BASELINE_JSONL" \
    --probe_force_manual_fullseq "$PROBE_FORCE_MANUAL_FULLSEQ" \
    --probe_preview_max_new_tokens "$PROBE_PREVIEW_MAX_NEW_TOKENS" \
    --probe_preview_reuse_baseline "$PROBE_PREVIEW_REUSE_BASELINE" \
    --probe_preview_fallback_to_prompt_last "$PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST" \
    --seed "$SEED" \
    --max_samples "$MAX_SAMPLES"
else
  echo "[3/10] reuse AMBER discovery probe features"
fi

echo "[4/10] build AMBER discovery pregate table"
mkdir -p "$AMBER_DISC_TABLE_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_table.py \
  --probe_features_csv "$AMBER_DISC_PROBE_DIR/probe_features.csv" \
  --baseline_pred_jsonl "$AMBER_BASELINE_JSONL" \
  --intervention_pred_jsonl "$AMBER_INTERVENTION_JSONL" \
  --question_file "$AMBER_DISC_Q_FILE" \
  --gt_csv "$AMBER_DISC_GT_CSV" \
  --benchmark_name amber_discriminative \
  --split_name discovery \
  --out_csv "$AMBER_DISC_TABLE_DIR/table.csv" \
  --out_summary_json "$AMBER_DISC_TABLE_DIR/summary.json"

echo "[5/10] phase-1 discovery analysis"
mkdir -p "$DISC_ANALYSIS_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/analyze_vga_pregate_harm.py \
  --table_csvs "$POPE_DISC_TABLE_DIR/table.csv" "$AMBER_DISC_TABLE_DIR/table.csv" \
  --feature_cols "$FEATURE_COLS" \
  --out_dir "$DISC_ANALYSIS_DIR"

echo "[6/10] phase-2 unified discovery controller"
mkdir -p "$DISC_CONTROLLER_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_harm_controller.py \
  --discovery_table_csvs "$POPE_DISC_TABLE_DIR/table.csv" "$AMBER_DISC_TABLE_DIR/table.csv" \
  --out_dir "$DISC_CONTROLLER_DIR" \
  --min_feature_auroc "$MIN_FEATURE_AUROC" \
  --top_k "$TOP_K" \
  --max_baseline_rate "$MAX_BASELINE_RATE" \
  --feature_cols "$FEATURE_COLS"

if [[ "$REUSE_IF_EXISTS" != "true" || ! -f "$POPE_TEST_PROBE_DIR/probe_features.csv" ]]; then
  echo "[7/10] extract POPE held-out probe features"
  mkdir -p "$POPE_TEST_PROBE_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/extract_pnp_probe_features.py \
    --backend vga \
    --vga_root "$VGA_ROOT" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --image_folder "$POPE_IMAGE_FOLDER" \
    --question_file "$POPE_TEST_Q_FILE" \
    --out_dir "$POPE_TEST_PROBE_DIR" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --sampling "$SAMPLING" \
    --max_gen_len "$MAX_GEN_LEN" \
    --num_beams "$NUM_BEAMS" \
    --cd_alpha "$CD_ALPHA" \
    --attn_coef "$ATTN_COEF" \
    --start_layer "$START_LAYER" \
    --end_layer "$END_LAYER" \
    --head_balancing "$HEAD_BALANCING" \
    --attn_norm "$ATTN_NORM" \
    --late_start "$LATE_START" \
    --late_end "$LATE_END" \
    --probe_feature_mode "$PROBE_FEATURE_MODE" \
    --headset_json "$HEADSET_JSON" \
    --probe_position_mode "$PROBE_POSITION_MODE" \
    --probe_branch_source "$PROBE_BRANCH_SOURCE" \
    --branch_text_jsonl "$POPE_TEST_BASELINE_JSONL" \
    --probe_force_manual_fullseq "$PROBE_FORCE_MANUAL_FULLSEQ" \
    --probe_preview_max_new_tokens "$PROBE_PREVIEW_MAX_NEW_TOKENS" \
    --probe_preview_reuse_baseline "$PROBE_PREVIEW_REUSE_BASELINE" \
    --probe_preview_fallback_to_prompt_last "$PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST" \
    --seed "$SEED" \
    --max_samples "$MAX_SAMPLES"
else
  echo "[7/10] reuse POPE held-out probe features"
fi

echo "[8/10] build/apply POPE held-out gate"
mkdir -p "$POPE_TEST_TABLE_DIR" "$POPE_TEST_APPLY_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_table.py \
  --probe_features_csv "$POPE_TEST_PROBE_DIR/probe_features.csv" \
  --baseline_pred_jsonl "$POPE_TEST_BASELINE_JSONL" \
  --intervention_pred_jsonl "$POPE_TEST_INTERVENTION_JSONL" \
  --question_file "$POPE_TEST_Q_FILE" \
  --gt_csv "$POPE_TEST_GT_CSV" \
  --benchmark_name pope \
  --split_name heldout \
  --out_csv "$POPE_TEST_TABLE_DIR/table.csv" \
  --out_summary_json "$POPE_TEST_TABLE_DIR/summary.json"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_harm_controller.py \
  --table_csv "$POPE_TEST_TABLE_DIR/table.csv" \
  --policy_json "$DISC_CONTROLLER_DIR/selected_policy.json" \
  --out_dir "$POPE_TEST_APPLY_DIR"

if [[ "$REUSE_IF_EXISTS" != "true" || ! -f "$AMBER_TEST_PROBE_DIR/probe_features.csv" ]]; then
  echo "[9/10] extract AMBER held-out probe features"
  mkdir -p "$AMBER_TEST_PROBE_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/extract_pnp_probe_features.py \
    --backend vga \
    --vga_root "$VGA_ROOT" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --image_folder "$AMBER_IMAGE_FOLDER" \
    --question_file "$AMBER_TEST_Q_FILE" \
    --out_dir "$AMBER_TEST_PROBE_DIR" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --sampling "$SAMPLING" \
    --max_gen_len "$MAX_GEN_LEN" \
    --num_beams "$NUM_BEAMS" \
    --cd_alpha "$CD_ALPHA" \
    --attn_coef "$ATTN_COEF" \
    --start_layer "$START_LAYER" \
    --end_layer "$END_LAYER" \
    --head_balancing "$HEAD_BALANCING" \
    --attn_norm "$ATTN_NORM" \
    --late_start "$LATE_START" \
    --late_end "$LATE_END" \
    --probe_feature_mode "$PROBE_FEATURE_MODE" \
    --headset_json "$HEADSET_JSON" \
    --probe_position_mode "$PROBE_POSITION_MODE" \
    --probe_branch_source "$PROBE_BRANCH_SOURCE" \
    --branch_text_jsonl "$AMBER_BASELINE_JSONL" \
    --probe_force_manual_fullseq "$PROBE_FORCE_MANUAL_FULLSEQ" \
    --probe_preview_max_new_tokens "$PROBE_PREVIEW_MAX_NEW_TOKENS" \
    --probe_preview_reuse_baseline "$PROBE_PREVIEW_REUSE_BASELINE" \
    --probe_preview_fallback_to_prompt_last "$PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST" \
    --seed "$SEED" \
    --max_samples "$MAX_SAMPLES"
else
  echo "[9/10] reuse AMBER held-out probe features"
fi

echo "[10/10] build/apply AMBER held-out gate"
mkdir -p "$AMBER_TEST_TABLE_DIR" "$AMBER_TEST_APPLY_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_table.py \
  --probe_features_csv "$AMBER_TEST_PROBE_DIR/probe_features.csv" \
  --baseline_pred_jsonl "$AMBER_BASELINE_JSONL" \
  --intervention_pred_jsonl "$AMBER_INTERVENTION_JSONL" \
  --question_file "$AMBER_TEST_Q_FILE" \
  --gt_csv "$AMBER_TEST_GT_CSV" \
  --benchmark_name amber_discriminative \
  --split_name heldout \
  --out_csv "$AMBER_TEST_TABLE_DIR/table.csv" \
  --out_summary_json "$AMBER_TEST_TABLE_DIR/summary.json"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_harm_controller.py \
  --table_csv "$AMBER_TEST_TABLE_DIR/table.csv" \
  --policy_json "$DISC_CONTROLLER_DIR/selected_policy.json" \
  --out_dir "$AMBER_TEST_APPLY_DIR"

echo "[done] $DISC_ANALYSIS_DIR/summary.json"
echo "[done] $DISC_CONTROLLER_DIR/summary.json"
echo "[done] $POPE_TEST_APPLY_DIR/summary.json"
echo "[done] $AMBER_TEST_APPLY_DIR/summary.json"
