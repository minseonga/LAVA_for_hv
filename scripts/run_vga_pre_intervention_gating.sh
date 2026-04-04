#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
AMBER_ROOT="${AMBER_ROOT:-/home/kms/data/AMBER}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DISCOVERY_Q_FILE="${DISCOVERY_Q_FILE:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_mix_train2014_2785/assets/discovery_q_with_object.jsonl}"
DISCOVERY_GT_CSV="${DISCOVERY_GT_CSV:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_gt.csv}"
DISCOVERY_BASELINE_JSONL="${DISCOVERY_BASELINE_JSONL:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_mix_train2014_2785/baseline/pred_baseline.jsonl}"
DISCOVERY_TAXONOMY_CSV="${DISCOVERY_TAXONOMY_CSV:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/taxonomy/per_case_compare.csv}"
DISCOVERY_PROBE_CSV="${DISCOVERY_PROBE_CSV:-$CAL_ROOT/experiments/tau_c_calibration_mix_train2014_2785/probe_features_online/probe_features.csv}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"

POPE_Q_FILE="${POPE_Q_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
POPE_GT_CSV="${POPE_GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
POPE_BASELINE_JSONL="${POPE_BASELINE_JSONL:-$CAL_ROOT/experiments/paper_main_b_c_v1_full/test_stageb/pred_baseline.jsonl}"
POPE_INTERVENTION_JSONL="${POPE_INTERVENTION_JSONL:-$CAL_ROOT/experiments/paper_main_b_c_v1_full/test_stageb/pred_vga.jsonl}"

AMBER_ASSET_DIR="${AMBER_ASSET_DIR:-$CAL_ROOT/experiments/amber_fixed_transfer_assets}"
AMBER_DISC_Q_FILE="${AMBER_DISC_Q_FILE:-$AMBER_ASSET_DIR/discriminative/assets/amber_discriminative_q_with_object.jsonl}"
AMBER_BASELINE_JSONL="${AMBER_BASELINE_JSONL:-$CAL_ROOT/experiments/amber_fixed_transfer_from_pope/discriminative/pred_baseline.jsonl}"
AMBER_INTERVENTION_JSONL="${AMBER_INTERVENTION_JSONL:-$CAL_ROOT/experiments/amber_fixed_transfer_from_pope/discriminative/pred_vga.jsonl}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_pre_intervention_gating}"
DISCOVERY_ROUTER_DIR="${DISCOVERY_ROUTER_DIR:-$OUT_ROOT/discovery/router}"
POPE_PROBE_DIR="${POPE_PROBE_DIR:-$OUT_ROOT/pope_test/probe}"
POPE_APPLY_DIR="${POPE_APPLY_DIR:-$OUT_ROOT/pope_test/apply}"
AMBER_PROBE_DIR="${AMBER_PROBE_DIR:-$OUT_ROOT/amber_discriminative/probe}"
AMBER_APPLY_DIR="${AMBER_APPLY_DIR:-$OUT_ROOT/amber_discriminative/apply}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
IMAGE_FOLDER_POPE="${IMAGE_FOLDER_POPE:-/home/kms/data/pope/val2014}"
IMAGE_FOLDER_AMBER="${IMAGE_FOLDER_AMBER:-$AMBER_ROOT/image}"
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
DEPLOYMENT_BUDGET="${DEPLOYMENT_BUDGET:-0.30}"
FEATURE_VARIANT="${FEATURE_VARIANT:-no_abs}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

mkdir -p "$OUT_ROOT"
cd "$ROOT_DIR"

if [[ ! -f "$AMBER_DISC_Q_FILE" ]]; then
  echo "[prepare] AMBER fixed-transfer assets"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/prepare_amber_fixed_transfer_assets.py \
    --amber_root "$AMBER_ROOT" \
    --out_dir "$AMBER_ASSET_DIR"
fi

if [[ "$REUSE_IF_EXISTS" != "true" || ! -f "$DISCOVERY_ROUTER_DIR/router_metadata.json" ]]; then
  echo "[1/5] build POPE-discovery pre-intervention router"
  mkdir -p "$DISCOVERY_ROUTER_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_cost_aware_gain_router.py \
    --probe_log_csv "$DISCOVERY_PROBE_CSV" \
    --taxonomy_csv "$DISCOVERY_TAXONOMY_CSV" \
    --out_dir "$DISCOVERY_ROUTER_DIR" \
    --tau -0.0068411549792573 \
    --backend_name vga \
    --seed "$SEED" \
    --deployment_budget "$DEPLOYMENT_BUDGET" \
    --feature_variant "$FEATURE_VARIANT" \
    --save_router_artifact
else
  echo "[1/5] reuse discovery router -> $DISCOVERY_ROUTER_DIR"
fi

if [[ "$REUSE_IF_EXISTS" != "true" || ! -f "$POPE_PROBE_DIR/probe_features.csv" ]]; then
  echo "[2/5] extract POPE held-out pre-intervention probe features"
  mkdir -p "$POPE_PROBE_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/extract_pnp_probe_features.py \
    --backend vga \
    --vga_root "$VGA_ROOT" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --image_folder "$IMAGE_FOLDER_POPE" \
    --question_file "$POPE_Q_FILE" \
    --out_dir "$POPE_PROBE_DIR" \
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
    --branch_text_jsonl "$POPE_BASELINE_JSONL" \
    --probe_force_manual_fullseq "$PROBE_FORCE_MANUAL_FULLSEQ" \
    --probe_preview_max_new_tokens "$PROBE_PREVIEW_MAX_NEW_TOKENS" \
    --probe_preview_reuse_baseline "$PROBE_PREVIEW_REUSE_BASELINE" \
    --probe_preview_fallback_to_prompt_last "$PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST" \
    --seed "$SEED" \
    --max_samples "$MAX_SAMPLES"
else
  echo "[2/5] reuse POPE held-out probe features -> $POPE_PROBE_DIR/probe_features.csv"
fi

echo "[3/5] apply pre-intervention router on POPE held-out"
mkdir -p "$POPE_APPLY_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pre_intervention_router.py \
  --router_dir "$DISCOVERY_ROUTER_DIR" \
  --probe_features_csv "$POPE_PROBE_DIR/probe_features.csv" \
  --baseline_pred_jsonl "$POPE_BASELINE_JSONL" \
  --intervention_pred_jsonl "$POPE_INTERVENTION_JSONL" \
  --gt_csv "$POPE_GT_CSV" \
  --benchmark_name pope \
  --out_dir "$POPE_APPLY_DIR"

if [[ "$REUSE_IF_EXISTS" != "true" || ! -f "$AMBER_PROBE_DIR/probe_features.csv" ]]; then
  echo "[4/5] extract AMBER-discriminative pre-intervention probe features"
  mkdir -p "$AMBER_PROBE_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/extract_pnp_probe_features.py \
    --backend vga \
    --vga_root "$VGA_ROOT" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --image_folder "$IMAGE_FOLDER_AMBER" \
    --question_file "$AMBER_DISC_Q_FILE" \
    --out_dir "$AMBER_PROBE_DIR" \
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
  echo "[4/5] reuse AMBER-discriminative probe features -> $AMBER_PROBE_DIR/probe_features.csv"
fi

echo "[5/5] apply pre-intervention router on AMBER-discriminative"
mkdir -p "$AMBER_APPLY_DIR"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pre_intervention_router.py \
  --router_dir "$DISCOVERY_ROUTER_DIR" \
  --probe_features_csv "$AMBER_PROBE_DIR/probe_features.csv" \
  --baseline_pred_jsonl "$AMBER_BASELINE_JSONL" \
  --intervention_pred_jsonl "$AMBER_INTERVENTION_JSONL" \
  --amber_root "$AMBER_ROOT" \
  --benchmark_name amber_discriminative \
  --out_dir "$AMBER_APPLY_DIR"

echo "[done] $POPE_APPLY_DIR/summary.json"
echo "[done] $AMBER_APPLY_DIR/summary.json"
