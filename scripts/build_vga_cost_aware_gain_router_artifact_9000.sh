#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
QUESTION_FILE="${QUESTION_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
TAXONOMY_CSV="${TAXONOMY_CSV:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/vga/taxonomy/per_case_compare.csv}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"

OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/pope_full_9000/vga_cost_aware_gain_router_artifact_9000}"
PROBE_OUT_DIR="${PROBE_OUT_DIR:-$OUT_DIR/probe_features}"
ROUTER_OUT_DIR="${ROUTER_OUT_DIR:-$OUT_DIR/router}"

CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
TEMPERATURE="${TEMPERATURE:-0}"
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
PROBE_POSITION_MODE="${PROBE_POSITION_MODE:-baseline_yesno_preview}"
PROBE_BRANCH_SOURCE="${PROBE_BRANCH_SOURCE:-preview}"
PROBE_FORCE_MANUAL_FULLSEQ="${PROBE_FORCE_MANUAL_FULLSEQ:-false}"
PROBE_PREVIEW_MAX_NEW_TOKENS="${PROBE_PREVIEW_MAX_NEW_TOKENS:-3}"
PROBE_PREVIEW_REUSE_BASELINE="${PROBE_PREVIEW_REUSE_BASELINE:-true}"
PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST="${PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST:-true}"

TAU="${TAU:--0.0068411549792573}"
FEATURE_VARIANT="${FEATURE_VARIANT:-no_abs}"
DEPLOYMENT_BUDGET="${DEPLOYMENT_BUDGET:-0.30}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

mkdir -p "$PROBE_OUT_DIR" "$ROUTER_OUT_DIR"
cd "$CAL_ROOT"

python scripts/extract_pnp_probe_features.py \
  --backend vga \
  --vga_root "$VGA_ROOT" \
  --model_path "$MODEL_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --question_file "$QUESTION_FILE" \
  --out_dir "$PROBE_OUT_DIR" \
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
  --probe_force_manual_fullseq "$PROBE_FORCE_MANUAL_FULLSEQ" \
  --probe_preview_max_new_tokens "$PROBE_PREVIEW_MAX_NEW_TOKENS" \
  --probe_preview_reuse_baseline "$PROBE_PREVIEW_REUSE_BASELINE" \
  --probe_preview_fallback_to_prompt_last "$PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST" \
  --seed "$SEED" \
  --max_samples "$MAX_SAMPLES"

python scripts/build_cost_aware_gain_router.py \
  --probe_log_csv "$PROBE_OUT_DIR/probe_features.csv" \
  --taxonomy_csv "$TAXONOMY_CSV" \
  --out_dir "$ROUTER_OUT_DIR" \
  --tau "$TAU" \
  --backend_name vga \
  --feature_variant "$FEATURE_VARIANT" \
  --deployment_budget "$DEPLOYMENT_BUDGET" \
  --seed "$SEED" \
  --save_router_artifact

echo "[done] $OUT_DIR"
echo "[saved] $PROBE_OUT_DIR/probe_features.csv"
echo "[saved] $ROUTER_OUT_DIR/router_model.pkl"
echo "[saved] $ROUTER_OUT_DIR/router_metadata.json"
