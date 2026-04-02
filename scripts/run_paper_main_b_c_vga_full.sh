#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"

DISCOVERY_IMAGE_FOLDER="${DISCOVERY_IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
DISCOVERY_QUESTION_FILE="${DISCOVERY_QUESTION_FILE:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q_with_object.jsonl}"
DISCOVERY_BASELINE_QUESTION_FILE="${DISCOVERY_BASELINE_QUESTION_FILE:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q.jsonl}"
DISCOVERY_GT_CSV="${DISCOVERY_GT_CSV:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_gt.csv}"
DISCOVERY_TAXONOMY_CSV="${DISCOVERY_TAXONOMY_CSV:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/taxonomy/per_case_compare.csv}"

TEST_IMAGE_FOLDER="${TEST_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
TEST_QUESTION_FILE="${TEST_QUESTION_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
TEST_BASELINE_QUESTION_FILE="${TEST_BASELINE_QUESTION_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q.jsonl}"
TEST_GT_CSV="${TEST_GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
TEST_TAXONOMY_CSV="${TEST_TAXONOMY_CSV:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/vga/taxonomy/per_case_compare.csv}"

HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/paper_main_b_c_v1_full}"
DISCOVERY_STAGEB_OUT="${DISCOVERY_STAGEB_OUT:-$OUT_ROOT/discovery_stageb}"
TEST_STAGEB_OUT="${TEST_STAGEB_OUT:-$OUT_ROOT/test_stageb}"

MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
USE_ADD="${USE_ADD:-true}"
ATTN_COEF="${ATTN_COEF:-0.2}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
SAMPLING="${SAMPLING:-false}"
CD_ALPHA="${CD_ALPHA:-0.02}"
START_LAYER="${START_LAYER:-2}"
END_LAYER="${END_LAYER:-15}"
SEED="${SEED:-42}"

BETA="${BETA:-1.0}"
LAMBDA_A="${LAMBDA_A:-0.5}"
LAMBDA_B="${LAMBDA_B:-0.5}"
BLUR_RADIUS="${BLUR_RADIUS:-12.0}"
REUSE_SCORES="${REUSE_SCORES:-false}"
MAKE_PLOTS="${MAKE_PLOTS:-false}"

SUBSET_PERCENTS="${SUBSET_PERCENTS:-1,2,5}"
FEATURE_COLS="${FEATURE_COLS:-cheap_lp_content_min,cheap_target_gap_content_min,cheap_lp_content_std,cheap_entropy_content_mean,cheap_margin_content_mean,cheap_target_gap_content_std,cheap_conflict_lp_minus_entropy}"
MAX_RESCUE_RATE="${MAX_RESCUE_RATE:-0.03}"
REUSE_CHEAP="${REUSE_CHEAP:-false}"
LOG_EVERY="${LOG_EVERY:-25}"

mkdir -p "$OUT_ROOT"

echo "[1/3] fresh discovery Stage-B run"
CAL_ROOT="$CAL_ROOT" \
VGA_ROOT="$VGA_ROOT" \
GPU="$GPU" \
DEVICE="$DEVICE" \
MODEL_PATH="$MODEL_PATH" \
IMAGE_FOLDER="$DISCOVERY_IMAGE_FOLDER" \
QUESTION_FILE="$DISCOVERY_QUESTION_FILE" \
BASELINE_QUESTION_FILE="$DISCOVERY_BASELINE_QUESTION_FILE" \
GT_CSV="$DISCOVERY_GT_CSV" \
HEADSET_JSON="$HEADSET_JSON" \
OUT_DIR="$DISCOVERY_STAGEB_OUT" \
CONV_MODE="$CONV_MODE" \
MAX_GEN_LEN="$MAX_GEN_LEN" \
USE_ADD="$USE_ADD" \
ATTN_COEF="$ATTN_COEF" \
HEAD_BALANCING="$HEAD_BALANCING" \
SAMPLING="$SAMPLING" \
CD_ALPHA="$CD_ALPHA" \
START_LAYER="$START_LAYER" \
END_LAYER="$END_LAYER" \
SEED="$SEED" \
BETA="$BETA" \
LAMBDA_A="$LAMBDA_A" \
LAMBDA_B="$LAMBDA_B" \
BLUR_RADIUS="$BLUR_RADIUS" \
REUSE_SCORES="$REUSE_SCORES" \
MAKE_PLOTS="$MAKE_PLOTS" \
bash scripts/run_stage_b_signal_validation_vga.sh

echo "[2/3] fresh held-out Stage-B run"
CAL_ROOT="$CAL_ROOT" \
VGA_ROOT="$VGA_ROOT" \
GPU="$GPU" \
DEVICE="$DEVICE" \
MODEL_PATH="$MODEL_PATH" \
IMAGE_FOLDER="$TEST_IMAGE_FOLDER" \
QUESTION_FILE="$TEST_QUESTION_FILE" \
BASELINE_QUESTION_FILE="$TEST_BASELINE_QUESTION_FILE" \
GT_CSV="$TEST_GT_CSV" \
HEADSET_JSON="$HEADSET_JSON" \
OUT_DIR="$TEST_STAGEB_OUT" \
CONV_MODE="$CONV_MODE" \
MAX_GEN_LEN="$MAX_GEN_LEN" \
USE_ADD="$USE_ADD" \
ATTN_COEF="$ATTN_COEF" \
HEAD_BALANCING="$HEAD_BALANCING" \
SAMPLING="$SAMPLING" \
CD_ALPHA="$CD_ALPHA" \
START_LAYER="$START_LAYER" \
END_LAYER="$END_LAYER" \
SEED="$SEED" \
BETA="$BETA" \
LAMBDA_A="$LAMBDA_A" \
LAMBDA_B="$LAMBDA_B" \
BLUR_RADIUS="$BLUR_RADIUS" \
REUSE_SCORES="$REUSE_SCORES" \
MAKE_PLOTS="$MAKE_PLOTS" \
bash scripts/run_stage_b_signal_validation_vga.sh

echo "[3/3] discovery calibration -> held-out fixed evaluation"
GPU="$GPU" \
DEVICE="$DEVICE" \
MODEL_PATH="$MODEL_PATH" \
MODEL_BASE="$MODEL_BASE" \
CONV_MODE="$CONV_MODE" \
DISCOVERY_IMAGE_FOLDER="$DISCOVERY_IMAGE_FOLDER" \
DISCOVERY_QUESTION_FILE="$DISCOVERY_QUESTION_FILE" \
DISCOVERY_INTERVENTION_PRED_JSONL="$DISCOVERY_STAGEB_OUT/pred_vga.jsonl" \
DISCOVERY_SCORES_CSV="$DISCOVERY_STAGEB_OUT/sample_scores.csv" \
DISCOVERY_TAXONOMY_CSV="$DISCOVERY_TAXONOMY_CSV" \
TEST_IMAGE_FOLDER="$TEST_IMAGE_FOLDER" \
TEST_QUESTION_FILE="$TEST_QUESTION_FILE" \
TEST_INTERVENTION_PRED_JSONL="$TEST_STAGEB_OUT/pred_vga.jsonl" \
TEST_SCORES_CSV="$TEST_STAGEB_OUT/sample_scores.csv" \
TEST_TAXONOMY_CSV="$TEST_TAXONOMY_CSV" \
OUT_ROOT="$OUT_ROOT" \
SUBSET_PERCENTS="$SUBSET_PERCENTS" \
FEATURE_COLS="$FEATURE_COLS" \
MAX_RESCUE_RATE="$MAX_RESCUE_RATE" \
REUSE_CHEAP="$REUSE_CHEAP" \
LOG_EVERY="$LOG_EVERY" \
bash scripts/run_main_b_c_fixed_policy_vga.sh

echo "[done] full paper protocol complete -> $OUT_ROOT"
