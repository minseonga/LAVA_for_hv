#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
AMBER_ROOT="${AMBER_ROOT:-/home/kms/data/AMBER}"
AMBER_IMAGE_FOLDER="${AMBER_IMAGE_FOLDER:-$AMBER_ROOT/image}"
AMBER_QUERY_JSON="${AMBER_QUERY_JSON:-$AMBER_ROOT/data/query/query_discriminative.json}"
AMBER_ANNOTATIONS_JSON="${AMBER_ANNOTATIONS_JSON:-$AMBER_ROOT/data/annotations.json}"

ASSET_ROOT="${ASSET_ROOT:-$CAL_ROOT/experiments/amber_discriminative_split}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/amber_discriminative_b_c_v1_full}"
ALIGNED_OUT_ROOT="${ALIGNED_OUT_ROOT:-$CAL_ROOT/experiments/aligned_cheap_proxy_amber}"

DISCOVERY_RATIO="${DISCOVERY_RATIO:-0.2}"
SEED="${SEED:-42}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"

MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
USE_ADD="${USE_ADD:-true}"
ATTN_COEF="${ATTN_COEF:-0.2}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
SAMPLING="${SAMPLING:-false}"
CD_ALPHA="${CD_ALPHA:-0.02}"
START_LAYER="${START_LAYER:-2}"
END_LAYER="${END_LAYER:-15}"

BETA="${BETA:-1.0}"
LAMBDA_A="${LAMBDA_A:-0.5}"
LAMBDA_B="${LAMBDA_B:-0.5}"
BLUR_RADIUS="${BLUR_RADIUS:-12.0}"
REUSE_SCORES="${REUSE_SCORES:-false}"
REUSE_CHEAP="${REUSE_CHEAP:-false}"
MAKE_PLOTS="${MAKE_PLOTS:-false}"

SUBSET_PERCENTS="${SUBSET_PERCENTS:-1,2,5}"
FEATURE_COLS="${FEATURE_COLS:-cheap_lp_content_min,cheap_lp_content_tail_gap,cheap_lp_content_tail_z,cheap_lp_content_q10,cheap_lp_content_min_len_corr,cheap_target_gap_content_min,cheap_lp_content_std,cheap_entropy_content_mean,cheap_margin_content_mean,cheap_target_gap_content_std,cheap_conflict_lp_minus_entropy}"
MAX_RESCUE_RATE="${MAX_RESCUE_RATE:-0.03}"
LOG_EVERY="${LOG_EVERY:-25}"
RUN_ALIGNED_PROXY="${RUN_ALIGNED_PROXY:-true}"

DISCOVERY_ASSETS="$ASSET_ROOT/discovery/assets"
TEST_ASSETS="$ASSET_ROOT/test/assets"

DISCOVERY_STAGEB_OUT="${OUT_ROOT}/discovery_stageb"
TEST_STAGEB_OUT="${OUT_ROOT}/test_stageb"
DISCOVERY_TAX_DIR="${OUT_ROOT}/discovery/taxonomy"
TEST_TAX_DIR="${OUT_ROOT}/test/taxonomy"

mkdir -p "$ASSET_ROOT" "$OUT_ROOT"

if [[ ! -d "$AMBER_ROOT" ]]; then
  echo "[error] AMBER_ROOT does not exist: $AMBER_ROOT" >&2
  exit 1
fi

if [[ ! -d "$AMBER_IMAGE_FOLDER" ]]; then
  echo "[error] AMBER_IMAGE_FOLDER does not exist: $AMBER_IMAGE_FOLDER" >&2
  exit 1
fi

if [[ ! -d "$VGA_ROOT" ]]; then
  echo "[error] VGA_ROOT does not exist: $VGA_ROOT" >&2
  echo "[hint] This wrapper requires the external VGA_origin checkout on the machine where it runs." >&2
  exit 1
fi

echo "[1/6] prepare AMBER discriminative assets"
PYTHONPATH="$ROOT_DIR" \
python scripts/prepare_amber_discriminative_assets.py \
  --amber_root "$AMBER_ROOT" \
  --query_json "$AMBER_QUERY_JSON" \
  --annotations_json "$AMBER_ANNOTATIONS_JSON" \
  --out_dir "$ASSET_ROOT" \
  --discovery_ratio "$DISCOVERY_RATIO" \
  --seed "$SEED"

echo "[2/6] discovery Stage-B run on AMBER"
CAL_ROOT="$CAL_ROOT" \
VGA_ROOT="$VGA_ROOT" \
GPU="$GPU" \
DEVICE="$DEVICE" \
MODEL_PATH="$MODEL_PATH" \
CONV_MODE="$CONV_MODE" \
IMAGE_FOLDER="$AMBER_IMAGE_FOLDER" \
QUESTION_FILE="$DISCOVERY_ASSETS/discovery_q_with_object.jsonl" \
BASELINE_QUESTION_FILE="$DISCOVERY_ASSETS/discovery_q.jsonl" \
GT_CSV="$DISCOVERY_ASSETS/discovery_gt.csv" \
HEADSET_JSON="$HEADSET_JSON" \
OUT_DIR="$DISCOVERY_STAGEB_OUT" \
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

echo "[3/6] discovery taxonomy from fresh AMBER preds"
PYTHONPATH="$ROOT_DIR" \
python scripts/build_vga_failure_taxonomy.py \
  --gt_csv "$DISCOVERY_ASSETS/discovery_gt.csv" \
  --baseline_pred_jsonl "$DISCOVERY_STAGEB_OUT/pred_baseline.jsonl" \
  --vga_pred_jsonl "$DISCOVERY_STAGEB_OUT/pred_vga.jsonl" \
  --baseline_pred_text_key text \
  --vga_pred_text_key output \
  --out_dir "$DISCOVERY_TAX_DIR"

echo "[4/6] held-out Stage-B run on AMBER"
CAL_ROOT="$CAL_ROOT" \
VGA_ROOT="$VGA_ROOT" \
GPU="$GPU" \
DEVICE="$DEVICE" \
MODEL_PATH="$MODEL_PATH" \
CONV_MODE="$CONV_MODE" \
IMAGE_FOLDER="$AMBER_IMAGE_FOLDER" \
QUESTION_FILE="$TEST_ASSETS/test_q_with_object.jsonl" \
BASELINE_QUESTION_FILE="$TEST_ASSETS/test_q.jsonl" \
GT_CSV="$TEST_ASSETS/test_gt.csv" \
HEADSET_JSON="$HEADSET_JSON" \
OUT_DIR="$TEST_STAGEB_OUT" \
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

echo "[5/6] held-out taxonomy from fresh AMBER preds"
PYTHONPATH="$ROOT_DIR" \
python scripts/build_vga_failure_taxonomy.py \
  --gt_csv "$TEST_ASSETS/test_gt.csv" \
  --baseline_pred_jsonl "$TEST_STAGEB_OUT/pred_baseline.jsonl" \
  --vga_pred_jsonl "$TEST_STAGEB_OUT/pred_vga.jsonl" \
  --baseline_pred_text_key text \
  --vga_pred_text_key output \
  --out_dir "$TEST_TAX_DIR"

echo "[6/6] discovery calibration -> held-out fixed eval on AMBER"
GPU="$GPU" \
DEVICE="$DEVICE" \
MODEL_PATH="$MODEL_PATH" \
MODEL_BASE="$MODEL_BASE" \
CONV_MODE="$CONV_MODE" \
DISCOVERY_IMAGE_FOLDER="$AMBER_IMAGE_FOLDER" \
DISCOVERY_QUESTION_FILE="$DISCOVERY_ASSETS/discovery_q_with_object.jsonl" \
DISCOVERY_INTERVENTION_PRED_JSONL="$DISCOVERY_STAGEB_OUT/pred_vga.jsonl" \
DISCOVERY_SCORES_CSV="$DISCOVERY_STAGEB_OUT/sample_scores.csv" \
DISCOVERY_TAXONOMY_CSV="$DISCOVERY_TAX_DIR/per_case_compare.csv" \
TEST_IMAGE_FOLDER="$AMBER_IMAGE_FOLDER" \
TEST_QUESTION_FILE="$TEST_ASSETS/test_q_with_object.jsonl" \
TEST_INTERVENTION_PRED_JSONL="$TEST_STAGEB_OUT/pred_vga.jsonl" \
TEST_SCORES_CSV="$TEST_STAGEB_OUT/sample_scores.csv" \
TEST_TAXONOMY_CSV="$TEST_TAX_DIR/per_case_compare.csv" \
OUT_ROOT="$OUT_ROOT" \
SUBSET_PERCENTS="$SUBSET_PERCENTS" \
FEATURE_COLS="$FEATURE_COLS" \
MAX_RESCUE_RATE="$MAX_RESCUE_RATE" \
REUSE_CHEAP="$REUSE_CHEAP" \
LOG_EVERY="$LOG_EVERY" \
bash scripts/run_main_b_c_fixed_policy_vga.sh

if [[ "$RUN_ALIGNED_PROXY" == "true" ]]; then
  echo "[7/7] aligned cheap proxy on AMBER canonical VGA outputs"
  REFERENCE_ROOT="$OUT_ROOT" \
  OUT_ROOT="$ALIGNED_OUT_ROOT" \
  DISCOVERY_GT_CSV="$DISCOVERY_ASSETS/discovery_gt.csv" \
  TEST_GT_CSV="$TEST_ASSETS/test_gt.csv" \
  DISCOVERY_BASELINE_PRED_JSONL="$DISCOVERY_STAGEB_OUT/pred_baseline.jsonl" \
  DISCOVERY_VGA_PRED_JSONL="$DISCOVERY_STAGEB_OUT/pred_vga.jsonl" \
  TEST_BASELINE_PRED_JSONL="$TEST_STAGEB_OUT/pred_baseline.jsonl" \
  TEST_VGA_PRED_JSONL="$TEST_STAGEB_OUT/pred_vga.jsonl" \
  DISCOVERY_FEATURES_CSV="$OUT_ROOT/discovery/cheap_online_features.csv" \
  TEST_FEATURES_CSV="$OUT_ROOT/test/cheap_online_features.csv" \
  FEATURE_COLS="$FEATURE_COLS" \
  MAX_RESCUE_RATE="$MAX_RESCUE_RATE" \
  TARGET_LABEL="actual_rescue" \
  bash scripts/run_aligned_cheap_proxy_from_reference_vga.sh
fi

echo "[done] reference root -> $OUT_ROOT"
if [[ "$RUN_ALIGNED_PROXY" == "true" ]]; then
  echo "[done] aligned cheap proxy -> $ALIGNED_OUT_ROOT"
fi
