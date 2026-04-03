#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"

DISCOVERY_IMAGE_FOLDER="${DISCOVERY_IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
DISCOVERY_QUESTION_FILE="${DISCOVERY_QUESTION_FILE:-$ROOT_DIR/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q_with_object.jsonl}"
DISCOVERY_REF_PRED_JSONL="${DISCOVERY_REF_PRED_JSONL:-$ROOT_DIR/experiments/pope_discovery/stage_b_signal_validation_vga/pred_vga.jsonl}"

TEST_IMAGE_FOLDER="${TEST_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
TEST_QUESTION_FILE="${TEST_QUESTION_FILE:-$ROOT_DIR/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
TEST_REF_PRED_JSONL="${TEST_REF_PRED_JSONL:-$ROOT_DIR/experiments/pope_full_9000/stage_b_signal_validation_vga/pred_vga.jsonl}"

HEADSET_JSON="${HEADSET_JSON:-$ROOT_DIR/experiments/pope_discovery/discovery_headset.json}"

REFERENCE_ROOT="${REFERENCE_ROOT:-$ROOT_DIR/experiments/main_b_c_fixed_policy_vga}"
REFERENCE_POLICY_JSON="${REFERENCE_POLICY_JSON:-$REFERENCE_ROOT/discovery/calibration/selected_policy.json}"

DISCOVERY_REF_SCORES_CSV="${DISCOVERY_REF_SCORES_CSV:-$ROOT_DIR/experiments/pope_discovery/stage_b_signal_validation_vga/sample_scores.csv}"
DISCOVERY_REF_TAXONOMY_CSV="${DISCOVERY_REF_TAXONOMY_CSV:-$ROOT_DIR/experiments/pope_discovery/tau_c_calibration_adversarial/taxonomy/per_case_compare.csv}"
DISCOVERY_REF_FEATURES_CSV="${DISCOVERY_REF_FEATURES_CSV:-$REFERENCE_ROOT/discovery/cheap_online_features.csv}"
DISCOVERY_REF_DECISION_DIR="${DISCOVERY_REF_DECISION_DIR:-$REFERENCE_ROOT/discovery/fixed_eval}"
DISCOVERY_REF_DECISIONS_CSV="${DISCOVERY_REF_DECISIONS_CSV:-$DISCOVERY_REF_DECISION_DIR/decision_rows.csv}"

TEST_REF_SCORES_CSV="${TEST_REF_SCORES_CSV:-$ROOT_DIR/experiments/pope_full_9000/stage_b_signal_validation_vga/sample_scores.csv}"
TEST_REF_TAXONOMY_CSV="${TEST_REF_TAXONOMY_CSV:-$ROOT_DIR/experiments/pope_full_9000/all_models_full_strict/vga/taxonomy/per_case_compare.csv}"
TEST_REF_FEATURES_CSV="${TEST_REF_FEATURES_CSV:-$REFERENCE_ROOT/test/cheap_online_features.csv}"
TEST_REF_DECISION_DIR="${TEST_REF_DECISION_DIR:-$REFERENCE_ROOT/test/fixed_eval}"
TEST_REF_DECISIONS_CSV="${TEST_REF_DECISIONS_CSV:-$TEST_REF_DECISION_DIR/decision_rows.csv}"

OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/experiments/decode_time_proxy_vga}"
DISCOVERY_OUT_DIR="${DISCOVERY_OUT_DIR:-$OUT_ROOT/discovery}"
TEST_OUT_DIR="${TEST_OUT_DIR:-$OUT_ROOT/test}"

FEATURE_COLS="${FEATURE_COLS:-proxy_lp_content_mean,proxy_lp_content_std,proxy_lp_content_min,proxy_lp_content_lastk_mean,proxy_lp_content_lastk_std,proxy_margin_content_mean,proxy_margin_content_min,proxy_margin_content_lastk_mean,proxy_margin_content_lastk_std,proxy_entropy_content_mean,proxy_entropy_content_std,proxy_low_margin_ratio_content}"
PAIR_FEATURE_TOPN="${PAIR_FEATURE_TOPN:-6}"
MAX_RESCUE_RATE="${MAX_RESCUE_RATE:-0.03}"

MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
SAMPLING="${SAMPLING:-false}"
NUM_BEAMS="${NUM_BEAMS:-1}"
CD_ALPHA="${CD_ALPHA:-0.02}"
ATTN_COEF="${ATTN_COEF:-0.2}"
START_LAYER="${START_LAYER:-16}"
END_LAYER="${END_LAYER:-24}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
ATTN_NORM="${ATTN_NORM:-false}"
PROXY_TRACE_LATE_START="${PROXY_TRACE_LATE_START:-16}"
PROXY_TRACE_LATE_END="${PROXY_TRACE_LATE_END:-24}"
PROXY_TRACE_LAST_K="${PROXY_TRACE_LAST_K:-8}"
PROXY_TRACE_MARGIN_LOW="${PROXY_TRACE_MARGIN_LOW:-1.0}"
PREFER_LOCAL_LLAVA="${PREFER_LOCAL_LLAVA:-false}"
PROXY_TRACE_ENABLED="${PROXY_TRACE_ENABLED:-false}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
LOG_EVERY="${LOG_EVERY:-25}"

mkdir -p "$DISCOVERY_OUT_DIR" "$TEST_OUT_DIR"

if [[ ! -f "$DISCOVERY_REF_DECISIONS_CSV" ]]; then
  echo "[prep] materialize discovery reference decisions"
  PYTHONPATH="$ROOT_DIR" \
  python scripts/run_b_c_fixed_policy.py apply \
    --scores_csv "$DISCOVERY_REF_SCORES_CSV" \
    --taxonomy_csv "$DISCOVERY_REF_TAXONOMY_CSV" \
    --features_csv "$DISCOVERY_REF_FEATURES_CSV" \
    --policy_json "$REFERENCE_POLICY_JSON" \
    --out_dir "$DISCOVERY_REF_DECISION_DIR"
fi

if [[ ! -f "$TEST_REF_DECISIONS_CSV" ]]; then
  echo "[prep] materialize held-out reference decisions"
  PYTHONPATH="$ROOT_DIR" \
  python scripts/run_b_c_fixed_policy.py apply \
    --scores_csv "$TEST_REF_SCORES_CSV" \
    --taxonomy_csv "$TEST_REF_TAXONOMY_CSV" \
    --features_csv "$TEST_REF_FEATURES_CSV" \
    --policy_json "$REFERENCE_POLICY_JSON" \
    --out_dir "$TEST_REF_DECISION_DIR"
fi

echo "[1/4] extract decode-time proxy features on discovery"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT_DIR" \
python scripts/extract_decode_time_proxy_vga.py \
  --vga_root "$VGA_ROOT" \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --image_folder "$DISCOVERY_IMAGE_FOLDER" \
  --question_file "$DISCOVERY_QUESTION_FILE" \
  --out_dir "$DISCOVERY_OUT_DIR" \
  --headset_json "$HEADSET_JSON" \
  --reference_pred_jsonl "$DISCOVERY_REF_PRED_JSONL" \
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
  --proxy_trace_late_start "$PROXY_TRACE_LATE_START" \
  --proxy_trace_late_end "$PROXY_TRACE_LATE_END" \
  --proxy_trace_last_k "$PROXY_TRACE_LAST_K" \
  --proxy_trace_margin_low "$PROXY_TRACE_MARGIN_LOW" \
  --prefer_local_llava "$PREFER_LOCAL_LLAVA" \
  --proxy_trace_enabled "$PROXY_TRACE_ENABLED" \
  --seed "$SEED" \
  --max_samples "$MAX_SAMPLES" \
  --log_every "$LOG_EVERY"

echo "[2/4] calibrate decode-time proxy policy on discovery"
PYTHONPATH="$ROOT_DIR" \
python scripts/run_decode_time_proxy_policy.py calibrate \
  --features_csv "$DISCOVERY_OUT_DIR/decode_time_proxy_features.csv" \
  --reference_decisions_csv "$DISCOVERY_REF_DECISIONS_CSV" \
  --feature_cols "$FEATURE_COLS" \
  --pair_feature_topn "$PAIR_FEATURE_TOPN" \
  --max_rescue_rate "$MAX_RESCUE_RATE" \
  --out_dir "$DISCOVERY_OUT_DIR/calibration"

echo "[3/4] extract decode-time proxy features on held-out test"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT_DIR" \
python scripts/extract_decode_time_proxy_vga.py \
  --vga_root "$VGA_ROOT" \
  --model_path "$MODEL_PATH" \
  --model_base "$MODEL_BASE" \
  --image_folder "$TEST_IMAGE_FOLDER" \
  --question_file "$TEST_QUESTION_FILE" \
  --out_dir "$TEST_OUT_DIR" \
  --headset_json "$HEADSET_JSON" \
  --reference_pred_jsonl "$TEST_REF_PRED_JSONL" \
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
  --proxy_trace_late_start "$PROXY_TRACE_LATE_START" \
  --proxy_trace_late_end "$PROXY_TRACE_LATE_END" \
  --proxy_trace_last_k "$PROXY_TRACE_LAST_K" \
  --proxy_trace_margin_low "$PROXY_TRACE_MARGIN_LOW" \
  --prefer_local_llava "$PREFER_LOCAL_LLAVA" \
  --proxy_trace_enabled "$PROXY_TRACE_ENABLED" \
  --seed "$SEED" \
  --max_samples "$MAX_SAMPLES" \
  --log_every "$LOG_EVERY"

echo "[4/4] apply decode-time proxy policy on held-out test"
PYTHONPATH="$ROOT_DIR" \
python scripts/run_decode_time_proxy_policy.py apply \
  --features_csv "$TEST_OUT_DIR/decode_time_proxy_features.csv" \
  --reference_decisions_csv "$TEST_REF_DECISIONS_CSV" \
  --policy_json "$DISCOVERY_OUT_DIR/calibration/selected_policy.json" \
  --out_dir "$TEST_OUT_DIR/fixed_eval"

echo "[done] discovery calibration -> $DISCOVERY_OUT_DIR/calibration"
echo "[done] held-out fixed eval -> $TEST_OUT_DIR/fixed_eval"
