#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
DISCOVERY_ASSET_ROOT="${DISCOVERY_ASSET_ROOT:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets}"
Q_DISC="${Q_DISC:-$DISCOVERY_ASSET_ROOT/discovery_q_with_object.jsonl}"
Q_NOOBJ="${Q_NOOBJ:-$DISCOVERY_ASSET_ROOT/discovery_q.jsonl}"
GT_CSV="${GT_CSV:-$DISCOVERY_ASSET_ROOT/discovery_gt.csv}"

BASE_OUT_ROOT="${BASE_OUT_ROOT:-$CAL_ROOT/experiments/common_pope_discovery_harm_miner_v2}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_claim_subset_probe_v1}"

RUN_PREREQ="${RUN_PREREQ:-1}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

DISC_LIMIT="${DISC_LIMIT:-500}"
GEN_LIMIT="${GEN_LIMIT:-200}"
USE_BLUR_CONTROL="${USE_BLUR_CONTROL:-true}"
BLUR_RADIUS="${BLUR_RADIUS:-8.0}"
MAX_ANSWER_WORDS="${MAX_ANSWER_WORDS:-2}"
MAX_CLAIM_WORDS="${MAX_CLAIM_WORDS:-24}"

DISC_GPU="${DISC_GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
GEN_GPU="${GEN_GPU:-$DISC_GPU}"
PARALLEL_EXTRACT="${PARALLEL_EXTRACT:-1}"

FEATURE_COLS="${FEATURE_COLS:-probe_lp_min_real,probe_target_gap_min_real,probe_entropy_max_real,probe_lp_min_real_minus_blur,probe_target_gap_min_real_minus_blur,probe_entropy_max_real_minus_blur}"

DISC_OUT_DIR="$OUT_ROOT/discriminative"
GEN_OUT_DIR="$OUT_ROOT/generative"
DISC_ANALYSIS_DIR="$DISC_OUT_DIR/analysis"
GEN_ANALYSIS_DIR="$GEN_OUT_DIR/analysis"

DISC_FEATURE_CSV="$DISC_OUT_DIR/subset_features.csv"
DISC_FEATURE_SUMMARY="$DISC_OUT_DIR/subset_features.summary.json"
DISC_TABLE_CSV="$DISC_OUT_DIR/vga_table.csv"
DISC_TABLE_SUMMARY="$DISC_OUT_DIR/vga_table.summary.json"

GEN_FEATURE_CSV="$GEN_OUT_DIR/subset_features.csv"
GEN_FEATURE_SUMMARY="$GEN_OUT_DIR/subset_features.summary.json"
GEN_TABLE_CSV="$GEN_OUT_DIR/vga_table.csv"
GEN_TABLE_SUMMARY="$GEN_OUT_DIR/vga_table.summary.json"

BASE_DISC_BASELINE_JSONL="$BASE_OUT_ROOT/discriminative/baseline/pred_vanilla_discovery.jsonl"
BASE_DISC_VGA_JSONL="$BASE_OUT_ROOT/discriminative/vga/pred_vga_discovery.jsonl"
BASE_CAPTION_Q_JSONL="$BASE_OUT_ROOT/generative/assets/discovery_caption_q.jsonl"
BASE_GEN_BASELINE_JSONL="$BASE_OUT_ROOT/generative/baseline/pred_vanilla_caption.jsonl"
BASE_GEN_VGA_JSONL="$BASE_OUT_ROOT/generative/vga/pred_vga_caption.jsonl"
BASE_GEN_BASELINE_CHAIR_JSON="$BASE_OUT_ROOT/generative/baseline/chair_baseline.json"
BASE_GEN_VGA_CHAIR_JSON="$BASE_OUT_ROOT/generative/vga/chair_vga.json"

mkdir -p "$DISC_OUT_DIR" "$GEN_OUT_DIR" "$DISC_ANALYSIS_DIR" "$GEN_ANALYSIS_DIR"

if [[ "$RUN_PREREQ" == "1" ]]; then
  echo "[0/6] ensure baseline/VGA discovery artifacts"
  (
    cd "$CAL_ROOT"
    CAL_ROOT="$CAL_ROOT" \
    CAL_PYTHON_BIN="$CAL_PYTHON_BIN" \
    MODEL_PATH="$MODEL_PATH" \
    IMAGE_FOLDER="$IMAGE_FOLDER" \
    DISCOVERY_ASSET_ROOT="$DISCOVERY_ASSET_ROOT" \
    OUT_ROOT="$BASE_OUT_ROOT" \
    RUN_BASELINE=1 \
    RUN_VGA=1 \
    RUN_VISTA=0 \
    RUN_EAZY=0 \
    REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
    bash scripts/run_common_pope_harm_miner.sh
  )
fi

if [[ ! -f "$Q_DISC" ]]; then
  echo "[error] missing discriminative discovery question file: $Q_DISC" >&2
  exit 1
fi
if [[ ! -f "$Q_NOOBJ" ]]; then
  echo "[error] missing no-object discovery question file: $Q_NOOBJ" >&2
  exit 1
fi
if [[ ! -f "$GT_CSV" ]]; then
  echo "[error] missing discovery gt csv: $GT_CSV" >&2
  exit 1
fi
if [[ ! -f "$BASE_DISC_BASELINE_JSONL" || ! -f "$BASE_DISC_VGA_JSONL" ]]; then
  echo "[error] missing discriminative baseline/VGA artifacts under $BASE_OUT_ROOT" >&2
  exit 1
fi
if [[ ! -f "$BASE_CAPTION_Q_JSONL" ]]; then
  echo "[1/6] build caption prompts"
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/build_discovery_caption_questions.py \
      --question_file "$Q_NOOBJ" \
      --out_jsonl "$BASE_CAPTION_Q_JSONL" \
      --out_summary_json "$BASE_OUT_ROOT/generative/assets/discovery_caption_q.summary.json"
  )
fi
if [[ ! -f "$BASE_GEN_BASELINE_JSONL" || ! -f "$BASE_GEN_VGA_JSONL" || ! -f "$BASE_GEN_BASELINE_CHAIR_JSON" || ! -f "$BASE_GEN_VGA_CHAIR_JSON" ]]; then
  echo "[error] missing generative baseline/VGA/CHAIR artifacts under $BASE_OUT_ROOT" >&2
  echo "[hint] rerun common_pope_harm_miner with RUN_VGA=1 and generative steps enabled first" >&2
  exit 1
fi

run_disc_extract() {
  echo "[1/6] extract discriminative subset features"
  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$DISC_GPU" "$CAL_PYTHON_BIN" scripts/extract_vga_claim_subset_features.py \
      --task_mode discriminative \
      --question_file "$Q_DISC" \
      --image_folder "$IMAGE_FOLDER" \
      --baseline_pred_jsonl "$BASE_DISC_BASELINE_JSONL" \
      --out_csv "$DISC_FEATURE_CSV" \
      --out_summary_json "$DISC_FEATURE_SUMMARY" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --conv_mode "$CONV_MODE" \
      --device cuda \
      --limit "$DISC_LIMIT" \
      --reuse_if_exists "$REUSE_IF_EXISTS" \
      --use_blur_control "$USE_BLUR_CONTROL" \
      --blur_radius "$BLUR_RADIUS" \
      --max_answer_words "$MAX_ANSWER_WORDS" \
      --max_claim_words "$MAX_CLAIM_WORDS"
  )
}

run_gen_extract() {
  echo "[2/6] extract generative subset features"
  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GEN_GPU" "$CAL_PYTHON_BIN" scripts/extract_vga_claim_subset_features.py \
      --task_mode generative \
      --question_file "$BASE_CAPTION_Q_JSONL" \
      --image_folder "$IMAGE_FOLDER" \
      --baseline_pred_jsonl "$BASE_GEN_BASELINE_JSONL" \
      --out_csv "$GEN_FEATURE_CSV" \
      --out_summary_json "$GEN_FEATURE_SUMMARY" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --conv_mode "$CONV_MODE" \
      --device cuda \
      --limit "$GEN_LIMIT" \
      --reuse_if_exists "$REUSE_IF_EXISTS" \
      --use_blur_control "$USE_BLUR_CONTROL" \
      --blur_radius "$BLUR_RADIUS" \
      --max_answer_words "$MAX_ANSWER_WORDS" \
      --max_claim_words "$MAX_CLAIM_WORDS"
  )
}

if [[ "$PARALLEL_EXTRACT" == "1" && "$DISC_GPU" != "$GEN_GPU" ]]; then
  run_disc_extract &
  disc_pid=$!
  run_gen_extract &
  gen_pid=$!
  wait "$disc_pid"
  wait "$gen_pid"
else
  run_disc_extract
  run_gen_extract
fi

echo "[3/6] build discriminative VGA table"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/build_method_harm_table.py \
    --baseline_features_csv "$DISC_FEATURE_CSV" \
    --baseline_pred_jsonl "$BASE_DISC_BASELINE_JSONL" \
    --intervention_pred_jsonl "$BASE_DISC_VGA_JSONL" \
    --gt_csv "$GT_CSV" \
    --method_name vga \
    --benchmark_name pope_discovery \
    --split_name subset_probe \
    --out_csv "$DISC_TABLE_CSV" \
    --out_summary_json "$DISC_TABLE_SUMMARY"
)

echo "[4/6] build generative VGA table"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/build_method_chair_table.py \
    --baseline_features_csv "$GEN_FEATURE_CSV" \
    --baseline_pred_jsonl "$BASE_GEN_BASELINE_JSONL" \
    --intervention_pred_jsonl "$BASE_GEN_VGA_JSONL" \
    --baseline_chair_json "$BASE_GEN_BASELINE_CHAIR_JSON" \
    --intervention_chair_json "$BASE_GEN_VGA_CHAIR_JSON" \
    --method_name vga \
    --benchmark_name pope_discovery_caption \
    --split_name subset_probe \
    --out_csv "$GEN_TABLE_CSV" \
    --out_summary_json "$GEN_TABLE_SUMMARY"
)

echo "[5/6] analyze discriminative subset features"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/analyze_common_method_harm_miner.py \
    --table_csvs "$DISC_TABLE_CSV" \
    --out_dir "$DISC_ANALYSIS_DIR" \
    --feature_cols "$FEATURE_COLS"
) &
disc_analysis_pid=$!

echo "[6/6] analyze generative subset features"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/analyze_common_method_harm_miner.py \
    --table_csvs "$GEN_TABLE_CSV" \
    --out_dir "$GEN_ANALYSIS_DIR" \
    --feature_cols "$FEATURE_COLS"
) &
gen_analysis_pid=$!

wait "$disc_analysis_pid"
wait "$gen_analysis_pid"

echo "[done] discriminative summary: $DISC_ANALYSIS_DIR/summary.json"
echo "[done] generative summary: $GEN_ANALYSIS_DIR/summary.json"
