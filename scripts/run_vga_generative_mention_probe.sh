#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
BASE_OUT_ROOT="${BASE_OUT_ROOT:-$CAL_ROOT/experiments/common_pope_discovery_harm_miner_v2}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_generative_mention_probe_v1}"

GEN_LIMIT="${GEN_LIMIT:-200}"
MAX_MENTIONS="${MAX_MENTIONS:-12}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

FEATURE_COLS="${FEATURE_COLS:-probe_mention_lp_min_real,probe_mention_target_gap_min_real,probe_mention_entropy_max_real,probe_mention_lp_tail_gap_real}"

CAPTION_Q_JSONL="$BASE_OUT_ROOT/generative/assets/discovery_caption_q.jsonl"
BASELINE_JSONL="$BASE_OUT_ROOT/generative/baseline/pred_vanilla_caption.jsonl"
VGA_JSONL="$BASE_OUT_ROOT/generative/vga/pred_vga_caption.jsonl"
BASELINE_CHAIR_JSON="$BASE_OUT_ROOT/generative/baseline/chair_baseline.json"
VGA_CHAIR_JSON="$BASE_OUT_ROOT/generative/vga/chair_vga.json"

FEATURE_CSV="$OUT_ROOT/subset_features.csv"
FEATURE_SUMMARY="$OUT_ROOT/subset_features.summary.json"
TABLE_CSV="$OUT_ROOT/vga_table.csv"
TABLE_SUMMARY="$OUT_ROOT/vga_table.summary.json"
ANALYSIS_DIR="$OUT_ROOT/analysis"

mkdir -p "$OUT_ROOT" "$ANALYSIS_DIR"

if [[ ! -f "$CAPTION_Q_JSONL" ]]; then
  echo "[error] missing caption discovery question file: $CAPTION_Q_JSONL" >&2
  exit 1
fi
if [[ ! -f "$BASELINE_JSONL" || ! -f "$VGA_JSONL" ]]; then
  echo "[error] missing baseline/VGA caption outputs under $BASE_OUT_ROOT" >&2
  exit 1
fi
if [[ ! -f "$BASELINE_CHAIR_JSON" || ! -f "$VGA_CHAIR_JSON" ]]; then
  echo "[error] missing baseline/VGA CHAIR outputs under $BASE_OUT_ROOT" >&2
  exit 1
fi

echo "[1/3] extract mention-level generative features"
(
  cd "$CAL_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" "$CAL_PYTHON_BIN" scripts/extract_vga_generative_mention_features.py \
    --question_file "$CAPTION_Q_JSONL" \
    --image_folder "$IMAGE_FOLDER" \
    --baseline_pred_jsonl "$BASELINE_JSONL" \
    --out_csv "$FEATURE_CSV" \
    --out_summary_json "$FEATURE_SUMMARY" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device cuda \
    --limit "$GEN_LIMIT" \
    --max_mentions "$MAX_MENTIONS" \
    --reuse_if_exists "$REUSE_IF_EXISTS"
)

echo "[2/3] build VGA generative CHAIR table"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/build_method_chair_table.py \
    --baseline_features_csv "$FEATURE_CSV" \
    --baseline_pred_jsonl "$BASELINE_JSONL" \
    --intervention_pred_jsonl "$VGA_JSONL" \
    --baseline_chair_json "$BASELINE_CHAIR_JSON" \
    --intervention_chair_json "$VGA_CHAIR_JSON" \
    --method_name vga \
    --benchmark_name pope_discovery_caption \
    --split_name mention_probe \
    --out_csv "$TABLE_CSV" \
    --out_summary_json "$TABLE_SUMMARY"
)

echo "[3/3] analyze mention-level generative features"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/analyze_common_method_harm_miner.py \
    --table_csvs "$TABLE_CSV" \
    --out_dir "$ANALYSIS_DIR" \
    --feature_cols "$FEATURE_COLS"
)

echo "[done] $ANALYSIS_DIR/summary.json"
