#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
BASE_OUT_ROOT="${BASE_OUT_ROOT:-$CAL_ROOT/experiments/common_pope_discovery_harm_miner_v2}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_generative_coverage_probe_v1}"

GEN_LIMIT="${GEN_LIMIT:-200}"
MAX_MENTIONS="${MAX_MENTIONS:-12}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

SUPPORTED_WEIGHT="${SUPPORTED_WEIGHT:-1.0}"
HALL_WEIGHT="${HALL_WEIGHT:-1.0}"
LENGTH_WEIGHT="${LENGTH_WEIGHT:-0.25}"

PROXY_FEATURE_COLS="${PROXY_FEATURE_COLS:-probe_n_content_tokens,probe_n_mentions_total,probe_n_object_mentions,probe_n_noun_phrases,probe_n_attribute_phrases,probe_n_relation_phrases,probe_n_count_phrases,probe_object_diversity,probe_mention_diversity,probe_first_half_object_mentions,probe_second_half_object_mentions,probe_tail_tokens_after_last_mention,probe_last_mention_pos_frac,probe_lp_head_mean_real,probe_lp_tail_mean_real,probe_lp_tail_minus_head_real,probe_gap_head_mean_real,probe_gap_tail_mean_real,probe_gap_tail_minus_head_real,probe_entropy_head_mean_real,probe_entropy_tail_mean_real,probe_entropy_tail_minus_head_real}"
ORACLE_FEATURE_COLS="${ORACLE_FEATURE_COLS:-n_gt_terms,n_base_supported,n_base_hall,base_supported_recall,base_hall_rate,delta_supported_recall,delta_hall_rate,length_collapse_penalty}"

CAPTION_Q_JSONL="$BASE_OUT_ROOT/generative/assets/discovery_caption_q.jsonl"
BASELINE_JSONL="$BASE_OUT_ROOT/generative/baseline/pred_vanilla_caption.jsonl"
VGA_JSONL="$BASE_OUT_ROOT/generative/vga/pred_vga_caption.jsonl"
BASELINE_CHAIR_JSON="$BASE_OUT_ROOT/generative/baseline/chair_baseline.json"
VGA_CHAIR_JSON="$BASE_OUT_ROOT/generative/vga/chair_vga.json"

FEATURE_CSV="$OUT_ROOT/coverage_features.csv"
FEATURE_SUMMARY="$OUT_ROOT/coverage_features.summary.json"
TABLE_CSV="$OUT_ROOT/vga_claim_aware_table.csv"
TABLE_SUMMARY="$OUT_ROOT/vga_claim_aware_table.summary.json"
PROXY_ANALYSIS_DIR="$OUT_ROOT/proxy_analysis"
ORACLE_ANALYSIS_DIR="$OUT_ROOT/oracle_analysis"

mkdir -p "$OUT_ROOT" "$PROXY_ANALYSIS_DIR" "$ORACLE_ANALYSIS_DIR"

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

echo "[1/4] extract generative coverage features"
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

echo "[2/4] build claim-aware VGA table"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/build_method_claim_aware_table.py \
    --baseline_features_csv "$FEATURE_CSV" \
    --baseline_pred_jsonl "$BASELINE_JSONL" \
    --intervention_pred_jsonl "$VGA_JSONL" \
    --baseline_chair_json "$BASELINE_CHAIR_JSON" \
    --intervention_chair_json "$VGA_CHAIR_JSON" \
    --method_name vga \
    --benchmark_name pope_discovery_caption \
    --split_name coverage_probe \
    --supported_weight "$SUPPORTED_WEIGHT" \
    --hall_weight "$HALL_WEIGHT" \
    --length_weight "$LENGTH_WEIGHT" \
    --out_csv "$TABLE_CSV" \
    --out_summary_json "$TABLE_SUMMARY"
)

echo "[3/4] analyze proxy coverage / omission features"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/analyze_common_method_harm_miner.py \
    --table_csvs "$TABLE_CSV" \
    --out_dir "$PROXY_ANALYSIS_DIR" \
    --feature_cols "$PROXY_FEATURE_COLS"
)

echo "[4/4] analyze oracle diagnostic features"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/analyze_common_method_harm_miner.py \
    --table_csvs "$TABLE_CSV" \
    --out_dir "$ORACLE_ANALYSIS_DIR" \
    --feature_cols "$ORACLE_FEATURE_COLS"
)

echo "[done] $PROXY_ANALYSIS_DIR/summary.json"
echo "[done] $ORACLE_ANALYSIS_DIR/summary.json"
