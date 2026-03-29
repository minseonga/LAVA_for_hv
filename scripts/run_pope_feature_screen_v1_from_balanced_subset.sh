#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for feature families A-E on the 25/25/25/25 balanced POPE subset.
# - A (visual confidence): VGA-style VSC/G extraction
# - B/C/D (temporal + faithful/harmful routing): from all-layer/head trace
# - E (guidance mismatch): composed from A/C/D
#
# Notes:
# - Performance-impacting VGA defaults are aligned to VGA_origin eval defaults:
#   use_add=true, cd_alpha=0.02, attn_coef=0.2, start_layer=2, end_layer=15,
#   head_balancing=simg, attn_norm=false, sampling=false.

ROOT="${ROOT:-/home/kms/LLaVA_calibration}"
OUT_DIR="${OUT_DIR:-$ROOT/experiments/pope_feature_screen_v1}"

SUBSET_DIR="${SUBSET_DIR:-$ROOT/experiments/pope_full_9000/vcs_vga_balanced_1000}"
SUBSET_GT_CSV="${SUBSET_GT_CSV:-$SUBSET_DIR/subset_gt.csv}"
SUBSET_IDS_CSV="${SUBSET_IDS_CSV:-$SUBSET_DIR/subset_ids.csv}"
SUBSET_Q_OBJ_JSONL="${SUBSET_Q_OBJ_JSONL:-$SUBSET_DIR/subset_questions_with_object.jsonl}"
SAMPLES_CSV="${SAMPLES_CSV:-}"
SPLIT_CSV="${SPLIT_CSV:-}"
USE_SPLIT="${USE_SPLIT:-all}"

# Use a full-9000 pred jsonl; converter keeps only subset ids.
PRED_JSONL="${PRED_JSONL:-$ROOT/experiments/pope_full_9000/vanilla_vs_vga_taxonomy/pred_vanilla_9000.jsonl}"
PRED_TEXT_KEY="${PRED_TEXT_KEY:-text}"   # vanilla file uses "text"; VGA usually uses "output"

IMAGE_ROOT="${IMAGE_ROOT:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
CONV_MODE="${CONV_MODE:-llava_v1}"

HEADSET_JSON="${HEADSET_JSON:-$ROOT/experiments/pope_headsets_v1/headset.json}"
ROLE_CSV="${ROLE_CSV:-}"                 # optional
COUNTERFACTUAL_CSV="${COUNTERFACTUAL_CSV:-}"  # optional
SOURCE_REPO_A="${SOURCE_REPO_A:-/home/kms/VGA_origin}"
SOURCE_REPO_B="${SOURCE_REPO_B:-/home/kms/VISTA}"
SOURCE_REPO_C="${SOURCE_REPO_C:-/home/kms/VHR}"
SOURCE_REPO_D="${SOURCE_REPO_D:-/home/kms/EAZY}"

# B temporal range defaults (override if needed)
EARLY_START="${EARLY_START:-10}"
EARLY_END="${EARLY_END:-15}"
LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
LAYER_FOCUS="${LAYER_FOCUS:-17}"

CONDA_ENV="${CONDA_ENV:-vocot}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VSC_PYTHON_BIN="${VSC_PYTHON_BIN:-/home/kms/miniconda3/envs/vga/bin/python}"
INCLUDE_AUX_TARGETS="${INCLUDE_AUX_TARGETS:-false}"

mkdir -p "$OUT_DIR"

SAMPLES_CSV_AUTO="$OUT_DIR/samples_from_pred.csv"
SAMPLES_SUMMARY="$OUT_DIR/samples_from_pred_summary.json"

TRACE_DIR="$OUT_DIR/traces_alllayers"
TRACE_SUMMARY="$TRACE_DIR/summary.json"
PER_LAYER_TRACE="$TRACE_DIR/per_layer_yes_trace.csv"
PER_HEAD_TRACE="$TRACE_DIR/per_head_yes_trace.csv"

VSC_DIR="$OUT_DIR/vsc_vga"
VSC_CSV="$VSC_DIR/features_visual_confidence_raw.csv"
VSC_SUMMARY="$VSC_DIR/summary.json"

FEATURE_DIR="$OUT_DIR/features"
FEATURE_SUMMARY="$FEATURE_DIR/summary.json"

if [[ -n "${SAMPLES_CSV}" ]]; then
  echo "[1/4] use fixed samples_csv: ${SAMPLES_CSV}"
else
  echo "[1/4] build samples_csv from subset_gt + pred_jsonl"
fi
cd "$ROOT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if [[ -n "${SAMPLES_CSV}" ]]; then
  if [[ ! -f "${SAMPLES_CSV}" ]]; then
    echo "[error] SAMPLES_CSV not found: ${SAMPLES_CSV}"
    exit 2
  fi
else
  "$PYTHON_BIN" "$ROOT/scripts/build_pope_samples_from_gt_and_pred.py" \
    --subset_gt_csv "$SUBSET_GT_CSV" \
    --pred_jsonl "$PRED_JSONL" \
    --pred_text_key "$PRED_TEXT_KEY" \
    --out_csv "$SAMPLES_CSV_AUTO" \
    --out_summary "$SAMPLES_SUMMARY"
  SAMPLES_CSV="$SAMPLES_CSV_AUTO"
fi

echo "[2/4] extract all-layer/per-head trace on the balanced subset"
mkdir -p "$TRACE_DIR"
"$PYTHON_BIN" "$ROOT/analyze_pope_visual_disconnect.py" \
  --samples_csv "$SAMPLES_CSV" \
  --image_root "$IMAGE_ROOT" \
  --out_dir "$TRACE_DIR" \
  --dataset_mode pope \
  --model_path "$MODEL_PATH" \
  --conv_mode "$CONV_MODE" \
  --topk_local 16 \
  --object_patch_topk 64 \
  --hidden_layer_idx -1 \
  --attn_layer_idx -1 \
  --save_layer_trace \
  --save_head_trace \
  --head_layer_start 10 \
  --head_layer_end 24 \
  --control_modes blur,shuffle \
  --shuffle_grid 4 \
  --blur_radius 12 \
  --bootstrap 500 \
  --seed 42

echo "[3/4] extract VGA-style visual confidence (A family)"
mkdir -p "$VSC_DIR"
"$VSC_PYTHON_BIN" "$ROOT/scripts/extract_vga_vsc_features.py" \
  --vga_root /home/kms/VGA_origin \
  --model_path "$MODEL_PATH" \
  --image_folder "$IMAGE_ROOT" \
  --question_file "$SUBSET_Q_OBJ_JSONL" \
  --conv_mode "$CONV_MODE" \
  --out_csv "$VSC_CSV" \
  --out_summary "$VSC_SUMMARY" \
  --obj_topk 5 \
  --entropy_topk 10 \
  --use_add true \
  --cd_alpha 0.02 \
  --attn_coef 0.2 \
  --start_layer 2 \
  --end_layer 15 \
  --head_balancing simg \
  --attn_norm false \
  --sampling false \
  --seed 42

echo "[4/4] build A-E unified feature tables"
mkdir -p "$FEATURE_DIR"
CMD=(
  "$PYTHON_BIN" "$ROOT/scripts/build_pope_feature_screen_v1.py"
  --subset_ids_csv "$SUBSET_IDS_CSV"
  --subset_gt_csv "$SUBSET_GT_CSV"
  --per_layer_trace_csv "$PER_LAYER_TRACE"
  --per_head_trace_csv "$PER_HEAD_TRACE"
  --headset_json "$HEADSET_JSON"
  --vsc_csv "$VSC_CSV"
  --samples_csv "$SAMPLES_CSV"
  --use_split "$USE_SPLIT"
  --out_dir "$FEATURE_DIR"
  --early_start "$EARLY_START"
  --early_end "$EARLY_END"
  --late_start "$LATE_START"
  --late_end "$LATE_END"
  --layer_focus "$LAYER_FOCUS"
  --source_repo_A "$SOURCE_REPO_A"
  --source_repo_B "$SOURCE_REPO_B"
  --source_repo_C "$SOURCE_REPO_C"
  --source_repo_D "$SOURCE_REPO_D"
  --eps 1e-6
)
if [[ -n "${SPLIT_CSV}" ]]; then
  CMD+=(--split_csv "$SPLIT_CSV")
fi
if [[ "${INCLUDE_AUX_TARGETS}" == "true" ]]; then
  CMD+=(--include_aux_targets)
fi
if [[ -n "${COUNTERFACTUAL_CSV}" ]]; then
  CMD+=(--counterfactual_csv "$COUNTERFACTUAL_CSV")
fi
if [[ -n "${ROLE_CSV}" ]]; then
  CMD+=(--role_csv "$ROLE_CSV")
fi
"${CMD[@]}"

echo "[done] $OUT_DIR"
echo "[saved] $SAMPLES_SUMMARY"
echo "[saved] $TRACE_SUMMARY"
echo "[saved] $VSC_SUMMARY"
echo "[saved] $FEATURE_SUMMARY"
