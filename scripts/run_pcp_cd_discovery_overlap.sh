#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
BACKBONE="${BACKBONE:-llava15}"  # llava15 | llava_next
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/paper_pcp_cd}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

DISC_Q_NOOBJ="${DISC_Q_NOOBJ:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q.jsonl}"
DISC_Q_WITHOBJ="${DISC_Q_WITHOBJ:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q_with_object.jsonl}"
DISC_GT_CSV="${DISC_GT_CSV:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_gt.csv}"
DISC_IMAGE_FOLDER="${DISC_IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"

C_FEATURE_COLS="${C_FEATURE_COLS:-cheap_target_gap_content_min,cheap_lp_content_min,cheap_lp_content_std}"
D_FEATURE_COLS="${D_FEATURE_COLS:-cheap_decision_candidate_minus_alt,cheap_decision_candidate_prob_binary,cheap_decision_candidate_label_lp,cheap_decision_candidate_kl_uniform}"
MIN_PRESENT_RATE="${MIN_PRESENT_RATE:-0.8}"
MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.0}"
TOP_K_C="${TOP_K_C:-3}"
TOP_K_D="${TOP_K_D:-4}"
ALPHA_GRID="${ALPHA_GRID:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}"
TAU_OBJECTIVE="${TAU_OBJECTIVE:-final_acc}"
CANDIDATE_FILTER="${CANDIDATE_FILTER:-changed_answer}"

LLAVA15_PY="${LLAVA15_PY:-/home/kms/miniconda3/envs/vga_base/bin/python}"
LLAVA_NEXT_PY="${LLAVA_NEXT_PY:-/home/kms/miniconda3/envs/llava_next_official/bin/python}"
LLAVA_NEXT_ROOT="${LLAVA_NEXT_ROOT:-/home/kms/LLaVA-NeXT}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] missing file: $path" >&2
    exit 2
  fi
}

run_extract_only() {
  local py_bin="$1"
  local runtime_backend="$2"
  local model_path="$3"
  local conv_mode="$4"
  local baseline_pred="$5"
  local baseline_key="$6"
  local intervention_pred="$7"
  local intervention_key="$8"
  local out_dir="$9"

  CLEANROOM_IMAGE_PREPROCESS_MODE=process_images \
  CLEANROOM_TF_FORWARD_MODE=model \
  CAL_ROOT="$CAL_ROOT" \
  PY_BIN="$py_bin" \
  GPU="$GPU" \
  DEVICE=cuda \
  MODEL_PATH="$model_path" \
  CONV_MODE="$conv_mode" \
  RUNTIME_BACKEND="$runtime_backend" \
  LLAVA_NEXT_ROOT="$LLAVA_NEXT_ROOT" \
  LLAVA_NEXT_TORCH_TYPE=fp16 \
  LLAVA_NEXT_ATTN_IMPLEMENTATION=sdpa \
  QUESTION_FILE="$DISC_Q_WITHOBJ" \
  IMAGE_FOLDER="$DISC_IMAGE_FOLDER" \
  GT_CSV="$DISC_GT_CSV" \
  HEADSET_JSON="$HEADSET_JSON" \
  POLICY_BUNDLE_JSON=none \
  INTERVENTION_PRED_JSONL="$intervention_pred" \
  INTERVENTION_PRED_KEY="$intervention_key" \
  BASELINE_PRED_JSONL="$baseline_pred" \
  BASELINE_PRED_KEY="$baseline_key" \
  OUT_DIR="$out_dir" \
  FEATURE_ORDER=cheap_first \
  CONTROLLER_MODE=meta_strong \
  EXTRACT_ONLY=true \
  SKIP_STAGE_A=true \
  REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
  bash "$CAL_ROOT/scripts/run_discriminative_meta_strong_online.sh"
}

if [[ "$BACKBONE" == "llava15" ]]; then
  PY_BIN="$LLAVA15_PY"
  MODEL_PATH="liuhaotian/llava-v1.5-7b"
  CONV_MODE="llava_v1"
  RUNTIME_BACKEND="llava15_cleanroom"
  BASELINE_PRED="$CAL_ROOT/experiments/common_pope_discovery_v3_panel_v1/discriminative/baseline/pred_vanilla_discovery.jsonl"
  BASELINE_KEY="text"
  INTERVENTION_PRED="$CAL_ROOT/experiments/common_pope_discovery_v3_panel_v1/discriminative/vga/pred_vga_discovery.jsonl"
  INTERVENTION_KEY="output"
elif [[ "$BACKBONE" == "llava_next" ]]; then
  PY_BIN="$LLAVA_NEXT_PY"
  MODEL_PATH="/home/kms/models/llama3-llava-next-8b"
  CONV_MODE="llava_llama_3"
  RUNTIME_BACKEND="llava_next_official"
  RAW_ROOT="$OUT_ROOT/llava_next/discovery/raw"

  CAL_PYTHON_BIN="$PY_BIN" LLAVA_NEXT_PYTHON_BIN="$PY_BIN" VGA_PYTHON_BIN="$PY_BIN" \
  GPU="$GPU" BACKBONE=llava_next METHOD=baseline TASK=pope \
  MODEL_PATH="$MODEL_PATH" CONV_MODE="$CONV_MODE" \
  LLAVA_NEXT_ROOT="$LLAVA_NEXT_ROOT" LLAVA_NEXT_ATTN_IMPLEMENTATION=sdpa LLAVA_NEXT_TORCH_TYPE=fp16 \
  IMAGE_FOLDER="$DISC_IMAGE_FOLDER" QUESTION_FILE="$DISC_Q_NOOBJ" GT_CSV="$DISC_GT_CSV" \
  OUT_ROOT="$RAW_ROOT/baseline" MAX_NEW_TOKENS=8 REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
  bash "$CAL_ROOT/scripts/run_multibackbone_method_prediction.sh"

  CAL_PYTHON_BIN="$PY_BIN" LLAVA_NEXT_PYTHON_BIN="$PY_BIN" VGA_PYTHON_BIN="$PY_BIN" \
  GPU="$GPU" BACKBONE=llava_next METHOD=vga TASK=pope \
  MODEL_PATH="$MODEL_PATH" CONV_MODE="$CONV_MODE" \
  LLAVA_NEXT_ROOT="$LLAVA_NEXT_ROOT" VGA_ATTN_TYPE=sdpa VGA_TORCH_TYPE=fp16 VGA_START_LAYER=0 VGA_END_LAYER=16 \
  IMAGE_FOLDER="$DISC_IMAGE_FOLDER" QUESTION_FILE="$DISC_Q_WITHOBJ" GT_CSV="$DISC_GT_CSV" \
  OUT_ROOT="$RAW_ROOT/vga" MAX_NEW_TOKENS=8 REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
  bash "$CAL_ROOT/scripts/run_multibackbone_method_prediction.sh"

  BASELINE_PRED="$RAW_ROOT/baseline/pred_baseline.jsonl"
  BASELINE_KEY="output"
  INTERVENTION_PRED="$RAW_ROOT/vga/pred_vga.jsonl"
  INTERVENTION_KEY="output"
else
  echo "[error] unsupported BACKBONE=$BACKBONE" >&2
  exit 2
fi

require_file "$DISC_Q_WITHOBJ"
require_file "$DISC_GT_CSV"
require_file "$HEADSET_JSON"
require_file "$BASELINE_PRED"
require_file "$INTERVENTION_PRED"

FEATURE_OUT_DIR="$OUT_ROOT/$BACKBONE/discovery/features"
CALIB_OUT_DIR="$OUT_ROOT/$BACKBONE/discovery/pcp_cd_calibration"
OVERLAP_JSON="$OUT_ROOT/$BACKBONE/discovery/pcp_cd_overlap.json"

echo "[1/3] extract-only feature rows -> $FEATURE_OUT_DIR"
run_extract_only \
  "$PY_BIN" \
  "$RUNTIME_BACKEND" \
  "$MODEL_PATH" \
  "$CONV_MODE" \
  "$BASELINE_PRED" \
  "$BASELINE_KEY" \
  "$INTERVENTION_PRED" \
  "$INTERVENTION_KEY" \
  "$FEATURE_OUT_DIR"

echo "[2/3] calibrate c_only / d_only / cd_fusion -> $CALIB_OUT_DIR"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" "$CAL_ROOT/scripts/build_pcp_c_d_controller.py" \
  --rows_csv "$FEATURE_OUT_DIR/online_feature_rows.csv" \
  --out_dir "$CALIB_OUT_DIR" \
  --c_feature_cols "$C_FEATURE_COLS" \
  --d_feature_cols "$D_FEATURE_COLS" \
  --derive_decision_kl true \
  --min_present_rate "$MIN_PRESENT_RATE" \
  --min_feature_auroc "$MIN_FEATURE_AUROC" \
  --top_k_c "$TOP_K_C" \
  --top_k_d "$TOP_K_D" \
  --alpha_grid "$ALPHA_GRID" \
  --tau_objective "$TAU_OBJECTIVE" \
  --candidate_filter "$CANDIDATE_FILTER"

echo "[3/3] overlap analysis -> $OVERLAP_JSON"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" "$CAL_ROOT/scripts/analyze_pcp_c_d_overlap.py" \
  --rows_csv "$FEATURE_OUT_DIR/online_feature_rows.csv" \
  --policy_json "$CALIB_OUT_DIR/selected_policy.json" \
  --out_json "$OVERLAP_JSON"

echo "[done] calibration summary: $CALIB_OUT_DIR/summary.json"
echo "[done] overlap summary: $OVERLAP_JSON"
