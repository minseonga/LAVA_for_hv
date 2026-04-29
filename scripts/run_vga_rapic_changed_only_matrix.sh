#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/paper_pcp_cd_changedfast/vga}"
BACKBONES="${BACKBONES:-llava15,llava_next}"
DATASETS="${DATASETS:-mscoco,aokvqa,gqa}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-python}"
LLAVA15_PYTHON_BIN="${LLAVA15_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
LLAVA_NEXT_PYTHON_BIN="${LLAVA_NEXT_PYTHON_BIN:-/home/kms/miniconda3/envs/llava_next_official/bin/python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE="${DEVICE:-cuda}"

HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"
POLICY_LLAVA15="${POLICY_LLAVA15:-$CAL_ROOT/experiments/paper_pcp_cd/llava15/discovery/pcp_cd_compact_oracle/selected_policy.json}"
POLICY_LLAVA_NEXT="${POLICY_LLAVA_NEXT:-$CAL_ROOT/experiments/paper_pcp_cd_tokfix_newline/llava_next/discovery/pcp_cd_content_pool_changed/selected_policy.json}"

LLAVA15_MODEL_PATH="${LLAVA15_MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
LLAVA_NEXT_MODEL_PATH="${LLAVA_NEXT_MODEL_PATH:-/home/kms/models/llama3-llava-next-8b}"
LLAVA_NEXT_ROOT="${LLAVA_NEXT_ROOT:-/home/kms/LLaVA-NeXT}"
LLAVA_NEXT_TORCH_TYPE="${LLAVA_NEXT_TORCH_TYPE:-fp16}"
LLAVA_NEXT_ATTN_IMPLEMENTATION="${LLAVA_NEXT_ATTN_IMPLEMENTATION:-sdpa}"

DATA_ROOT="${DATA_ROOT:-$CAL_ROOT/experiments/pope_hf_multidataset}"
PAPER_RAW="${PAPER_RAW:-$CAL_ROOT/experiments/paper_raw}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-false}"
LOG_EVERY="${LOG_EVERY:-25}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] missing file: $path" >&2
    exit 2
  fi
}

dataset_config() {
  local backbone="$1"
  local dataset="$2"

  case "$dataset" in
    mscoco)
      Q_WITHOBJ="${MSCOCO_Q_WITHOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
      GT_CSV="${MSCOCO_GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
      IMAGE_FOLDER="${MSCOCO_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
      if [[ "$backbone" == "llava15" ]]; then
        BASELINE_PRED="${LLAVA15_MSCOCO_BASELINE:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/baseline/pred_vanilla_9000.jsonl}"
        INTERVENTION_PRED="${LLAVA15_MSCOCO_VGA:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/vga/pred_vga_9000.jsonl}"
        BASELINE_KEY="text"
        INTERVENTION_KEY="output"
      else
        BASELINE_PRED="${NEXT_MSCOCO_BASELINE:-$PAPER_RAW/pope/llava_next_llama3_8b/baseline_sdpa_tok8_full9000/pred_baseline.jsonl}"
        INTERVENTION_PRED="${NEXT_MSCOCO_VGA:-$PAPER_RAW/pope/llava_next_llama3_8b/vga_sdpa_tok8_layers0_16_full9000/pred_vga.jsonl}"
        BASELINE_KEY="output"
        INTERVENTION_KEY="output"
      fi
      ;;
    aokvqa)
      Q_WITHOBJ="${AOKVQA_Q_WITHOBJ:-$DATA_ROOT/aokvqa/pope_aokvqa_9000_q_with_object.jsonl}"
      GT_CSV="${AOKVQA_GT_CSV:-$DATA_ROOT/aokvqa/pope_aokvqa_9000_gt.csv}"
      IMAGE_FOLDER="${AOKVQA_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
      if [[ "$backbone" == "llava15" ]]; then
        BASELINE_PRED="${LLAVA15_AOKVQA_BASELINE:-$PAPER_RAW/pope_transfer_llava15_mscoco_policy/aokvqa/llava15_7b/baseline_full9000/pred_baseline.jsonl}"
        INTERVENTION_PRED="${LLAVA15_AOKVQA_VGA:-$PAPER_RAW/pope_transfer_llava15_mscoco_policy/aokvqa/llava15_7b/vga_full9000/pred_vga.jsonl}"
        BASELINE_KEY="text"
        INTERVENTION_KEY="output"
      else
        BASELINE_PRED="${NEXT_AOKVQA_BASELINE:-$PAPER_RAW/pope/aokvqa/llava_next_llama3_8b/baseline_sdpa_tok8_full9000/pred_baseline.jsonl}"
        INTERVENTION_PRED="${NEXT_AOKVQA_VGA:-$PAPER_RAW/pope/aokvqa/llava_next_llama3_8b/vga_sdpa_tok8_layers0_16_full9000/pred_vga.jsonl}"
        BASELINE_KEY="output"
        INTERVENTION_KEY="output"
      fi
      ;;
    gqa)
      Q_WITHOBJ="${GQA_Q_WITHOBJ:-$DATA_ROOT/gqa/pope_gqa_9000_q_with_object.jsonl}"
      GT_CSV="${GQA_GT_CSV:-$DATA_ROOT/gqa/pope_gqa_9000_gt.csv}"
      IMAGE_FOLDER="${GQA_IMAGE_FOLDER:-/home/kms/data/GQA}"
      if [[ "$backbone" == "llava15" ]]; then
        BASELINE_PRED="${LLAVA15_GQA_BASELINE:-$PAPER_RAW/pope_transfer_llava15_mscoco_policy/gqa/llava15_7b/baseline_full9000/pred_baseline.jsonl}"
        INTERVENTION_PRED="${LLAVA15_GQA_VGA:-$PAPER_RAW/pope_transfer_llava15_mscoco_policy/gqa/llava15_7b/vga_full9000/pred_vga.jsonl}"
        BASELINE_KEY="text"
        INTERVENTION_KEY="output"
      else
        BASELINE_PRED="${NEXT_GQA_BASELINE:-$PAPER_RAW/pope/gqa/llava_next_llama3_8b/baseline_sdpa_tok8_full9000/pred_baseline.jsonl}"
        INTERVENTION_PRED="${NEXT_GQA_VGA:-$PAPER_RAW/pope/gqa/llava_next_llama3_8b/vga_sdpa_tok8_layers0_16_full9000/pred_vga.jsonl}"
        BASELINE_KEY="output"
        INTERVENTION_KEY="output"
      fi
      ;;
    *)
      echo "[error] unsupported dataset: $dataset" >&2
      exit 2
      ;;
  esac
}

backbone_config() {
  local backbone="$1"
  if [[ "$backbone" == "llava15" ]]; then
    PY_BIN="$LLAVA15_PYTHON_BIN"
    MODEL_PATH="$LLAVA15_MODEL_PATH"
    MODEL_BASE=""
    CONV_MODE="llava_v1"
    RUNTIME_BACKEND="llava15_cleanroom"
    POLICY_JSON="$POLICY_LLAVA15"
    EXTRA_RUNTIME_ENV=()
  elif [[ "$backbone" == "llava_next" ]]; then
    PY_BIN="$LLAVA_NEXT_PYTHON_BIN"
    MODEL_PATH="$LLAVA_NEXT_MODEL_PATH"
    MODEL_BASE=""
    CONV_MODE="llava_llama_3"
    RUNTIME_BACKEND="llava_next_official"
    POLICY_JSON="$POLICY_LLAVA_NEXT"
    EXTRA_RUNTIME_ENV=(
      "LLAVA_NEXT_ROOT=$LLAVA_NEXT_ROOT"
      "LLAVA_NEXT_TORCH_TYPE=$LLAVA_NEXT_TORCH_TYPE"
      "LLAVA_NEXT_ATTN_IMPLEMENTATION=$LLAVA_NEXT_ATTN_IMPLEMENTATION"
    )
  else
    echo "[error] unsupported backbone: $backbone" >&2
    exit 2
  fi
}

run_one() {
  local backbone="$1"
  local dataset="$2"

  backbone_config "$backbone"
  dataset_config "$backbone" "$dataset"

  require_file "$Q_WITHOBJ"
  require_file "$GT_CSV"
  require_file "$BASELINE_PRED"
  require_file "$INTERVENTION_PRED"
  require_file "$HEADSET_JSON"
  require_file "$POLICY_JSON"

  local run_root="$OUT_ROOT/$backbone/$dataset"
  local subset_dir="$run_root/changed_subset"
  local feat_dir="$run_root/features_changed"
  local apply_dir="$run_root/apply_changed_policy"

  echo "== $backbone / $dataset"
  echo "[paths] base=$BASELINE_PRED"
  echo "[paths] vga =$INTERVENTION_PRED"
  echo "[paths] out =$run_root"

  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/build_changed_pope_subset.py" \
    --question_jsonl "$Q_WITHOBJ" \
    --gt_csv "$GT_CSV" \
    --baseline_pred_jsonl "$BASELINE_PRED" \
    --intervention_pred_jsonl "$INTERVENTION_PRED" \
    --baseline_pred_text_key "$BASELINE_KEY" \
    --intervention_pred_text_key "$INTERVENTION_KEY" \
    --out_dir "$subset_dir"

  env \
    CAL_ROOT="$CAL_ROOT" \
    PY_BIN="$PY_BIN" \
    GPU="$GPU" \
    DEVICE="$DEVICE" \
    MODEL_PATH="$MODEL_PATH" \
    MODEL_BASE="$MODEL_BASE" \
    CONV_MODE="$CONV_MODE" \
    RUNTIME_BACKEND="$RUNTIME_BACKEND" \
    QUESTION_FILE="$subset_dir/changed_q_with_object.jsonl" \
    IMAGE_FOLDER="$IMAGE_FOLDER" \
    GT_CSV="$GT_CSV" \
    INTERVENTION_PRED_JSONL="$INTERVENTION_PRED" \
    INTERVENTION_PRED_KEY="$INTERVENTION_KEY" \
    BASELINE_PRED_JSONL="$BASELINE_PRED" \
    BASELINE_PRED_KEY="$BASELINE_KEY" \
    HEADSET_JSON="$HEADSET_JSON" \
    OUT_DIR="$feat_dir" \
    EXTRACT_ONLY=true \
    SKIP_STAGE_A=true \
    REUSE_IF_EXISTS="$REUSE_IF_EXISTS" \
    LOG_EVERY="$LOG_EVERY" \
    "${EXTRA_RUNTIME_ENV[@]}" \
    bash "$CAL_ROOT/scripts/run_discriminative_meta_strong_online.sh"

  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/apply_pcp_c_d_controller.py" \
    --rows_csv "$feat_dir/online_feature_rows.csv" \
    --policy_json "$POLICY_JSON" \
    --out_dir "$apply_dir" \
    --family selected \
    --candidate_filter all \
    --derive_decision_kl true

  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/eval_pope_meta_by_category.py" \
    --gt_csv "$GT_CSV" \
    --baseline_pred_jsonl "$BASELINE_PRED" \
    --intervention_pred_jsonl "$INTERVENTION_PRED" \
    --meta_route_rows_csv "$apply_dir/pcp_route_rows.csv" \
    --baseline_pred_text_key "$BASELINE_KEY" \
    --intervention_pred_text_key "$INTERVENTION_KEY" \
    --out_json "$apply_dir/metrics_meta_by_category.json" \
    --out_csv "$apply_dir/metrics_meta_by_category.csv"

  "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/summarize_pcp_deployment_from_routes.py" \
    --gt_csv "$GT_CSV" \
    --baseline_pred_jsonl "$BASELINE_PRED" \
    --intervention_pred_jsonl "$INTERVENTION_PRED" \
    --route_rows_csv "$apply_dir/pcp_route_rows.csv" \
    --baseline_pred_text_key "$BASELINE_KEY" \
    --intervention_pred_text_key "$INTERVENTION_KEY" \
    --out_json "$apply_dir/deployment_summary.json"

  echo "[done] $apply_dir"
}

IFS=',' read -r -a backbone_list <<< "$BACKBONES"
IFS=',' read -r -a dataset_list <<< "$DATASETS"
for backbone in "${backbone_list[@]}"; do
  for dataset in "${dataset_list[@]}"; do
    run_one "$backbone" "$dataset"
  done
done
