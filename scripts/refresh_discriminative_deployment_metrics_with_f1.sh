#!/usr/bin/env bash
set -euo pipefail

# Recompute POPE deployment summaries with precision/recall/F1 from existing
# RaPiC route rows. This is intentionally summary-only: it does not rebuild
# features, policies, or predictions.

CAL="${CAL:-/home/kms/LLaVA_calibration}"
CAL_PY="${CAL_PY:-/home/kms/miniconda3/envs/vga_base/bin/python}"

APPLY_ROOTS="${APPLY_ROOTS:-$CAL/experiments/paper_pcp_cd_transition_split_calib_existing/apply}"
DATASETS="${DATASETS:-mscoco aokvqa gqa}"
TARGETS="${TARGETS:-}"
EXCLUDE_TARGETS="${EXCLUDE_TARGETS:-}"

contains_word() {
  local needle="$1"
  local haystack="$2"
  [[ -z "$haystack" ]] && return 1
  [[ " $haystack " == *" $needle "* ]]
}

dataset_gt() {
  local dataset="$1"
  case "$dataset" in
    mscoco) echo "$CAL/experiments/pope_full_9000/pope_9000_gt.csv" ;;
    aokvqa) echo "$CAL/experiments/pope_hf_multidataset/aokvqa/pope_aokvqa_9000_gt.csv" ;;
    gqa) echo "$CAL/experiments/pope_hf_multidataset/gqa/pope_gqa_9000_gt.csv" ;;
    *) echo "[error] unsupported dataset: $dataset" >&2; exit 2 ;;
  esac
}

target_backbone() {
  case "$1" in
    vga_llava15|llava15_*) echo "llava15" ;;
    vga_llava_next|llava_next_*) echo "llava_next" ;;
    vga_qwen25_vl_7b|qwen25_*) echo "qwen25" ;;
    *) echo "[error] unsupported target: $1" >&2; exit 2 ;;
  esac
}

baseline_pred_path() {
  local backbone="$1"
  local dataset="$2"
  if [[ "$backbone" == "llava15" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/pope_full_9000/stage_b_signal_validation_vga/pred_baseline.jsonl"
  elif [[ "$backbone" == "llava15" && "$dataset" == "aokvqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope_transfer_llava15_mscoco_policy/aokvqa/llava15_7b/baseline_full9000/pred_baseline.jsonl"
  elif [[ "$backbone" == "llava15" && "$dataset" == "gqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope_transfer_llava15_mscoco_policy/gqa/llava15_7b/baseline_full9000/pred_baseline.jsonl"
  elif [[ "$backbone" == "llava_next" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/llava_next_llama3_8b/baseline_sdpa_tok8_full9000/pred_baseline.jsonl"
  elif [[ "$backbone" == "llava_next" && "$dataset" == "aokvqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope/aokvqa/llava_next_llama3_8b/baseline_sdpa_tok8_full9000/pred_baseline.jsonl"
  elif [[ "$backbone" == "llava_next" && "$dataset" == "gqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope/gqa/llava_next_llama3_8b/baseline_sdpa_tok8_full9000/pred_baseline.jsonl"
  elif [[ "$backbone" == "qwen25" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/qwen25_vl_7b/baseline_eager_tok8_full9000/pred_baseline.jsonl"
  elif [[ "$backbone" == "qwen25" && "$dataset" == "aokvqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope/aokvqa/qwen25_vl_7b/baseline_eager_tok8_full9000/pred_baseline.jsonl"
  elif [[ "$backbone" == "qwen25" && "$dataset" == "gqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope/gqa/qwen25_vl_7b/baseline_eager_tok8_full9000/pred_baseline.jsonl"
  else
    echo "[error] unsupported baseline path: $backbone $dataset" >&2
    exit 2
  fi
}

method_pred_path() {
  local target="$1"
  local dataset="$2"
  if [[ "$target" == "vga_llava15" && "$dataset" == "mscoco" ]]; then
    echo "${VGA_LLAVA15_MSCOCO_PRED:-$CAL/experiments/pope_full_9000/all_models_full_strict/vga/pred_vga_9000.jsonl}"
  elif [[ "$target" == "vga_llava15" && "$dataset" == "aokvqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope_transfer_llava15_mscoco_policy/aokvqa/llava15_7b/vga_full9000/pred_vga.jsonl"
  elif [[ "$target" == "vga_llava15" && "$dataset" == "gqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope_transfer_llava15_mscoco_policy/gqa/llava15_7b/vga_full9000/pred_vga.jsonl"
  elif [[ "$target" == "vga_llava_next" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/llava_next_llama3_8b/vga_sdpa_tok8_layers0_16_full9000/pred_vga.jsonl"
  elif [[ "$target" == "vga_llava_next" ]]; then
    echo "$CAL/experiments/paper_raw/pope/$dataset/llava_next_llama3_8b/vga_sdpa_tok8_layers0_16_full9000/pred_vga.jsonl"
  elif [[ "$target" == "vga_qwen25_vl_7b" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/qwen25_vl_7b/vga_eager_tok8_layers4_16_full9000/pred_vga.jsonl"
  elif [[ "$target" == "vga_qwen25_vl_7b" ]]; then
    echo "$CAL/experiments/paper_raw/pope/$dataset/qwen25_vl_7b/vga_eager_tok8_layers4_16_full9000/pred_vga.jsonl"
  elif [[ "$target" == "llava15_vaf" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/llava15_7b/vaf_clearsight_full9000/pred_vaf.jsonl"
  elif [[ "$target" == "llava15_vaf" ]]; then
    echo "$CAL/experiments/paper_raw/pope_transfer_llava15_mscoco_policy/$dataset/llava15_7b/vaf_clearsight_full9000/pred_vaf.jsonl"
  elif [[ "$target" == "llava15_pai_attn" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/llava15_7b/pai_attn_full9000/pred_pai_attn.jsonl"
  elif [[ "$target" == "llava15_pai_attn" ]]; then
    echo "$CAL/experiments/paper_raw/pope_transfer_llava15_mscoco_policy/$dataset/llava15_7b/pai_attn_full9000/pred_pai_attn.jsonl"
  elif [[ "$target" == "llava_next_vaf" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/llava_next_llama3_8b/vaf_sdpa_tok8_layers9_14_full9000/pred_vaf.jsonl"
  elif [[ "$target" == "llava_next_vaf" ]]; then
    echo "$CAL/experiments/paper_raw/pope/$dataset/llava_next_llama3_8b/vaf_sdpa_tok8_layers9_14_full9000/pred_vaf.jsonl"
  elif [[ "$target" == "llava_next_pai_attn" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/llava_next_llama3_8b/pai_attn_sdpa_tok8_layers0_16_full9000/pred_pai_attn.jsonl"
  elif [[ "$target" == "llava_next_pai_attn" ]]; then
    echo "$CAL/experiments/paper_raw/pope/$dataset/llava_next_llama3_8b/pai_attn_sdpa_tok8_layers0_16_full9000/pred_pai_attn.jsonl"
  elif [[ "$target" == "qwen25_vaf" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/qwen25_vl_7b/vaf_eager_tok8_layers9_14_full9000/pred_vaf.jsonl"
  elif [[ "$target" == "qwen25_vaf" ]]; then
    echo "$CAL/experiments/paper_raw/pope/$dataset/qwen25_vl_7b/vaf_eager_tok8_layers9_14_full9000/pred_vaf.jsonl"
  elif [[ "$target" == "qwen25_pai_attn" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/qwen25_vl_7b/pai_attn_eager_tok8_layers4_16_full9000/pred_pai_attn.jsonl"
  elif [[ "$target" == "qwen25_pai_attn" ]]; then
    echo "$CAL/experiments/paper_raw/pope/$dataset/qwen25_vl_7b/pai_attn_eager_tok8_layers4_16_full9000/pred_pai_attn.jsonl"
  else
    echo "[error] unsupported method path: $target $dataset" >&2
    exit 2
  fi
}

for root in $APPLY_ROOTS; do
  [[ -d "$root" ]] || { echo "[skip missing root] $root" >&2; continue; }
  while IFS= read -r route_rows; do
    route_dir="$(dirname "$route_rows")"
    dataset="$(basename "$route_dir")"
    target="$(basename "$(dirname "$route_dir")")"
    case "$dataset" in yes_to_no|no_to_yes) continue ;; esac
    contains_word "$dataset" "$DATASETS" || continue
    if [[ -n "$TARGETS" ]] && ! contains_word "$target" "$TARGETS"; then
      continue
    fi
    if contains_word "$target" "$EXCLUDE_TARGETS"; then
      continue
    fi

    backbone="$(target_backbone "$target")"
    gt_csv="$(dataset_gt "$dataset")"
    base_pred="$(baseline_pred_path "$backbone" "$dataset")"
    method_pred="$(method_pred_path "$target" "$dataset")"
    for path in "$gt_csv" "$base_pred" "$method_pred" "$route_rows"; do
      [[ -f "$path" ]] || { echo "[missing] $target $dataset $path" >&2; continue 2; }
    done
    echo "== refresh $target $dataset"
    "$CAL_PY" "$CAL/scripts/summarize_pcp_deployment_from_routes.py" \
      --gt_csv "$gt_csv" \
      --baseline_pred_jsonl "$base_pred" \
      --intervention_pred_jsonl "$method_pred" \
      --route_rows_csv "$route_rows" \
      --baseline_pred_text_key auto \
      --intervention_pred_text_key auto \
      --out_json "$route_dir/deployment_summary.json"
  done < <(find "$root" -type f -name pcp_route_rows.csv | sort)
done
