#!/usr/bin/env bash
set -euo pipefail

# Run paper-main RaPiC calibration for raw PAI/VAF results that already exist
# or can be generated without LLaVA-NeXT.
#
# Main protocol:
#   - discovery split only for family/alpha/tau selection
#   - candidate_filter=changed_answer
#   - objective=final_acc
#   - family is free among c_only, d_only, cd_fusion
#   - cd_fusion alpha grid: 0.025, 0.050, ..., 0.975
#
# Default targets:
#   llava15_vaf        reuses existing VAF discovery/test feature rows
#   llava15_pai_attn   generates discovery raw/features, uses existing full raw
#   qwen25_vaf         generates discovery raw/features, uses existing full raw
#   qwen25_pai_attn    generates discovery raw/features, uses existing full raw
#
# Example:
#   bash scripts/run_finalacc_pai_vaf_main.sh
#   TARGETS="qwen25_vaf qwen25_pai_attn" FEAT_GPU=2 RAW_GPU=2 bash scripts/run_finalacc_pai_vaf_main.sh

CAL="${CAL:-/home/kms/LLaVA_calibration}"
CAL_PY="${CAL_PY:-/home/kms/miniconda3/envs/vga_base/bin/python}"
QWEN_PY="${QWEN_PY:-/home/kms/miniconda3/envs/qwen25_vl/bin/python}"
PAI_PY="${PAI_PY:-/home/kms/miniconda3/envs/pai_base/bin/python}"
PAI_ROOT="${PAI_ROOT:-/home/kms/PAI}"

RAW_GPU="${RAW_GPU:-2}"
FEAT_GPU="${FEAT_GPU:-2}"

RUN_ROOT="${RUN_ROOT:-$CAL/experiments/paper_pcp_cd_finalacc_alpha0p025_pai_vaf_main}"
POL_ROOT="${POL_ROOT:-$RUN_ROOT/policies}"
mkdir -p "$POL_ROOT"

OLD="${OLD:-$CAL/experiments/pope_discovery/tau_c_calibration_adversarial/assets}"
DISC_BASE="${DISC_BASE:-$CAL/experiments/pope_discovery/tau_c_calibration_adversarial/baseline/pred_baseline.jsonl}"
DISC_IMG="${DISC_IMG:-}"
if [[ -z "$DISC_IMG" ]]; then
  DISC_SAMPLE="${DISC_SAMPLE:-COCO_train2014_000000126408.jpg}"
  DISC_FOUND="$(find /home/kms/data -name "$DISC_SAMPLE" -print -quit)"
  if [[ -z "$DISC_FOUND" ]]; then
    echo "[error] could not find discovery image sample: $DISC_SAMPLE" >&2
    exit 2
  fi
  DISC_IMG="$(dirname "$DISC_FOUND")"
fi

HEADSET="${HEADSET:-$CAL/experiments/pope_headsets_v1/headset.json}"
LLAVA15_MODEL="${LLAVA15_MODEL:-liuhaotian/llava-v1.5-7b}"
QWEN_MODEL="${QWEN_MODEL:-/home/kms/models/Qwen2.5-VL-7B-Instruct}"

C_COLS="${C_COLS:-cheap_lp_content_min,cheap_lp_content_std,cheap_entropy_content_mean,cheap_first_target_gap,cheap_target_gap_content_min}"
D_COLS="${D_COLS:-cheap_decision_candidate_minus_alt,cheap_decision_candidate_prob_binary,cheap_decision_candidate_label_lp,cheap_decision_candidate_kl_uniform}"
ALPHA_GRID="${ALPHA_GRID:-$("$CAL_PY" - <<'PY'
print(",".join(f"{i/40:.3f}" for i in range(1, 40)))
PY
)}"

TARGETS="${TARGETS:-llava15_vaf llava15_pai_attn qwen25_vaf qwen25_pai_attn}"

log() {
  printf '\n== %s ==\n' "$*"
}

check_file() {
  if [[ ! -f "$1" ]]; then
    echo "[missing] $1" >&2
    exit 2
  fi
}

dataset_paths() {
  local dataset="$1"
  if [[ "$dataset" == "mscoco" ]]; then
    DS_Q="$CAL/experiments/pope_full_9000/pope_9000_q_with_object.jsonl"
    DS_GT="$CAL/experiments/pope_full_9000/pope_9000_gt.csv"
    DS_IMG="/home/kms/data/pope/val2014"
  elif [[ "$dataset" == "aokvqa" ]]; then
    DS_Q="$CAL/experiments/pope_hf_multidataset/aokvqa/pope_aokvqa_9000_q_with_object.jsonl"
    DS_GT="$CAL/experiments/pope_hf_multidataset/aokvqa/pope_aokvqa_9000_gt.csv"
    DS_IMG="/home/kms/data/pope/val2014"
  elif [[ "$dataset" == "gqa" ]]; then
    DS_Q="$CAL/experiments/pope_hf_multidataset/gqa/pope_gqa_9000_q_with_object.jsonl"
    DS_GT="$CAL/experiments/pope_hf_multidataset/gqa/pope_gqa_9000_gt.csv"
    DS_IMG="/home/kms/data/gqa/images"
  else
    echo "[error] unknown dataset: $dataset" >&2
    exit 2
  fi
}

base_pred_path() {
  local backbone="$1"
  local dataset="$2"
  if [[ "$backbone" == "llava15" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/pope_full_9000/stage_b_signal_validation_vga/pred_baseline.jsonl"
  elif [[ "$backbone" == "llava15" && "$dataset" == "aokvqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope_transfer_llava15_mscoco_policy/aokvqa/llava15_7b/baseline_full9000/pred_baseline.jsonl"
  elif [[ "$backbone" == "llava15" && "$dataset" == "gqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope_transfer_llava15_mscoco_policy/gqa/llava15_7b/baseline_full9000/pred_baseline.jsonl"
  elif [[ "$backbone" == "qwen25" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/qwen25_vl_7b/baseline_eager_tok8_full9000/pred_baseline.jsonl"
  elif [[ "$backbone" == "qwen25" && "$dataset" == "aokvqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope/aokvqa/qwen25_vl_7b/baseline_eager_tok8_full9000/pred_baseline.jsonl"
  elif [[ "$backbone" == "qwen25" && "$dataset" == "gqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope/gqa/qwen25_vl_7b/baseline_eager_tok8_full9000/pred_baseline.jsonl"
  else
    echo "[error] unsupported base path: $backbone $dataset" >&2
    exit 2
  fi
}

full_method_pred_path() {
  local target="$1"
  local dataset="$2"
  if [[ "$target" == "llava15_vaf" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/llava15_7b/vaf_clearsight_full9000/pred_vaf.jsonl"
  elif [[ "$target" == "llava15_vaf" && "$dataset" == "aokvqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope_transfer_llava15_mscoco_policy/aokvqa/llava15_7b/vaf_clearsight_full9000/pred_vaf.jsonl"
  elif [[ "$target" == "llava15_vaf" && "$dataset" == "gqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope_transfer_llava15_mscoco_policy/gqa/llava15_7b/vaf_clearsight_full9000/pred_vaf.jsonl"
  elif [[ "$target" == "llava15_pai_attn" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/llava15_7b/pai_attn_full9000/pred_pai_attn.jsonl"
  elif [[ "$target" == "llava15_pai_attn" && "$dataset" == "aokvqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope_transfer_llava15_mscoco_policy/aokvqa/llava15_7b/pai_attn_full9000/pred_pai_attn.jsonl"
  elif [[ "$target" == "llava15_pai_attn" && "$dataset" == "gqa" ]]; then
    echo "$CAL/experiments/paper_raw/pope_transfer_llava15_mscoco_policy/gqa/llava15_7b/pai_attn_full9000/pred_pai_attn.jsonl"
  elif [[ "$target" == "qwen25_vaf" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/qwen25_vl_7b/vaf_eager_tok8_layers9_14_full9000/pred_vaf.jsonl"
  elif [[ "$target" == "qwen25_vaf" ]]; then
    echo "$CAL/experiments/paper_raw/pope/$dataset/qwen25_vl_7b/vaf_eager_tok8_layers9_14_full9000/pred_vaf.jsonl"
  elif [[ "$target" == "qwen25_pai_attn" && "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/qwen25_vl_7b/pai_attn_eager_tok8_layers4_16_full9000/pred_pai_attn.jsonl"
  elif [[ "$target" == "qwen25_pai_attn" ]]; then
    echo "$CAL/experiments/paper_raw/pope/$dataset/qwen25_vl_7b/pai_attn_eager_tok8_layers4_16_full9000/pred_pai_attn.jsonl"
  else
    echo "[error] unsupported method pred path: $target $dataset" >&2
    exit 2
  fi
}

target_backbone() {
  case "$1" in
    llava15_*) echo "llava15" ;;
    qwen25_*) echo "qwen25" ;;
    *) echo "[error] unsupported target: $1" >&2; exit 2 ;;
  esac
}

target_raw_method() {
  case "$1" in
    *_vaf) echo "vaf" ;;
    *_pai_attn) echo "pai_attn" ;;
    *) echo "[error] unsupported target: $1" >&2; exit 2 ;;
  esac
}

target_pred_key() {
  case "$1" in
    llava15_vaf) echo "text" ;;
    *) echo "auto" ;;
  esac
}

target_model_path() {
  case "$1" in
    llava15_*) echo "$LLAVA15_MODEL" ;;
    qwen25_*) echo "$QWEN_MODEL" ;;
  esac
}

target_runtime_backend() {
  case "$1" in
    llava15_*) echo "llava15_cleanroom" ;;
    qwen25_*) echo "qwen25_vl_official" ;;
  esac
}

discovery_pred_for_target() {
  local target="$1"
  local raw_method
  raw_method="$(target_raw_method "$target")"
  if [[ "$target" == "llava15_vaf" ]]; then
    echo "$CAL/experiments/paper_pcp_cd/llava15/pai_vaf_oldcompact/vaf/raw_discovery/pred_vaf.jsonl"
  elif [[ "$raw_method" == "vaf" ]]; then
    echo "$RUN_ROOT/methods/$target/raw_discovery/pred_vaf.jsonl"
  else
    echo "$RUN_ROOT/methods/$target/raw_discovery/pred_pai_attn.jsonl"
  fi
}

ensure_discovery_raw() {
  local target="$1"
  local backbone raw_method out pred
  backbone="$(target_backbone "$target")"
  raw_method="$(target_raw_method "$target")"
  pred="$(discovery_pred_for_target "$target")"

  if [[ -s "$pred" ]]; then
    log "reuse discovery raw $target"
    return
  fi

  if [[ "$target" == "llava15_vaf" ]]; then
    echo "[missing] expected existing LLaVA-1.5 VAF discovery raw: $pred" >&2
    exit 2
  fi

  out="$RUN_ROOT/methods/$target/raw_discovery"
  mkdir -p "$out"
  log "generate discovery raw $target on GPU $RAW_GPU"

  if [[ "$backbone" == "llava15" ]]; then
    env \
      GPU="$RAW_GPU" \
      CAL_ROOT="$CAL" \
      CAL_PYTHON_BIN="$CAL_PY" \
      PAI_PYTHON_BIN="$PAI_PY" \
      PAI_ROOT="$PAI_ROOT" \
      BACKBONE=llava15 \
      METHOD="$raw_method" \
      TASK=pope \
      MODEL_PATH="$LLAVA15_MODEL" \
      IMAGE_FOLDER="$DISC_IMG" \
      QUESTION_FILE="$OLD/discovery_q_with_object.jsonl" \
      GT_CSV="$OLD/discovery_gt.csv" \
      OUT_ROOT="$out" \
      MAX_NEW_TOKENS=8 \
      PAI_USE_ATTN=1 \
      PAI_USE_CFG=0 \
      PAI_START_LAYER=2 \
      PAI_END_LAYER=15 \
      REUSE_IF_EXISTS=false \
      bash "$CAL/scripts/run_multibackbone_method_prediction.sh" \
      > "$out/run.log" 2>&1
  else
    env \
      GPU="$RAW_GPU" \
      PYTHONPATH="$CAL:${PYTHONPATH:-}" \
      CAL_ROOT="$CAL" \
      CAL_PYTHON_BIN="$CAL_PY" \
      QWEN25_PYTHON_BIN="$QWEN_PY" \
      BACKBONE=qwen25_vl \
      METHOD="$raw_method" \
      TASK=pope \
      MODEL_PATH="$QWEN_MODEL" \
      IMAGE_FOLDER="$DISC_IMG" \
      QUESTION_FILE="$OLD/discovery_q_with_object.jsonl" \
      GT_CSV="$OLD/discovery_gt.csv" \
      OUT_ROOT="$out" \
      MAX_NEW_TOKENS=8 \
      VGA_TORCH_TYPE=bf16 \
      QWEN25_DEVICE_MAP=cuda \
      VAF_START_LAYER=9 \
      VAF_END_LAYER=14 \
      PAI_USE_ATTN=1 \
      PAI_USE_CFG=0 \
      PAI_START_LAYER=4 \
      PAI_END_LAYER=16 \
      REUSE_IF_EXISTS=false \
      bash "$CAL/scripts/run_multibackbone_method_prediction.sh" \
      > "$out/run.log" 2>&1
  fi

  check_file "$pred"
}

build_changed_subset() {
  local target="$1"
  local question_jsonl="$2"
  local gt_csv="$3"
  local base_pred="$4"
  local method_pred="$5"
  local pred_key="$6"
  local out_dir="$7"

  if [[ -f "$out_dir/summary.json" ]]; then
    log "reuse changed subset $target -> $out_dir"
    return
  fi

  log "changed subset $target -> $out_dir"
  "$CAL_PY" "$CAL/scripts/build_changed_pope_subset.py" \
    --question_jsonl "$question_jsonl" \
    --gt_csv "$gt_csv" \
    --baseline_pred_jsonl "$base_pred" \
    --intervention_pred_jsonl "$method_pred" \
    --baseline_pred_text_key auto \
    --intervention_pred_text_key "$pred_key" \
    --mode changed_answer \
    --out_dir "$out_dir"
}

extract_features() {
  local target="$1"
  local question_jsonl="$2"
  local gt_csv="$3"
  local image_folder="$4"
  local base_pred="$5"
  local method_pred="$6"
  local pred_key="$7"
  local out_dir="$8"
  local backend model_path
  backend="$(target_runtime_backend "$target")"
  model_path="$(target_model_path "$target")"

  if [[ -f "$out_dir/online_feature_rows.csv" ]]; then
    log "reuse features $target -> $out_dir"
    return
  fi

  mkdir -p "$out_dir"
  log "extract features $target -> $out_dir on GPU $FEAT_GPU"

  if [[ "$backend" == "qwen25_vl_official" ]]; then
    CUDA_VISIBLE_DEVICES="$FEAT_GPU" PYTHONPATH="$CAL:${PYTHONPATH:-}" \
      "$QWEN_PY" "$CAL/scripts/run_discriminative_meta_strong_online.py" \
      --question_file "$question_jsonl" \
      --image_folder "$image_folder" \
      --intervention_pred_jsonl "$method_pred" \
      --headset_json "$HEADSET" \
      --out_dir "$out_dir" \
      --baseline_pred_jsonl "$base_pred" \
      --gt_csv "$gt_csv" \
      --model_path "$model_path" \
      --runtime_backend qwen25_vl_official \
      --qwen25_torch_type bf16 \
      --qwen25_attn_implementation eager \
      --qwen25_device_map cuda \
      --baseline_pred_key auto \
      --intervention_pred_key "$pred_key" \
      --extract_only true \
      --skip_stage_a true \
      --reuse_if_exists false \
      --log_every 50
  else
    CUDA_VISIBLE_DEVICES="$FEAT_GPU" PYTHONPATH="$CAL:${PYTHONPATH:-}" \
      "$CAL_PY" "$CAL/scripts/run_discriminative_meta_strong_online.py" \
      --question_file "$question_jsonl" \
      --image_folder "$image_folder" \
      --intervention_pred_jsonl "$method_pred" \
      --headset_json "$HEADSET" \
      --out_dir "$out_dir" \
      --baseline_pred_jsonl "$base_pred" \
      --gt_csv "$gt_csv" \
      --model_path "$model_path" \
      --runtime_backend llava15_cleanroom \
      --baseline_pred_key auto \
      --intervention_pred_key "$pred_key" \
      --extract_only true \
      --skip_stage_a true \
      --reuse_if_exists false \
      --log_every 50
  fi
}

existing_llava15_vaf_features() {
  local dataset="$1"
  if [[ "$dataset" == "discovery" ]]; then
    echo "$CAL/experiments/paper_pcp_cd/llava15/pai_vaf_oldcompact/vaf/discovery_features_changed/online_feature_rows.csv"
  else
    echo "$CAL/experiments/paper_pcp_cd/llava15/pai_vaf_oldcompact/vaf/apply_$dataset/features_changed/online_feature_rows.csv"
  fi
}

build_policy() {
  local target="$1"
  local rows_csv="$2"
  local out_dir="$POL_ROOT/$target"
  check_file "$rows_csv"
  mkdir -p "$out_dir"

  if [[ -f "$out_dir/selected_policy.json" ]]; then
    log "reuse policy $target"
    return
  fi

  log "build final_acc policy $target"
  "$CAL_PY" "$CAL/scripts/build_pcp_c_d_controller.py" \
    --rows_csv "$rows_csv" \
    --c_feature_cols "$C_COLS" \
    --d_feature_cols "$D_COLS" \
    --derive_decision_kl true \
    --min_present_rate 0.8 \
    --min_feature_auroc 0.55 \
    --top_k_c 3 \
    --top_k_d 4 \
    --alpha_grid "$ALPHA_GRID" \
    --tau_objective final_acc \
    --min_baseline_rate 0.0 \
    --max_baseline_rate 1.0 \
    --min_selected_count 5 \
    --candidate_filter changed_answer \
    --out_dir "$out_dir"

  python -m json.tool "$out_dir/summary.json" \
    | rg '"family"|"alpha"|"tau"|"selected_count"|"selected_harm"|"selected_help"|"net"|"selected_harm_precision"|"selected_harm_recall"|"delta_vs_intervention"' || true
}

apply_policy() {
  local target="$1"
  local dataset="$2"
  local rows_csv="$3"
  local gt_csv="$4"
  local base_pred="$5"
  local method_pred="$6"
  local pred_key="$7"
  local out_dir="$RUN_ROOT/apply/$target/$dataset"

  check_file "$rows_csv"
  check_file "$gt_csv"
  check_file "$base_pred"
  check_file "$method_pred"
  check_file "$POL_ROOT/$target/selected_policy.json"

  mkdir -p "$out_dir"
  log "apply policy $target $dataset"

  "$CAL_PY" "$CAL/scripts/apply_pcp_c_d_controller.py" \
    --rows_csv "$rows_csv" \
    --policy_json "$POL_ROOT/$target/selected_policy.json" \
    --out_dir "$out_dir" \
    --family selected \
    --candidate_filter changed_answer \
    --derive_decision_kl true

  "$CAL_PY" "$CAL/scripts/summarize_pcp_deployment_from_routes.py" \
    --gt_csv "$gt_csv" \
    --baseline_pred_jsonl "$base_pred" \
    --intervention_pred_jsonl "$method_pred" \
    --route_rows_csv "$out_dir/pcp_route_rows.csv" \
    --baseline_pred_text_key auto \
    --intervention_pred_text_key "$pred_key" \
    --out_json "$out_dir/deployment_summary.json"
}

run_target() {
  local target="$1"
  local backbone pred_key disc_pred method_root disc_changed disc_features
  backbone="$(target_backbone "$target")"
  pred_key="$(target_pred_key "$target")"
  method_root="$RUN_ROOT/methods/$target"

  log "target $target"
  ensure_discovery_raw "$target"
  disc_pred="$(discovery_pred_for_target "$target")"

  if [[ "$target" == "llava15_vaf" ]]; then
    disc_features="$(existing_llava15_vaf_features discovery)"
    check_file "$disc_features"
  else
    disc_changed="$method_root/discovery_changed"
    build_changed_subset "$target/discovery" \
      "$OLD/discovery_q_with_object.jsonl" \
      "$OLD/discovery_gt.csv" \
      "$DISC_BASE" \
      "$disc_pred" \
      "$pred_key" \
      "$disc_changed"

    disc_features="$method_root/discovery_features"
    extract_features "$target/discovery" \
      "$disc_changed/changed_q_with_object.jsonl" \
      "$disc_changed/changed_gt.csv" \
      "$DISC_IMG" \
      "$DISC_BASE" \
      "$disc_pred" \
      "$pred_key" \
      "$disc_features"
    disc_features="$disc_features/online_feature_rows.csv"
  fi

  build_policy "$target" "$disc_features"

  for dataset in mscoco aokvqa gqa; do
    local base_pred method_pred test_changed test_features rows_csv
    dataset_paths "$dataset"
    base_pred="$(base_pred_path "$backbone" "$dataset")"
    method_pred="$(full_method_pred_path "$target" "$dataset")"
    check_file "$base_pred"
    check_file "$method_pred"

    if [[ "$target" == "llava15_vaf" ]]; then
      rows_csv="$(existing_llava15_vaf_features "$dataset")"
      check_file "$rows_csv"
    else
      test_changed="$method_root/apply_$dataset/changed"
      build_changed_subset "$target/$dataset" \
        "$DS_Q" \
        "$DS_GT" \
        "$base_pred" \
        "$method_pred" \
        "$pred_key" \
        "$test_changed"

      test_features="$method_root/apply_$dataset/features"
      extract_features "$target/$dataset" \
        "$test_changed/changed_q_with_object.jsonl" \
        "$test_changed/changed_gt.csv" \
        "$DS_IMG" \
        "$base_pred" \
        "$method_pred" \
        "$pred_key" \
        "$test_features"
      rows_csv="$test_features/online_feature_rows.csv"
    fi

    apply_policy "$target" "$dataset" "$rows_csv" "$DS_GT" "$base_pred" "$method_pred" "$pred_key"
  done
}

write_summary_table() {
  log "summary table"
  RUN_ROOT="$RUN_ROOT" POL_ROOT="$POL_ROOT" "$CAL_PY" - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ["RUN_ROOT"])
pol = Path(os.environ["POL_ROOT"])
targets = ["llava15_vaf", "llava15_pai_attn", "qwen25_vaf", "qwen25_pai_attn"]
labels = {
    "llava15_vaf": "VAF / LLaVA-1.5",
    "llava15_pai_attn": "PAI-attn / LLaVA-1.5",
    "qwen25_vaf": "VAF / Qwen2.5-VL-7B",
    "qwen25_pai_attn": "PAI-attn / Qwen2.5-VL-7B",
}

lines = [
    "| Method / Backbone | Dataset | Family | Alpha | Tau | Base | Method | RaPiC | dMethod | dBase | Fallback | H/G/Net | Hrec | Grec |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for target in targets:
    policy_path = pol / target / "selected_policy.json"
    if not policy_path.exists():
        continue
    sp = json.load(open(policy_path))["selected_policy"]
    for dataset in ["mscoco", "aokvqa", "gqa"]:
        p = run / "apply" / target / dataset / "deployment_summary.json"
        if not p.exists():
            lines.append(f"| {labels[target]} | {dataset} | missing | | | | | | | | | | | |")
            continue
        d = json.load(open(p))
        hrec = d["selected_harm"] / d["total_harm"] if d["total_harm"] else 0.0
        grec = d["selected_help"] / d["total_help"] if d["total_help"] else 0.0
        lines.append(
            f"| {labels[target]} | {dataset} | {sp['family']} | {float(sp.get('alpha', 0.0)):.3f} | {float(sp['tau']):.4f} | "
            f"{100*d['baseline_acc']:.2f} | {100*d['intervention_acc']:.2f} | {100*d['pcp_deploy_acc']:.2f} | "
            f"{100*d['delta_vs_intervention']:+.2f} | {100*(d['pcp_deploy_acc'] - d['baseline_acc']):+.2f} | "
            f"{d['baseline_generated']} | {d['selected_harm']}/{d['selected_help']}/{d['net']} | {100*hrec:.2f} | {100*grec:.2f} |"
        )

out_md = run / "pai_vaf_finalacc_alpha0p025_summary.md"
out_md.parent.mkdir(parents=True, exist_ok=True)
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print("[saved]", out_md)
PY
}

log "config"
echo "CAL=$CAL"
echo "RUN_ROOT=$RUN_ROOT"
echo "RAW_GPU=$RAW_GPU FEAT_GPU=$FEAT_GPU"
echo "DISC_IMG=$DISC_IMG"
echo "TARGETS=$TARGETS"

check_file "$CAL/scripts/run_multibackbone_method_prediction.sh"
check_file "$CAL/scripts/build_changed_pope_subset.py"
check_file "$CAL/scripts/run_discriminative_meta_strong_online.py"
check_file "$CAL/scripts/build_pcp_c_d_controller.py"
check_file "$CAL/scripts/apply_pcp_c_d_controller.py"
check_file "$CAL/scripts/summarize_pcp_deployment_from_routes.py"
check_file "$DISC_BASE"
check_file "$OLD/discovery_q_with_object.jsonl"
check_file "$OLD/discovery_gt.csv"
check_file "$HEADSET"

for target in $TARGETS; do
  run_target "$target"
done

write_summary_table
