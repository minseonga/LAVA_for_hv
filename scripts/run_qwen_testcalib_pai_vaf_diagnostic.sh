#!/usr/bin/env bash
set -euo pipefail

# Oracle diagnostic: calibrate RaPiC on each Qwen test split and apply to the
# same split. This is not a paper protocol; it checks whether failure comes from
# discovery calibration mismatch or from non-separable harm/help on the method.
#
# Example:
#   bash scripts/run_qwen_testcalib_pai_vaf_diagnostic.sh
#   CANDIDATE_FILTER=yes_to_no bash scripts/run_qwen_testcalib_pai_vaf_diagnostic.sh
#   ALLOW_NOOP_POLICY=true bash scripts/run_qwen_testcalib_pai_vaf_diagnostic.sh

CAL="${CAL:-/home/kms/LLaVA_calibration}"
CAL_PY="${CAL_PY:-/home/kms/miniconda3/envs/vga_base/bin/python}"

RUN_ROOT="${RUN_ROOT:-$CAL/experiments/paper_pcp_cd_finalacc_alpha0p025_pai_vaf_main}"
METHOD_ROOT="${METHOD_ROOT:-$RUN_ROOT/methods}"
CANDIDATE_FILTER="${CANDIDATE_FILTER:-changed_answer}"
ALLOW_NOOP_POLICY="${ALLOW_NOOP_POLICY:-false}"
MIN_SELECTED_COUNT="${MIN_SELECTED_COUNT:-5}"
MAX_HELP_RECALL="${MAX_HELP_RECALL:-1.0}"

if [[ "$ALLOW_NOOP_POLICY" == "true" ]]; then
  NOOP_TAG="gated"
else
  NOOP_TAG="forced"
fi

OUT_ROOT="${OUT_ROOT:-$RUN_ROOT/testcalib_${CANDIDATE_FILTER}_${NOOP_TAG}}"
POL_ROOT="$OUT_ROOT/policies"
APPLY_ROOT="$OUT_ROOT/apply"

C_COLS="${C_COLS:-cheap_lp_content_min,cheap_lp_content_std,cheap_entropy_content_mean,cheap_first_target_gap,cheap_target_gap_content_min}"
D_COLS="${D_COLS:-cheap_decision_candidate_minus_alt,cheap_decision_candidate_prob_binary,cheap_decision_candidate_label_lp,cheap_decision_candidate_kl_uniform}"
ALPHA_GRID="${ALPHA_GRID:-$("$CAL_PY" - <<'PY'
print(",".join(f"{i/40:.3f}" for i in range(1, 40)))
PY
)}"

TARGETS="${TARGETS:-qwen25_vaf qwen25_pai_attn}"
DATASETS="${DATASETS:-mscoco aokvqa gqa}"

log() {
  printf '\n== %s ==\n' "$*"
}

check_file() {
  if [[ ! -f "$1" ]]; then
    echo "[missing] $1" >&2
    exit 2
  fi
}

dataset_gt() {
  local dataset="$1"
  if [[ "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/pope_full_9000/pope_9000_gt.csv"
  else
    echo "$CAL/experiments/pope_hf_multidataset/$dataset/pope_${dataset}_9000_gt.csv"
  fi
}

base_pred_path() {
  local dataset="$1"
  if [[ "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/qwen25_vl_7b/baseline_eager_tok8_full9000/pred_baseline.jsonl"
  else
    echo "$CAL/experiments/paper_raw/pope/$dataset/qwen25_vl_7b/baseline_eager_tok8_full9000/pred_baseline.jsonl"
  fi
}

method_pred_path() {
  local target="$1"
  local dataset="$2"
  local method_dir pred_name
  if [[ "$target" == "qwen25_vaf" ]]; then
    method_dir="vaf_eager_tok8_layers9_14_full9000"
    pred_name="pred_vaf.jsonl"
  elif [[ "$target" == "qwen25_pai_attn" ]]; then
    method_dir="pai_attn_eager_tok8_layers4_16_full9000"
    pred_name="pred_pai_attn.jsonl"
  else
    echo "[error] unsupported target: $target" >&2
    exit 2
  fi

  if [[ "$dataset" == "mscoco" ]]; then
    echo "$CAL/experiments/paper_raw/pope/qwen25_vl_7b/$method_dir/$pred_name"
  else
    echo "$CAL/experiments/paper_raw/pope/$dataset/qwen25_vl_7b/$method_dir/$pred_name"
  fi
}

feature_rows_path() {
  local target="$1"
  local dataset="$2"
  echo "$METHOD_ROOT/$target/apply_$dataset/features/online_feature_rows.csv"
}

build_policy_for_split() {
  local target="$1"
  local dataset="$2"
  local rows_csv="$3"
  local out_dir="$POL_ROOT/$target/$dataset"

  check_file "$rows_csv"
  mkdir -p "$out_dir"
  log "test-calibrate policy $target/$dataset"

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
    --min_selected_count "$MIN_SELECTED_COUNT" \
    --max_help_recall "$MAX_HELP_RECALL" \
    --allow_noop_policy "$ALLOW_NOOP_POLICY" \
    --candidate_filter "$CANDIDATE_FILTER" \
    --out_dir "$out_dir"

  python -m json.tool "$out_dir/summary.json" \
    | rg '"family"|"disabled"|"alpha"|"tau"|"selected_count"|"selected_harm"|"selected_help"|"net"|"final_acc"|"delta_vs_intervention"' || true
}

apply_policy_for_split() {
  local target="$1"
  local dataset="$2"
  local rows_csv="$3"
  local policy_json="$4"
  local gt_csv base_pred method_pred out_dir

  gt_csv="$(dataset_gt "$dataset")"
  base_pred="$(base_pred_path "$dataset")"
  method_pred="$(method_pred_path "$target" "$dataset")"
  out_dir="$APPLY_ROOT/$target/$dataset"

  check_file "$rows_csv"
  check_file "$policy_json"
  check_file "$gt_csv"
  check_file "$base_pred"
  check_file "$method_pred"

  mkdir -p "$out_dir"
  log "apply test-calibrated policy $target/$dataset"

  "$CAL_PY" "$CAL/scripts/apply_pcp_c_d_controller.py" \
    --rows_csv "$rows_csv" \
    --policy_json "$policy_json" \
    --out_dir "$out_dir" \
    --family selected \
    --candidate_filter "$CANDIDATE_FILTER" \
    --derive_decision_kl true

  "$CAL_PY" "$CAL/scripts/summarize_pcp_deployment_from_routes.py" \
    --gt_csv "$gt_csv" \
    --baseline_pred_jsonl "$base_pred" \
    --intervention_pred_jsonl "$method_pred" \
    --route_rows_csv "$out_dir/pcp_route_rows.csv" \
    --baseline_pred_text_key auto \
    --intervention_pred_text_key auto \
    --out_json "$out_dir/deployment_summary.json"
}

write_summary() {
  log "summary table"
  RUN_ROOT="$RUN_ROOT" OUT_ROOT="$OUT_ROOT" POL_ROOT="$POL_ROOT" APPLY_ROOT="$APPLY_ROOT" \
  TARGETS="$TARGETS" DATASETS="$DATASETS" "$CAL_PY" - <<'PY'
import json
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
pol_root = Path(os.environ["POL_ROOT"])
apply_root = Path(os.environ["APPLY_ROOT"])
targets = os.environ["TARGETS"].split()
datasets = os.environ["DATASETS"].split()
labels = {
    "qwen25_vaf": "VAF / Qwen2.5-VL-7B",
    "qwen25_pai_attn": "PAI-attn / Qwen2.5-VL-7B",
}

lines = [
    "| Method / Backbone | Dataset | Family | Alpha | Tau | Base | Method | Test-Calib RaPiC | dMethod | dBase | Fallback | H/G/Net | Hrec | Grec |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for target in targets:
    for dataset in datasets:
        policy_path = pol_root / target / dataset / "selected_policy.json"
        summary_path = apply_root / target / dataset / "deployment_summary.json"
        if not policy_path.exists() or not summary_path.exists():
            lines.append(f"| {labels.get(target, target)} | {dataset} | missing | | | | | | | | | | | |")
            continue
        sp = json.load(open(policy_path))["selected_policy"]
        d = json.load(open(summary_path))
        hrec = d["selected_harm"] / d["total_harm"] if d["total_harm"] else 0.0
        grec = d["selected_help"] / d["total_help"] if d["total_help"] else 0.0
        family = sp.get("family", "")
        alpha = sp.get("alpha", 0.0)
        tau = sp.get("tau", 0.0)
        lines.append(
            f"| {labels.get(target, target)} | {dataset} | {family} | {float(alpha):.3f} | {float(tau):.4f} | "
            f"{100*d['baseline_acc']:.2f} | {100*d['intervention_acc']:.2f} | {100*d['pcp_deploy_acc']:.2f} | "
            f"{100*d['delta_vs_intervention']:+.2f} | {100*(d['pcp_deploy_acc'] - d['baseline_acc']):+.2f} | "
            f"{d['baseline_generated']} | {d['selected_harm']}/{d['selected_help']}/{d['net']} | {100*hrec:.2f} | {100*grec:.2f} |"
        )

out_md = out_root / "qwen_testcalib_summary.md"
out_md.parent.mkdir(parents=True, exist_ok=True)
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print("[saved]", out_md)
PY
}

log "config"
echo "CAL=$CAL"
echo "RUN_ROOT=$RUN_ROOT"
echo "METHOD_ROOT=$METHOD_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "CANDIDATE_FILTER=$CANDIDATE_FILTER"
echo "ALLOW_NOOP_POLICY=$ALLOW_NOOP_POLICY"
echo "MIN_SELECTED_COUNT=$MIN_SELECTED_COUNT"
echo "MAX_HELP_RECALL=$MAX_HELP_RECALL"
echo "TARGETS=$TARGETS"
echo "DATASETS=$DATASETS"

for target in $TARGETS; do
  for dataset in $DATASETS; do
    rows_csv="$(feature_rows_path "$target" "$dataset")"
    build_policy_for_split "$target" "$dataset" "$rows_csv"
    apply_policy_for_split "$target" "$dataset" "$rows_csv" "$POL_ROOT/$target/$dataset/selected_policy.json"
  done
done

write_summary
