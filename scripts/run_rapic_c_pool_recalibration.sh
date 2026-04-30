#!/usr/bin/env bash
set -Eeuo pipefail

# Recalibrate a RAPIC C-only policy from existing online_feature_rows.csv.
#
# This is intentionally feature-row only: it does not run model inference.
# Use it after extracting LLaVA-NeXT changed-answer rows, especially when
# testing expanded intervention-only C pools.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-python3}"
ROWS_CSV="${ROWS_CSV:-}"
OUT_DIR="${OUT_DIR:-}"
C_POOL_PRESET="${C_POOL_PRESET:-next_content_v2}"  # compact | next_content_v2 | next_hidden_object | next_hidden_object_trace
CANDIDATE_FILTER="${CANDIDATE_FILTER:-changed_answer}"
TAU_OBJECTIVE="${TAU_OBJECTIVE:-net}"
MIN_PRESENT_RATE="${MIN_PRESENT_RATE:-0.8}"
MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
TOP_K_C="${TOP_K_C:-4}"
MIN_SELECTED_COUNT="${MIN_SELECTED_COUNT:-5}"

if [[ -z "$ROWS_CSV" || -z "$OUT_DIR" ]]; then
  echo "usage: ROWS_CSV=/path/online_feature_rows.csv OUT_DIR=/path/out $0" >&2
  exit 2
fi

case "$C_POOL_PRESET" in
  compact)
    C_FEATURE_COLS="cheap_entropy_content_mean,cheap_first_target_gap,cheap_target_gap_content_min"
    ;;
  next_content_v2)
    # Single replay, intervention-only features already emitted by the cheap extractor.
    # This extends the original content pool with mean/tail trajectory instability
    # features that were stronger on LLaVA-NeXT discovery than min-only C.
    C_FEATURE_COLS="cheap_entropy_content_mean,cheap_first_target_gap,cheap_target_gap_content_min,cheap_lp_all_mean,cheap_lp_content_tail_gap,cheap_lp_content_min,cheap_margin_all_mean,cheap_margin_content_mean,cheap_conflict_gap_minus_entropy,cheap_target_gap_content_mean,cheap_entropy_content_max"
    ;;
  next_hidden_object)
    # Superset for rows extracted with CHEAP_HIDDEN_FEATURES=true and the object-token
    # extractor enabled. Missing columns are ignored by build_pcp_c_d_controller via
    # min_present_rate, so this preset is also safe on content-only rows.
    C_FEATURE_COLS="cheap_entropy_content_mean,cheap_first_target_gap,cheap_target_gap_content_min,cheap_lp_all_mean,cheap_lp_content_tail_gap,cheap_lp_content_min,cheap_margin_all_mean,cheap_margin_content_mean,cheap_conflict_gap_minus_entropy,cheap_target_gap_content_mean,cheap_entropy_content_max,cheap_hidden_answer_to_prompt_cos,cheap_hidden_answer_mean_norm,cheap_hidden_answer_vision_minus_prompt,cheap_hidden_answer_to_vision_cos,cheap_hidden_first_answer_to_prompt_cos,cheap_hidden_first_answer_norm,cheap_hidden_first_answer_to_vision_cos,cheap_margin_object_max,cheap_target_gap_object_max,cheap_object_gap_minus_entropy,cheap_margin_object_mean,cheap_target_gap_object_mean,cheap_first_object_target_gap,cheap_first_object_top1_margin,cheap_hidden_object_to_prompt_cos,cheap_hidden_object_mean_norm,cheap_hidden_object_to_vision_cos"
    ;;
  next_hidden_object_trace)
    # Adds VGA process-trace features joined from process_trace_features.csv.
    # These are still intervention-run features, not GT or baseline-answer features.
    C_FEATURE_COLS="cheap_entropy_content_mean,cheap_first_target_gap,cheap_target_gap_content_min,cheap_lp_all_mean,cheap_lp_content_tail_gap,cheap_lp_content_min,cheap_margin_all_mean,cheap_margin_content_mean,cheap_conflict_gap_minus_entropy,cheap_target_gap_content_mean,cheap_entropy_content_max,cheap_hidden_answer_to_prompt_cos,cheap_hidden_answer_mean_norm,cheap_hidden_answer_vision_minus_prompt,cheap_hidden_answer_to_vision_cos,cheap_hidden_first_answer_to_prompt_cos,cheap_hidden_first_answer_norm,cheap_hidden_first_answer_to_vision_cos,cheap_margin_object_max,cheap_target_gap_object_max,cheap_object_gap_minus_entropy,cheap_margin_object_mean,cheap_target_gap_object_mean,cheap_first_object_target_gap,cheap_first_object_top1_margin,cheap_hidden_object_to_prompt_cos,cheap_hidden_object_mean_norm,cheap_hidden_object_to_vision_cos,proc_label_add_candidate_minus_alt,proc_label_noadd_candidate_minus_alt,proc_label_add_candidate_lp,proc_label_noadd_candidate_lp,proc_label_margin_boost,proc_label_candidate_lp_boost,proc_label_add_kl_times_margin_boost,proc_eos_boost_max,proc_early_kl_add_to_noadd_mean,proc_kl_add_to_noadd_mean,proc_entropy_delta_max,proc_actual_visual_prob_max_max"
    ;;
  *)
    echo "[error] unknown C_POOL_PRESET=$C_POOL_PRESET" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR"

echo "[rapic-c-pool] rows=$ROWS_CSV"
echo "[rapic-c-pool] out=$OUT_DIR"
echo "[rapic-c-pool] preset=$C_POOL_PRESET top_k_c=$TOP_K_C objective=$TAU_OBJECTIVE candidate_filter=$CANDIDATE_FILTER"

PYTHONPATH="$ROOT_DIR" "$PY_BIN" "$ROOT_DIR/scripts/build_pcp_c_d_controller.py" \
  --rows_csv "$ROWS_CSV" \
  --out_dir "$OUT_DIR" \
  --c_feature_cols "$C_FEATURE_COLS" \
  --d_feature_cols "" \
  --derive_decision_kl true \
  --min_present_rate "$MIN_PRESENT_RATE" \
  --min_feature_auroc "$MIN_FEATURE_AUROC" \
  --top_k_c "$TOP_K_C" \
  --top_k_d 0 \
  --tau_objective "$TAU_OBJECTIVE" \
  --min_selected_count "$MIN_SELECTED_COUNT" \
  --candidate_filter "$CANDIDATE_FILTER"

echo "[done] $OUT_DIR/summary.json"
