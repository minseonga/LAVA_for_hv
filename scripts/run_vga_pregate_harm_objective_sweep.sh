#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
AMBER_ROOT="${AMBER_ROOT:-/home/kms/data/AMBER}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_ROOT="${SOURCE_ROOT:-$CAL_ROOT/experiments/vga_pregate_harm_v2}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_pregate_harm_objective_sweep}"
DISCOVERY_ONLY="${DISCOVERY_ONLY:-true}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
SAMPLING="${SAMPLING:-false}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
NUM_BEAMS="${NUM_BEAMS:-1}"
CD_ALPHA="${CD_ALPHA:-0.02}"
ATTN_COEF="${ATTN_COEF:-0.2}"
START_LAYER="${START_LAYER:-16}"
END_LAYER="${END_LAYER:-24}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
ATTN_NORM="${ATTN_NORM:-false}"
LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
PROBE_FEATURE_MODE="${PROBE_FEATURE_MODE:-static_headset}"
PROBE_POSITION_MODE="${PROBE_POSITION_MODE:-baseline_yesno_offline_fullseq}"
PROBE_BRANCH_SOURCE="${PROBE_BRANCH_SOURCE:-baseline_jsonl}"
PROBE_FORCE_MANUAL_FULLSEQ="${PROBE_FORCE_MANUAL_FULLSEQ:-false}"
PROBE_PREVIEW_MAX_NEW_TOKENS="${PROBE_PREVIEW_MAX_NEW_TOKENS:-3}"
PROBE_PREVIEW_REUSE_BASELINE="${PROBE_PREVIEW_REUSE_BASELINE:-true}"
PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST="${PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST:-true}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
TOP_K="${TOP_K:-3}"
FEATURE_COLS="${FEATURE_COLS:-frg,frg_shared_mean,frg_shared_topk,g_top5_mass,gmi,c_agg_cos,c_agg_ip,e_agg_combo,e_agg_js,late_head_vis_ratio_mean,late_head_vis_ratio_topkmean}"

POPE_DISC_TABLE="${POPE_DISC_TABLE:-$SOURCE_ROOT/discovery/pope/table/table.csv}"
AMBER_DISC_TABLE="${AMBER_DISC_TABLE:-$SOURCE_ROOT/discovery/amber_disc/table/table.csv}"
POPE_TEST_TABLE="${POPE_TEST_TABLE:-$SOURCE_ROOT/test/pope/table/table.csv}"
AMBER_TEST_TABLE="${AMBER_TEST_TABLE:-$SOURCE_ROOT/test/amber_disc/table/table.csv}"

POPE_IMAGE_FOLDER="${POPE_IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
AMBER_IMAGE_FOLDER="${AMBER_IMAGE_FOLDER:-$AMBER_ROOT/image}"
POPE_TEST_Q_FILE="${POPE_TEST_Q_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
POPE_TEST_GT_CSV="${POPE_TEST_GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
POPE_TEST_BASELINE_JSONL="${POPE_TEST_BASELINE_JSONL:-$CAL_ROOT/experiments/paper_main_b_c_v1_full/test_stageb/pred_baseline.jsonl}"
POPE_TEST_INTERVENTION_JSONL="${POPE_TEST_INTERVENTION_JSONL:-$CAL_ROOT/experiments/paper_main_b_c_v1_full/test_stageb/pred_vga.jsonl}"
AMBER_SPLIT_DIR="${AMBER_SPLIT_DIR:-$SOURCE_ROOT/assets/amber_discriminative_split}"
AMBER_TEST_Q_FILE="${AMBER_TEST_Q_FILE:-$AMBER_SPLIT_DIR/test/assets/test_q_with_object.jsonl}"
AMBER_TEST_GT_CSV="${AMBER_TEST_GT_CSV:-$AMBER_SPLIT_DIR/test/assets/test_gt.csv}"
AMBER_BASELINE_JSONL="${AMBER_BASELINE_JSONL:-$CAL_ROOT/experiments/amber_fixed_transfer_from_pope/discriminative/pred_baseline.jsonl}"
AMBER_INTERVENTION_JSONL="${AMBER_INTERVENTION_JSONL:-$CAL_ROOT/experiments/amber_fixed_transfer_from_pope/discriminative/pred_vga.jsonl}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"

POPE_TEST_PROBE_DIR="${POPE_TEST_PROBE_DIR:-$SOURCE_ROOT/test/pope/probe}"
AMBER_TEST_PROBE_DIR="${AMBER_TEST_PROBE_DIR:-$SOURCE_ROOT/test/amber_disc/probe}"

mkdir -p "$OUT_ROOT"
cd "$ROOT_DIR"

for path in "$POPE_DISC_TABLE" "$AMBER_DISC_TABLE"; do
  if [[ ! -f "$path" ]]; then
    echo "[error] missing required table: $path" >&2
    exit 1
  fi
done

ensure_pope_test_table() {
  if [[ -f "$POPE_TEST_TABLE" ]]; then
    return
  fi
  if [[ ! -f "$POPE_TEST_PROBE_DIR/probe_features.csv" ]]; then
    echo "[prep] extract POPE held-out probe features"
    mkdir -p "$POPE_TEST_PROBE_DIR"
    PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/extract_pnp_probe_features.py \
      --backend vga \
      --vga_root "$VGA_ROOT" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --image_folder "$POPE_IMAGE_FOLDER" \
      --question_file "$POPE_TEST_Q_FILE" \
      --out_dir "$POPE_TEST_PROBE_DIR" \
      --conv_mode "$CONV_MODE" \
      --device "$DEVICE" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --sampling "$SAMPLING" \
      --max_gen_len "$MAX_GEN_LEN" \
      --num_beams "$NUM_BEAMS" \
      --cd_alpha "$CD_ALPHA" \
      --attn_coef "$ATTN_COEF" \
      --start_layer "$START_LAYER" \
      --end_layer "$END_LAYER" \
      --head_balancing "$HEAD_BALANCING" \
      --attn_norm "$ATTN_NORM" \
      --late_start "$LATE_START" \
      --late_end "$LATE_END" \
      --probe_feature_mode "$PROBE_FEATURE_MODE" \
      --headset_json "$HEADSET_JSON" \
      --probe_position_mode "$PROBE_POSITION_MODE" \
      --probe_branch_source "$PROBE_BRANCH_SOURCE" \
      --branch_text_jsonl "$POPE_TEST_BASELINE_JSONL" \
      --probe_force_manual_fullseq "$PROBE_FORCE_MANUAL_FULLSEQ" \
      --probe_preview_max_new_tokens "$PROBE_PREVIEW_MAX_NEW_TOKENS" \
      --probe_preview_reuse_baseline "$PROBE_PREVIEW_REUSE_BASELINE" \
      --probe_preview_fallback_to_prompt_last "$PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST" \
      --seed "$SEED" \
      --max_samples "$MAX_SAMPLES"
  fi
  echo "[prep] build POPE held-out table"
  mkdir -p "$(dirname "$POPE_TEST_TABLE")"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_table.py \
    --probe_features_csv "$POPE_TEST_PROBE_DIR/probe_features.csv" \
    --baseline_pred_jsonl "$POPE_TEST_BASELINE_JSONL" \
    --intervention_pred_jsonl "$POPE_TEST_INTERVENTION_JSONL" \
    --question_file "$POPE_TEST_Q_FILE" \
    --gt_csv "$POPE_TEST_GT_CSV" \
    --benchmark_name pope \
    --split_name heldout \
    --out_csv "$POPE_TEST_TABLE" \
    --out_summary_json "$(dirname "$POPE_TEST_TABLE")/summary.json"
}

ensure_amber_test_table() {
  if [[ -f "$AMBER_TEST_TABLE" ]]; then
    return
  fi
  if [[ ! -f "$AMBER_TEST_Q_FILE" || ! -f "$AMBER_TEST_GT_CSV" ]]; then
    echo "[error] missing AMBER held-out split assets under $AMBER_SPLIT_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$AMBER_TEST_PROBE_DIR/probe_features.csv" ]]; then
    echo "[prep] extract AMBER-disc held-out probe features"
    mkdir -p "$AMBER_TEST_PROBE_DIR"
    PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/extract_pnp_probe_features.py \
      --backend vga \
      --vga_root "$VGA_ROOT" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --image_folder "$AMBER_IMAGE_FOLDER" \
      --question_file "$AMBER_TEST_Q_FILE" \
      --out_dir "$AMBER_TEST_PROBE_DIR" \
      --conv_mode "$CONV_MODE" \
      --device "$DEVICE" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --sampling "$SAMPLING" \
      --max_gen_len "$MAX_GEN_LEN" \
      --num_beams "$NUM_BEAMS" \
      --cd_alpha "$CD_ALPHA" \
      --attn_coef "$ATTN_COEF" \
      --start_layer "$START_LAYER" \
      --end_layer "$END_LAYER" \
      --head_balancing "$HEAD_BALANCING" \
      --attn_norm "$ATTN_NORM" \
      --late_start "$LATE_START" \
      --late_end "$LATE_END" \
      --probe_feature_mode "$PROBE_FEATURE_MODE" \
      --headset_json "$HEADSET_JSON" \
      --probe_position_mode "$PROBE_POSITION_MODE" \
      --probe_branch_source "$PROBE_BRANCH_SOURCE" \
      --branch_text_jsonl "$AMBER_BASELINE_JSONL" \
      --probe_force_manual_fullseq "$PROBE_FORCE_MANUAL_FULLSEQ" \
      --probe_preview_max_new_tokens "$PROBE_PREVIEW_MAX_NEW_TOKENS" \
      --probe_preview_reuse_baseline "$PROBE_PREVIEW_REUSE_BASELINE" \
      --probe_preview_fallback_to_prompt_last "$PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST" \
      --seed "$SEED" \
      --max_samples "$MAX_SAMPLES"
  fi
  echo "[prep] build AMBER-disc held-out table"
  mkdir -p "$(dirname "$AMBER_TEST_TABLE")"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_table.py \
    --probe_features_csv "$AMBER_TEST_PROBE_DIR/probe_features.csv" \
    --baseline_pred_jsonl "$AMBER_BASELINE_JSONL" \
    --intervention_pred_jsonl "$AMBER_INTERVENTION_JSONL" \
    --question_file "$AMBER_TEST_Q_FILE" \
    --gt_csv "$AMBER_TEST_GT_CSV" \
    --benchmark_name amber_discriminative \
    --split_name heldout \
    --out_csv "$AMBER_TEST_TABLE" \
    --out_summary_json "$(dirname "$AMBER_TEST_TABLE")/summary.json"
}

if [[ "$DISCOVERY_ONLY" != "true" ]]; then
  ensure_pope_test_table
  ensure_amber_test_table
fi

run_variant() {
  local name="$1"
  local objective="$2"
  local min_rate="$3"
  local max_rate="$4"
  local min_selected="$5"

  local variant_root="$OUT_ROOT/$name"
  local controller_dir="$variant_root/discovery/unified_controller"
  local pope_apply_dir="$variant_root/test/pope/apply"
  local amber_apply_dir="$variant_root/test/amber_disc/apply"

  echo "[sweep:$name] build controller"
  mkdir -p "$controller_dir"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/build_vga_pregate_harm_controller.py \
    --discovery_table_csvs "$POPE_DISC_TABLE" "$AMBER_DISC_TABLE" \
    --out_dir "$controller_dir" \
    --min_feature_auroc "$MIN_FEATURE_AUROC" \
    --top_k "$TOP_K" \
    --feature_cols "$FEATURE_COLS" \
    --tau_objective "$objective" \
    --min_baseline_rate "$min_rate" \
    --max_baseline_rate "$max_rate" \
    --min_selected_count "$min_selected"

  if [[ "$DISCOVERY_ONLY" != "true" ]]; then
    echo "[sweep:$name] apply on POPE held-out"
    mkdir -p "$pope_apply_dir"
    PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_harm_controller.py \
      --table_csv "$POPE_TEST_TABLE" \
      --policy_json "$controller_dir/selected_policy.json" \
      --out_dir "$pope_apply_dir"

    echo "[sweep:$name] apply on AMBER-disc held-out"
    mkdir -p "$amber_apply_dir"
    PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/apply_vga_pregate_harm_controller.py \
      --table_csv "$AMBER_TEST_TABLE" \
      --policy_json "$controller_dir/selected_policy.json" \
      --out_dir "$amber_apply_dir"
  fi
}

run_variant "acc_default" "final_acc" "0.0" "1.0" "0"
run_variant "harm_f1_b3" "harm_f1" "0.005" "0.03" "10"
run_variant "harm_precision_b3" "harm_precision" "0.005" "0.03" "10"
run_variant "harm_recall_b3" "harm_recall" "0.005" "0.03" "10"

if [[ "$DISCOVERY_ONLY" == "true" ]]; then
  echo "[done] $OUT_ROOT/acc_default/discovery/unified_controller/summary.json"
  echo "[done] $OUT_ROOT/harm_f1_b3/discovery/unified_controller/summary.json"
  echo "[done] $OUT_ROOT/harm_precision_b3/discovery/unified_controller/summary.json"
  echo "[done] $OUT_ROOT/harm_recall_b3/discovery/unified_controller/summary.json"
else
  echo "[done] $OUT_ROOT/acc_default/test/pope/apply/summary.json"
  echo "[done] $OUT_ROOT/harm_f1_b3/test/pope/apply/summary.json"
  echo "[done] $OUT_ROOT/harm_precision_b3/test/pope/apply/summary.json"
  echo "[done] $OUT_ROOT/harm_recall_b3/test/pope/apply/summary.json"
fi
