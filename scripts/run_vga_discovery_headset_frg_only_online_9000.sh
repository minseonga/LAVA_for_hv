#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
QUESTION_FILE="${QUESTION_FILE:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"
OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/pope_full_9000/vga_discovery_headset_frg_only_online_9000}"

ONLINE_PROBE_CONTROLLER_SUMMARY_JSON="$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/controller_online_probe/summary.json"
OFFLINE_CONTROLLER_SUMMARY_JSON="$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/controller/summary.json"
TAU_SOURCE="${TAU_SOURCE:-online_probe}"
if [[ -z "${CONTROLLER_SUMMARY_JSON:-}" ]]; then
  case "$TAU_SOURCE" in
    offline)
      CONTROLLER_SUMMARY_JSON="$OFFLINE_CONTROLLER_SUMMARY_JSON"
      ;;
    online_probe)
      if [[ -f "$ONLINE_PROBE_CONTROLLER_SUMMARY_JSON" ]]; then
        CONTROLLER_SUMMARY_JSON="$ONLINE_PROBE_CONTROLLER_SUMMARY_JSON"
      else
        CONTROLLER_SUMMARY_JSON="$OFFLINE_CONTROLLER_SUMMARY_JSON"
      fi
      ;;
    *)
      echo "[error] unsupported TAU_SOURCE=$TAU_SOURCE (expected offline or online_probe)" >&2
      exit 1
      ;;
  esac
fi

CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SAMPLING="${SAMPLING:-false}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
NUM_BEAMS="${NUM_BEAMS:-1}"

CD_ALPHA="${CD_ALPHA:-0.02}"
ATTN_COEF="${ATTN_COEF:-0.2}"
START_LAYER="${START_LAYER:-2}"
END_LAYER="${END_LAYER:-15}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
ATTN_NORM="${ATTN_NORM:-false}"
LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
PROBE_POSITION_MODE="${PROBE_POSITION_MODE:-baseline_yesno_offline_fullseq}"
PROBE_BRANCH_SOURCE="${PROBE_BRANCH_SOURCE:-baseline_output}"
PROBE_BRANCH_JSONL="${PROBE_BRANCH_JSONL:-}"
PROBE_BRANCH_CSV="${PROBE_BRANCH_CSV:-}"
PROBE_BRANCH_ID_COL="${PROBE_BRANCH_ID_COL:-question_id}"
PROBE_BRANCH_TEXT_COL="${PROBE_BRANCH_TEXT_COL:-output}"
PROBE_PREVIEW_MAX_NEW_TOKENS="${PROBE_PREVIEW_MAX_NEW_TOKENS:-3}"
PROBE_PREVIEW_REUSE_BASELINE="${PROBE_PREVIEW_REUSE_BASELINE:-true}"
PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST="${PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST:-true}"
PROBE_FORCE_MANUAL_FULLSEQ="${PROBE_FORCE_MANUAL_FULLSEQ:-false}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

mkdir -p "$OUT_DIR"
cd "$CAL_ROOT"

EXTRA_ARGS=()
if [[ -n "$PROBE_BRANCH_JSONL" ]]; then
  EXTRA_ARGS+=(--probe_branch_jsonl "$PROBE_BRANCH_JSONL")
fi
if [[ -n "$PROBE_BRANCH_CSV" ]]; then
  EXTRA_ARGS+=(--probe_branch_csv "$PROBE_BRANCH_CSV")
  EXTRA_ARGS+=(--probe_branch_id_col "$PROBE_BRANCH_ID_COL")
  EXTRA_ARGS+=(--probe_branch_text_col "$PROBE_BRANCH_TEXT_COL")
fi

python scripts/run_pnp_hard_veto_online.py \
  --backend vga \
  --vga_root "$VGA_ROOT" \
  --model_path "$MODEL_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --question_file "$QUESTION_FILE" \
  --out_dir "$OUT_DIR" \
  --conv_mode "$CONV_MODE" \
  --device "$DEVICE" \
  --controller_summary_json "$CONTROLLER_SUMMARY_JSON" \
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
  --probe_feature_mode static_headset \
  --probe_position_mode "$PROBE_POSITION_MODE" \
  --probe_branch_source "$PROBE_BRANCH_SOURCE" \
  --probe_force_manual_fullseq "$PROBE_FORCE_MANUAL_FULLSEQ" \
  --probe_preview_max_new_tokens "$PROBE_PREVIEW_MAX_NEW_TOKENS" \
  --probe_preview_reuse_baseline "$PROBE_PREVIEW_REUSE_BASELINE" \
  --probe_preview_fallback_to_prompt_last "$PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST" \
  --headset_json "$HEADSET_JSON" \
  --use_gmi false \
  --seed "$SEED" \
  --max_samples "$MAX_SAMPLES" \
  --gt_csv "$GT_CSV" \
  --gt_id_col id \
  --gt_label_col answer \
  "${EXTRA_ARGS[@]}"

echo "[done] $OUT_DIR"
echo "[saved] $OUT_DIR/summary.json"
