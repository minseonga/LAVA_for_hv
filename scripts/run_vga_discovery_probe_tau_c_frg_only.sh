#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"

DISCOVERY_JSONL="${DISCOVERY_JSONL:-$CAL_ROOT/experiments/pope_discovery/discovery_adversarial.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial}"
ASSET_DIR="${ASSET_DIR:-$OUT_ROOT/assets}"
BASELINE_DIR="${BASELINE_DIR:-$OUT_ROOT/baseline}"
VGA_DIR="${VGA_DIR:-$OUT_ROOT/vga}"
TAX_DIR="${TAX_DIR:-$OUT_ROOT/taxonomy}"
PROBE_DIR="${PROBE_DIR:-$OUT_ROOT/probe_features_online}"
CONTROLLER_DIR="${CONTROLLER_DIR:-$OUT_ROOT/controller_online_probe}"

GT_CSV="${GT_CSV:-$ASSET_DIR/discovery_gt.csv}"
Q_JSONL="${Q_JSONL:-$ASSET_DIR/discovery_q.jsonl}"
Q_WITH_OBJECT_JSONL="${Q_WITH_OBJECT_JSONL:-$ASSET_DIR/discovery_q_with_object.jsonl}"
BASELINE_PRED_JSONL="${BASELINE_PRED_JSONL:-$BASELINE_DIR/pred_baseline.jsonl}"
VGA_PRED_JSONL="${VGA_PRED_JSONL:-$VGA_DIR/pred_vga.jsonl}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"
PROBE_FEATURES_CSV="${PROBE_FEATURES_CSV:-$PROBE_DIR/probe_features.csv}"

OBJECT_PRIOR_THR="${OBJECT_PRIOR_THR:-0.55}"
CALIB_RATIO="${CALIB_RATIO:-0.3}"
SEED="${SEED:-42}"
LAMBDA_D1="${LAMBDA_D1:-1.0}"
MAX_D1_WRONG_RATE="${MAX_D1_WRONG_RATE:-0.35}"
Q_GRID="${Q_GRID:-0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95}"

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
PROBE_BRANCH_SOURCE="${PROBE_BRANCH_SOURCE:-baseline_jsonl}"
PROBE_FORCE_MANUAL_FULLSEQ="${PROBE_FORCE_MANUAL_FULLSEQ:-false}"
PROBE_PREVIEW_MAX_NEW_TOKENS="${PROBE_PREVIEW_MAX_NEW_TOKENS:-3}"
PROBE_PREVIEW_REUSE_BASELINE="${PROBE_PREVIEW_REUSE_BASELINE:-true}"
PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST="${PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST:-true}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
FORCE_PROBE="${FORCE_PROBE:-0}"

mkdir -p "$ASSET_DIR" "$BASELINE_DIR" "$VGA_DIR" "$TAX_DIR" "$PROBE_DIR" "$CONTROLLER_DIR"
cd "$CAL_ROOT"

if [[ ! -f "$GT_CSV" || ! -f "$Q_JSONL" || ! -f "$Q_WITH_OBJECT_JSONL" ]]; then
  echo "[1/6] build discovery assets"
  python scripts/build_pope_style_discovery_assets.py \
    --in_jsonl "$DISCOVERY_JSONL" \
    --out_dir "$ASSET_DIR"
else
  echo "[1/6] skip discovery assets (already exist)"
fi

if [[ ! -f "$BASELINE_PRED_JSONL" ]]; then
  echo "[2/6] run baseline on discovery set"
  PYTHONPATH="$CAL_ROOT" python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$Q_JSONL" \
    --answers-file "$BASELINE_PRED_JSONL" \
    --conv-mode "$CONV_MODE" \
    --temperature 0 \
    --num_beams 1 \
    --max_new_tokens 8
else
  echo "[2/6] skip baseline predictions (already exist)"
fi

if [[ ! -f "$VGA_PRED_JSONL" ]]; then
  echo "[3/6] run VGA on discovery set"
  (
    cd "$VGA_ROOT"
    python eval/object_hallucination_vqa_llava.py \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$Q_WITH_OBJECT_JSONL" \
      --answers-file "$VGA_PRED_JSONL" \
      --conv-mode "$CONV_MODE" \
      --max_gen_len 8 \
      --use_add true \
      --attn_coef 0.2 \
      --head_balancing simg \
      --sampling false \
      --cd_alpha 0.02 \
      --seed "$SEED" \
      --start_layer 2 \
      --end_layer 15
  )
else
  echo "[3/6] skip VGA predictions (already exist)"
fi

if [[ ! -f "$TAX_DIR/per_case_compare.csv" ]]; then
  echo "[4/6] build taxonomy on discovery set"
  python scripts/build_vga_failure_taxonomy.py \
    --gt_csv "$GT_CSV" \
    --baseline_pred_jsonl "$BASELINE_PRED_JSONL" \
    --vga_pred_jsonl "$VGA_PRED_JSONL" \
    --baseline_pred_text_key text \
    --vga_pred_text_key output \
    --object_prior_thr "$OBJECT_PRIOR_THR" \
    --out_dir "$TAX_DIR"
else
  echo "[4/6] skip taxonomy (already exist)"
fi

if [[ "$FORCE_PROBE" == "1" || ! -f "$PROBE_FEATURES_CSV" ]]; then
  echo "[5/6] extract online probe FRG on discovery set"
  python scripts/extract_pnp_probe_features.py \
    --backend vga \
    --vga_root "$VGA_ROOT" \
    --model_path "$MODEL_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --question_file "$Q_WITH_OBJECT_JSONL" \
    --out_dir "$PROBE_DIR" \
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
    --probe_feature_mode static_headset \
    --probe_position_mode "$PROBE_POSITION_MODE" \
    --probe_branch_source "$PROBE_BRANCH_SOURCE" \
    --probe_force_manual_fullseq "$PROBE_FORCE_MANUAL_FULLSEQ" \
    --branch_text_jsonl "$BASELINE_PRED_JSONL" \
    --probe_preview_max_new_tokens "$PROBE_PREVIEW_MAX_NEW_TOKENS" \
    --probe_preview_reuse_baseline "$PROBE_PREVIEW_REUSE_BASELINE" \
    --probe_preview_fallback_to_prompt_last "$PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST" \
    --headset_json "$HEADSET_JSON" \
    --seed "$SEED" \
    --max_samples "$MAX_SAMPLES"
else
  echo "[5/6] skip probe feature extraction (already exist)"
fi

echo "[6/6] calibrate tau_c on online probe FRG"
python scripts/run_vga_hard_veto_controller.py \
  --per_case_csv "$TAX_DIR/per_case_compare.csv" \
  --features_csv "$PROBE_FEATURES_CSV" \
  --out_dir "$CONTROLLER_DIR" \
  --c_col frg \
  --e_col gmi \
  --calib_ratio "$CALIB_RATIO" \
  --seed "$SEED" \
  --lambda_d1 "$LAMBDA_D1" \
  --max_d1_wrong_rate "$MAX_D1_WRONG_RATE" \
  --q_grid "$Q_GRID" \
  --fallback_when_missing_feature vga \
  --use_c 1 \
  --use_e 0

echo "[done] $OUT_ROOT"
echo "[saved] $CONTROLLER_DIR/summary.json"
