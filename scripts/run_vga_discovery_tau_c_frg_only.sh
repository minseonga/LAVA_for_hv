#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"

DISCOVERY_JSONL="${DISCOVERY_JSONL:-$CAL_ROOT/experiments/pope_discovery/discovery_adversarial.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
CONV_MODE="${CONV_MODE:-llava_v1}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial}"
ASSET_DIR="${ASSET_DIR:-$OUT_ROOT/assets}"
BASELINE_DIR="${BASELINE_DIR:-$OUT_ROOT/baseline}"
VGA_DIR="${VGA_DIR:-$OUT_ROOT/vga}"
TAX_DIR="${TAX_DIR:-$OUT_ROOT/taxonomy}"
SAMPLES_DIR="${SAMPLES_DIR:-$OUT_ROOT/samples}"
TRACE_DIR="${TRACE_DIR:-$OUT_ROOT/traces_baseline}"
FEATURE_DIR="${FEATURE_DIR:-$OUT_ROOT/features}"
CONTROLLER_DIR="${CONTROLLER_DIR:-$OUT_ROOT/controller}"

GT_CSV="${GT_CSV:-$ASSET_DIR/discovery_gt.csv}"
SUBSET_IDS_CSV="${SUBSET_IDS_CSV:-$ASSET_DIR/discovery_subset_ids.csv}"
Q_JSONL="${Q_JSONL:-$ASSET_DIR/discovery_q.jsonl}"
Q_WITH_OBJECT_JSONL="${Q_WITH_OBJECT_JSONL:-$ASSET_DIR/discovery_q_with_object.jsonl}"

BASELINE_PRED_JSONL="${BASELINE_PRED_JSONL:-$BASELINE_DIR/pred_baseline.jsonl}"
VGA_PRED_JSONL="${VGA_PRED_JSONL:-$VGA_DIR/pred_vga.jsonl}"
SAMPLES_CSV="${SAMPLES_CSV:-$SAMPLES_DIR/samples_baseline.csv}"
PER_LAYER_TRACE_CSV="${PER_LAYER_TRACE_CSV:-$TRACE_DIR/per_layer_yes_trace.csv}"
PER_HEAD_TRACE_CSV="${PER_HEAD_TRACE_CSV:-$TRACE_DIR/per_head_yes_trace.csv}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"

EARLY_START="${EARLY_START:-10}"
EARLY_END="${EARLY_END:-15}"
LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
LAYER_FOCUS="${LAYER_FOCUS:-17}"
TRACE_HEAD_LAYER_START="${TRACE_HEAD_LAYER_START:-10}"
TRACE_HEAD_LAYER_END="${TRACE_HEAD_LAYER_END:-24}"
MAX_TRACE_SAMPLES="${MAX_TRACE_SAMPLES:-0}"

OBJECT_PRIOR_THR="${OBJECT_PRIOR_THR:-0.55}"
CALIB_RATIO="${CALIB_RATIO:-0.3}"
SEED="${SEED:-42}"
LAMBDA_D1="${LAMBDA_D1:-1.0}"
MAX_D1_WRONG_RATE="${MAX_D1_WRONG_RATE:-0.35}"
Q_GRID="${Q_GRID:-0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95}"

mkdir -p "$ASSET_DIR" "$BASELINE_DIR" "$VGA_DIR" "$TAX_DIR" "$SAMPLES_DIR" "$TRACE_DIR" "$FEATURE_DIR" "$CONTROLLER_DIR"
cd "$CAL_ROOT"

if [[ ! -f "$GT_CSV" || ! -f "$SUBSET_IDS_CSV" || ! -f "$Q_JSONL" || ! -f "$Q_WITH_OBJECT_JSONL" ]]; then
  echo "[1/7] build discovery assets"
  python scripts/build_pope_style_discovery_assets.py \
    --in_jsonl "$DISCOVERY_JSONL" \
    --out_dir "$ASSET_DIR"
else
  echo "[1/7] skip discovery assets (already exist)"
fi

if [[ ! -f "$BASELINE_PRED_JSONL" ]]; then
  echo "[2/7] run baseline on discovery set"
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
  echo "[2/7] skip baseline predictions (already exist)"
fi

if [[ ! -f "$VGA_PRED_JSONL" ]]; then
  echo "[3/7] run VGA on discovery set"
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
  echo "[3/7] skip VGA predictions (already exist)"
fi

if [[ ! -f "$TAX_DIR/per_case_compare.csv" ]]; then
  echo "[4/7] build taxonomy on discovery set"
  python scripts/build_vga_failure_taxonomy.py \
    --gt_csv "$GT_CSV" \
    --baseline_pred_jsonl "$BASELINE_PRED_JSONL" \
    --vga_pred_jsonl "$VGA_PRED_JSONL" \
    --baseline_pred_text_key text \
    --vga_pred_text_key output \
    --object_prior_thr "$OBJECT_PRIOR_THR" \
    --out_dir "$TAX_DIR"
else
  echo "[4/7] skip taxonomy (already exist)"
fi

if [[ ! -f "$SAMPLES_CSV" ]]; then
  echo "[5/7] build baseline samples csv"
  python scripts/build_pope_samples_from_gt_and_pred.py \
    --subset_gt_csv "$GT_CSV" \
    --pred_jsonl "$BASELINE_PRED_JSONL" \
    --pred_text_key text \
    --out_csv "$SAMPLES_CSV" \
    --out_summary "$SAMPLES_DIR/summary.json"
else
  echo "[5/7] skip samples csv (already exist)"
fi

if [[ ! -f "$PER_HEAD_TRACE_CSV" || ! -f "$PER_LAYER_TRACE_CSV" ]]; then
  echo "[6/7] extract baseline traces on discovery set"
  TRACE_CMD=(
    python "$CAL_ROOT/analyze_pope_visual_disconnect.py"
    --samples_csv "$SAMPLES_CSV"
    --image_root "$IMAGE_FOLDER"
    --out_dir "$TRACE_DIR"
    --dataset_mode pope
    --model_path "$MODEL_PATH"
    --conv_mode "$CONV_MODE"
    --topk_local 16
    --object_patch_topk 64
    --hidden_layer_idx -1
    --attn_layer_idx -1
    --save_layer_trace
    --save_head_trace
    --head_layer_start "$TRACE_HEAD_LAYER_START"
    --head_layer_end "$TRACE_HEAD_LAYER_END"
    --control_modes blur,shuffle
    --shuffle_grid 4
    --blur_radius 12
    --bootstrap 500
    --seed "$SEED"
  )
  if [[ "$MAX_TRACE_SAMPLES" != "0" ]]; then
    TRACE_CMD+=(--num_samples "$MAX_TRACE_SAMPLES")
  fi
  PYTHONPATH="$CAL_ROOT" "${TRACE_CMD[@]}"
else
  echo "[6/7] skip baseline traces (already exist)"
fi

echo "[7/7] build FRG-only features and calibrate tau_c"
if [[ ! -f "$FEATURE_DIR/features_unified_table.csv" ]]; then
  python scripts/build_pope_feature_screen_v1.py \
    --subset_ids_csv "$SUBSET_IDS_CSV" \
    --subset_gt_csv "$GT_CSV" \
    --per_layer_trace_csv "$PER_LAYER_TRACE_CSV" \
    --per_head_trace_csv "$PER_HEAD_TRACE_CSV" \
    --headset_json "$HEADSET_JSON" \
    --samples_csv "$SAMPLES_CSV" \
    --use_split all \
    --out_dir "$FEATURE_DIR" \
    --early_start "$EARLY_START" \
    --early_end "$EARLY_END" \
    --late_start "$LATE_START" \
    --late_end "$LATE_END" \
    --layer_focus "$LAYER_FOCUS" \
    --eps 1e-6
else
  echo "  [skip] feature table already exists"
fi

python scripts/run_vga_hard_veto_controller.py \
  --per_case_csv "$TAX_DIR/per_case_compare.csv" \
  --features_csv "$FEATURE_DIR/features_unified_table.csv" \
  --out_dir "$CONTROLLER_DIR" \
  --c_col faithful_minus_global_attn \
  --e_col guidance_mismatch_score \
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
