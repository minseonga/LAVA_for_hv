#!/usr/bin/env bash
set -euo pipefail

SOURCE_OUT="${SOURCE_OUT:-/home/kms/LLaVA_calibration/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments/coco_chair_v79_intervention_process_trace}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
INTERVENTION_PRED="${INTERVENTION_PRED:-$SOURCE_OUT/test/pred_origin_entropy_simg_caption.jsonl}"
QUESTION_FILE="${QUESTION_FILE:-$SOURCE_OUT/splits/test_caption_q_limited500.jsonl}"
ORACLE_ROWS="${ORACLE_ROWS:-$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg/unique_safe_oracle_rows.csv}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_ci_unique_noworse}"
LIMIT="${LIMIT:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-140}"
VSS_MODE="${VSS_MODE:-entropy}"
VSS_TOPK="${VSS_TOPK:-10}"
CD_ALPHA="${CD_ALPHA:-0.02}"
ATTN_COEF="${ATTN_COEF:-0.2}"
START_LAYER="${START_LAYER:-2}"
END_LAYER="${END_LAYER:-15}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
TOPK="${TOPK:-10}"

mkdir -p "$OUT_ROOT"
export PYTHONDONTWRITEBYTECODE=1

echo "[settings] source=$SOURCE_OUT"
echo "[settings] out=$OUT_ROOT"
echo "[settings] limit=$LIMIT max_new_tokens=$MAX_NEW_TOKENS"
echo "[settings] intervention_pred=$INTERVENTION_PRED"

python scripts/extract_vga_intervention_process_features.py \
  --model-path "$MODEL_PATH" \
  --image-folder "$IMAGE_FOLDER" \
  --question-file "$QUESTION_FILE" \
  --intervention-pred-jsonl "$INTERVENTION_PRED" \
  --oracle-rows-csv "$ORACLE_ROWS" \
  --target-col "$TARGET_COL" \
  --limit "$LIMIT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --vss-mode "$VSS_MODE" \
  --vss-topk "$VSS_TOPK" \
  --use-add true \
  --cd-alpha "$CD_ALPHA" \
  --attn-coef "$ATTN_COEF" \
  --start-layer "$START_LAYER" \
  --end-layer "$END_LAYER" \
  --head-balancing "$HEAD_BALANCING" \
  --topk "$TOPK" \
  --out-steps-csv "$OUT_ROOT/intervention_process_steps.csv" \
  --out-features-csv "$OUT_ROOT/intervention_process_features.csv" \
  --out-metrics-csv "$OUT_ROOT/intervention_process_feature_metrics.csv" \
  --out-summary-json "$OUT_ROOT/summary.json"

echo "[done] $OUT_ROOT"
