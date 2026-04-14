#!/usr/bin/env bash
set -euo pipefail

SOURCE_OUT="${SOURCE_OUT:-/home/kms/LLaVA_calibration/experiments/coco_chair_v59_repro_vss_ablation_full500}"
OUT_ROOT="${OUT_ROOT:-/home/kms/LLaVA_calibration/experiments/coco_chair_v66_lost_object_guidance_debug}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
TARGET_COL="${TARGET_COL:-oracle_recall_gain_f1_nondecrease_ci_unique_noworse}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-15}"
SAMPLES_PER_MODE="${SAMPLES_PER_MODE:-5}"
MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-220}"
VSS_MODE="${VSS_MODE:-entropy}"
VSS_TOPK="${VSS_TOPK:-10}"
CD_ALPHA="${CD_ALPHA:-0.02}"
ATTN_COEF="${ATTN_COEF:-0.2}"
START_LAYER="${START_LAYER:-2}"
END_LAYER="${END_LAYER:-15}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
TOPK_PATCHES="${TOPK_PATCHES:-10}"

mkdir -p "$OUT_ROOT"
export PYTHONDONTWRITEBYTECODE=1

echo "[settings] source=$SOURCE_OUT"
echo "[settings] out=$OUT_ROOT"
echo "[settings] target=$TARGET_COL limit_samples=$LIMIT_SAMPLES samples_per_mode=$SAMPLES_PER_MODE"
echo "[settings] vss=$VSS_MODE topk=$VSS_TOPK alpha=$CD_ALPHA attn_coef=$ATTN_COEF layers=$START_LAYER-$END_LAYER head=$HEAD_BALANCING"

python scripts/diagnose_vga_lost_object_guidance.py \
  --model-path "$MODEL_PATH" \
  --image-folder "$IMAGE_FOLDER" \
  --question-file "$SOURCE_OUT/splits/test_caption_q_limited500.jsonl" \
  --oracle-rows-csv "$SOURCE_OUT/unique_safe_oracle_test_origin_entropy_simg/unique_safe_oracle_rows.csv" \
  --target-col "$TARGET_COL" \
  --samples-per-mode "$SAMPLES_PER_MODE" \
  --limit-samples "$LIMIT_SAMPLES" \
  --max-caption-tokens "$MAX_CAPTION_TOKENS" \
  --vss-mode "$VSS_MODE" \
  --vss-topk "$VSS_TOPK" \
  --use-add true \
  --cd-alpha "$CD_ALPHA" \
  --attn-coef "$ATTN_COEF" \
  --start-layer "$START_LAYER" \
  --end-layer "$END_LAYER" \
  --head-balancing "$HEAD_BALANCING" \
  --topk-patches "$TOPK_PATCHES" \
  --out-csv "$OUT_ROOT/lost_object_guidance.csv" \
  --out-json "$OUT_ROOT/lost_object_guidance_summary.json"

echo "[done] $OUT_ROOT"
