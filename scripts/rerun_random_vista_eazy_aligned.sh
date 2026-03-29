#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
GPU="${GPU:-6}"

VISTA_SCRIPT="$CAL_ROOT/scripts/run_vista_vs_baseline_taxonomy_9000.sh"
EAZY_SCRIPT="$CAL_ROOT/scripts/run_eazy_vs_baseline_taxonomy_9000.sh"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict}"

GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
BASELINE_JSONL="${BASELINE_JSONL:-$OUT_ROOT/baseline/pred_vanilla_9000.jsonl}"
FEATURES_CSV="${FEATURES_CSV:-$CAL_ROOT/experiments/pope_feature_screen_v1_e1_4_l16_24/features/features_unified_table.csv}"

# 0) Ensure random category file is GT-aligned 3000
/home/kms/miniconda3/envs/vocot/bin/python "$CAL_ROOT/scripts/rebuild_pope_category_from_gt.py" \
  --gt_csv "$GT_CSV" \
  --category random \
  --out_jsonl /home/kms/VISTA/pope_coco/coco_pope_random.json \
  --backup_old

# 1) VISTA random-only rerun + rebuild pred/taxonomy/d1d2
CUDA_VISIBLE_DEVICES="$GPU" \
GPU="$GPU" \
ONLY_CATEGORY=random \
CATEGORY_SIZE=3000 \
GT_CSV="$GT_CSV" \
BASELINE_JSONL="$BASELINE_JSONL" \
FEATURES_CSV="$FEATURES_CSV" \
OUT_DIR="$OUT_ROOT/vista" \
TAX_OUT_DIR="$OUT_ROOT/vista/taxonomy" \
D1D2_OUT_DIR="$OUT_ROOT/vista/d1d2_audit" \
bash "$VISTA_SCRIPT"

# 2) EAZY random-only rerun + rebuild pred/taxonomy/d1d2
CUDA_VISIBLE_DEVICES="$GPU" \
GPU="$GPU" \
ONLY_CATEGORY=random \
CATEGORY_SIZE=3000 \
GT_CSV="$GT_CSV" \
BASELINE_JSONL="$BASELINE_JSONL" \
FEATURES_CSV="$FEATURES_CSV" \
OUT_DIR="$OUT_ROOT/eazy" \
TAX_OUT_DIR="$OUT_ROOT/eazy/taxonomy" \
D1D2_OUT_DIR="$OUT_ROOT/eazy/d1d2_audit" \
bash "$EAZY_SCRIPT"

echo "[done] random-only rerun completed"
