#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  GPU="$1"
  shift
fi

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
PY_BIN="${PY_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
DEVICE="${DEVICE:-cuda}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
LIMIT="${LIMIT:-0}"

SRC="${SRC:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_first_next_len512}"
OUT="${OUT:-$CAL_ROOT/experiments/coco_chair_vga_linear_v49_yesonly_competition}"
TOP_N_FEATURES="${TOP_N_FEATURES:-32}"
MAX_BASELINE_RATE="${MAX_BASELINE_RATE:-0.08}"
MIN_TEACHER_PRECISION="${MIN_TEACHER_PRECISION:-0.7}"
MIN_TEACHER_RECALL="${MIN_TEACHER_RECALL:-0.03}"

echo "[config] CAL_ROOT=$CAL_ROOT"
echo "[config] SRC=$SRC"
echo "[config] OUT=$OUT"
echo "[config] GPU=$GPU"
echo "[config] PY_BIN=$PY_BIN"
echo "[config] LIMIT=$LIMIT"

for path in \
  "$PY_BIN" \
  "$SRC/splits/val_caption_q.jsonl" \
  "$SRC/splits/test_caption_q.jsonl" \
  "$SRC/val/chair_baseline.json" \
  "$SRC/val/chair_vga_full_pvg.json" \
  "$SRC/test/chair_baseline.json" \
  "$SRC/test/chair_vga_full_pvg.json"; do
  if [[ ! -e "$path" ]]; then
    echo "[error] missing required path: $path" >&2
    exit 1
  fi
done

for split in val test; do
  split_out="$OUT/features/$split"
  mkdir -p "$split_out"

  echo "[1/3][$split] extract CHAIR object delta yes-only support features"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_chair_object_delta_yesno_features.py \
    --question_file "$SRC/splits/${split}_caption_q.jsonl" \
    --image_folder "$IMAGE_FOLDER" \
    --baseline_chair_json "$SRC/$split/chair_baseline.json" \
    --intervention_chair_json "$SRC/$split/chair_vga_full_pvg.json" \
    --out_csv "$split_out/chair_object_delta_yesonly.csv" \
    --out_summary_json "$split_out/chair_object_delta_yesonly.summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --score_mode yes_only \
    --reuse_if_exists "$REUSE_IF_EXISTS"
done

echo "[2/3][val] fit yes-only competition gate"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/evaluate_chair_pairwise_proxy.py \
  --baseline_chair_json "$SRC/val/chair_baseline.json" \
  --intervention_chair_json "$SRC/val/chair_vga_full_pvg.json" \
  --extra_features_csv "$OUT/features/val/chair_object_delta_yesonly.csv" \
  --out_dir "$OUT/pairwise_val_yesonly" \
  --chair_i_eps 0.005 \
  --chair_s_eps 0.005 \
  --min_recall_gain 0.005 \
  --require_f1_nondecrease \
  --max_baseline_rate "$MAX_BASELINE_RATE" \
  --min_teacher_precision "$MIN_TEACHER_PRECISION" \
  --min_teacher_recall "$MIN_TEACHER_RECALL" \
  --fit_gated \
  --anchor_feature pair_chairyn_base_only_object_count \
  --gate_feature pair_chairyn_comp_base_advantage_yes_lp_mean \
  --gate_feature pair_chairyn_comp_yes_lp_mean_margin \
  --gate_feature pair_chairyn_comp_yes_lp_sum_margin \
  --top_n_features "$TOP_N_FEATURES"

echo "[3/3][test] apply yes-only competition gate"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/evaluate_chair_pairwise_proxy.py \
  --baseline_chair_json "$SRC/test/chair_baseline.json" \
  --intervention_chair_json "$SRC/test/chair_vga_full_pvg.json" \
  --extra_features_csv "$OUT/features/test/chair_object_delta_yesonly.csv" \
  --out_dir "$OUT/pairwise_test_apply_yesonly" \
  --selected_policy_json "$OUT/pairwise_val_yesonly/selected_policy.json" \
  --chair_i_eps 0.005 \
  --chair_s_eps 0.005 \
  --min_recall_gain 0.005 \
  --require_f1_nondecrease \
  --max_baseline_rate "$MAX_BASELINE_RATE"

echo "[done] val summary: $OUT/pairwise_val_yesonly/summary.json"
echo "[done] test summary: $OUT/pairwise_test_apply_yesonly/summary.json"
