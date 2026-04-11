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
MAX_MENTIONS="${MAX_MENTIONS:-12}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-false}"

SRC="${SRC:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_first_next_len512}"
OUT="${OUT:-$CAL_ROOT/experiments/coco_chair_vga_linear_v42_confidence_collapse}"

CHAIR_I_EPS="${CHAIR_I_EPS:-0.01}"
CHAIR_S_EPS="${CHAIR_S_EPS:-0.0}"
MAX_BASELINE_RATE="${MAX_BASELINE_RATE:-0.08}"
TOP_N_FEATURES="${TOP_N_FEATURES:-6}"
ANCHOR_FEATURE="${ANCHOR_FEATURE:-proxy_chairgen_generated_unique_drop}"

GATE_FEATURE_ARGS=(
  --gate_feature pair_intprobe_lp_content_min_real
  --gate_feature pair_intprobe_target_gap_content_min_real
  --gate_feature pair_intprobe_lp_content_std_real
  --gate_feature pair_probe_bad_shift_lp_content_min
  --gate_feature pair_probe_bad_shift_target_gap_content_min
  --gate_feature pair_probe_bad_shift_lp_content_std
)

echo "[config] CAL_ROOT=$CAL_ROOT"
echo "[config] SRC=$SRC"
echo "[config] OUT=$OUT"
echo "[config] GPU=$GPU"
echo "[config] PY_BIN=$PY_BIN"
echo "[config] REUSE_IF_EXISTS=$REUSE_IF_EXISTS"

for path in "$PY_BIN" "$SRC/splits/val_caption_q.jsonl" "$SRC/splits/test_caption_q.jsonl"; do
  if [[ ! -e "$path" ]]; then
    echo "[error] missing required path: $path" >&2
    exit 1
  fi
done

for split in val test; do
  split_out="$OUT/features/$split"
  mkdir -p "$split_out"

  echo "[1/4][$split] extract baseline confidence-collapse probe features"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_vga_generative_mention_features.py \
    --question_file "$SRC/splits/${split}_caption_q.jsonl" \
    --image_folder "$IMAGE_FOLDER" \
    --baseline_pred_jsonl "$SRC/$split/pred_vanilla_caption.jsonl" \
    --out_csv "$split_out/coverage_features.probe.csv" \
    --out_summary_json "$split_out/coverage_features.probe.summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --max_mentions "$MAX_MENTIONS" \
    --reuse_if_exists "$REUSE_IF_EXISTS"

  echo "[2/4][$split] extract intervention-relative confidence-collapse probe features"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_generative_relative_probe_features.py \
    --question_file "$SRC/splits/${split}_caption_q.jsonl" \
    --image_folder "$IMAGE_FOLDER" \
    --intervention_pred_jsonl "$SRC/$split/pred_vga_full_pvg_caption.jsonl" \
    --base_features_csv "$split_out/coverage_features.probe.csv" \
    --out_csv "$split_out/coverage_features.relprobe.csv" \
    --out_summary_json "$split_out/coverage_features.relprobe.summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --max_mentions "$MAX_MENTIONS" \
    --pred_text_key auto \
    --reuse_if_exists "$REUSE_IF_EXISTS"
done

echo "[3/4][val] fit recall-anchor confidence gate"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/evaluate_chair_pairwise_proxy.py \
  --baseline_chair_json "$SRC/val/chair_baseline.json" \
  --intervention_chair_json "$SRC/val/chair_vga_full_pvg.json" \
  --extra_features_csv "$OUT/features/val/coverage_features.relprobe.csv" \
  --out_dir "$OUT/val_recall_anchor_confidence_gate" \
  --chair_i_eps "$CHAIR_I_EPS" \
  --chair_s_eps "$CHAIR_S_EPS" \
  --max_baseline_rate "$MAX_BASELINE_RATE" \
  --require_f1_nondecrease \
  --fit_gated \
  --anchor_feature "$ANCHOR_FEATURE" \
  "${GATE_FEATURE_ARGS[@]}" \
  --top_n_features "$TOP_N_FEATURES"

echo "[4/4][test] apply frozen val-selected confidence gate"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/evaluate_chair_pairwise_proxy.py \
  --baseline_chair_json "$SRC/test/chair_baseline.json" \
  --intervention_chair_json "$SRC/test/chair_vga_full_pvg.json" \
  --extra_features_csv "$OUT/features/test/coverage_features.relprobe.csv" \
  --out_dir "$OUT/test_apply_recall_anchor_confidence_gate" \
  --selected_policy_json "$OUT/val_recall_anchor_confidence_gate/selected_policy.json" \
  --chair_i_eps "$CHAIR_I_EPS" \
  --chair_s_eps "$CHAIR_S_EPS" \
  --max_baseline_rate "$MAX_BASELINE_RATE" \
  --require_f1_nondecrease

echo "[done] val summary: $OUT/val_recall_anchor_confidence_gate/summary.json"
echo "[done] test summary: $OUT/test_apply_recall_anchor_confidence_gate/summary.json"
