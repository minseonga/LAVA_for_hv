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
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
LIMIT="${LIMIT:-0}"

SRC="${SRC:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_first_next_len512}"
V46_ROOT="${V46_ROOT:-$CAL_ROOT/experiments/coco_chair_vga_linear_v46_competition_distill}"
OUT="${OUT:-$CAL_ROOT/experiments/coco_chair_vga_linear_v48_intervention_trace_distill}"
METHOD_PRED_NAME="${METHOD_PRED_NAME:-pred_vga_full_pvg_caption.jsonl}"

TARGET_COL="${TARGET_COL:-v46_safe_target}"
TARGET_REQUIRES_TEACHER_POSITIVE="${TARGET_REQUIRES_TEACHER_POSITIVE:-true}"
TOP_N_FEATURES="${TOP_N_FEATURES:-64}"
MAX_COMBO_SIZE="${MAX_COMBO_SIZE:-3}"
DIAG_MAX_BASELINE_RATE="${DIAG_MAX_BASELINE_RATE:-0.12}"
STRICT_MAX_BASELINE_RATE="${STRICT_MAX_BASELINE_RATE:-0.08}"
STRICT_MIN_RECALL_GAIN="${STRICT_MIN_RECALL_GAIN:-0.005}"
STRICT_MAX_CHAIR_I_DELTA="${STRICT_MAX_CHAIR_I_DELTA:-0.005}"
STRICT_MAX_CHAIR_S_DELTA="${STRICT_MAX_CHAIR_S_DELTA:-0.005}"

echo "[config] CAL_ROOT=$CAL_ROOT"
echo "[config] SRC=$SRC"
echo "[config] V46_ROOT=$V46_ROOT"
echo "[config] OUT=$OUT"
echo "[config] GPU=$GPU"
echo "[config] PY_BIN=$PY_BIN"
echo "[config] METHOD_PRED_NAME=$METHOD_PRED_NAME"
echo "[config] TARGET_COL=$TARGET_COL TARGET_REQUIRES_TEACHER_POSITIVE=$TARGET_REQUIRES_TEACHER_POSITIVE"
echo "[config] LIMIT=$LIMIT REUSE_IF_EXISTS=$REUSE_IF_EXISTS"

for path in "$PY_BIN" "$SRC/splits/val_caption_q.jsonl" "$SRC/splits/test_caption_q.jsonl"; do
  if [[ ! -e "$path" ]]; then
    echo "[error] missing required path: $path" >&2
    exit 1
  fi
done

for split in val test; do
  if [[ ! -f "$SRC/$split/$METHOD_PRED_NAME" ]]; then
    echo "[error] missing prediction file: $SRC/$split/$METHOD_PRED_NAME" >&2
    exit 1
  fi
  if [[ ! -f "$V46_ROOT/pairwise_${split}_apply/decision_rows.csv" ]]; then
    echo "[error] missing v46 route rows: $V46_ROOT/pairwise_${split}_apply/decision_rows.csv" >&2
    exit 1
  fi
done

target_args=()
if [[ "$TARGET_REQUIRES_TEACHER_POSITIVE" == "true" || "$TARGET_REQUIRES_TEACHER_POSITIVE" == "1" ]]; then
  target_args+=(--target_requires_teacher_positive)
fi

for split in val test; do
  split_out="$OUT/features/$split"
  mkdir -p "$split_out" "$OUT/distill"

  echo "[1/4][$split] extract intervention caption trace features"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/extract_vga_generative_mention_features.py \
    --question_file "$SRC/splits/${split}_caption_q.jsonl" \
    --image_folder "$IMAGE_FOLDER" \
    --baseline_pred_jsonl "$SRC/$split/$METHOD_PRED_NAME" \
    --out_csv "$split_out/intervention_trace.probe.csv" \
    --out_summary_json "$split_out/intervention_trace.probe.summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device "$DEVICE" \
    --limit "$LIMIT" \
    --pred_text_key auto \
    --max_mentions "$MAX_MENTIONS" \
    --reuse_if_exists "$REUSE_IF_EXISTS"

  echo "[2/4][$split] merge v46 route target with intervention trace features"
  PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/merge_generative_distill_features.py \
    --route_rows_csv "$V46_ROOT/pairwise_${split}_apply/decision_rows.csv" \
    --feature_rows_csv "$split_out/intervention_trace.probe.csv" \
    --feature_prefix probe_ \
    --route_col proxy_route \
    --positive_route baseline \
    --target_col "$TARGET_COL" \
    "${target_args[@]}" \
    --out_csv "$OUT/distill/${split}_rows.csv" \
    --out_summary_json "$OUT/distill/${split}_rows.summary.json" \
    --drop_unmatched_feature_rows
done

echo "[3/4][val] fit relaxed intervention-trace distill diagnostic"
set +e
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/build_generative_route_distill_proxy.py \
  --decision_rows_csv "$OUT/distill/val_rows.csv" \
  --route_col proxy_route \
  --positive_route baseline \
  --target_col "$TARGET_COL" \
  --use_existing_target \
  --feature_prefix probe_ \
  --top_n_features "$TOP_N_FEATURES" \
  --max_combo_size "$MAX_COMBO_SIZE" \
  --min_baseline_rate 0.0 \
  --max_baseline_rate "$DIAG_MAX_BASELINE_RATE" \
  --min_f1_gain_vs_intervention -1.0 \
  --max_chair_i_delta_vs_intervention 1.0 \
  --max_chair_s_delta_vs_intervention 1.0 \
  --selection_objective recall_minus_chairi \
  --out_dir "$OUT/distill/trace_val_diag"
diag_status=$?
set -e

if [[ "$diag_status" -eq 0 && -f "$OUT/distill/trace_val_diag/selected_policy.json" ]]; then
  echo "[4/4][test] apply relaxed intervention-trace diagnostic policy"
  PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/apply_generative_route_distill_proxy_to_rows.py \
    --rows_csv "$OUT/distill/test_rows.csv" \
    --selected_policy_json "$OUT/distill/trace_val_diag/selected_policy.json" \
    --out_dir "$OUT/distill/test_apply_diag" \
    --target_col "$TARGET_COL"
else
  echo "[4/4] relaxed diagnostic policy was not feasible; inspect $OUT/distill/trace_val_diag/summary.json if present"
fi

echo "[optional][val] fit strict intervention-trace policy"
set +e
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/build_generative_route_distill_proxy.py \
  --decision_rows_csv "$OUT/distill/val_rows.csv" \
  --route_col proxy_route \
  --positive_route baseline \
  --target_col "$TARGET_COL" \
  --use_existing_target \
  --feature_prefix probe_ \
  --top_n_features "$TOP_N_FEATURES" \
  --max_combo_size "$MAX_COMBO_SIZE" \
  --min_baseline_rate 0.0 \
  --max_baseline_rate "$STRICT_MAX_BASELINE_RATE" \
  --min_recall_gain_vs_intervention "$STRICT_MIN_RECALL_GAIN" \
  --max_chair_i_delta_vs_intervention "$STRICT_MAX_CHAIR_I_DELTA" \
  --max_chair_s_delta_vs_intervention "$STRICT_MAX_CHAIR_S_DELTA" \
  --selection_objective recall_minus_chairi \
  --out_dir "$OUT/distill/trace_val_strict"
strict_status=$?
set -e

if [[ "$strict_status" -eq 0 && -f "$OUT/distill/trace_val_strict/selected_policy.json" ]]; then
  echo "[optional][test] apply strict intervention-trace policy"
  PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/apply_generative_route_distill_proxy_to_rows.py \
    --rows_csv "$OUT/distill/test_rows.csv" \
    --selected_policy_json "$OUT/distill/trace_val_strict/selected_policy.json" \
    --out_dir "$OUT/distill/test_apply_strict" \
    --target_col "$TARGET_COL"
else
  echo "[optional] strict intervention-trace policy was not feasible; inspect $OUT/distill/trace_val_strict/summary.json if present"
fi

echo "[done] diag summary: $OUT/distill/test_apply_diag/summary.json"
echo "[done] strict summary: $OUT/distill/test_apply_strict/summary.json"
