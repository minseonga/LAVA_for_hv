#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"

PANEL_ROOT="${PANEL_ROOT:-$CAL_ROOT/experiments/common_pope_discovery_v3_panel_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/common_pope_discovery_posthoc_panel_v1}"

QUESTION_FILE="${QUESTION_FILE:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q_with_object.jsonl}"
GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_gt.csv}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"

METHODS="${METHODS:-vga,pai,vcd}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
LOG_EVERY="${LOG_EVERY:-50}"

FEATURE_COLS="${FEATURE_COLS:-cheap_lp_content_min,cheap_lp_content_tail_gap,cheap_lp_content_tail_z,cheap_lp_content_q10,cheap_lp_content_min_len_corr,cheap_target_gap_content_min,cheap_lp_content_std,cheap_entropy_content_mean,cheap_margin_content_mean,cheap_target_gap_content_std,cheap_conflict_lp_minus_entropy}"
MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
TOP_K="${TOP_K:-3}"
TAU_OBJECTIVE="${TAU_OBJECTIVE:-final_acc}"
MIN_BASELINE_RATE="${MIN_BASELINE_RATE:-0.0}"
MAX_BASELINE_RATE="${MAX_BASELINE_RATE:-1.0}"
MIN_SELECTED_COUNT="${MIN_SELECTED_COUNT:-0}"

BASELINE_JSONL="${BASELINE_JSONL:-$PANEL_ROOT/discriminative/baseline/pred_vanilla_discovery.jsonl}"

FEATURE_DIR="$OUT_ROOT/features"
TABLE_DIR="$OUT_ROOT/tables"
CONTROLLER_DIR="$OUT_ROOT/discovery/harm_controller"
MANIFEST_JSON="$OUT_ROOT/manifest.json"

mkdir -p "$FEATURE_DIR" "$TABLE_DIR" "$CONTROLLER_DIR"

if [[ ! -f "$BASELINE_JSONL" ]]; then
  echo "[error] missing baseline prediction jsonl: $BASELINE_JSONL" >&2
  exit 1
fi
if [[ ! -f "$QUESTION_FILE" ]]; then
  echo "[error] missing discovery question file: $QUESTION_FILE" >&2
  exit 1
fi
if [[ ! -f "$GT_CSV" ]]; then
  echo "[error] missing discovery gt csv: $GT_CSV" >&2
  exit 1
fi

IFS=',' read -r -a METHOD_LIST <<< "$METHODS"
TABLES=()

for method in "${METHOD_LIST[@]}"; do
  method="$(echo "$method" | xargs)"
  [[ -z "$method" ]] && continue

  pred_jsonl="$PANEL_ROOT/discriminative/$method/pred_${method}_discovery.jsonl"
  if [[ ! -f "$pred_jsonl" ]]; then
    echo "[warn] skip method=$method; missing prediction file: $pred_jsonl" >&2
    continue
  fi

  cheap_csv="$FEATURE_DIR/${method}_cheap_features.csv"
  cheap_summary="$FEATURE_DIR/${method}_cheap_features.summary.json"
  table_csv="$TABLE_DIR/${method}_table.csv"
  table_summary="$TABLE_DIR/${method}_table.summary.json"

  if [[ "$REUSE_IF_EXISTS" == "true" && -f "$cheap_csv" ]]; then
    echo "[reuse] $cheap_csv"
  else
    echo "[posthoc] extract cheap output features: method=$method"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    PYTHONPATH="$ROOT_DIR" "$CAL_PYTHON_BIN" scripts/extract_c_stage_cheap_online_features.py \
      --question_file "$QUESTION_FILE" \
      --image_folder "$IMAGE_FOLDER" \
      --intervention_pred_jsonl "$pred_jsonl" \
      --out_csv "$cheap_csv" \
      --out_summary_json "$cheap_summary" \
      --model_path "$MODEL_PATH" \
      --model_base "$MODEL_BASE" \
      --conv_mode "$CONV_MODE" \
      --device "$DEVICE" \
      --reuse_if_exists "$REUSE_IF_EXISTS" \
      --log_every "$LOG_EVERY"
  fi

  echo "[posthoc] build harm table: method=$method"
  PYTHONPATH="$ROOT_DIR" "$CAL_PYTHON_BIN" scripts/build_method_harm_table.py \
    --baseline_features_csv "$cheap_csv" \
    --baseline_pred_jsonl "$BASELINE_JSONL" \
    --intervention_pred_jsonl "$pred_jsonl" \
    --gt_csv "$GT_CSV" \
    --method_name "$method" \
    --benchmark_name pope \
    --split_name discovery \
    --out_csv "$table_csv" \
    --out_summary_json "$table_summary" \
    --baseline_pred_text_key auto \
    --intervention_pred_text_key auto

  TABLES+=("$table_csv")
done

if [[ "${#TABLES[@]}" -eq 0 ]]; then
  echo "[error] no method posthoc tables were built" >&2
  exit 1
fi

echo "[posthoc] calibrate pooled harm-veto controller"
PYTHONPATH="$ROOT_DIR" "$CAL_PYTHON_BIN" scripts/build_vga_pregate_harm_controller.py \
  --discovery_table_csvs "${TABLES[@]}" \
  --out_dir "$CONTROLLER_DIR" \
  --source_key method \
  --feature_cols "$FEATURE_COLS" \
  --min_feature_auroc "$MIN_FEATURE_AUROC" \
  --top_k "$TOP_K" \
  --tau_objective "$TAU_OBJECTIVE" \
  --min_baseline_rate "$MIN_BASELINE_RATE" \
  --max_baseline_rate "$MAX_BASELINE_RATE" \
  --min_selected_count "$MIN_SELECTED_COUNT"

CAL_ROOT_ENV="$CAL_ROOT" \
PANEL_ROOT_ENV="$PANEL_ROOT" \
OUT_ROOT_ENV="$OUT_ROOT" \
QUESTION_FILE_ENV="$QUESTION_FILE" \
GT_CSV_ENV="$GT_CSV" \
IMAGE_FOLDER_ENV="$IMAGE_FOLDER" \
METHODS_ENV="$METHODS" \
FEATURE_COLS_ENV="$FEATURE_COLS" \
MANIFEST_JSON_ENV="$MANIFEST_JSON" \
"$CAL_PYTHON_BIN" - <<'PY'
import json
import os

manifest = {
    "cal_root": os.environ["CAL_ROOT_ENV"],
    "panel_root": os.environ["PANEL_ROOT_ENV"],
    "out_root": os.environ["OUT_ROOT_ENV"],
    "question_file": os.environ["QUESTION_FILE_ENV"],
    "gt_csv": os.environ["GT_CSV_ENV"],
    "image_folder": os.environ["IMAGE_FOLDER_ENV"],
    "methods": [m.strip() for m in os.environ["METHODS_ENV"].split(",") if m.strip()],
    "feature_cols": os.environ["FEATURE_COLS_ENV"],
}
with open(os.environ["MANIFEST_JSON_ENV"], "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
PY

echo "[saved] $MANIFEST_JSON"
echo "[done] $CONTROLLER_DIR/summary.json"
