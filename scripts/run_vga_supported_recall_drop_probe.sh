#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"

BASE_OUT_ROOT="${BASE_OUT_ROOT:-$CAL_ROOT/experiments/vga_generative_coverage_probe_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_supported_recall_drop_probe_v1}"

TABLE_CSV="${TABLE_CSV:-$BASE_OUT_ROOT/vga_claim_aware_table.csv}"
FEATURE_COLS="${FEATURE_COLS:-probe_n_content_tokens,probe_tail_tokens_after_last_mention,probe_entropy_tail_mean_real,probe_lp_tail_mean_real,probe_last_mention_pos_frac,probe_gap_tail_mean_real,probe_object_diversity,probe_mention_diversity,probe_entropy_tail_minus_head_real,probe_gap_tail_minus_head_real}"
TARGET_SPECS="${TARGET_SPECS:-delta_supported_recall:lt:0,claim_supported_dropped}"
TOP_K_VALUES="${TOP_K_VALUES:-1,2,3,4,5}"

mkdir -p "$OUT_ROOT"

if [[ ! -f "$TABLE_CSV" ]]; then
  echo "[error] missing coverage table: $TABLE_CSV" >&2
  exit 1
fi

cd "$CAL_ROOT"
"$CAL_PYTHON_BIN" scripts/analyze_generative_target_composite.py \
  --table_csv "$TABLE_CSV" \
  --out_dir "$OUT_ROOT" \
  --feature_cols "$FEATURE_COLS" \
  --target_specs "$TARGET_SPECS" \
  --top_k_values "$TOP_K_VALUES"

echo "[done] $OUT_ROOT/summary.json"
