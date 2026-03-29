#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
PY_BIN="${PY_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"

PER_CASE_CSV="${PER_CASE_CSV:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/vga/taxonomy/per_case_compare.csv}"
OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict/vga_oracle_controller_9000}"

mkdir -p "$OUT_DIR"

cd "$CAL_ROOT"
"$PY_BIN" scripts/build_vga_decision_oracles.py \
  --per_case_csv "$PER_CASE_CSV" \
  --out_dir "$OUT_DIR" \
  --tie_preference vga \
  --regime_safe_action vga \
  --regime_hard_action vga \
  --regime_d1_action vga \
  --regime_d2_action baseline

echo "[done] $OUT_DIR"

