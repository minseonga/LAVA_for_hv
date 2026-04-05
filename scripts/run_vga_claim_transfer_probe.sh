#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"

DISC_ROOT="${DISC_ROOT:-$CAL_ROOT/experiments/vga_claim_subset_probe_v1/discriminative}"
GEN_ROOT="${GEN_ROOT:-$CAL_ROOT/experiments/vga_generative_mention_probe_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_claim_transfer_probe_v1}"

DISC_TABLE_CSV="${DISC_TABLE_CSV:-$DISC_ROOT/vga_table.csv}"
GEN_TABLE_CSV="${GEN_TABLE_CSV:-$GEN_ROOT/vga_table.csv}"

mkdir -p "$OUT_ROOT"

if [[ ! -f "$DISC_TABLE_CSV" ]]; then
  echo "[error] missing discriminative table: $DISC_TABLE_CSV" >&2
  exit 1
fi
if [[ ! -f "$GEN_TABLE_CSV" ]]; then
  echo "[error] missing generative table: $GEN_TABLE_CSV" >&2
  exit 1
fi

cd "$CAL_ROOT"
"$CAL_PYTHON_BIN" scripts/analyze_vga_claim_transfer.py \
  --disc_table_csv "$DISC_TABLE_CSV" \
  --gen_table_csv "$GEN_TABLE_CSV" \
  --out_dir "$OUT_ROOT"

echo "[done] $OUT_ROOT/summary.json"
