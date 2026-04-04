#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
SOURCE_ROOT="${SOURCE_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_harm_v1}"
OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/vga_pregate_semantic_v3_analysis_compare}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUT_ROOT"
cd "$ROOT_DIR"

echo "[1/3] discovery analysis with POPE+AMBER mix"
CAL_ROOT="$CAL_ROOT" \
SOURCE_ROOT="$SOURCE_ROOT" \
OUT_ROOT="$OUT_ROOT/mix" \
DISCOVERY_ONLY=true \
INCLUDE_AMBER_DISCOVERY=true \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_vga_pregate_semantic_v3.sh

echo "[2/3] discovery analysis with POPE-only"
CAL_ROOT="$CAL_ROOT" \
SOURCE_ROOT="$SOURCE_ROOT" \
OUT_ROOT="$OUT_ROOT/pope_only" \
DISCOVERY_ONLY=true \
INCLUDE_AMBER_DISCOVERY=false \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_vga_pregate_semantic_v3.sh

echo "[3/3] compare summaries"
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" scripts/compare_vga_pregate_v3_summaries.py \
  --mix_summary_json "$OUT_ROOT/mix/discovery/unified_controller/summary.json" \
  --pope_only_summary_json "$OUT_ROOT/pope_only/discovery/unified_controller/summary.json" \
  --out_json "$OUT_ROOT/comparison/summary.json"

echo "[done] $OUT_ROOT/comparison/summary.json"
