#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
DISCOVERY_ROOT="${DISCOVERY_ROOT:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_mix_train2014_2785}"

VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
HEADSET_JSON="${HEADSET_JSON:-$CAL_ROOT/experiments/pope_discovery/discovery_headset.json}"
OUT_DIR="${OUT_DIR:-$DISCOVERY_ROOT/vga_cost_aware_gain_router_artifact}"

DEFAULT_Q_WITH_OBJECT="$DISCOVERY_ROOT/discovery_q_with_object.jsonl"
DEFAULT_Q_MIX="$DISCOVERY_ROOT/discovery_mix_train2014.jsonl"
DEFAULT_TAX_ROOT="$DISCOVERY_ROOT/per_case_compare.csv"
DEFAULT_TAX_DIR="$DISCOVERY_ROOT/taxonomy/per_case_compare.csv"
DEFAULT_GT_ROOT="$DISCOVERY_ROOT/discovery_gt.csv"
DEFAULT_GT_ASSETS="$DISCOVERY_ROOT/assets/discovery_gt.csv"
DEFAULT_BASELINE_DIR="$DISCOVERY_ROOT/baseline/pred_baseline.jsonl"
DEFAULT_BASELINE_ROOT="$DISCOVERY_ROOT/pred_baseline.jsonl"
DEFAULT_VGA_DIR="$DISCOVERY_ROOT/vga/pred_vga.jsonl"
DEFAULT_VGA_ROOT="$DISCOVERY_ROOT/pred_vga.jsonl"
if [[ -z "${QUESTION_FILE:-}" ]]; then
  if [[ -f "$DEFAULT_Q_WITH_OBJECT" ]]; then
    QUESTION_FILE="$DEFAULT_Q_WITH_OBJECT"
  elif [[ -f "$DEFAULT_Q_MIX" ]]; then
    QUESTION_FILE="$DEFAULT_Q_MIX"
  else
    QUESTION_FILE="$DEFAULT_Q_WITH_OBJECT"
  fi
else
  QUESTION_FILE="${QUESTION_FILE}"
fi

if [[ -z "${TAXONOMY_CSV:-}" ]]; then
  if [[ -f "$DEFAULT_TAX_ROOT" ]]; then
    TAXONOMY_CSV="$DEFAULT_TAX_ROOT"
  elif [[ -f "$DEFAULT_TAX_DIR" ]]; then
    TAXONOMY_CSV="$DEFAULT_TAX_DIR"
  else
    TAXONOMY_CSV="$DEFAULT_TAX_ROOT"
  fi
else
  TAXONOMY_CSV="${TAXONOMY_CSV}"
fi

if [[ -z "${GT_CSV:-}" ]]; then
  if [[ -f "$DEFAULT_GT_ROOT" ]]; then
    GT_CSV="$DEFAULT_GT_ROOT"
  elif [[ -f "$DEFAULT_GT_ASSETS" ]]; then
    GT_CSV="$DEFAULT_GT_ASSETS"
  else
    GT_CSV="$DEFAULT_GT_ROOT"
  fi
fi

if [[ -z "${BASELINE_PRED_JSONL:-}" ]]; then
  if [[ -f "$DEFAULT_BASELINE_DIR" ]]; then
    BASELINE_PRED_JSONL="$DEFAULT_BASELINE_DIR"
  elif [[ -f "$DEFAULT_BASELINE_ROOT" ]]; then
    BASELINE_PRED_JSONL="$DEFAULT_BASELINE_ROOT"
  else
    BASELINE_PRED_JSONL="$DEFAULT_BASELINE_DIR"
  fi
fi

if [[ -z "${VGA_PRED_JSONL:-}" ]]; then
  if [[ -f "$DEFAULT_VGA_DIR" ]]; then
    VGA_PRED_JSONL="$DEFAULT_VGA_DIR"
  elif [[ -f "$DEFAULT_VGA_ROOT" ]]; then
    VGA_PRED_JSONL="$DEFAULT_VGA_ROOT"
else
  VGA_PRED_JSONL="$DEFAULT_VGA_DIR"
  fi
fi

if [[ "$(basename "$QUESTION_FILE")" == "discovery_mix_train2014.jsonl" ]]; then
  NORMALIZED_ASSET_DIR="${NORMALIZED_ASSET_DIR:-$OUT_DIR/discovery_assets}"
  NORMALIZED_Q_WITH_OBJECT="$NORMALIZED_ASSET_DIR/discovery_q_with_object.jsonl"
  NORMALIZED_GT="$NORMALIZED_ASSET_DIR/discovery_gt.csv"
  if [[ ! -f "$NORMALIZED_Q_WITH_OBJECT" || ! -f "$NORMALIZED_GT" ]]; then
    mkdir -p "$NORMALIZED_ASSET_DIR"
    python "$CAL_ROOT/scripts/build_pope_style_discovery_assets.py" \
      --in_jsonl "$QUESTION_FILE" \
      --out_dir "$NORMALIZED_ASSET_DIR"
  fi
  QUESTION_FILE="$NORMALIZED_Q_WITH_OBJECT"
  GT_CSV="$NORMALIZED_GT"
fi

CONV_MODE="${CONV_MODE:-llava_v1}"
DEVICE="${DEVICE:-cuda}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SAMPLING="${SAMPLING:-false}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8}"
NUM_BEAMS="${NUM_BEAMS:-1}"

CD_ALPHA="${CD_ALPHA:-0.02}"
ATTN_COEF="${ATTN_COEF:-0.2}"
START_LAYER="${START_LAYER:-16}"
END_LAYER="${END_LAYER:-24}"
HEAD_BALANCING="${HEAD_BALANCING:-simg}"
ATTN_NORM="${ATTN_NORM:-false}"
LATE_START="${LATE_START:-16}"
LATE_END="${LATE_END:-24}"
PROBE_FEATURE_MODE="${PROBE_FEATURE_MODE:-static_headset}"
PROBE_POSITION_MODE="${PROBE_POSITION_MODE:-baseline_yesno_preview}"
PROBE_BRANCH_SOURCE="${PROBE_BRANCH_SOURCE:-preview}"
PROBE_FORCE_MANUAL_FULLSEQ="${PROBE_FORCE_MANUAL_FULLSEQ:-false}"
PROBE_PREVIEW_MAX_NEW_TOKENS="${PROBE_PREVIEW_MAX_NEW_TOKENS:-3}"
PROBE_PREVIEW_REUSE_BASELINE="${PROBE_PREVIEW_REUSE_BASELINE:-true}"
PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST="${PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST:-true}"

TAU="${TAU:--0.0068411549792573}"
FEATURE_VARIANT="${FEATURE_VARIANT:-no_abs}"
DEPLOYMENT_BUDGET="${DEPLOYMENT_BUDGET:-0.30}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

mkdir -p "$OUT_DIR"
cd "$CAL_ROOT"

echo "[info] DISCOVERY_ROOT=$DISCOVERY_ROOT"
echo "[info] QUESTION_FILE=$QUESTION_FILE"
echo "[info] TAXONOMY_CSV=$TAXONOMY_CSV"
if [[ "$QUESTION_FILE" == "$DEFAULT_Q_MIX" ]]; then
  echo "[warn] discovery_q_with_object.jsonl not found; falling back to discovery_mix_train2014.jsonl"
fi
if [[ "$TAXONOMY_CSV" == "$DEFAULT_TAX_DIR" ]]; then
  echo "[warn] root per_case_compare.csv not found; falling back to taxonomy/per_case_compare.csv"
fi
if [[ -n "${NORMALIZED_ASSET_DIR:-}" ]]; then
  echo "[info] normalized discovery mix into $NORMALIZED_ASSET_DIR"
fi

TAX_VALIDATION_STATUS="$(python - "$TAXONOMY_CSV" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
required = ["id", "gt", "pred_baseline", "pred_vga", "baseline_ok", "vga_ok", "case_type"]
try:
    df = pd.read_csv(path, nrows=1)
except Exception as exc:
    print(f"read_error:{exc}")
    raise SystemExit(0)
missing = [c for c in required if c not in df.columns]
if missing:
    print("missing:" + ",".join(missing))
else:
    print("ok")
PY
)"

if [[ "$TAX_VALIDATION_STATUS" != "ok" ]]; then
  echo "[warn] TAXONOMY_CSV does not look like VGA branch taxonomy: $TAX_VALIDATION_STATUS"
  if [[ -f "$GT_CSV" && -f "$BASELINE_PRED_JSONL" && -f "$VGA_PRED_JSONL" ]]; then
    REBUILT_TAX_DIR="$OUT_DIR/taxonomy_rebuilt"
    mkdir -p "$REBUILT_TAX_DIR"
    echo "[info] rebuilding taxonomy at $REBUILT_TAX_DIR from discovery_gt/baseline/VGA predictions"
    python scripts/build_vga_failure_taxonomy.py \
      --gt_csv "$GT_CSV" \
      --baseline_pred_jsonl "$BASELINE_PRED_JSONL" \
      --vga_pred_jsonl "$VGA_PRED_JSONL" \
      --baseline_pred_text_key text \
      --vga_pred_text_key output \
      --out_dir "$REBUILT_TAX_DIR"
    TAXONOMY_CSV="$REBUILT_TAX_DIR/per_case_compare.csv"
    echo "[info] using rebuilt TAXONOMY_CSV=$TAXONOMY_CSV"
  else
    echo "[error] cannot rebuild taxonomy automatically."
    echo "[error] expected GT_CSV=$GT_CSV"
    echo "[error] expected BASELINE_PRED_JSONL=$BASELINE_PRED_JSONL"
    echo "[error] expected VGA_PRED_JSONL=$VGA_PRED_JSONL"
    exit 1
  fi
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
CAL_ROOT="$CAL_ROOT" \
VGA_ROOT="$VGA_ROOT" \
MODEL_PATH="$MODEL_PATH" \
IMAGE_FOLDER="$IMAGE_FOLDER" \
QUESTION_FILE="$QUESTION_FILE" \
TAXONOMY_CSV="$TAXONOMY_CSV" \
HEADSET_JSON="$HEADSET_JSON" \
OUT_DIR="$OUT_DIR" \
CONV_MODE="$CONV_MODE" \
DEVICE="$DEVICE" \
TEMPERATURE="$TEMPERATURE" \
TOP_P="$TOP_P" \
SAMPLING="$SAMPLING" \
MAX_GEN_LEN="$MAX_GEN_LEN" \
NUM_BEAMS="$NUM_BEAMS" \
CD_ALPHA="$CD_ALPHA" \
ATTN_COEF="$ATTN_COEF" \
START_LAYER="$START_LAYER" \
END_LAYER="$END_LAYER" \
HEAD_BALANCING="$HEAD_BALANCING" \
ATTN_NORM="$ATTN_NORM" \
LATE_START="$LATE_START" \
LATE_END="$LATE_END" \
PROBE_FEATURE_MODE="$PROBE_FEATURE_MODE" \
PROBE_POSITION_MODE="$PROBE_POSITION_MODE" \
PROBE_BRANCH_SOURCE="$PROBE_BRANCH_SOURCE" \
PROBE_FORCE_MANUAL_FULLSEQ="$PROBE_FORCE_MANUAL_FULLSEQ" \
PROBE_PREVIEW_MAX_NEW_TOKENS="$PROBE_PREVIEW_MAX_NEW_TOKENS" \
PROBE_PREVIEW_REUSE_BASELINE="$PROBE_PREVIEW_REUSE_BASELINE" \
PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST="$PROBE_PREVIEW_FALLBACK_TO_PROMPT_LAST" \
TAU="$TAU" \
FEATURE_VARIANT="$FEATURE_VARIANT" \
DEPLOYMENT_BUDGET="$DEPLOYMENT_BUDGET" \
SEED="$SEED" \
MAX_SAMPLES="$MAX_SAMPLES" \
bash "$CAL_ROOT/scripts/build_vga_cost_aware_gain_router_artifact_9000.sh"

echo "[done] $OUT_DIR"
echo "[saved] $OUT_DIR/router/summary.json"
