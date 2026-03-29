#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# Config (override via env vars)
# -------------------------------
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy/bin/python}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"
POPE_COCO_FALLBACK="${POPE_COCO_FALLBACK:-/home/kms/VISTA/pope_coco}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
DATA_PATH="${DATA_PATH:-/home/kms/data/pope/val2014}"

GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
BASELINE_JSONL="${BASELINE_JSONL:-$CAL_ROOT/experiments/pope_full_9000/vanilla_vs_vga_taxonomy/pred_vanilla_9000.jsonl}"
FEATURES_CSV="${FEATURES_CSV:-$CAL_ROOT/experiments/pope_feature_screen_v1_e1_4_l16_24/features/features_unified_table.csv}"

OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/pope_full_9000/eazy_vs_baseline_9000}"
TAX_OUT_DIR="${TAX_OUT_DIR:-$OUT_DIR/taxonomy}"
D1D2_OUT_DIR="${D1D2_OUT_DIR:-$OUT_DIR/d1d2_audit}"

# EAZY run config
MODEL="${MODEL:-llava-1.5}"
BEAM="${BEAM:-1}"
TOPK_K="${TOPK_K:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CATEGORY_SIZE="${CATEGORY_SIZE:-3000}"
ONLY_CATEGORY="${ONLY_CATEGORY:-all}"
EAZY_REQUIRE_CHAIR="${EAZY_REQUIRE_CHAIR:-1}"

mkdir -p "$OUT_DIR" "$TAX_OUT_DIR" "$D1D2_OUT_DIR"

if [[ ! -f "$EAZY_PYTHON_BIN" ]]; then
  echo "[warn] EAZY_PYTHON_BIN not found: $EAZY_PYTHON_BIN"
  echo "[warn] fallback to CAL_PYTHON_BIN: $CAL_PYTHON_BIN"
  EAZY_PYTHON_BIN="$CAL_PYTHON_BIN"
fi

# Guardrail: onepass EAZY depends on CHAIR detector.
# If CHAIR import fails, runtime degrades to no-zeroout (near-vanilla behavior).
if ! PYTHONPATH="$EAZY_ROOT:${PYTHONPATH:-}" "$EAZY_PYTHON_BIN" - <<'PY'
ok = True
try:
    from utils.chair_detector import CHAIR_detector  # noqa: F401
except Exception:
    ok = False
print("ok" if ok else "missing")
raise SystemExit(0 if ok else 1)
PY
then
  echo "[warn] CHAIR detector import failed in EAZY env."
  echo "[warn] This run will not be true EAZY intervention (fallback no-zeroout)."
  if [[ "$EAZY_REQUIRE_CHAIR" == "1" ]]; then
    echo "[error] Set EAZY_REQUIRE_CHAIR=0 to allow fallback mode intentionally."
    exit 1
  fi
fi

# EAZY eval script uses relative path pope_coco/*.json.
# If absent in EAZY repo, reuse VISTA's pope_coco via symlink.
if [[ ! -d "$EAZY_ROOT/pope_coco" ]]; then
  if [[ -d "$POPE_COCO_FALLBACK" ]]; then
    ln -sfn "$POPE_COCO_FALLBACK" "$EAZY_ROOT/pope_coco"
    echo "[info] linked pope_coco -> $POPE_COCO_FALLBACK"
  else
    echo "[error] pope_coco not found in EAZY and fallback missing: $POPE_COCO_FALLBACK"
    exit 1
  fi
fi

run_one_category() {
  local cat="$1"
  local save_path="$OUT_DIR/raw_${cat}.jsonl"

  if [[ -f "$save_path" ]]; then
    local nline
    nline=$(wc -l < "$save_path" | tr -d ' ')
    if [[ "$nline" == "$CATEGORY_SIZE" ]]; then
      echo "[reuse] existing EAZY output (matched category_size=${CATEGORY_SIZE}): $save_path"
      return
    fi

    # Some EAZY dumps are one-line files with literal "\n" separators.
    # In that case wc -l is 0, so check literal separator count as fallback.
    local literal_n
    literal_n=$(grep -o '\\n' "$save_path" | wc -l | tr -d ' ' || true)
    if [[ "$literal_n" == "$CATEGORY_SIZE" ]]; then
      echo "[reuse] existing EAZY output (matched literal \n count=${CATEGORY_SIZE}): $save_path"
      return
    fi

    echo "[stale] existing output lines=${nline}, literal_n=${literal_n}; expected category_size=${CATEGORY_SIZE}. regenerating: $save_path"
    rm -f "$save_path"
  fi

  echo "[run] EAZY category=${cat}"
  cd "$EAZY_ROOT"

  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$EAZY_ROOT:${PYTHONPATH:-}" "$EAZY_PYTHON_BIN" eval_script/pope_eval_eazy_onepass.py     --model "$MODEL"     --pope-type "$cat"     --gpu-id 0     --data_path "$DATA_PATH"     --batch_size "$BATCH_SIZE"     --num_workers "$NUM_WORKERS"     --beam "$BEAM"     --k "$TOPK_K"     --save-jsonl "$save_path"

  if [[ ! -f "$save_path" ]]; then
    echo "[error] expected output not found: $save_path"
    exit 1
  fi
  echo "[saved] $save_path"
}

if [[ "$ONLY_CATEGORY" == "all" ]]; then
  run_one_category adversarial
  run_one_category popular
  run_one_category random
else
  case "$ONLY_CATEGORY" in
    adversarial)
      run_one_category adversarial
      ;;
    popular)
      run_one_category popular
      ;;
    random)
      run_one_category random
      ;;
    *)
      echo "[error] ONLY_CATEGORY must be one of: all, adversarial, popular, random"
      exit 1
      ;;
  esac
fi

cd "$CAL_ROOT"

# Normalize 3x category outputs into question_id 0..8999 aligned jsonl
"$CAL_PYTHON_BIN" scripts/normalize_pope_category_preds.py \
  --adversarial_jsonl "$OUT_DIR/raw_adversarial.jsonl" \
  --popular_jsonl "$OUT_DIR/raw_popular.jsonl" \
  --random_jsonl "$OUT_DIR/raw_random.jsonl" \
  --out_jsonl "$OUT_DIR/pred_eazy_9000.jsonl" \
  --text_keys "ans,output,text,answer" \
  --category_size "$CATEGORY_SIZE"

# Build taxonomy vs vanilla baseline
"$CAL_PYTHON_BIN" scripts/build_vga_failure_taxonomy.py \
  --gt_csv "$GT_CSV" \
  --baseline_pred_jsonl "$BASELINE_JSONL" \
  --vga_pred_jsonl "$OUT_DIR/pred_eazy_9000.jsonl" \
  --baseline_pred_text_key auto \
  --vga_pred_text_key output \
  --out_dir "$TAX_OUT_DIR"

# D1/D2 yes-dampening audit
"$CAL_PYTHON_BIN" scripts/analyze_yes_dampening_d1d2.py \
  --per_case_csv "$TAX_OUT_DIR/per_case_compare.csv" \
  --features_csv "$FEATURES_CSV" \
  --out_dir "$D1D2_OUT_DIR"

echo "[done] $OUT_DIR"
