#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# Config (override via env vars)
# -------------------------------
CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VISTA_ROOT="${VISTA_ROOT:-/home/kms/VISTA}"
VISTA_PYTHON_BIN="${VISTA_PYTHON_BIN:-/home/kms/miniconda3/envs/vista/bin/python}"
CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
DATA_PATH="${DATA_PATH:-/home/kms/data/pope/val2014}"
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-liuhaotian/llava-v1.5-7b}"

GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"
BASELINE_JSONL="${BASELINE_JSONL:-$CAL_ROOT/experiments/pope_full_9000/vanilla_vs_vga_taxonomy/pred_vanilla_9000.jsonl}"
FEATURES_CSV="${FEATURES_CSV:-$CAL_ROOT/experiments/pope_feature_screen_v1_e1_4_l16_24/features/features_unified_table.csv}"

OUT_DIR="${OUT_DIR:-$CAL_ROOT/experiments/pope_full_9000/vista_vs_baseline_9000}"
TAX_OUT_DIR="${TAX_OUT_DIR:-$OUT_DIR/taxonomy}"
D1D2_OUT_DIR="${D1D2_OUT_DIR:-$OUT_DIR/d1d2_audit}"

# VISTA run config
EXP_FOLDER="${EXP_FOLDER:-pope_9000_vista_eval}"
MODEL="${MODEL:-llava-1.5}"
SEED="${SEED:-1994}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
VSV_LAMBDA="${VSV_LAMBDA:-0.01}"
LOGITS_LAYERS="${LOGITS_LAYERS:-25,30}"
LOGITS_ALPHA="${LOGITS_ALPHA:-0.3}"
VISTA_ENABLE_VSV="${VISTA_ENABLE_VSV:-1}"
VISTA_ENABLE_LOGITS_AUG="${VISTA_ENABLE_LOGITS_AUG:-1}"

# subset controls
ADV_SUBSET_SIZE="${ADV_SUBSET_SIZE:--1}"
POP_SUBSET_SIZE="${POP_SUBSET_SIZE:--1}"
RND_SUBSET_SIZE="${RND_SUBSET_SIZE:--1}"
CATEGORY_SIZE="${CATEGORY_SIZE:-3000}"
ONLY_CATEGORY="${ONLY_CATEGORY:-all}"

mkdir -p "$OUT_DIR" "$TAX_OUT_DIR" "$D1D2_OUT_DIR"

run_one_category() {
  local cat="$1"
  local subset_size="$2"

  local method_tag="org"
  if [[ "$VISTA_ENABLE_VSV" == "1" ]]; then
    method_tag="vsv_lambda_${VSV_LAMBDA}"
  fi

  local logits_tag=""
  if [[ "$VISTA_ENABLE_LOGITS_AUG" == "1" ]]; then
    logits_tag="_logaug_loglayer_${LOGITS_LAYERS}_logalpha_${LOGITS_ALPHA}"
  fi

  local file_name="seed${SEED}_${cat}_${method_tag}${logits_tag}_greedy_max_new_tokens_${MAX_NEW_TOKENS}.jsonl"
  local src="$VISTA_ROOT/exp_results/${EXP_FOLDER}/${MODEL}/${file_name}"
  local dst="$OUT_DIR/raw_${cat}.jsonl"

  if [[ -f "$src" ]]; then
    if [[ "$subset_size" != "-1" ]]; then
      local nline
      nline=$(wc -l < "$src" | tr -d ' ')
      if [[ "$nline" == "$subset_size" ]]; then
        echo "[reuse] existing VISTA output (matched subset_size=${subset_size}): $src"
        cp -f "$src" "$dst"
        echo "[saved] $dst"
        return
      else
        echo "[stale] existing output lines=${nline} but subset_size=${subset_size}; regenerating: $src"
        rm -f "$src"
      fi
    else
      local nline
      nline=$(wc -l < "$src" | tr -d ' ')
      if [[ "$nline" == "$CATEGORY_SIZE" ]]; then
        echo "[reuse] existing VISTA output (full mode matched category_size=${CATEGORY_SIZE}): $src"
        cp -f "$src" "$dst"
        echo "[saved] $dst"
        return
      else
        echo "[stale] existing output lines=${nline} but expected category_size=${CATEGORY_SIZE}; regenerating: $src"
        rm -f "$src"
      fi
    fi
  fi

  echo "[run] VISTA category=${cat} subset_size=${subset_size}"
  cd "$VISTA_ROOT"

  local cmd=(
    pope_eval.py
    --exp_folder "$EXP_FOLDER"
    --model "$MODEL"
    --pope-type "$cat"
    --data-path "$DATA_PATH"
    --batch-size 1
    --subset-size "$subset_size"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --num-beams 1
    --temperature 1.0
    --seed "$SEED"
  )

  if [[ "$VISTA_ENABLE_VSV" == "1" ]]; then
    cmd+=(--vsv --vsv-lambda "$VSV_LAMBDA")
  fi

  if [[ "$VISTA_ENABLE_LOGITS_AUG" == "1" ]]; then
    cmd+=(--logits-aug --logits-layers "$LOGITS_LAYERS" --logits-alpha "$LOGITS_ALPHA")
  fi

  CUDA_VISIBLE_DEVICES="$GPU" LLAVA_MODEL_PATH="$LLAVA_MODEL_PATH" "$VISTA_PYTHON_BIN" "${cmd[@]}"

  if [[ ! -f "$src" ]]; then
    echo "[error] expected output not found: $src"
    exit 1
  fi
  cp -f "$src" "$dst"
  echo "[saved] $dst"
}

if [[ "$ONLY_CATEGORY" == "all" ]]; then
  run_one_category adversarial "$ADV_SUBSET_SIZE"
  run_one_category popular "$POP_SUBSET_SIZE"
  run_one_category random "$RND_SUBSET_SIZE"
else
  case "$ONLY_CATEGORY" in
    adversarial)
      run_one_category adversarial "$ADV_SUBSET_SIZE"
      ;;
    popular)
      run_one_category popular "$POP_SUBSET_SIZE"
      ;;
    random)
      run_one_category random "$RND_SUBSET_SIZE"
      ;;
    *)
      echo "[error] ONLY_CATEGORY must be one of: all, adversarial, popular, random"
      exit 1
      ;;
  esac
fi

cd "$CAL_ROOT"

# Normalize category outputs into aligned question_id jsonl
"$CAL_PYTHON_BIN" scripts/normalize_pope_category_preds.py \
  --adversarial_jsonl "$OUT_DIR/raw_adversarial.jsonl" \
  --popular_jsonl "$OUT_DIR/raw_popular.jsonl" \
  --random_jsonl "$OUT_DIR/raw_random.jsonl" \
  --out_jsonl "$OUT_DIR/pred_vista_9000.jsonl" \
  --text_keys "ans,output,text,answer" \
  --category_size "$CATEGORY_SIZE" \
  --adv_subset_size "$ADV_SUBSET_SIZE" \
  --pop_subset_size "$POP_SUBSET_SIZE" \
  --rnd_subset_size "$RND_SUBSET_SIZE" \
  --sample_seed "$SEED"

# Build taxonomy vs vanilla baseline
"$CAL_PYTHON_BIN" scripts/build_vga_failure_taxonomy.py \
  --gt_csv "$GT_CSV" \
  --baseline_pred_jsonl "$BASELINE_JSONL" \
  --vga_pred_jsonl "$OUT_DIR/pred_vista_9000.jsonl" \
  --baseline_pred_text_key auto \
  --vga_pred_text_key output \
  --out_dir "$TAX_OUT_DIR"

# D1/D2 yes-dampening audit
if [[ -n "${FEATURES_CSV}" && -f "${FEATURES_CSV}" ]]; then
  "$CAL_PYTHON_BIN" scripts/analyze_yes_dampening_d1d2.py \
    --per_case_csv "$TAX_OUT_DIR/per_case_compare.csv" \
    --features_csv "$FEATURES_CSV" \
    --out_dir "$D1D2_OUT_DIR"
else
  "$CAL_PYTHON_BIN" scripts/analyze_yes_dampening_d1d2.py \
    --per_case_csv "$TAX_OUT_DIR/per_case_compare.csv" \
    --out_dir "$D1D2_OUT_DIR"
fi

echo "[done] $OUT_DIR"
