#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  GPU="$1"
  shift
fi

CAL_ROOT="${CAL_ROOT:-$ROOT_DIR}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"
PY_BIN="${PY_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

SRC="${SRC:-$CAL_ROOT/experiments/coco_chair_vga_pvg_ablation_first_next_len512}"
OUT="${OUT:-$CAL_ROOT/experiments/coco_chair_suffix_repair_smoke}"
TRACE_ROOT="${TRACE_ROOT:-$CAL_ROOT/experiments/coco_chair_vga_linear_v50_trace_novelty}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
COCO_ANN_ROOT="${COCO_ANN_ROOT:-/home/kms/data/COCO/annotations_trainval2014/annotations}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-llava_v1}"

SPLITS="${SPLITS:-test}"
METHOD_PRED_NAME="${METHOD_PRED_NAME:-pred_vga_full_pvg_caption.jsonl}"
LIMIT="${LIMIT:-50}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"
REPAIR_ALL="${REPAIR_ALL:-false}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
LAST_NEW_POS_FRAC_MAX="${LAST_NEW_POS_FRAC_MAX:-0.65}"
TAIL_NEW_WORD_RATE_MAX="${TAIL_NEW_WORD_RATE_MAX:-0.20}"
TAIL_REPEAT_RATE_MIN="${TAIL_REPEAT_RATE_MIN:-0.80}"
MIN_CONTENT_WORDS_FOR_REPAIR="${MIN_CONTENT_WORDS_FOR_REPAIR:-6}"
MIN_SUFFIX_CONTENT_WORDS="${MIN_SUFFIX_CONTENT_WORDS:-1}"
MIN_PREFIX_CONTENT_WORDS="${MIN_PREFIX_CONTENT_WORDS:-4}"
PREFIX_MARGIN_CONTENT_WORDS="${PREFIX_MARGIN_CONTENT_WORDS:-1}"

chair_ann_ready() {
  local root="$1"
  [[ -f "$root/instances_val2014.json" ]] && \
  [[ -f "$root/instances_train2014.json" ]] && \
  [[ -f "$root/captions_val2014.json" ]] && \
  [[ -f "$root/captions_train2014.json" ]]
}

resolve_coco_ann_root() {
  local raw="$1"
  local image_parent=""
  local image_grandparent=""
  image_parent="$(dirname "$IMAGE_FOLDER")"
  image_grandparent="$(dirname "$image_parent")"
  local candidates=(
    "$raw"
    "$raw/annotations"
    "$raw/annotations_trainval2014"
    "$raw/annotations_trainval2014/annotations"
    "$image_parent/annotations"
    "$image_parent/annotations_trainval2014/annotations"
    "$image_grandparent/annotations"
    "$image_grandparent/annotations_trainval2014/annotations"
    "/home/kms/data/COCO/annotations"
    "/home/kms/data/COCO/annotations_trainval2014/annotations"
    "/home/kms/data/images/mscoco/annotations"
    "/home/kms/data/images/mscoco/annotations_trainval2014/annotations"
  )
  local cand=""
  for cand in "${candidates[@]}"; do
    if [[ -n "$cand" ]] && chair_ann_ready "$cand"; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

reuse_file() {
  local path="$1"
  if [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]; then
    return 0
  fi
  return 1
}

run_chair_eval() {
  local cap_file="$1"
  local image_id_key="$2"
  local caption_key="$3"
  local save_path="$4"
  local prepared_cap_file="$5"
  if reuse_file "$save_path"; then
    echo "[reuse] $save_path"
    return
  fi
  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/prepare_chair_caption_jsonl.py \
      --in_file "$cap_file" \
      --out_file "$prepared_cap_file" \
      --image_id_key "$image_id_key" \
      --image_key image \
      --drop_missing
    PYTHONPATH="$EAZY_ROOT:${PYTHONPATH:-}" "$EAZY_PYTHON_BIN" "$EAZY_ROOT/eval_script/chair.py" \
      --cap_file "$prepared_cap_file" \
      --image_id_key "$image_id_key" \
      --caption_key "$caption_key" \
      --coco_path "$COCO_ANN_ROOT" \
      --cache "$CHAIR_CACHE" \
      --save_path "$save_path"
  )
}

bool_flag=()
if [[ "$REPAIR_ALL" == "true" || "$REPAIR_ALL" == "1" ]]; then
  bool_flag+=(--repair_all)
fi

COCO_ANN_ROOT="$(resolve_coco_ann_root "$COCO_ANN_ROOT" || true)"
if [[ -z "$COCO_ANN_ROOT" ]]; then
  echo "[error] could not locate COCO annotation root for CHAIR eval" >&2
  echo "[hint] set COCO_ANN_ROOT to the directory that directly contains COCO 2014 instances/captions annotations" >&2
  exit 1
fi

mkdir -p "$OUT/summary"
CHAIR_CACHE="$OUT/chair_cache.pkl"

echo "[config] CAL_ROOT=$CAL_ROOT"
echo "[config] SRC=$SRC"
echo "[config] OUT=$OUT"
echo "[config] TRACE_ROOT=$TRACE_ROOT"
echo "[config] GPU=$GPU LIMIT=$LIMIT SPLITS=$SPLITS"
echo "[config] repair_all=$REPAIR_ALL max_new_tokens=$MAX_NEW_TOKENS"

summary_args=(
  --out_csv "$OUT/summary/chair_suffix_repair.csv"
  --out_json "$OUT/summary/chair_suffix_repair.json"
)

for split in $SPLITS; do
  split_dir="$OUT/$split"
  mkdir -p "$split_dir"
  q_jsonl="$SRC/splits/${split}_caption_q.jsonl"
  int_pred="$SRC/$split/$METHOD_PRED_NAME"
  baseline_pred="$SRC/$split/pred_vanilla_caption.jsonl"
  baseline_chair="$SRC/$split/chair_baseline.json"
  int_chair="$SRC/$split/chair_vga_full_pvg.json"
  trace_csv="$TRACE_ROOT/features/$split/intervention_trace.probe.csv"
  if [[ ! -f "$trace_csv" ]]; then
    trace_csv=""
  fi

  if [[ ! -f "$q_jsonl" || ! -f "$int_pred" ]]; then
    echo "[error][$split] missing q_jsonl or intervention prediction under $SRC" >&2
    exit 1
  fi

  pred_repair="$split_dir/pred_suffix_repair_caption.jsonl"
  chair_repair="$split_dir/chair_suffix_repair.json"
  chair_input="$split_dir/chair_input_suffix_repair.jsonl"
  baseline_chair_for_summary="$baseline_chair"
  int_chair_for_summary="$int_chair"

  if [[ "$LIMIT" != "0" ]]; then
    baseline_limited="$split_dir/pred_vanilla_caption.limited.jsonl"
    int_limited="$split_dir/pred_vga_full_pvg_caption.limited.jsonl"
    baseline_chair_limited="$split_dir/chair_baseline_limited.json"
    int_chair_limited="$split_dir/chair_vga_full_pvg_limited.json"
    baseline_chair_input_limited="$split_dir/chair_input_baseline_limited.jsonl"
    int_chair_input_limited="$split_dir/chair_input_vga_full_pvg_limited.jsonl"

    if [[ -f "$baseline_pred" ]]; then
      PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/filter_jsonl_by_question_ids.py \
        --question_file "$q_jsonl" \
        --in_jsonl "$baseline_pred" \
        --out_jsonl "$baseline_limited" \
        --limit "$LIMIT" \
        --missing_ok
      run_chair_eval "$baseline_limited" image_id text "$baseline_chair_limited" "$baseline_chair_input_limited"
      baseline_chair_for_summary="$baseline_chair_limited"
    fi

    PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/filter_jsonl_by_question_ids.py \
      --question_file "$q_jsonl" \
      --in_jsonl "$int_pred" \
      --out_jsonl "$int_limited" \
      --limit "$LIMIT" \
      --missing_ok
    run_chair_eval "$int_limited" image_id output "$int_chair_limited" "$int_chair_input_limited"
    int_chair_for_summary="$int_chair_limited"
  fi

  echo "[1/2][$split] generate prefix-preserving suffix repair captions"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/generate_prefix_suffix_repair_captions.py \
    --question_file "$q_jsonl" \
    --image_folder "$IMAGE_FOLDER" \
    --intervention_pred_jsonl "$int_pred" \
    --trace_features_csv "$trace_csv" \
    --out_jsonl "$pred_repair" \
    --out_summary_json "$split_dir/pred_suffix_repair_caption.summary.json" \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --conv_mode "$CONV_MODE" \
    --device cuda \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --limit "$LIMIT" \
    --reuse_if_exists "$REUSE_IF_EXISTS" \
    --last_new_pos_frac_max "$LAST_NEW_POS_FRAC_MAX" \
    --tail_new_word_rate_max "$TAIL_NEW_WORD_RATE_MAX" \
    --tail_repeat_rate_min "$TAIL_REPEAT_RATE_MIN" \
    --min_content_words_for_repair "$MIN_CONTENT_WORDS_FOR_REPAIR" \
    --min_suffix_content_words "$MIN_SUFFIX_CONTENT_WORDS" \
    --min_prefix_content_words "$MIN_PREFIX_CONTENT_WORDS" \
    --prefix_margin_content_words "$PREFIX_MARGIN_CONTENT_WORDS" \
    "${bool_flag[@]}"

  echo "[2/2][$split] CHAIR eval for suffix repair"
  run_chair_eval "$pred_repair" image_id output "$chair_repair" "$chair_input"

  if [[ -f "$baseline_chair_for_summary" ]]; then
    summary_args+=(--entry "baseline::$split::$baseline_chair_for_summary")
  fi
  if [[ -f "$int_chair_for_summary" ]]; then
    summary_args+=(--entry "vga_full_pvg::$split::$int_chair_for_summary")
  fi
  summary_args+=(--entry "suffix_repair::$split::$chair_repair")
done

echo "[summary] summarize CHAIR repair comparison"
PYTHONPATH="$CAL_ROOT" "$PY_BIN" scripts/summarize_chair_main_table.py "${summary_args[@]}"

echo "[done] repair predictions under $OUT"
echo "[done] summary: $OUT/summary/chair_suffix_repair.json"
