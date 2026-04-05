#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
VISTA_ROOT="${VISTA_ROOT:-/home/kms/VISTA}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"
VISTA_PYTHON_BIN="${VISTA_PYTHON_BIN:-/home/kms/miniconda3/envs/vista_base/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
DISCOVERY_ASSET_ROOT="${DISCOVERY_ASSET_ROOT:-$CAL_ROOT/experiments/tau_c_calibration_mix_train2014_2785/assets}"
DISCOVERY_ASSET_FALLBACK_ROOT="${DISCOVERY_ASSET_FALLBACK_ROOT:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets}"
Q_NOOBJ="${Q_NOOBJ:-}"
Q_WITHOBJ="${Q_WITHOBJ:-}"
GT_CSV="${GT_CSV:-}"
COCO_ANN_ROOT="${COCO_ANN_ROOT:-/home/kms/data/COCO/annotations_trainval2014/annotations}"

LEGACY_FEATURES_CSV="${LEGACY_FEATURES_CSV:-$CAL_ROOT/experiments/pope_feature_screen_v1_e1_4_l16_24/features/features_unified_table.csv}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/common_pope_discovery_harm_miner_v1}"
DISC_ROOT="$OUT_ROOT/discriminative"
GEN_ROOT="$OUT_ROOT/generative"
BASELINE_DIR="$DISC_ROOT/baseline"
VGA_DIR="$DISC_ROOT/vga"
VISTA_DIR="$DISC_ROOT/vista"
EAZY_DIR="$DISC_ROOT/eazy"
TABLE_DIR="$DISC_ROOT/tables"
ANALYSIS_DIR="$DISC_ROOT/analysis"
GEN_ASSET_DIR="$GEN_ROOT/assets"
GEN_BASELINE_DIR="$GEN_ROOT/baseline"
GEN_VGA_DIR="$GEN_ROOT/vga"
GEN_VISTA_DIR="$GEN_ROOT/vista"
GEN_EAZY_DIR="$GEN_ROOT/eazy"
GEN_TABLE_DIR="$GEN_ROOT/tables"
GEN_ANALYSIS_DIR="$GEN_ROOT/analysis"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_VGA="${RUN_VGA:-1}"
RUN_VISTA="${RUN_VISTA:-1}"
RUN_EAZY="${RUN_EAZY:-1}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

VGA_CONDA_SH="${VGA_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
VGA_ENV="${VGA_ENV:-vga_base}"
VGA_CONV_MODE="${VGA_CONV_MODE:-llava_v1}"
VGA_MAX_GEN_LEN="${VGA_MAX_GEN_LEN:-8}"
VGA_USE_ADD="${VGA_USE_ADD:-true}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_HEAD_BALANCING="${VGA_HEAD_BALANCING:-simg}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"
SEED="${SEED:-1994}"

VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-pope_discovery_vista_eval}"
VISTA_MODEL="${VISTA_MODEL:-llava-1.5}"
VISTA_ENABLE_VSV="${VISTA_ENABLE_VSV:-1}"
VISTA_ENABLE_LOGITS_AUG="${VISTA_ENABLE_LOGITS_AUG:-1}"

EAZY_MODEL="${EAZY_MODEL:-llava-1.5}"
EAZY_BEAM="${EAZY_BEAM:-1}"
EAZY_TOPK_K="${EAZY_TOPK_K:-2}"
EAZY_REQUIRE_CHAIR="${EAZY_REQUIRE_CHAIR:-1}"

FEATURE_COLS="${FEATURE_COLS:-base_lp_content_mean,base_target_argmax_content_mean,base_target_gap_content_min,base_entropy_content_mean,base_conflict_lp_minus_entropy}"

chair_ann_ready() {
  local root="$1"
  [[ -f "$root/instances_val2014.json" ]] && \
  [[ -f "$root/instances_train2014.json" ]] && \
  [[ -f "$root/captions_val2014.json" ]] && \
  [[ -f "$root/captions_train2014.json" ]]
}

resolve_coco_ann_root() {
  local raw="$1"
  local candidates=()
  local image_parent=""
  local image_grandparent=""
  image_parent="$(dirname "$IMAGE_FOLDER")"
  image_grandparent="$(dirname "$image_parent")"
  candidates+=("$raw")
  candidates+=("$raw/annotations")
  candidates+=("$raw/annotations_trainval2014")
  candidates+=("$raw/annotations_trainval2014/annotations")
  candidates+=("$image_parent/annotations")
  candidates+=("$image_parent/annotations_trainval2014/annotations")
  candidates+=("$image_grandparent/annotations")
  candidates+=("$image_grandparent/annotations_trainval2014/annotations")
  candidates+=("/home/kms/data/COCO/annotations")
  candidates+=("/home/kms/data/COCO/annotations_trainval2014/annotations")
  candidates+=("/home/kms/data/images/mscoco/annotations")
  candidates+=("/home/kms/data/images/mscoco/annotations_trainval2014/annotations")
  candidates+=("/home/kms/data/coco/annotations")
  candidates+=("/home/kms/data/mscoco/annotations")
  local cand
  for cand in "${candidates[@]}"; do
    if chair_ann_ready "$cand"; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  return 1
}

VISTA_FLAGS=()
if [[ "$VISTA_ENABLE_VSV" == "1" ]]; then
  VISTA_FLAGS+=(--vsv)
fi
if [[ "$VISTA_ENABLE_LOGITS_AUG" == "1" ]]; then
  VISTA_FLAGS+=(--logits_aug)
fi

if [[ ! -d "$DISCOVERY_ASSET_ROOT" && -d "$DISCOVERY_ASSET_FALLBACK_ROOT" ]]; then
  DISCOVERY_ASSET_ROOT="$DISCOVERY_ASSET_FALLBACK_ROOT"
fi

if [[ -z "$Q_NOOBJ" ]]; then
  if [[ -f "$DISCOVERY_ASSET_ROOT/discovery_q.jsonl" ]]; then
    Q_NOOBJ="$DISCOVERY_ASSET_ROOT/discovery_q.jsonl"
  elif [[ -f "$DISCOVERY_ASSET_ROOT/discovery_q_with_object.jsonl" ]]; then
    Q_NOOBJ="$DISCOVERY_ASSET_ROOT/discovery_q_with_object.jsonl"
  fi
fi

if [[ -z "$Q_WITHOBJ" ]]; then
  if [[ -f "$DISCOVERY_ASSET_ROOT/discovery_q_with_object.jsonl" ]]; then
    Q_WITHOBJ="$DISCOVERY_ASSET_ROOT/discovery_q_with_object.jsonl"
  elif [[ -f "$DISCOVERY_ASSET_ROOT/discovery_q.jsonl" ]]; then
    Q_WITHOBJ="$DISCOVERY_ASSET_ROOT/discovery_q.jsonl"
  fi
fi

if [[ -z "$GT_CSV" ]]; then
  GT_CSV="$DISCOVERY_ASSET_ROOT/discovery_gt.csv"
fi

if [[ ! -f "$Q_NOOBJ" ]]; then
  echo "[error] missing discovery question file for caption prompts: $Q_NOOBJ" >&2
  exit 1
fi
if [[ ! -f "$Q_WITHOBJ" ]]; then
  echo "[error] missing discovery question file for discriminative run: $Q_WITHOBJ" >&2
  exit 1
fi
if [[ ! -f "$GT_CSV" ]]; then
  echo "[error] missing discovery gt csv: $GT_CSV" >&2
  exit 1
fi

COCO_ANN_ROOT="$(resolve_coco_ann_root "$COCO_ANN_ROOT" || true)"
if [[ -z "$COCO_ANN_ROOT" ]]; then
  echo "[error] could not locate COCO annotation root containing train/val 2014 instances+captions jsons" >&2
  echo "[hint] set COCO_ANN_ROOT to the directory that directly contains instances_val2014.json and captions_val2014.json" >&2
  exit 1
fi

echo "[assets] root=$DISCOVERY_ASSET_ROOT"
echo "[assets] q_noobj=$Q_NOOBJ"
echo "[assets] q_withobj=$Q_WITHOBJ"
echo "[assets] gt_csv=$GT_CSV"
echo "[assets] coco_ann_root=$COCO_ANN_ROOT"

DISC_METHODS=()
GEN_METHODS=()
if [[ "$RUN_VGA" == "1" ]]; then
  DISC_METHODS+=(vga)
  GEN_METHODS+=(vga)
fi
if [[ "$RUN_VISTA" == "1" ]]; then
  DISC_METHODS+=(vista)
  GEN_METHODS+=(vista)
fi
if [[ "$RUN_EAZY" == "1" ]]; then
  DISC_METHODS+=(eazy)
  GEN_METHODS+=(eazy)
fi

mkdir -p "$BASELINE_DIR" "$VGA_DIR" "$VISTA_DIR" "$EAZY_DIR" "$TABLE_DIR" "$ANALYSIS_DIR"
mkdir -p "$GEN_ASSET_DIR" "$GEN_BASELINE_DIR" "$GEN_VGA_DIR" "$GEN_VISTA_DIR" "$GEN_EAZY_DIR" "$GEN_TABLE_DIR" "$GEN_ANALYSIS_DIR"

BASELINE_JSONL="$BASELINE_DIR/pred_vanilla_discovery.jsonl"
BASELINE_FEATURES_CSV="$BASELINE_DIR/base_semantic_features.csv"
BASELINE_FEATURES_SUMMARY="$BASELINE_DIR/base_semantic_features.summary.json"
VGA_JSONL="$VGA_DIR/pred_vga_discovery.jsonl"
VISTA_JSONL="$VISTA_DIR/pred_vista_discovery.jsonl"
EAZY_JSONL="$EAZY_DIR/pred_eazy_discovery.jsonl"
CAPTION_Q_JSONL="$GEN_ASSET_DIR/discovery_caption_q.jsonl"
CAPTION_Q_SUMMARY="$GEN_ASSET_DIR/discovery_caption_q.summary.json"
GEN_BASELINE_JSONL="$GEN_BASELINE_DIR/pred_vanilla_caption.jsonl"
GEN_BASELINE_FEATURES_CSV="$GEN_BASELINE_DIR/base_semantic_features.csv"
GEN_BASELINE_FEATURES_SUMMARY="$GEN_BASELINE_DIR/base_semantic_features.summary.json"
GEN_VGA_JSONL="$GEN_VGA_DIR/pred_vga_caption.jsonl"
GEN_VISTA_JSONL="$GEN_VISTA_DIR/pred_vista_caption.jsonl"
GEN_EAZY_JSONL="$GEN_EAZY_DIR/pred_eazy_caption.jsonl"
GEN_BASELINE_CHAIR_JSON="$GEN_BASELINE_DIR/chair_baseline.json"
GEN_VGA_CHAIR_JSON="$GEN_VGA_DIR/chair_vga.json"
GEN_VISTA_CHAIR_JSON="$GEN_VISTA_DIR/chair_vista.json"
GEN_EAZY_CHAIR_JSON="$GEN_EAZY_DIR/chair_eazy.json"
GEN_CHAIR_CACHE="$GEN_ROOT/chair_cache.pkl"
GEN_BASELINE_CHAIR_INPUT="$GEN_BASELINE_DIR/chair_input.jsonl"
GEN_VGA_CHAIR_INPUT="$GEN_VGA_DIR/chair_input.jsonl"
GEN_VISTA_CHAIR_INPUT="$GEN_VISTA_DIR/chair_input.jsonl"
GEN_EAZY_CHAIR_INPUT="$GEN_EAZY_DIR/chair_input.jsonl"

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
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/prepare_chair_caption_jsonl.py \
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
    --cache "$GEN_CHAIR_CACHE" \
    --save_path "$save_path"
}

echo "[1/14] build discovery caption prompts"
if ! reuse_file "$CAPTION_Q_JSONL"; then
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/build_discovery_caption_questions.py \
    --question_file "$Q_NOOBJ" \
    --out_jsonl "$CAPTION_Q_JSONL" \
    --out_summary_json "$CAPTION_Q_SUMMARY"
else
  echo "[reuse] $CAPTION_Q_JSONL"
fi

echo "[2/14] baseline discriminative prediction"
if [[ "$RUN_BASELINE" == "1" ]] && ! reuse_file "$BASELINE_JSONL"; then
  cd "$CAL_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$Q_WITHOBJ" \
    --answers-file "$BASELINE_JSONL" \
    --conv-mode llava_v1 \
    --temperature 0 \
    --num_beams 1 \
    --max_new_tokens 8
else
  echo "[reuse] $BASELINE_JSONL"
fi

echo "[3/14] baseline discriminative semantic features"
if [[ "$RUN_BASELINE" == "1" ]] && ! reuse_file "$BASELINE_FEATURES_CSV"; then
  cd "$CAL_ROOT"
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/extract_baseline_semantic_features.py \
    --question_file "$Q_WITHOBJ" \
    --image_folder "$IMAGE_FOLDER" \
    --baseline_pred_jsonl "$BASELINE_JSONL" \
    --out_csv "$BASELINE_FEATURES_CSV" \
    --out_summary_json "$BASELINE_FEATURES_SUMMARY" \
    --model_path "$MODEL_PATH" \
    --conv_mode llava_v1 \
    --device cuda \
    --reuse_if_exists "$REUSE_IF_EXISTS"
else
  echo "[reuse] $BASELINE_FEATURES_CSV"
fi

echo "[4/14] baseline generative prediction"
if [[ "$RUN_BASELINE" == "1" ]] && ! reuse_file "$GEN_BASELINE_JSONL"; then
  cd "$CAL_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$CAPTION_Q_JSONL" \
    --answers-file "$GEN_BASELINE_JSONL" \
    --conv-mode llava_v1 \
    --temperature 0 \
    --num_beams 1 \
    --max_new_tokens 128
else
  echo "[reuse] $GEN_BASELINE_JSONL"
fi

echo "[5/14] baseline generative semantic features"
if [[ "$RUN_BASELINE" == "1" ]] && ! reuse_file "$GEN_BASELINE_FEATURES_CSV"; then
  cd "$CAL_ROOT"
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/extract_baseline_semantic_features.py \
    --question_file "$CAPTION_Q_JSONL" \
    --image_folder "$IMAGE_FOLDER" \
    --baseline_pred_jsonl "$GEN_BASELINE_JSONL" \
    --out_csv "$GEN_BASELINE_FEATURES_CSV" \
    --out_summary_json "$GEN_BASELINE_FEATURES_SUMMARY" \
    --model_path "$MODEL_PATH" \
    --conv_mode llava_v1 \
    --device cuda \
    --reuse_if_exists "$REUSE_IF_EXISTS"
else
  echo "[reuse] $GEN_BASELINE_FEATURES_CSV"
fi

echo "[6/14] VGA discriminative prediction"
if [[ "$RUN_VGA" == "1" ]] && ! reuse_file "$VGA_JSONL"; then
  if [[ ! -f "$VGA_CONDA_SH" ]]; then
    echo "[error] missing conda.sh: $VGA_CONDA_SH" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$VGA_CONDA_SH"
  conda activate "$VGA_ENV"
  cd "$VGA_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" python eval/object_hallucination_vqa_llava.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$Q_WITHOBJ" \
    --answers-file "$VGA_JSONL" \
    --conv-mode "$VGA_CONV_MODE" \
    --max_gen_len "$VGA_MAX_GEN_LEN" \
    --use_add "$VGA_USE_ADD" \
    --attn_coef "$VGA_ATTN_COEF" \
    --head_balancing "$VGA_HEAD_BALANCING" \
    --sampling "$VGA_SAMPLING" \
    --cd_alpha "$VGA_CD_ALPHA" \
    --seed "$SEED" \
    --start_layer "$VGA_START_LAYER" \
    --end_layer "$VGA_END_LAYER"
else
  echo "[reuse] $VGA_JSONL"
fi

echo "[7/14] VGA generative prediction"
if [[ "$RUN_VGA" == "1" ]] && ! reuse_file "$GEN_VGA_JSONL"; then
  if [[ ! -f "$VGA_CONDA_SH" ]]; then
    echo "[error] missing conda.sh: $VGA_CONDA_SH" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$VGA_CONDA_SH"
  conda activate "$VGA_ENV"
  cd "$VGA_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" python eval/object_hallucination_vqa_llava.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$CAPTION_Q_JSONL" \
    --answers-file "$GEN_VGA_JSONL" \
    --conv-mode "$VGA_CONV_MODE" \
    --max_gen_len 128 \
    --use_add "$VGA_USE_ADD" \
    --attn_coef "$VGA_ATTN_COEF" \
    --head_balancing "$VGA_HEAD_BALANCING" \
    --sampling "$VGA_SAMPLING" \
    --cd_alpha "$VGA_CD_ALPHA" \
    --seed "$SEED" \
    --start_layer "$VGA_START_LAYER" \
    --end_layer "$VGA_END_LAYER"
else
  echo "[reuse] $GEN_VGA_JSONL"
fi

echo "[8/14] VISTA discriminative prediction"
if [[ "$RUN_VISTA" == "1" ]] && ! reuse_file "$VISTA_JSONL"; then
  cd "$CAL_ROOT"
  PYTHONPATH="$CAL_ROOT" "$VISTA_PYTHON_BIN" "$CAL_ROOT/scripts/run_vista_question_subset.py" \
    --vista_root "$VISTA_ROOT" \
    --question_file "$Q_WITHOBJ" \
    --image_folder "$IMAGE_FOLDER" \
    --answers_file "$VISTA_JSONL" \
    --model "$VISTA_MODEL" \
    --vsv_lambda 0.01 \
    --logits_layers 25,30 \
    --logits_alpha 0.3 \
    --max_new_tokens 8 \
    --num_beams 1 \
    "${VISTA_FLAGS[@]}" \
    --seed "$SEED"
else
  echo "[reuse] $VISTA_JSONL"
fi

echo "[9/14] VISTA generative prediction"
if [[ "$RUN_VISTA" == "1" ]] && ! reuse_file "$GEN_VISTA_JSONL"; then
  cd "$CAL_ROOT"
  PYTHONPATH="$CAL_ROOT" "$VISTA_PYTHON_BIN" "$CAL_ROOT/scripts/run_vista_question_subset.py" \
    --vista_root "$VISTA_ROOT" \
    --question_file "$CAPTION_Q_JSONL" \
    --image_folder "$IMAGE_FOLDER" \
    --answers_file "$GEN_VISTA_JSONL" \
    --model "$VISTA_MODEL" \
    --vsv_lambda 0.01 \
    --logits_layers 25,30 \
    --logits_alpha 0.3 \
    --max_new_tokens 128 \
    --num_beams 1 \
    "${VISTA_FLAGS[@]}" \
    --seed "$SEED"
else
  echo "[reuse] $GEN_VISTA_JSONL"
fi

if [[ "$RUN_EAZY" == "1" ]]; then
  echo "[10/14] EAZY discriminative prediction"
  if ! reuse_file "$EAZY_JSONL"; then
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$EAZY_PYTHON_BIN" "$CAL_ROOT/scripts/run_eazy_question_subset.py" \
      --eazy_root "$EAZY_ROOT" \
      --question_file "$Q_WITHOBJ" \
      --image_folder "$IMAGE_FOLDER" \
      --answers_file "$EAZY_JSONL" \
      --model "$EAZY_MODEL" \
      --gpu_id 0 \
      --beam "$EAZY_BEAM" \
      --k "$EAZY_TOPK_K" \
      --seed "$SEED"
  else
    echo "[reuse] $EAZY_JSONL"
  fi

  echo "[11/14] EAZY generative prediction"
  if ! reuse_file "$GEN_EAZY_JSONL"; then
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$EAZY_PYTHON_BIN" "$CAL_ROOT/scripts/run_eazy_question_subset.py" \
      --eazy_root "$EAZY_ROOT" \
      --question_file "$CAPTION_Q_JSONL" \
      --image_folder "$IMAGE_FOLDER" \
      --answers_file "$GEN_EAZY_JSONL" \
      --model "$EAZY_MODEL" \
      --gpu_id 0 \
      --beam "$EAZY_BEAM" \
      --k "$EAZY_TOPK_K" \
      --seed "$SEED"
  else
    echo "[reuse] $GEN_EAZY_JSONL"
  fi
else
  echo "[10/14] EAZY prediction steps skipped (RUN_EAZY=0)"
fi

echo "[12/14] build discriminative method tables"
cd "$CAL_ROOT"
DISC_TABLES=()
if [[ "$RUN_VGA" == "1" ]]; then
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_harm_table.py \
    --baseline_features_csv "$BASELINE_FEATURES_CSV" \
    --baseline_pred_jsonl "$BASELINE_JSONL" \
    --intervention_pred_jsonl "$VGA_JSONL" \
    --gt_csv "$GT_CSV" \
    --method_name vga \
    --benchmark_name pope \
    --split_name discovery \
    --out_csv "$TABLE_DIR/vga_table.csv" \
    --out_summary_json "$TABLE_DIR/vga_table.summary.json" \
    --baseline_pred_text_key auto \
    --intervention_pred_text_key output
  DISC_TABLES+=("$TABLE_DIR/vga_table.csv")
fi

if [[ "$RUN_VISTA" == "1" ]]; then
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_harm_table.py \
    --baseline_features_csv "$BASELINE_FEATURES_CSV" \
    --baseline_pred_jsonl "$BASELINE_JSONL" \
    --intervention_pred_jsonl "$VISTA_JSONL" \
    --gt_csv "$GT_CSV" \
    --method_name vista \
    --benchmark_name pope \
    --split_name discovery \
    --out_csv "$TABLE_DIR/vista_table.csv" \
    --out_summary_json "$TABLE_DIR/vista_table.summary.json" \
    --baseline_pred_text_key auto \
    --intervention_pred_text_key output
  DISC_TABLES+=("$TABLE_DIR/vista_table.csv")
fi

if [[ "$RUN_EAZY" == "1" ]]; then
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_harm_table.py \
    --baseline_features_csv "$BASELINE_FEATURES_CSV" \
    --baseline_pred_jsonl "$BASELINE_JSONL" \
    --intervention_pred_jsonl "$EAZY_JSONL" \
    --gt_csv "$GT_CSV" \
    --method_name eazy \
    --benchmark_name pope \
    --split_name discovery \
    --out_csv "$TABLE_DIR/eazy_table.csv" \
    --out_summary_json "$TABLE_DIR/eazy_table.summary.json" \
    --baseline_pred_text_key auto \
    --intervention_pred_text_key output
  DISC_TABLES+=("$TABLE_DIR/eazy_table.csv")
fi

echo "[12.5/14] shared discriminative harm analysis"
if [[ "${#DISC_TABLES[@]}" -gt 0 ]]; then
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/analyze_common_method_harm_miner.py \
    --table_csvs "${DISC_TABLES[@]}" \
    --feature_cols "$FEATURE_COLS" \
    --out_dir "$ANALYSIS_DIR"
fi

echo "[13/14] score generative CHAIR and build method tables"
run_chair_eval "$GEN_BASELINE_JSONL" image_id text "$GEN_BASELINE_CHAIR_JSON" "$GEN_BASELINE_CHAIR_INPUT"
GEN_TABLES=()
if [[ "$RUN_VGA" == "1" ]]; then
  run_chair_eval "$GEN_VGA_JSONL" image_id output "$GEN_VGA_CHAIR_JSON" "$GEN_VGA_CHAIR_INPUT"
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_chair_table.py \
    --baseline_features_csv "$GEN_BASELINE_FEATURES_CSV" \
    --baseline_pred_jsonl "$GEN_BASELINE_JSONL" \
    --intervention_pred_jsonl "$GEN_VGA_JSONL" \
    --baseline_chair_json "$GEN_BASELINE_CHAIR_JSON" \
    --intervention_chair_json "$GEN_VGA_CHAIR_JSON" \
    --method_name vga \
    --benchmark_name pope_discovery_caption \
    --split_name discovery \
    --chair_metric CHAIRi \
    --out_csv "$GEN_TABLE_DIR/vga_table.csv" \
    --out_summary_json "$GEN_TABLE_DIR/vga_table.summary.json" \
    --baseline_pred_text_key auto \
    --intervention_pred_text_key output
  GEN_TABLES+=("$GEN_TABLE_DIR/vga_table.csv")
fi

if [[ "$RUN_VISTA" == "1" ]]; then
  run_chair_eval "$GEN_VISTA_JSONL" image_id caption "$GEN_VISTA_CHAIR_JSON" "$GEN_VISTA_CHAIR_INPUT"
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_chair_table.py \
    --baseline_features_csv "$GEN_BASELINE_FEATURES_CSV" \
    --baseline_pred_jsonl "$GEN_BASELINE_JSONL" \
    --intervention_pred_jsonl "$GEN_VISTA_JSONL" \
    --baseline_chair_json "$GEN_BASELINE_CHAIR_JSON" \
    --intervention_chair_json "$GEN_VISTA_CHAIR_JSON" \
    --method_name vista \
    --benchmark_name pope_discovery_caption \
    --split_name discovery \
    --chair_metric CHAIRi \
    --out_csv "$GEN_TABLE_DIR/vista_table.csv" \
    --out_summary_json "$GEN_TABLE_DIR/vista_table.summary.json" \
    --baseline_pred_text_key auto \
    --intervention_pred_text_key output
  GEN_TABLES+=("$GEN_TABLE_DIR/vista_table.csv")
fi

if [[ "$RUN_EAZY" == "1" ]]; then
  run_chair_eval "$GEN_EAZY_JSONL" image_id caption "$GEN_EAZY_CHAIR_JSON" "$GEN_EAZY_CHAIR_INPUT"
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_chair_table.py \
    --baseline_features_csv "$GEN_BASELINE_FEATURES_CSV" \
    --baseline_pred_jsonl "$GEN_BASELINE_JSONL" \
    --intervention_pred_jsonl "$GEN_EAZY_JSONL" \
    --baseline_chair_json "$GEN_BASELINE_CHAIR_JSON" \
    --intervention_chair_json "$GEN_EAZY_CHAIR_JSON" \
    --method_name eazy \
    --benchmark_name pope_discovery_caption \
    --split_name discovery \
    --chair_metric CHAIRi \
    --out_csv "$GEN_TABLE_DIR/eazy_table.csv" \
    --out_summary_json "$GEN_TABLE_DIR/eazy_table.summary.json" \
    --baseline_pred_text_key auto \
    --intervention_pred_text_key output
  GEN_TABLES+=("$GEN_TABLE_DIR/eazy_table.csv")
fi

echo "[14/14] shared generative harm analysis"
if [[ "${#GEN_TABLES[@]}" -gt 0 ]]; then
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/analyze_common_method_harm_miner.py \
    --table_csvs "${GEN_TABLES[@]}" \
    --feature_cols "$FEATURE_COLS" \
    --out_dir "$GEN_ANALYSIS_DIR"
fi

echo "[done] $OUT_ROOT"
