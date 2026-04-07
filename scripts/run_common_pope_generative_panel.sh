#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
PAI_ROOT="${PAI_ROOT:-/home/kms/PAI}"
VCD_ROOT="${VCD_ROOT:-/home/kms/VCD}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
PAI_PYTHON_BIN="${PAI_PYTHON_BIN:-/home/kms/miniconda3/envs/pai_base/bin/python}"
VCD_PYTHON_BIN="${VCD_PYTHON_BIN:-/home/kms/miniconda3/envs/vcd_base/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VGA_MODEL_PATH="${VGA_MODEL_PATH:-$MODEL_PATH}"
PAI_MODEL_PATH="${PAI_MODEL_PATH:-$MODEL_PATH}"
VCD_MODEL_PATH="${VCD_MODEL_PATH:-$MODEL_PATH}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
DISCOVERY_ASSET_ROOT="${DISCOVERY_ASSET_ROOT:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets}"
DISCOVERY_ASSET_FALLBACK_ROOT="${DISCOVERY_ASSET_FALLBACK_ROOT:-$CAL_ROOT/experiments/tau_c_calibration_mix_train2014_2785/assets}"
Q_WITHOBJ="${Q_WITHOBJ:-}"
CAPTION_PROMPT="${CAPTION_PROMPT:-Please describe this image in detail.}"
COCO_ANN_ROOT="${COCO_ANN_ROOT:-/home/kms/data/COCO/annotations_trainval2014/annotations}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/common_pope_generative_panel_v1}"
ASSET_DIR="$OUT_ROOT/assets"
BASELINE_DIR="$OUT_ROOT/baseline"
VGA_DIR="$OUT_ROOT/vga"
PAI_DIR="$OUT_ROOT/pai"
VCD_DIR="$OUT_ROOT/vcd"
CHAIR_TABLE_DIR="$OUT_ROOT/chair_tables"
CLAIM_TABLE_DIR="$OUT_ROOT/claim_tables"
MANIFEST_JSON="$OUT_ROOT/manifest.json"
COMPACT_JSON="$OUT_ROOT/summary_compact.json"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_VGA="${RUN_VGA:-1}"
RUN_PAI="${RUN_PAI:-1}"
RUN_VCD="${RUN_VCD:-1}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

VGA_CONV_MODE="${VGA_CONV_MODE:-llava_v1}"
VGA_MAX_GEN_LEN="${VGA_MAX_GEN_LEN:-128}"
VGA_USE_ADD="${VGA_USE_ADD:-true}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_HEAD_BALANCING="${VGA_HEAD_BALANCING:-simg}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"

PAI_MODEL="${PAI_MODEL:-llava-1.5}"
PAI_USE_ATTN="${PAI_USE_ATTN:-1}"
PAI_USE_CFG="${PAI_USE_CFG:-1}"
PAI_BEAM="${PAI_BEAM:-1}"
PAI_SAMPLE="${PAI_SAMPLE:-0}"
PAI_ALPHA="${PAI_ALPHA:-0.2}"
PAI_GAMMA="${PAI_GAMMA:-1.1}"
PAI_START_LAYER="${PAI_START_LAYER:-2}"
PAI_END_LAYER="${PAI_END_LAYER:-32}"
PAI_MAX_NEW_TOKENS="${PAI_MAX_NEW_TOKENS:-128}"

VCD_CONV_MODE="${VCD_CONV_MODE:-llava_v1}"
VCD_USE_CD="${VCD_USE_CD:-1}"
VCD_NOISE_STEP="${VCD_NOISE_STEP:-500}"
VCD_CD_ALPHA="${VCD_CD_ALPHA:-1.0}"
VCD_CD_BETA="${VCD_CD_BETA:-0.1}"
VCD_DO_SAMPLE="${VCD_DO_SAMPLE:-false}"
VCD_TEMPERATURE="${VCD_TEMPERATURE:-0}"
VCD_TOP_P="${VCD_TOP_P:-1.0}"
VCD_TOP_K="${VCD_TOP_K:-0}"
VCD_MAX_NEW_TOKENS="${VCD_MAX_NEW_TOKENS:-128}"
VCD_QUESTION_SUFFIX="${VCD_QUESTION_SUFFIX:-}"

SUPPORTED_WEIGHT="${SUPPORTED_WEIGHT:-1.0}"
HALL_WEIGHT="${HALL_WEIGHT:-1.0}"
LENGTH_WEIGHT="${LENGTH_WEIGHT:-0.25}"

SEED="${SEED:-1994}"

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
  local cand
  for cand in "${candidates[@]}"; do
    if chair_ann_ready "$cand"; then
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
      --cache "$CHAIR_CACHE" \
      --save_path "$save_path"
  )
}

if [[ ! -d "$DISCOVERY_ASSET_ROOT" && -d "$DISCOVERY_ASSET_FALLBACK_ROOT" ]]; then
  DISCOVERY_ASSET_ROOT="$DISCOVERY_ASSET_FALLBACK_ROOT"
fi
if [[ -z "$Q_WITHOBJ" ]]; then
  Q_WITHOBJ="$DISCOVERY_ASSET_ROOT/discovery_q_with_object.jsonl"
fi
if [[ ! -f "$Q_WITHOBJ" ]]; then
  echo "[error] missing object-bearing discovery question file: $Q_WITHOBJ" >&2
  exit 1
fi

COCO_ANN_ROOT="$(resolve_coco_ann_root "$COCO_ANN_ROOT" || true)"
if [[ -z "$COCO_ANN_ROOT" ]]; then
  echo "[error] could not locate COCO annotation root for CHAIR eval" >&2
  exit 1
fi

mkdir -p "$ASSET_DIR" "$BASELINE_DIR" "$VGA_DIR" "$PAI_DIR" "$VCD_DIR" "$CHAIR_TABLE_DIR" "$CLAIM_TABLE_DIR"

CAPTION_Q_JSONL="$ASSET_DIR/discovery_caption_q.jsonl"
CAPTION_Q_SUMMARY="$ASSET_DIR/discovery_caption_q.summary.json"

BASELINE_JSONL="$BASELINE_DIR/pred_vanilla_caption.jsonl"
BASELINE_FEATURES_CSV="$BASELINE_DIR/base_semantic_features.csv"
BASELINE_FEATURES_SUMMARY="$BASELINE_DIR/base_semantic_features.summary.json"
BASELINE_CHAIR_INPUT="$BASELINE_DIR/chair_input.jsonl"
BASELINE_CHAIR_JSON="$BASELINE_DIR/chair_baseline.json"

VGA_JSONL="$VGA_DIR/pred_vga_caption.jsonl"
VGA_CHAIR_INPUT="$VGA_DIR/chair_input.jsonl"
VGA_CHAIR_JSON="$VGA_DIR/chair_vga.json"

PAI_JSONL="$PAI_DIR/pred_pai_caption.jsonl"
PAI_CHAIR_INPUT="$PAI_DIR/chair_input.jsonl"
PAI_CHAIR_JSON="$PAI_DIR/chair_pai.json"

VCD_JSONL="$VCD_DIR/pred_vcd_caption.jsonl"
VCD_CHAIR_INPUT="$VCD_DIR/chair_input.jsonl"
VCD_CHAIR_JSON="$VCD_DIR/chair_vcd.json"

CHAIR_CACHE="$OUT_ROOT/chair_cache.pkl"

echo "[assets] q_withobj=$Q_WITHOBJ"
echo "[assets] image_folder=$IMAGE_FOLDER"
echo "[assets] coco_ann_root=$COCO_ANN_ROOT"

echo "[1/10] build caption prompts"
if ! reuse_file "$CAPTION_Q_JSONL"; then
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/build_discovery_caption_questions.py \
      --question_file "$Q_WITHOBJ" \
      --out_jsonl "$CAPTION_Q_JSONL" \
      --out_summary_json "$CAPTION_Q_SUMMARY" \
      --prompt "$CAPTION_PROMPT" \
      --include_objects true
  )
else
  echo "[reuse] $CAPTION_Q_JSONL"
fi

echo "[2/10] baseline generative prediction"
if [[ "$RUN_BASELINE" == "1" ]] && ! reuse_file "$BASELINE_JSONL"; then
  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" -m llava.eval.model_vqa_loader \
      --model-path "$MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$CAPTION_Q_JSONL" \
      --answers-file "$BASELINE_JSONL" \
      --conv-mode llava_v1 \
      --temperature 0 \
      --num_beams 1 \
      --max_new_tokens 128
  )
else
  echo "[reuse] $BASELINE_JSONL"
fi

echo "[3/10] baseline generative semantic features"
if [[ "$RUN_BASELINE" == "1" ]] && ! reuse_file "$BASELINE_FEATURES_CSV"; then
  (
    cd "$CAL_ROOT"
    PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/extract_baseline_semantic_features.py \
      --question_file "$CAPTION_Q_JSONL" \
      --image_folder "$IMAGE_FOLDER" \
      --baseline_pred_jsonl "$BASELINE_JSONL" \
      --out_csv "$BASELINE_FEATURES_CSV" \
      --out_summary_json "$BASELINE_FEATURES_SUMMARY" \
      --model_path "$MODEL_PATH" \
      --conv_mode llava_v1 \
      --device cuda \
      --reuse_if_exists "$REUSE_IF_EXISTS"
  )
else
  echo "[reuse] $BASELINE_FEATURES_CSV"
fi

echo "[4/10] VGA generative prediction"
if [[ "$RUN_VGA" == "1" ]] && ! reuse_file "$VGA_JSONL"; then
  (
    cd "$VGA_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" "$VGA_ROOT/eval/object_hallucination_vqa_llava.py" \
      --model-path "$VGA_MODEL_PATH" \
      --image-folder "$IMAGE_FOLDER" \
      --question-file "$CAPTION_Q_JSONL" \
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
  )
else
  echo "[reuse] $VGA_JSONL"
fi

echo "[5/10] PAI generative prediction"
PAI_FLAGS=()
if [[ "$PAI_USE_ATTN" == "1" ]]; then
  PAI_FLAGS+=(--use_attn)
fi
if [[ "$PAI_USE_CFG" == "1" ]]; then
  PAI_FLAGS+=(--use_cfg)
fi
if [[ "$PAI_SAMPLE" == "1" ]]; then
  PAI_FLAGS+=(--sample)
fi
if [[ "$RUN_PAI" == "1" ]] && ! reuse_file "$PAI_JSONL"; then
  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$PAI_PYTHON_BIN" "$CAL_ROOT/scripts/run_pai_question_subset.py" \
      --pai_root "$PAI_ROOT" \
      --question_file "$CAPTION_Q_JSONL" \
      --image_folder "$IMAGE_FOLDER" \
      --answers_file "$PAI_JSONL" \
      --model "$PAI_MODEL" \
      --model_path "$PAI_MODEL_PATH" \
      --gpu_id 0 \
      --beam "$PAI_BEAM" \
      --alpha "$PAI_ALPHA" \
      --gamma "$PAI_GAMMA" \
      --start_layer "$PAI_START_LAYER" \
      --end_layer "$PAI_END_LAYER" \
      --max_new_tokens "$PAI_MAX_NEW_TOKENS" \
      "${PAI_FLAGS[@]}" \
      --seed "$SEED"
  )
else
  echo "[reuse] $PAI_JSONL"
fi

echo "[6/10] VCD generative prediction"
VCD_FLAGS=()
if [[ "$VCD_USE_CD" == "1" ]]; then
  VCD_FLAGS+=(--use_cd)
fi
if [[ "$RUN_VCD" == "1" ]] && ! reuse_file "$VCD_JSONL"; then
  (
    cd "$CAL_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$VCD_PYTHON_BIN" "$CAL_ROOT/scripts/run_vcd_question_subset.py" \
      --vcd_root "$VCD_ROOT" \
      --question_file "$CAPTION_Q_JSONL" \
      --image_folder "$IMAGE_FOLDER" \
      --answers_file "$VCD_JSONL" \
      --model_path "$VCD_MODEL_PATH" \
      --conv_mode "$VCD_CONV_MODE" \
      --gpu_id 0 \
      --temperature "$VCD_TEMPERATURE" \
      --top_p "$VCD_TOP_P" \
      --top_k "$VCD_TOP_K" \
      --do_sample "$VCD_DO_SAMPLE" \
      --noise_step "$VCD_NOISE_STEP" \
      --cd_alpha "$VCD_CD_ALPHA" \
      --cd_beta "$VCD_CD_BETA" \
      --max_new_tokens "$VCD_MAX_NEW_TOKENS" \
      --question_suffix "$VCD_QUESTION_SUFFIX" \
      "${VCD_FLAGS[@]}" \
      --seed "$SEED"
  )
else
  echo "[reuse] $VCD_JSONL"
fi

echo "[7/10] score CHAIR"
run_chair_eval "$BASELINE_JSONL" image_id text "$BASELINE_CHAIR_JSON" "$BASELINE_CHAIR_INPUT"
if [[ "$RUN_VGA" == "1" ]]; then
  run_chair_eval "$VGA_JSONL" image_id output "$VGA_CHAIR_JSON" "$VGA_CHAIR_INPUT"
fi
if [[ "$RUN_PAI" == "1" ]]; then
  run_chair_eval "$PAI_JSONL" image_id output "$PAI_CHAIR_JSON" "$PAI_CHAIR_INPUT"
fi
if [[ "$RUN_VCD" == "1" ]]; then
  run_chair_eval "$VCD_JSONL" image_id output "$VCD_CHAIR_JSON" "$VCD_CHAIR_INPUT"
fi

echo "[8/10] build CHAIR tables"
if [[ "$RUN_VGA" == "1" ]]; then
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/build_method_chair_table.py \
      --baseline_features_csv "$BASELINE_FEATURES_CSV" \
      --baseline_pred_jsonl "$BASELINE_JSONL" \
      --intervention_pred_jsonl "$VGA_JSONL" \
      --baseline_chair_json "$BASELINE_CHAIR_JSON" \
      --intervention_chair_json "$VGA_CHAIR_JSON" \
      --method_name vga \
      --benchmark_name pope_discovery_caption \
      --split_name common_panel \
      --chair_metric CHAIRi \
      --out_csv "$CHAIR_TABLE_DIR/vga_table.csv" \
      --out_summary_json "$CHAIR_TABLE_DIR/vga_table.summary.json" \
      --baseline_pred_text_key auto \
      --intervention_pred_text_key output
  )
fi
if [[ "$RUN_PAI" == "1" ]]; then
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/build_method_chair_table.py \
      --baseline_features_csv "$BASELINE_FEATURES_CSV" \
      --baseline_pred_jsonl "$BASELINE_JSONL" \
      --intervention_pred_jsonl "$PAI_JSONL" \
      --baseline_chair_json "$BASELINE_CHAIR_JSON" \
      --intervention_chair_json "$PAI_CHAIR_JSON" \
      --method_name pai \
      --benchmark_name pope_discovery_caption \
      --split_name common_panel \
      --chair_metric CHAIRi \
      --out_csv "$CHAIR_TABLE_DIR/pai_table.csv" \
      --out_summary_json "$CHAIR_TABLE_DIR/pai_table.summary.json" \
      --baseline_pred_text_key auto \
      --intervention_pred_text_key output
  )
fi
if [[ "$RUN_VCD" == "1" ]]; then
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/build_method_chair_table.py \
      --baseline_features_csv "$BASELINE_FEATURES_CSV" \
      --baseline_pred_jsonl "$BASELINE_JSONL" \
      --intervention_pred_jsonl "$VCD_JSONL" \
      --baseline_chair_json "$BASELINE_CHAIR_JSON" \
      --intervention_chair_json "$VCD_CHAIR_JSON" \
      --method_name vcd \
      --benchmark_name pope_discovery_caption \
      --split_name common_panel \
      --chair_metric CHAIRi \
      --out_csv "$CHAIR_TABLE_DIR/vcd_table.csv" \
      --out_summary_json "$CHAIR_TABLE_DIR/vcd_table.summary.json" \
      --baseline_pred_text_key auto \
      --intervention_pred_text_key output
  )
fi

echo "[9/10] build claim-aware tables"
if [[ "$RUN_VGA" == "1" ]]; then
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/build_method_claim_aware_table.py \
      --baseline_features_csv "$BASELINE_FEATURES_CSV" \
      --baseline_pred_jsonl "$BASELINE_JSONL" \
      --intervention_pred_jsonl "$VGA_JSONL" \
      --baseline_chair_json "$BASELINE_CHAIR_JSON" \
      --intervention_chair_json "$VGA_CHAIR_JSON" \
      --method_name vga \
      --benchmark_name pope_discovery_caption \
      --split_name common_panel \
      --supported_weight "$SUPPORTED_WEIGHT" \
      --hall_weight "$HALL_WEIGHT" \
      --length_weight "$LENGTH_WEIGHT" \
      --out_csv "$CLAIM_TABLE_DIR/vga_table.csv" \
      --out_summary_json "$CLAIM_TABLE_DIR/vga_table.summary.json" \
      --baseline_pred_text_key auto \
      --intervention_pred_text_key output
  )
fi
if [[ "$RUN_PAI" == "1" ]]; then
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/build_method_claim_aware_table.py \
      --baseline_features_csv "$BASELINE_FEATURES_CSV" \
      --baseline_pred_jsonl "$BASELINE_JSONL" \
      --intervention_pred_jsonl "$PAI_JSONL" \
      --baseline_chair_json "$BASELINE_CHAIR_JSON" \
      --intervention_chair_json "$PAI_CHAIR_JSON" \
      --method_name pai \
      --benchmark_name pope_discovery_caption \
      --split_name common_panel \
      --supported_weight "$SUPPORTED_WEIGHT" \
      --hall_weight "$HALL_WEIGHT" \
      --length_weight "$LENGTH_WEIGHT" \
      --out_csv "$CLAIM_TABLE_DIR/pai_table.csv" \
      --out_summary_json "$CLAIM_TABLE_DIR/pai_table.summary.json" \
      --baseline_pred_text_key auto \
      --intervention_pred_text_key output
  )
fi
if [[ "$RUN_VCD" == "1" ]]; then
  (
    cd "$CAL_ROOT"
    "$CAL_PYTHON_BIN" scripts/build_method_claim_aware_table.py \
      --baseline_features_csv "$BASELINE_FEATURES_CSV" \
      --baseline_pred_jsonl "$BASELINE_JSONL" \
      --intervention_pred_jsonl "$VCD_JSONL" \
      --baseline_chair_json "$BASELINE_CHAIR_JSON" \
      --intervention_chair_json "$VCD_CHAIR_JSON" \
      --method_name vcd \
      --benchmark_name pope_discovery_caption \
      --split_name common_panel \
      --supported_weight "$SUPPORTED_WEIGHT" \
      --hall_weight "$HALL_WEIGHT" \
      --length_weight "$LENGTH_WEIGHT" \
      --out_csv "$CLAIM_TABLE_DIR/vcd_table.csv" \
      --out_summary_json "$CLAIM_TABLE_DIR/vcd_table.summary.json" \
      --baseline_pred_text_key auto \
      --intervention_pred_text_key output
  )
fi

echo "[10/10] collect compact summary"
(
  cd "$CAL_ROOT"
  "$CAL_PYTHON_BIN" scripts/collect_common_generative_panel_summary.py \
    --panel_root "$OUT_ROOT" \
    --methods "vga,pai,vcd" \
    --out_json "$COMPACT_JSON"
)

cat > "$MANIFEST_JSON" <<EOF
{
  "inputs": {
    "q_withobj": "$Q_WITHOBJ",
    "image_folder": "$IMAGE_FOLDER",
    "caption_prompt": "$CAPTION_PROMPT",
    "coco_ann_root": "$COCO_ANN_ROOT"
  },
  "outputs": {
    "caption_q_jsonl": "$CAPTION_Q_JSONL",
    "baseline_pred_jsonl": "$BASELINE_JSONL",
    "baseline_features_csv": "$BASELINE_FEATURES_CSV",
    "baseline_chair_json": "$BASELINE_CHAIR_JSON",
    "summary_compact_json": "$COMPACT_JSON"
  }
}
EOF

echo "[done] $OUT_ROOT"
