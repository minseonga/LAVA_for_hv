#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"
VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
VISTA_ROOT="${VISTA_ROOT:-/home/kms/VISTA}"
EAZY_ROOT="${EAZY_ROOT:-/home/kms/EAZY_origin}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/vocot/bin/python}"
VISTA_PYTHON_BIN="${VISTA_PYTHON_BIN:-/home/kms/miniconda3/envs/vista/bin/python}"
EAZY_PYTHON_BIN="${EAZY_PYTHON_BIN:-/home/kms/miniconda3/envs/eazy/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
Q_NOOBJ="${Q_NOOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q.jsonl}"
Q_WITHOBJ="${Q_WITHOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"

LEGACY_FEATURES_CSV="${LEGACY_FEATURES_CSV:-$CAL_ROOT/experiments/pope_feature_screen_v1_e1_4_l16_24/features/features_unified_table.csv}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/common_pope_harm_miner_v1}"
BASELINE_DIR="$OUT_ROOT/baseline"
VGA_DIR="$OUT_ROOT/vga"
VISTA_DIR="$OUT_ROOT/vista"
EAZY_DIR="$OUT_ROOT/eazy"
TABLE_DIR="$OUT_ROOT/tables"
ANALYSIS_DIR="$OUT_ROOT/analysis"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_VGA="${RUN_VGA:-1}"
RUN_VISTA="${RUN_VISTA:-1}"
RUN_EAZY="${RUN_EAZY:-1}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

VGA_CONDA_SH="${VGA_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
VGA_ENV="${VGA_ENV:-vga}"
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

VISTA_EXP_FOLDER="${VISTA_EXP_FOLDER:-pope_9000_vista_eval}"
VISTA_MODEL="${VISTA_MODEL:-llava-1.5}"
VISTA_ENABLE_VSV="${VISTA_ENABLE_VSV:-1}"
VISTA_ENABLE_LOGITS_AUG="${VISTA_ENABLE_LOGITS_AUG:-1}"

EAZY_MODEL="${EAZY_MODEL:-llava-1.5}"
EAZY_BEAM="${EAZY_BEAM:-1}"
EAZY_TOPK_K="${EAZY_TOPK_K:-2}"
EAZY_REQUIRE_CHAIR="${EAZY_REQUIRE_CHAIR:-1}"

FEATURE_COLS="${FEATURE_COLS:-base_lp_content_mean,base_target_argmax_content_mean,base_target_gap_content_min,base_entropy_content_mean,base_conflict_lp_minus_entropy}"

mkdir -p "$BASELINE_DIR" "$VGA_DIR" "$VISTA_DIR" "$EAZY_DIR" "$TABLE_DIR" "$ANALYSIS_DIR"

BASELINE_JSONL="$BASELINE_DIR/pred_vanilla_9000.jsonl"
BASELINE_FEATURES_CSV="$BASELINE_DIR/base_semantic_features.csv"
BASELINE_FEATURES_SUMMARY="$BASELINE_DIR/base_semantic_features.summary.json"
VGA_JSONL="$VGA_DIR/pred_vga_9000.jsonl"
VISTA_JSONL="$VISTA_DIR/pred_vista_9000.jsonl"
EAZY_JSONL="$EAZY_DIR/pred_eazy_9000.jsonl"

reuse_file() {
  local path="$1"
  if [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]; then
    return 0
  fi
  return 1
}

echo "[1/7] baseline prediction"
if [[ "$RUN_BASELINE" == "1" ]] && ! reuse_file "$BASELINE_JSONL"; then
  cd "$CAL_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$Q_NOOBJ" \
    --answers-file "$BASELINE_JSONL" \
    --conv-mode llava_v1 \
    --temperature 0 \
    --num_beams 1 \
    --max_new_tokens 8
else
  echo "[reuse] $BASELINE_JSONL"
fi

echo "[2/7] baseline semantic features"
if [[ "$RUN_BASELINE" == "1" ]] && ! reuse_file "$BASELINE_FEATURES_CSV"; then
  cd "$CAL_ROOT"
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/extract_baseline_semantic_features.py \
    --question_file "$Q_NOOBJ" \
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

echo "[3/7] VGA prediction"
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

echo "[4/7] VISTA prediction"
if [[ "$RUN_VISTA" == "1" ]] && ! reuse_file "$VISTA_JSONL"; then
  cd "$CAL_ROOT"
  GPU="$GPU" \
  CAL_ROOT="$CAL_ROOT" \
  VISTA_ROOT="$VISTA_ROOT" \
  VISTA_PYTHON_BIN="$VISTA_PYTHON_BIN" \
  CAL_PYTHON_BIN="$CAL_PYTHON_BIN" \
  DATA_PATH="$IMAGE_FOLDER" \
  LLAVA_MODEL_PATH="$MODEL_PATH" \
  GT_CSV="$GT_CSV" \
  BASELINE_JSONL="$BASELINE_JSONL" \
  FEATURES_CSV="$LEGACY_FEATURES_CSV" \
  OUT_DIR="$VISTA_DIR" \
  TAX_OUT_DIR="$VISTA_DIR/taxonomy" \
  D1D2_OUT_DIR="$VISTA_DIR/d1d2_audit" \
  EXP_FOLDER="$VISTA_EXP_FOLDER" \
  MODEL="$VISTA_MODEL" \
  SEED="$SEED" \
  ADV_SUBSET_SIZE=-1 \
  POP_SUBSET_SIZE=-1 \
  RND_SUBSET_SIZE=-1 \
  CATEGORY_SIZE=3000 \
  VISTA_ENABLE_VSV="$VISTA_ENABLE_VSV" \
  VISTA_ENABLE_LOGITS_AUG="$VISTA_ENABLE_LOGITS_AUG" \
  bash "$CAL_ROOT/scripts/run_vista_vs_baseline_taxonomy_9000.sh"
else
  echo "[reuse] $VISTA_JSONL"
fi

echo "[5/7] EAZY prediction"
if [[ "$RUN_EAZY" == "1" ]] && ! reuse_file "$EAZY_JSONL"; then
  cd "$CAL_ROOT"
  GPU="$GPU" \
  CAL_ROOT="$CAL_ROOT" \
  EAZY_ROOT="$EAZY_ROOT" \
  EAZY_PYTHON_BIN="$EAZY_PYTHON_BIN" \
  CAL_PYTHON_BIN="$CAL_PYTHON_BIN" \
  DATA_PATH="$IMAGE_FOLDER" \
  GT_CSV="$GT_CSV" \
  BASELINE_JSONL="$BASELINE_JSONL" \
  FEATURES_CSV="$LEGACY_FEATURES_CSV" \
  OUT_DIR="$EAZY_DIR" \
  TAX_OUT_DIR="$EAZY_DIR/taxonomy" \
  D1D2_OUT_DIR="$EAZY_DIR/d1d2_audit" \
  MODEL="$EAZY_MODEL" \
  BEAM="$EAZY_BEAM" \
  TOPK_K="$EAZY_TOPK_K" \
  EAZY_REQUIRE_CHAIR="$EAZY_REQUIRE_CHAIR" \
  CATEGORY_SIZE=3000 \
  bash "$CAL_ROOT/scripts/run_eazy_vs_baseline_taxonomy_9000.sh"
else
  echo "[reuse] $EAZY_JSONL"
fi

echo "[6/7] method tables"
cd "$CAL_ROOT"
PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_harm_table.py \
  --baseline_features_csv "$BASELINE_FEATURES_CSV" \
  --baseline_pred_jsonl "$BASELINE_JSONL" \
  --intervention_pred_jsonl "$VGA_JSONL" \
  --gt_csv "$GT_CSV" \
  --method_name vga \
  --benchmark_name pope \
  --split_name full \
  --out_csv "$TABLE_DIR/vga_table.csv" \
  --out_summary_json "$TABLE_DIR/vga_table.summary.json" \
  --baseline_pred_text_key auto \
  --intervention_pred_text_key output

PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_harm_table.py \
  --baseline_features_csv "$BASELINE_FEATURES_CSV" \
  --baseline_pred_jsonl "$BASELINE_JSONL" \
  --intervention_pred_jsonl "$VISTA_JSONL" \
  --gt_csv "$GT_CSV" \
  --method_name vista \
  --benchmark_name pope \
  --split_name full \
  --out_csv "$TABLE_DIR/vista_table.csv" \
  --out_summary_json "$TABLE_DIR/vista_table.summary.json" \
  --baseline_pred_text_key auto \
  --intervention_pred_text_key output

PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_harm_table.py \
  --baseline_features_csv "$BASELINE_FEATURES_CSV" \
  --baseline_pred_jsonl "$BASELINE_JSONL" \
  --intervention_pred_jsonl "$EAZY_JSONL" \
  --gt_csv "$GT_CSV" \
  --method_name eazy \
  --benchmark_name pope \
  --split_name full \
  --out_csv "$TABLE_DIR/eazy_table.csv" \
  --out_summary_json "$TABLE_DIR/eazy_table.summary.json" \
  --baseline_pred_text_key auto \
  --intervention_pred_text_key output

echo "[7/7] shared harm analysis"
PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/analyze_common_method_harm_miner.py \
  --table_csvs "$TABLE_DIR/vga_table.csv" "$TABLE_DIR/vista_table.csv" "$TABLE_DIR/eazy_table.csv" \
  --feature_cols "$FEATURE_COLS" \
  --out_dir "$ANALYSIS_DIR"

echo "[done] $OUT_ROOT"
