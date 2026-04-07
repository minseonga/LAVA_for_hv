#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"

VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
OPERA_ROOT="${OPERA_ROOT:-/home/kms/OPERA}"
PAI_ROOT="${PAI_ROOT:-/home/kms/PAI}"
VCD_ROOT="${VCD_ROOT:-/home/kms/VCD}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
OPERA_PYTHON_BIN="${OPERA_PYTHON_BIN:-/home/kms/miniconda3/envs/opera_base/bin/python}"
PAI_PYTHON_BIN="${PAI_PYTHON_BIN:-/home/kms/miniconda3/envs/pai_base/bin/python}"
VCD_PYTHON_BIN="${VCD_PYTHON_BIN:-/home/kms/miniconda3/envs/vcd_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VGA_MODEL_PATH="${VGA_MODEL_PATH:-$MODEL_PATH}"
OPERA_MODEL_PATH="${OPERA_MODEL_PATH:-$MODEL_PATH}"
PAI_MODEL_PATH="${PAI_MODEL_PATH:-$MODEL_PATH}"
VCD_MODEL_PATH="${VCD_MODEL_PATH:-$MODEL_PATH}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/images/mscoco/images/train2014}"
DISCOVERY_ASSET_ROOT="${DISCOVERY_ASSET_ROOT:-$CAL_ROOT/experiments/tau_c_calibration_mix_train2014_2785/assets}"
DISCOVERY_ASSET_FALLBACK_ROOT="${DISCOVERY_ASSET_FALLBACK_ROOT:-$CAL_ROOT/experiments/pope_discovery/tau_c_calibration_adversarial/assets}"
Q_WITHOBJ="${Q_WITHOBJ:-}"
GT_CSV="${GT_CSV:-}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/common_pope_discovery_v3_panel_v1}"
DISC_ROOT="$OUT_ROOT/discriminative"
BASELINE_DIR="$DISC_ROOT/baseline"
VGA_DIR="$DISC_ROOT/vga"
OPERA_DIR="$DISC_ROOT/opera"
PAI_DIR="$DISC_ROOT/pai"
VCD_DIR="$DISC_ROOT/vcd"
TABLE_DIR="$DISC_ROOT/tables"
ANALYSIS_DIR="$DISC_ROOT/analysis"
V3_DIR="$OUT_ROOT/discovery/unified_controller"
MANIFEST_JSON="$OUT_ROOT/manifest.json"
COMPACT_JSON="$OUT_ROOT/summary_compact.json"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_VGA="${RUN_VGA:-1}"
RUN_OPERA="${RUN_OPERA:-1}"
RUN_PAI="${RUN_PAI:-1}"
RUN_VCD="${RUN_VCD:-1}"
REUSE_IF_EXISTS="${REUSE_IF_EXISTS:-true}"

VGA_CONV_MODE="${VGA_CONV_MODE:-llava_v1}"
VGA_MAX_GEN_LEN="${VGA_MAX_GEN_LEN:-8}"
VGA_USE_ADD="${VGA_USE_ADD:-true}"
VGA_ATTN_COEF="${VGA_ATTN_COEF:-0.2}"
VGA_HEAD_BALANCING="${VGA_HEAD_BALANCING:-simg}"
VGA_SAMPLING="${VGA_SAMPLING:-false}"
VGA_CD_ALPHA="${VGA_CD_ALPHA:-0.02}"
VGA_START_LAYER="${VGA_START_LAYER:-2}"
VGA_END_LAYER="${VGA_END_LAYER:-15}"

OPERA_MODEL="${OPERA_MODEL:-llava-1.5}"
OPERA_BEAM="${OPERA_BEAM:-5}"
OPERA_SCALE_FACTOR="${OPERA_SCALE_FACTOR:-50}"
OPERA_THRESHOLD="${OPERA_THRESHOLD:-15}"
OPERA_NUM_ATTN_CANDIDATES="${OPERA_NUM_ATTN_CANDIDATES:-5}"
OPERA_PENALTY_WEIGHTS="${OPERA_PENALTY_WEIGHTS:-1.0}"
OPERA_MAX_NEW_TOKENS="${OPERA_MAX_NEW_TOKENS:-8}"

PAI_MODEL="${PAI_MODEL:-llava-1.5}"
PAI_USE_ATTN="${PAI_USE_ATTN:-1}"
PAI_USE_CFG="${PAI_USE_CFG:-1}"
PAI_BEAM="${PAI_BEAM:-1}"
PAI_SAMPLE="${PAI_SAMPLE:-0}"
PAI_ALPHA="${PAI_ALPHA:-0.2}"
PAI_GAMMA="${PAI_GAMMA:-1.1}"
PAI_START_LAYER="${PAI_START_LAYER:-2}"
PAI_END_LAYER="${PAI_END_LAYER:-32}"
PAI_MAX_NEW_TOKENS="${PAI_MAX_NEW_TOKENS:-8}"

VCD_CONV_MODE="${VCD_CONV_MODE:-llava_v1}"
VCD_USE_CD="${VCD_USE_CD:-1}"
VCD_NOISE_STEP="${VCD_NOISE_STEP:-500}"
VCD_CD_ALPHA="${VCD_CD_ALPHA:-1.0}"
VCD_CD_BETA="${VCD_CD_BETA:-0.1}"
VCD_DO_SAMPLE="${VCD_DO_SAMPLE:-true}"
VCD_TEMPERATURE="${VCD_TEMPERATURE:-1.0}"
VCD_TOP_P="${VCD_TOP_P:-1.0}"
VCD_TOP_K="${VCD_TOP_K:-0}"
VCD_MAX_NEW_TOKENS="${VCD_MAX_NEW_TOKENS:-8}"

SEED="${SEED:-1994}"

HELP_FEATURE_COLS="${HELP_FEATURE_COLS:-base_lp_content_mean,base_target_argmax_content_mean,base_target_gap_content_min,base_entropy_content_mean,base_conflict_lp_minus_entropy}"
HARM_FEATURE_COLS="${HARM_FEATURE_COLS:-base_lp_content_mean,base_target_argmax_content_mean,base_target_gap_content_min,base_entropy_content_mean,base_conflict_lp_minus_entropy}"
FEATURE_COLS="${FEATURE_COLS:-$HARM_FEATURE_COLS}"
MIN_FEATURE_AUROC="${MIN_FEATURE_AUROC:-0.55}"
TOP_K_HELP="${TOP_K_HELP:-3}"
TOP_K_HARM="${TOP_K_HARM:-3}"
LAMBDA_VALUES="${LAMBDA_VALUES:-0.5,1.0,1.5,2.0}"
TAU_QUANTILES="${TAU_QUANTILES:-0.01,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99}"
TAU_OBJECTIVE="${TAU_OBJECTIVE:-balanced_utility}"
MIN_METHOD_RATE="${MIN_METHOD_RATE:-0.0}"
MAX_METHOD_RATE="${MAX_METHOD_RATE:-1.0}"
MIN_SELECTED_COUNT="${MIN_SELECTED_COUNT:-0}"

if [[ ! -d "$DISCOVERY_ASSET_ROOT" && -d "$DISCOVERY_ASSET_FALLBACK_ROOT" ]]; then
  DISCOVERY_ASSET_ROOT="$DISCOVERY_ASSET_FALLBACK_ROOT"
fi

if [[ -z "$Q_WITHOBJ" ]]; then
  if [[ -f "$DISCOVERY_ASSET_ROOT/discovery_q_with_object.jsonl" ]]; then
    Q_WITHOBJ="$DISCOVERY_ASSET_ROOT/discovery_q_with_object.jsonl"
  else
    Q_WITHOBJ="$DISCOVERY_ASSET_ROOT/discovery_q.jsonl"
  fi
fi
if [[ -z "$GT_CSV" ]]; then
  GT_CSV="$DISCOVERY_ASSET_ROOT/discovery_gt.csv"
fi

if [[ ! -f "$Q_WITHOBJ" ]]; then
  echo "[error] missing discovery question file: $Q_WITHOBJ" >&2
  exit 1
fi
if [[ ! -f "$GT_CSV" ]]; then
  echo "[error] missing discovery gt csv: $GT_CSV" >&2
  exit 1
fi

mkdir -p "$BASELINE_DIR" "$VGA_DIR" "$OPERA_DIR" "$PAI_DIR" "$VCD_DIR" "$TABLE_DIR" "$ANALYSIS_DIR" "$V3_DIR"

BASELINE_JSONL="$BASELINE_DIR/pred_vanilla_discovery.jsonl"
BASELINE_FEATURES_CSV="$BASELINE_DIR/base_semantic_features.csv"
BASELINE_FEATURES_SUMMARY="$BASELINE_DIR/base_semantic_features.summary.json"
VGA_JSONL="$VGA_DIR/pred_vga_discovery.jsonl"
OPERA_JSONL="$OPERA_DIR/pred_opera_discovery.jsonl"
PAI_JSONL="$PAI_DIR/pred_pai_discovery.jsonl"
VCD_JSONL="$VCD_DIR/pred_vcd_discovery.jsonl"

reuse_file() {
  local path="$1"
  if [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]; then
    return 0
  fi
  return 1
}

echo "[assets] q_withobj=$Q_WITHOBJ"
echo "[assets] gt_csv=$GT_CSV"
echo "[assets] image_folder=$IMAGE_FOLDER"

echo "[1/10] baseline discriminative prediction"
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

echo "[2/10] baseline semantic features"
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

echo "[3/10] VGA discriminative prediction"
if [[ "$RUN_VGA" == "1" ]] && ! reuse_file "$VGA_JSONL"; then
  cd "$VGA_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" "$VGA_ROOT/eval/object_hallucination_vqa_llava.py" \
    --model-path "$VGA_MODEL_PATH" \
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

echo "[4/10] OPERA discriminative prediction"
if [[ "$RUN_OPERA" == "1" ]] && ! reuse_file "$OPERA_JSONL"; then
  cd "$CAL_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$OPERA_PYTHON_BIN" "$CAL_ROOT/scripts/run_opera_question_subset.py" \
    --opera_root "$OPERA_ROOT" \
    --question_file "$Q_WITHOBJ" \
    --image_folder "$IMAGE_FOLDER" \
    --answers_file "$OPERA_JSONL" \
    --model "$OPERA_MODEL" \
    --model_path "$OPERA_MODEL_PATH" \
    --gpu_id 0 \
    --beam "$OPERA_BEAM" \
    --scale_factor "$OPERA_SCALE_FACTOR" \
    --threshold "$OPERA_THRESHOLD" \
    --num_attn_candidates "$OPERA_NUM_ATTN_CANDIDATES" \
    --penalty_weights "$OPERA_PENALTY_WEIGHTS" \
    --max_new_tokens "$OPERA_MAX_NEW_TOKENS" \
    --seed "$SEED"
else
  echo "[reuse] $OPERA_JSONL"
fi

echo "[5/10] PAI discriminative prediction"
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
  cd "$CAL_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$PAI_PYTHON_BIN" "$CAL_ROOT/scripts/run_pai_question_subset.py" \
    --pai_root "$PAI_ROOT" \
    --question_file "$Q_WITHOBJ" \
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
else
  echo "[reuse] $PAI_JSONL"
fi

echo "[6/10] VCD discriminative prediction"
VCD_FLAGS=()
if [[ "$VCD_USE_CD" == "1" ]]; then
  VCD_FLAGS+=(--use_cd)
fi
if [[ "$RUN_VCD" == "1" ]] && ! reuse_file "$VCD_JSONL"; then
  cd "$CAL_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$VCD_PYTHON_BIN" "$CAL_ROOT/scripts/run_vcd_question_subset.py" \
    --vcd_root "$VCD_ROOT" \
    --question_file "$Q_WITHOBJ" \
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
    "${VCD_FLAGS[@]}" \
    --seed "$SEED"
else
  echo "[reuse] $VCD_JSONL"
fi

echo "[7/10] build method harm tables"
cd "$CAL_ROOT"
TABLES=()

build_table() {
  local method="$1"
  local pred_jsonl="$2"
  local out_csv="$TABLE_DIR/${method}_table.csv"
  local out_summary="$TABLE_DIR/${method}_table.summary.json"
  PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_method_harm_table.py \
    --baseline_features_csv "$BASELINE_FEATURES_CSV" \
    --baseline_pred_jsonl "$BASELINE_JSONL" \
    --intervention_pred_jsonl "$pred_jsonl" \
    --gt_csv "$GT_CSV" \
    --method_name "$method" \
    --benchmark_name pope \
    --split_name discovery \
    --out_csv "$out_csv" \
    --out_summary_json "$out_summary" \
    --baseline_pred_text_key auto \
    --intervention_pred_text_key auto
  TABLES+=("$out_csv")
}

if [[ "$RUN_VGA" == "1" ]]; then
  build_table vga "$VGA_JSONL"
fi
if [[ "$RUN_OPERA" == "1" ]]; then
  build_table opera "$OPERA_JSONL"
fi
if [[ "$RUN_PAI" == "1" ]]; then
  build_table pai "$PAI_JSONL"
fi
if [[ "$RUN_VCD" == "1" ]]; then
  build_table vcd "$VCD_JSONL"
fi

if [[ "${#TABLES[@]}" -eq 0 ]]; then
  echo "[error] no method tables built" >&2
  exit 1
fi

echo "[8/10] shared common-harm analysis"
PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/analyze_common_method_harm_miner.py \
  --table_csvs "${TABLES[@]}" \
  --feature_cols "$FEATURE_COLS" \
  --out_dir "$ANALYSIS_DIR"

echo "[9/10] pooled v3 discovery calibration"
PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/build_vga_pregate_v3_controller.py \
  --discovery_table_csvs "${TABLES[@]}" \
  --out_dir "$V3_DIR" \
  --source_key method \
  --help_feature_cols "$HELP_FEATURE_COLS" \
  --harm_feature_cols "$HARM_FEATURE_COLS" \
  --min_feature_auroc "$MIN_FEATURE_AUROC" \
  --top_k_help "$TOP_K_HELP" \
  --top_k_harm "$TOP_K_HARM" \
  --lambda_values "$LAMBDA_VALUES" \
  --tau_quantiles "$TAU_QUANTILES" \
  --tau_objective "$TAU_OBJECTIVE" \
  --min_method_rate "$MIN_METHOD_RATE" \
  --max_method_rate "$MAX_METHOD_RATE" \
  --min_selected_count "$MIN_SELECTED_COUNT"

echo "[10/10] manifest + compact summary"
SRC_ROOT_ENV="$OUT_ROOT" \
Q_WITHOBJ_ENV="$Q_WITHOBJ" \
GT_CSV_ENV="$GT_CSV" \
IMAGE_FOLDER_ENV="$IMAGE_FOLDER" \
TABLES_ENV="$(IFS=:; echo "${TABLES[*]}")" \
MANIFEST_JSON_ENV="$MANIFEST_JSON" \
"$CAL_PYTHON_BIN" - <<'PY'
import json
import os

manifest = {
    "out_root": os.environ["SRC_ROOT_ENV"],
    "question_file": os.environ["Q_WITHOBJ_ENV"],
    "gt_csv": os.environ["GT_CSV_ENV"],
    "image_folder": os.environ["IMAGE_FOLDER_ENV"],
    "table_csvs": [x for x in os.environ["TABLES_ENV"].split(":") if x],
}
with open(os.environ["MANIFEST_JSON_ENV"], "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
PY
PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" scripts/collect_vga_pregate_v3_compact.py \
  --root "$OUT_ROOT" \
  --out_json "$COMPACT_JSON"
echo "[saved] $MANIFEST_JSON"
echo "[done] $OUT_ROOT"
