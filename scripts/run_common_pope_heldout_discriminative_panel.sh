#!/usr/bin/env bash
set -euo pipefail

CAL_ROOT="${CAL_ROOT:-/home/kms/LLaVA_calibration}"

VGA_ROOT="${VGA_ROOT:-/home/kms/VGA_origin}"
PAI_ROOT="${PAI_ROOT:-/home/kms/PAI}"
VCD_ROOT="${VCD_ROOT:-/home/kms/VCD}"

CAL_PYTHON_BIN="${CAL_PYTHON_BIN:-/home/kms/miniconda3/envs/model_base/bin/python}"
VGA_PYTHON_BIN="${VGA_PYTHON_BIN:-/home/kms/miniconda3/envs/vga_base/bin/python}"
PAI_PYTHON_BIN="${PAI_PYTHON_BIN:-/home/kms/miniconda3/envs/pai_base/bin/python}"
VCD_PYTHON_BIN="${VCD_PYTHON_BIN:-/home/kms/miniconda3/envs/vcd_base/bin/python}"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-6}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
VGA_MODEL_PATH="${VGA_MODEL_PATH:-$MODEL_PATH}"
PAI_MODEL_PATH="${PAI_MODEL_PATH:-$MODEL_PATH}"
VCD_MODEL_PATH="${VCD_MODEL_PATH:-$MODEL_PATH}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
Q_NOOBJ="${Q_NOOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q.jsonl}"
Q_WITHOBJ="${Q_WITHOBJ:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_q_with_object.jsonl}"
GT_CSV="${GT_CSV:-$CAL_ROOT/experiments/pope_full_9000/pope_9000_gt.csv}"

OUT_ROOT="${OUT_ROOT:-$CAL_ROOT/experiments/pope_full_9000/all_models_full_strict}"
BASELINE_DIR="$OUT_ROOT/baseline"
VGA_DIR="$OUT_ROOT/vga"
PAI_DIR="$OUT_ROOT/pai"
VCD_DIR="$OUT_ROOT/vcd"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_VGA="${RUN_VGA:-1}"
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
SEED="${SEED:-1994}"

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
VCD_DO_SAMPLE="${VCD_DO_SAMPLE:-false}"
VCD_TEMPERATURE="${VCD_TEMPERATURE:-0}"
VCD_TOP_P="${VCD_TOP_P:-1.0}"
VCD_TOP_K="${VCD_TOP_K:-0}"
VCD_MAX_NEW_TOKENS="${VCD_MAX_NEW_TOKENS:-8}"

BASELINE_JSONL="$BASELINE_DIR/pred_vanilla_9000.jsonl"
BASELINE_METRICS_JSON="$BASELINE_DIR/metrics_vanilla_9000.json"
VGA_JSONL="$VGA_DIR/pred_vga_9000.jsonl"
VGA_METRICS_JSON="$VGA_DIR/metrics_vga_9000.json"
PAI_JSONL="$PAI_DIR/pred_pai_9000.jsonl"
PAI_METRICS_JSON="$PAI_DIR/metrics_pai_9000.json"
VCD_JSONL="$VCD_DIR/pred_vcd_9000.jsonl"
VCD_METRICS_JSON="$VCD_DIR/metrics_vcd_9000.json"

mkdir -p "$BASELINE_DIR" "$VGA_DIR" "$PAI_DIR" "$VCD_DIR" "$LOG_DIR"

for f in "$Q_NOOBJ" "$Q_WITHOBJ" "$GT_CSV"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] missing file: $f" >&2
    exit 1
  fi
done

reuse_file() {
  local path="$1"
  if [[ "$REUSE_IF_EXISTS" == "true" && -f "$path" ]]; then
    return 0
  fi
  return 1
}

run_step() {
  local name="$1"
  local logfile="$2"
  shift 2
  echo "[log] $logfile"
  "$@" 2>&1 | tee "$logfile"
}

echo "[assets] q_noobj=$Q_NOOBJ"
echo "[assets] q_withobj=$Q_WITHOBJ"
echo "[assets] gt_csv=$GT_CSV"
echo "[assets] image_folder=$IMAGE_FOLDER"

echo "[1/8] baseline held-out prediction"
if [[ "$RUN_BASELINE" == "1" ]] && ! reuse_file "$BASELINE_JSONL"; then
  run_step "baseline_predict" "$LOG_DIR/01_baseline_predict.log" \
    env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" -m llava.eval.model_vqa_loader \
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

echo "[2/8] baseline held-out metrics"
run_step "baseline_metrics" "$LOG_DIR/02_baseline_metrics.log" \
  env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/eval_pope_subset_yesno.py" \
    --gt_csv "$GT_CSV" \
    --pred_jsonl "$BASELINE_JSONL" \
    --pred_text_key text \
    --out_json "$BASELINE_METRICS_JSON"

echo "[3/8] VGA held-out prediction"
if [[ "$RUN_VGA" == "1" ]] && ! reuse_file "$VGA_JSONL"; then
  run_step "vga_predict" "$LOG_DIR/03_vga_predict.log" \
    env CUDA_VISIBLE_DEVICES="$GPU" "$VGA_PYTHON_BIN" "$VGA_ROOT/eval/object_hallucination_vqa_llava.py" \
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

echo "[4/8] VGA held-out metrics"
run_step "vga_metrics" "$LOG_DIR/04_vga_metrics.log" \
  env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/eval_pope_subset_yesno.py" \
    --gt_csv "$GT_CSV" \
    --pred_jsonl "$VGA_JSONL" \
    --pred_text_key output \
    --out_json "$VGA_METRICS_JSON"

echo "[5/8] PAI held-out prediction"
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
  run_step "pai_predict" "$LOG_DIR/05_pai_predict.log" \
    env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$PAI_PYTHON_BIN" "$CAL_ROOT/scripts/run_pai_question_subset.py" \
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

echo "[6/8] PAI held-out metrics"
run_step "pai_metrics" "$LOG_DIR/06_pai_metrics.log" \
  env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/eval_pope_subset_yesno.py" \
    --gt_csv "$GT_CSV" \
    --pred_jsonl "$PAI_JSONL" \
    --pred_text_key text \
    --out_json "$PAI_METRICS_JSON"

echo "[7/8] VCD held-out prediction"
VCD_FLAGS=()
if [[ "$VCD_USE_CD" == "1" ]]; then
  VCD_FLAGS+=(--use_cd)
fi
if [[ "$RUN_VCD" == "1" ]] && ! reuse_file "$VCD_JSONL"; then
  run_step "vcd_predict" "$LOG_DIR/07_vcd_predict.log" \
    env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CAL_ROOT" "$VCD_PYTHON_BIN" "$CAL_ROOT/scripts/run_vcd_question_subset.py" \
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
      "${VCD_FLAGS[@]}" \
      --cd_alpha "$VCD_CD_ALPHA" \
      --cd_beta "$VCD_CD_BETA" \
      --max_new_tokens "$VCD_MAX_NEW_TOKENS" \
      --seed "$SEED"
else
  echo "[reuse] $VCD_JSONL"
fi

echo "[8/8] VCD held-out metrics"
run_step "vcd_metrics" "$LOG_DIR/08_vcd_metrics.log" \
  env PYTHONPATH="$CAL_ROOT" "$CAL_PYTHON_BIN" "$CAL_ROOT/scripts/eval_pope_subset_yesno.py" \
    --gt_csv "$GT_CSV" \
    --pred_jsonl "$VCD_JSONL" \
    --pred_text_key text \
    --out_json "$VCD_METRICS_JSON"

echo "[done] baseline -> $BASELINE_JSONL"
echo "[done] vga -> $VGA_JSONL"
echo "[done] pai -> $PAI_JSONL"
echo "[done] vcd -> $VCD_JSONL"
