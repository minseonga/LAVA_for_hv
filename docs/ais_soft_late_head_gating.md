# AIS-aware Soft Late-Head Gating (Inference-time)

This repository includes a training-free attention-logit intervention for LLaVA/LLaMA generation:

- Early support map from early layers
- Late attraction from late layers
- Cheap AIS proxy from attention probabilities
- Soft penalty on **image-token columns only** in late layers

## Main idea (cheap-AIS)

For each decode step:

1. Compute early image support `R_t(p)` from early layers.
2. Compute late image attraction `L_{l,t}(p)` for each late layer.
3. Compute patch-wise score  
   `a_{l,t}(p) = log((L_{l,t}(p)+eps)/(R_t(p)+eps))`.
4. Trigger layer gating if top-k mean AIS exceeds `tau`.
5. Apply penalty only on image columns:
   `z_tilde(image) = z(image) - gamma * omega * relu(a)`.

Text columns are untouched.

## CLI (model_vqa_loader)

`llava/eval/model_vqa_loader.py` now supports:

- `--enable-ais-gating`
- `--ais-early-start`
- `--ais-early-end`
- `--ais-late-start`
- `--ais-late-end`
- `--ais-topk` (default `8`)
- `--ais-tau`
- `--ais-gamma`
- `--ais-eps` (default `1e-6`)
- `--ais-debug-log`
- `--ais-debug-dump <path.csv>`
- `--ais-arm {legacy,harmful_only,faithful_only,bipolar}`
- `--ais-harmful-heads "layer:head,..."`
- `--ais-faithful-heads "layer:head,..."`
- `--ais-headset-json <path.json>`
- `--ais-faithful-boost` (default `1.0`)
- `--ais-use-dynamic-omega` / `--no-ais-use-dynamic-omega`
- `--ais-use-budget-routing`
- `--ais-budget-total` (total per-step intervention mass across late layers)
- `--ais-harmful-top-ratio`
- `--ais-faithful-top-ratio`
- `--ais-bipolar-harmful-ratio`

Example:

```bash
python -m llava.eval.model_vqa_loader \
  --model-path liuhaotian/llava-v1.5-7b \
  --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
  --image-folder /path/to/images \
  --answers-file /tmp/pope_ais_answers.jsonl \
  --conv-mode llava_v1 \
  --temperature 0 \
  --num_beams 1 \
  --max_new_tokens 16 \
  --enable-ais-gating \
  --ais-early-start 0 \
  --ais-early-end 15 \
  --ais-late-start 16 \
  --ais-late-end 31 \
  --ais-topk 8 \
  --ais-tau 2.2 \
  --ais-gamma 0.2 \
  --ais-debug-log \
  --ais-debug-dump /tmp/ais_debug.csv
```

## Safety defaults

- Disabled (`--enable-ais-gating` absent): baseline path unchanged.
- `gamma=0`: baseline-equivalent behavior.
- Only late layers are modified; early layers are read-only.
- Only image columns are penalized; text columns are unchanged.
- Current CLI/model defaults are conservative (`tau=2.2`, `gamma=0.2`) to prevent always-on gating.

## POPE-1000 quick grid run

```bash
cd /home/kms/LLaVA_calibration
PYTHONPATH=. /home/kms/miniconda3/envs/vocot/bin/python scripts/run_pope_ais_grid.py \
  --model_path liuhaotian/llava-v1.5-7b \
  --image_folder /home/kms/data/pope/val2014 \
  --question_file /tmp/pope_1000_q.jsonl \
  --gt_csv /home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv \
  --out_dir /home/kms/LLaVA_calibration/experiments/pope_ais_grid_1000 \
  --grid 2.0:0.1,2.2:0.2,2.4:0.2 \
  --debug_dump
```

Single-result evaluation helper:

```bash
python scripts/eval_pope_subset_yesno.py \
  --gt_csv /home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv \
  --pred_jsonl /tmp/pope_1000_baseline.jsonl
```

Quantile-driven sweep (recommended):

```bash
PYTHONPATH=. /home/kms/miniconda3/envs/vocot/bin/python scripts/run_pope_ais_grid.py \
  --model_path liuhaotian/llava-v1.5-7b \
  --image_folder /home/kms/data/pope/val2014 \
  --question_file /tmp/pope_1000_q.jsonl \
  --gt_csv /home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv \
  --out_dir /home/kms/LLaVA_calibration/experiments/pope_ais_grid_quantile_1000 \
  --grid_from_baseline_quantiles \
  --baseline_debug_csv /tmp/pope_1000_ais_debug.csv \
  --tau_quantiles 0.90,0.95,0.97,0.99 \
  --gamma_list 0.05,0.1,0.2,0.4 \
  --debug_dump
```

## Head-set Ablation (baseline vs 3 arms)

1) Build faithful/harmful head sets JSON from prior analysis:

```bash
cd /home/kms/LLaVA_calibration
PYTHONPATH=. /home/kms/miniconda3/envs/vocot/bin/python scripts/make_head_sets_v1.py \
  --out_json /home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json
```

2) Run ablation:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 /home/kms/miniconda3/envs/vocot/bin/python scripts/run_pope_headset_ablation.py \
  --model_path liuhaotian/llava-v1.5-7b \
  --image_folder /home/kms/data/pope/val2014 \
  --question_file /tmp/pope_1000_q.jsonl \
  --gt_csv /home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv \
  --headset_json /home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json \
  --out_dir /home/kms/LLaVA_calibration/experiments/pope_headset_ablation_1000 \
  --ais_tau 2.2 \
  --ais_gamma 0.2 \
  --ais_faithful_boost 1.0 \
  --debug_dump
```

Outputs:
- `ablation_metrics.csv` (baseline + harmful_only + faithful_only + bipolar)
- `summary.json`

## Budget-Centered Mode (recommended for fair 3-arm comparison)

Instead of threshold-triggered intervention (`AIS > tau`), this mode fixes per-step intervention mass and routes it to:

- top `r%` harmful heads (suppression)
- top `q%` faithful heads (preservation)

Example:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 /home/kms/miniconda3/envs/vocot/bin/python scripts/run_pope_headset_ablation.py \
  --model_path liuhaotian/llava-v1.5-7b \
  --image_folder /home/kms/data/pope/val2014 \
  --question_file /tmp/pope_1000_q.jsonl \
  --gt_csv /home/kms/LLaVA_calibration/experiments/pope_fragility_1000_greedy_b1/per_sample.csv \
  --headset_json /home/kms/LLaVA_calibration/experiments/pope_headsets_v1/headset.json \
  --out_dir /home/kms/LLaVA_calibration/experiments/pope_headset_ablation_1000_budget \
  --ais_use_budget_routing \
  --ais_budget_total 0.06 \
  --ais_harmful_top_ratio 0.2 \
  --ais_faithful_top_ratio 0.2 \
  --ais_bipolar_harmful_ratio 0.5 \
  --ais_tau 999 \
  --ais_gamma 1.0 \
  --debug_dump
```
