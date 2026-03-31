# Server Online Experiment Guide

This file is the server-side execution guide for the current main routing candidate.

The goal is to let another Codex instance on the server reproduce the exact pipeline without reverse-engineering local notes or chat history.

## Current Main Candidate

Use this setting unless there is a deliberate reason to change it:

- backend: `VGA`
- dataset: `POPE-9000`
- router: `HistGradientBoostingRegressor`
- feature variant: `no_abs`
- deployment budget: `30%`
- probe feature source:
  - `probe_feature_mode = static_headset`
  - `probe_position_mode = baseline_yesno_preview`
  - `probe_branch_source = preview`
- headset: `experiments/pope_discovery/discovery_headset.json`

This is the best current cost-aware setting on the local side:

- OOF `HGB @ 30%`: about `0.8729`
- `baseline_only`: about `0.8522`
- `vga_only`: about `0.8661`

Important: this guide separates two experiment types.

1. `claim-safe offline estimate`
   Build the router from probe features and inspect OOF budget curves. This is the number to cite when you want a cleaner estimate of generalization on the same dataset.

2. `actual online end-to-end run`
   Save a full-data router artifact, then run real online inference with one fixed score cutoff. This is the deployment sanity check. It is useful, but if you fit and evaluate on the same 9000 set, do not present it as an unbiased held-out result.

## Paper-Valid Protocol

If the goal is a paper result, the correct split is:

1. fit on `discovery`
2. freeze the router artifact
3. run actual online inference on `POPE-9000`
4. report `POPE-9000` only as the final test result

Why this matters:

- the router target is `utility`
- `utility` is built from `gt`, `pred_baseline`, and `pred_method`
- so fitting the router on POPE-9000 uses POPE-9000 labels to shape the routing policy

That means:

- `POPE-fit -> POPE-online` = deployment sanity check
- `discovery-fit -> POPE-online` = paper-valid held-out online evaluation

The wrappers below can already run the paper-valid protocol through environment overrides.

## Files and Scripts

Relevant scripts:

- probe extraction: [scripts/extract_pnp_probe_features.py](../scripts/extract_pnp_probe_features.py)
- router build/export: [scripts/build_cost_aware_gain_router.py](../scripts/build_cost_aware_gain_router.py)
- actual online runner: [scripts/run_cost_aware_gain_router_online.py](../scripts/run_cost_aware_gain_router_online.py)
- artifact wrapper: [scripts/build_vga_cost_aware_gain_router_artifact_9000.sh](../scripts/build_vga_cost_aware_gain_router_artifact_9000.sh)
- online wrapper: [scripts/run_vga_cost_aware_gain_router_online_9000.sh](../scripts/run_vga_cost_aware_gain_router_online_9000.sh)

Relevant documents:

- method overview: [README.md](./README.md)
- 30% deployment story: [deployment_rule_30_budget.md](./deployment_rule_30_budget.md)
- CPU-only follow-ups: [cpu_only_followups.md](./cpu_only_followups.md)

## Recommended Server Workflow

### Stage A. Extract cheap probe features and build a router artifact

There are two versions of this stage.

#### A1. Deployment sanity check only

Build on POPE-9000 itself. Useful for validating the online code path, but not for the final held-out number.

This stage runs the actual cheap probe online, but does not generate final branch predictions.

Command:

```bash
cd ~/LLaVA_calibration

CUDA_VISIBLE_DEVICES=5 \
OUT_DIR=/home/kms/LLaVA_calibration/experiments/pope_full_9000/vga_cost_aware_gain_router_artifact_9000 \
bash scripts/build_vga_cost_aware_gain_router_artifact_9000.sh
```

Outputs:

- `.../probe_features/probe_features.csv`
- `.../probe_features/summary.json`
- `.../router/summary.json`
- `.../router/router_model.pkl`
- `.../router/router_metadata.json`

What this wrapper does:

1. extract probe features with the current main probe setting
2. fit the current best router (`HGB`, `no_abs`, `30% budget`)
3. save the deployable artifact

#### A2. Paper-valid artifact build

Build on discovery, then carry that frozen artifact to POPE-9000.

```bash
cd ~/LLaVA_calibration

CUDA_VISIBLE_DEVICES=5 \
QUESTION_FILE=/home/kms/LLaVA_calibration/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q_with_object.jsonl \
TAXONOMY_CSV=/home/kms/LLaVA_calibration/experiments/pope_discovery/tau_c_calibration_adversarial/taxonomy/per_case_compare.csv \
OUT_DIR=/home/kms/LLaVA_calibration/experiments/pope_discovery/vga_cost_aware_gain_router_artifact \
bash scripts/build_vga_cost_aware_gain_router_artifact_9000.sh
```

This is the command you want if the online POPE-9000 result is intended for the paper.

### Stage B. Inspect the offline estimate before the online run

Before spending GPU time on full end-to-end routing, check the saved router summary.

Main file:

- `.../router/summary.json`

Expect roughly:

- `feature_variant = "no_abs"`
- `deployment_budget = 0.3`
- `hgb_budget_row.acc ≈ 0.8729`
- `deployment_cutoff_hgb_fullfit ≈ 0.0141`

If these are far off, do not proceed until the probe extraction path matches the intended setting.

### Stage C. Run actual online inference on POPE-9000

This stage uses the saved router artifact and executes exactly one expensive branch per sample.

Command:

```bash
cd ~/LLaVA_calibration

CUDA_VISIBLE_DEVICES=5 \
ROUTER_DIR=/home/kms/LLaVA_calibration/experiments/pope_discovery/vga_cost_aware_gain_router_artifact/router \
OUT_DIR=/home/kms/LLaVA_calibration/experiments/pope_full_9000/vga_cost_aware_gain_router_online_from_discovery \
bash scripts/run_vga_cost_aware_gain_router_online_9000.sh
```

Outputs:

- `.../pred_online_controller.jsonl`
- `.../route_log.csv`
- `.../summary.json`

What to check:

- `counts.method_rate`
- `metrics.acc`
- route histogram by score band
- whether `method_rate` stays near the intended `30%`

Because the runtime policy is a fixed score threshold, the method rate may be near `30%`, not exactly `30.00%`.

## Direct Python Commands

If wrappers are inconvenient, the exact direct commands are below.

### Discovery probe extraction only

```bash
cd ~/LLaVA_calibration

CUDA_VISIBLE_DEVICES=5 python scripts/extract_pnp_probe_features.py \
  --backend vga \
  --vga_root /home/kms/VGA_origin \
  --model_path liuhaotian/llava-v1.5-7b \
  --image_folder /home/kms/data/pope/val2014 \
  --question_file /home/kms/LLaVA_calibration/experiments/pope_discovery/tau_c_calibration_adversarial/assets/discovery_q_with_object.jsonl \
  --out_dir /home/kms/LLaVA_calibration/experiments/pope_discovery/vga_cost_aware_gain_router_artifact/probe_features \
  --conv_mode llava_v1 \
  --device cuda \
  --temperature 0 \
  --top_p 1.0 \
  --sampling false \
  --max_gen_len 8 \
  --num_beams 1 \
  --cd_alpha 0.02 \
  --attn_coef 0.2 \
  --start_layer 16 \
  --end_layer 24 \
  --head_balancing simg \
  --attn_norm false \
  --late_start 16 \
  --late_end 24 \
  --probe_feature_mode static_headset \
  --headset_json /home/kms/LLaVA_calibration/experiments/pope_discovery/discovery_headset.json \
  --probe_position_mode baseline_yesno_preview \
  --probe_branch_source preview \
  --probe_preview_max_new_tokens 3 \
  --probe_preview_reuse_baseline true \
  --probe_preview_fallback_to_prompt_last true \
  --seed 42
```

### Build discovery router artifact

```bash
cd ~/LLaVA_calibration

python scripts/build_cost_aware_gain_router.py \
  --probe_log_csv /home/kms/LLaVA_calibration/experiments/pope_discovery/vga_cost_aware_gain_router_artifact/probe_features/probe_features.csv \
  --taxonomy_csv /home/kms/LLaVA_calibration/experiments/pope_discovery/tau_c_calibration_adversarial/taxonomy/per_case_compare.csv \
  --out_dir /home/kms/LLaVA_calibration/experiments/pope_discovery/vga_cost_aware_gain_router_artifact/router \
  --tau -0.0068411549792573 \
  --backend_name vga \
  --feature_variant no_abs \
  --deployment_budget 0.30 \
  --seed 42 \
  --save_router_artifact
```

### Actual online run

```bash
cd ~/LLaVA_calibration

CUDA_VISIBLE_DEVICES=5 python scripts/run_cost_aware_gain_router_online.py \
  --backend vga \
  --router_dir /home/kms/LLaVA_calibration/experiments/pope_discovery/vga_cost_aware_gain_router_artifact/router \
  --vga_root /home/kms/VGA_origin \
  --model_path liuhaotian/llava-v1.5-7b \
  --image_folder /home/kms/data/pope/val2014 \
  --question_file /home/kms/LLaVA_calibration/experiments/pope_full_9000/pope_9000_q_with_object.jsonl \
  --out_dir /home/kms/LLaVA_calibration/experiments/pope_full_9000/vga_cost_aware_gain_router_online_from_discovery \
  --conv_mode llava_v1 \
  --device cuda \
  --temperature 0 \
  --top_p 1.0 \
  --sampling false \
  --max_gen_len 8 \
  --num_beams 1 \
  --cd_alpha 0.02 \
  --attn_coef 0.2 \
  --start_layer 16 \
  --end_layer 24 \
  --head_balancing simg \
  --attn_norm false \
  --late_start 16 \
  --late_end 24 \
  --probe_feature_mode static_headset \
  --headset_json /home/kms/LLaVA_calibration/experiments/pope_discovery/discovery_headset.json \
  --probe_position_mode baseline_yesno_preview \
  --probe_branch_source preview \
  --probe_preview_max_new_tokens 3 \
  --probe_preview_reuse_baseline true \
  --probe_preview_fallback_to_prompt_last true \
  --use_gmi false \
  --seed 42 \
  --gt_csv /home/kms/LLaVA_calibration/experiments/pope_full_9000/pope_9000_gt.csv \
  --gt_id_col id \
  --gt_label_col answer
```

## Why This Is the Right Online Check

This is the first end-to-end run that actually tests the current main idea:

- cheap probe first
- router score next
- exactly one expensive branch per sample
- no both-run oracle at deployment

That matches the intended deployment story much better than replaying offline tables.

## Important Caveats

1. `POPE-fit -> POPE-online` is a sanity check, not the final paper number.
   - for the paper, use `discovery-fit -> POPE-online`

2. The current policy is tuned for yes/no POPE-style tasks.
   - `baseline_yesno_preview` plus preview reuse is acceptable here because the task output is short.

3. If the probe extraction path changes, the router artifact must be rebuilt.
   - Do not reuse a router trained from a different probe configuration.

## Minimal Checklist for Server Codex

Before running the online experiment, confirm:

- probe summary says `static_headset`
- probe summary says `baseline_yesno_preview`
- router summary says `feature_variant = no_abs`
- router summary says `deployment_budget = 0.3`
- router metadata exists:
  - `router_model.pkl`
  - `router_metadata.json`

After the online run, report:

- `metrics.acc`
- `counts.method_rate`
- path to `summary.json`
- whether the observed method rate stayed close to the trained `30%` target
