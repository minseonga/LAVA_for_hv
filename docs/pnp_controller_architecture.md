# PnP Controller Architecture

This note describes the new repo-independent plug-and-play controller layer added under:

- `/home/kms/LLaVA_calibration/pnp_controller`

The goal is to keep the controller family separate from model-internal method branches such as:

- `AIS`
- `RFHAR`
- `FRGG`
- `FRRS`

so that we can evaluate the same controller logic on top of multiple guidance-style backends:

- `VGA`
- `VISTA`
- `EAZY`
- future backends

without repeatedly rewriting the model core.


## 1. Design principle

We separate the system into two layers:

1. `controller core`
   - feature names
   - calibration
   - hard-veto / soft-gate decision logic
2. `backend adapters`
   - how a specific method family exposes:
     - baseline path
     - method path
     - probe features

This means the controller can stay method-agnostic while each backend only implements a thin compatibility layer.


## 2. Current package layout

### Core

- `/home/kms/LLaVA_calibration/pnp_controller/core/schemas.py`
- `/home/kms/LLaVA_calibration/pnp_controller/core/features.py`
- `/home/kms/LLaVA_calibration/pnp_controller/core/controller.py`

### Adapters

- `/home/kms/LLaVA_calibration/pnp_controller/adapters/base.py`
- `/home/kms/LLaVA_calibration/pnp_controller/adapters/offline_csv.py`
- `/home/kms/LLaVA_calibration/pnp_controller/adapters/vga.py`
- `/home/kms/LLaVA_calibration/pnp_controller/adapters/vista.py`
- `/home/kms/LLaVA_calibration/pnp_controller/adapters/eazy.py`

### Generic runner

- `/home/kms/LLaVA_calibration/scripts/run_pnp_hard_veto_offline.py`


## 3. What is implemented now

### Implemented

The current package supports:

- backend-agnostic **offline hard-veto control**
- common `FRG/GMI` threshold calibration
- common summary / per-id output format
- backend-specific offline adapters for:
  - `VGA`
  - `VISTA`
  - `EAZY`

### Partially implemented now

The package now also includes a first online backend:

- `/home/kms/LLaVA_calibration/pnp_controller/adapters/vga_online.py`
- `/home/kms/LLaVA_calibration/scripts/run_pnp_hard_veto_online.py`

This path implements:

- online `probe -> route -> selected generation`
- exact runtime `FRG/GMI` extraction for the current VGA-based definition
- wrapper-level hard-veto switching between:
  - baseline path (`use_add=False`)
  - VGA method path (`use_add=True`)

### Still not implemented

- online `VISTA` adapter
- online `EAZY` adapter
- fully generic online runner for all backends


## 4. Offline hard-veto controller

The controller supports two feature modes.

Legacy static-headset mode:

- `FRG := faithful_minus_global_attn`
- `GMI := guidance_mismatch_score`

Main aggregate mode:

- `FRG := C_agg`
- `GMI := E_agg`
- where `C_agg` / `E_agg` are sample-level late-window aggregate statistics computed from
  - the mean image-only attention distribution across all late heads
  - the guidance map `G`

Current default main instantiation:

- `FRG := 1 - TopKMean_(late heads)(attn_vis_ratio)`
- `GMI := JS(AggAttn || GuidanceMap_m)`

Double-mismatch variants such as `FRG := 1 - cos(AggAttn, G)` are preserved only as ablations.

and applies:

```text
veto = 1[ FRG >= tau_FRG or GMI >= tau_GMI ]
```

If veto is true:

- route to baseline

otherwise:

- route to method path

This is implemented in:

- `/home/kms/LLaVA_calibration/pnp_controller/core/controller.py`


## 5. Why this package exists separately from model core

There are already in-progress model-core modifications in:

- `/home/kms/LLaVA_calibration/llava/eval/model_vqa_loader.py`
- `/home/kms/LLaVA_calibration/llava/model/builder.py`
- `/home/kms/LLaVA_calibration/llava/model/language_model/llava_llama.py`
- `/home/kms/LLaVA_calibration/llava/model/llava_arch.py`

Those changes are intervention-runtime oriented.

The new `pnp_controller` package is intentionally separated so that:

- we do not couple the controller family to one backend
- we do not increase merge risk in already-dirty model-core files
- we can compare backends using the same controller logic


## 6. Online VGA path

The current online implementation is intentionally wrapper-level, not model-core invasive.

For each sample:

1. `probe`
   - run a VGA-style prompt prefill
   - compute `vis_logits`
   - build `vl_guidance`
   - run one extra one-token forward on the last prompt token with `output_attentions=True`
2. `feature extraction`
   - `static_headset` mode:
     - `FRG = faithful_head_attn_mean - global_late_head_attn_mean`
     - `GMI = (harmful_head_attn_mean * G_top5_mass) - (faithful_head_attn_mean * G_top5_mass)`
   - `aggregate` mode:
     - `bar(alpha_img)(p) = mean_(l,h in late) alpha_img_norm(l,h,p)`
     - shared `FRG = 1 - TopKMean_(late heads)(attn_vis_ratio)`
     - backend-conditioned `GMI = JS(bar(alpha_img) || G)` or `1-cos + lambda * (1 - coverage_topk(G))`
3. `route`
   - hard veto if `FRG >= tau_FRG` or `GMI >= tau_GMI`
4. `generation`
   - vetoed: baseline path (`use_add=False`)
   - otherwise: VGA path (`use_add=True`)

This keeps the controller logic outside the VGA model code and preserves the repo-independent design.


## 7. Immediate usage

The new generic runner can already be used for offline controller evaluation:

```bash
cd /home/kms/LLaVA_calibration

/home/kms/miniconda3/envs/vocot/bin/python scripts/run_pnp_hard_veto_offline.py \
  --backend vga \
  --per_case_csv /home/kms/LLaVA_calibration/experiments/pope_full_9000/all_models_full_strict/vga/taxonomy/per_case_compare.csv \
  --features_csv /home/kms/LLaVA_calibration/experiments/pope_feature_screen_v1_full9000/features_unified_table.csv \
  --out_dir /home/kms/LLaVA_calibration/experiments/pope_full_9000/pnp_vga_hard_veto
```

The same runner can be pointed at `VISTA` or `EAZY` outputs by switching the backend and input paths.


## 8. Online VGA usage

```bash
cd /home/kms/LLaVA_calibration

/home/kms/miniconda3/envs/vocot/bin/python scripts/run_pnp_hard_veto_online.py \
  --backend vga \
  --vga_root /home/kms/VGA_origin \
  --model_path liuhaotian/llava-v1.5-7b \
  --image_folder /home/kms/data/pope/val2014 \
  --question_file /tmp/pope_1000_q.jsonl \
  --probe_feature_mode aggregate \
  --aggregate_frg_metric frg_shared_topk \
  --aggregate_gmi_metric e_agg_js \
  --controller_summary_json /home/kms/LLaVA_calibration/experiments/pope_full_9000/vga_hard_veto_controller_9000/summary.json \
  --gt_csv /home/kms/LLaVA_calibration/experiments/pope_full_9000/pope_9000_gt.csv \
  --out_dir /home/kms/LLaVA_calibration/experiments/pnp_online_vga_example
```
