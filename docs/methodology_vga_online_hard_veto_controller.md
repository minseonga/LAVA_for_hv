# VGA Online Hard-Veto Controller: Exact Current Implementation

This note describes the **new online probe-and-route implementation** added under:

- `/home/kms/LLaVA_calibration/pnp_controller`

It is intentionally separate from the older offline decision-level emulation.


## 1. What this implementation is

The online method is a **training-free wrapper-level controller** that runs:

1. a lightweight **probe** on the current sample,
2. computes runtime controller features,
3. applies a **hard veto rule**,
4. routes the sample to either:
   - baseline path (`use_add=False`), or
   - VGA path (`use_add=True`).

This means the final generation is produced by **only one selected path**.
It is not a post-hoc selector that chooses between two already-finished predictions.


## 2. Main code path

### Online adapter

- `/home/kms/LLaVA_calibration/pnp_controller/adapters/vga_online.py`

### Runtime feature helpers

- `/home/kms/LLaVA_calibration/pnp_controller/core/runtime_features.py`

### Online runner

- `/home/kms/LLaVA_calibration/scripts/run_pnp_hard_veto_online.py`


## 3. Probe stage

For each sample, the adapter follows the VGA prompt construction exactly:

1. build the multimodal prompt
2. tokenize with `tokenizer_image_token`
3. preprocess the image
4. run a VGA-style prompt prefill:

```text
model(input_ids[:, :-1], images=..., use_cache=True)
```

This prefill provides:

- `past_key_values`
- `logits` over the prompt sequence

From these logits, the adapter computes:

- `vis_logits = softmax(logits[0, 35:611, :])`
- VGA-style grounding / guidance map `G`

The implementation currently keeps VGA-origin behavior:

- if an object field is present, use the first token of each object phrase
- otherwise use entropy fallback


## 4. Probe feature modes

Two probe feature modes are supported.

### 4.1 Legacy `static_headset`

This reproduces the earlier runtime definition as closely as possible using static faithful/harmful head sets.

### 4.2 Main `aggregate`

This is the preferred main controller direction.

Define the late-window mean image attention distribution:

```text
bar(alpha_img)(p) = mean_(l,h in late) alpha_img_norm(l,h,p)
```

Then compute a shared model-side FRG:

```text
FRG = 1 - TopKMean_(late heads)(attn_vis_ratio)
```

where `attn_vis_ratio = attn_to_image / attn_to_all_keys`.

Then compute backend-conditioned mismatch signals:

```text
C_agg = 1 - cos(bar(alpha_img), G)
```

or

```text
C_agg = 1 - <bar(alpha_img), G>
```

and

```text
E_agg = JS(bar(alpha_img) || G)
```

or

```text
E_agg = C_agg + lambda * (1 - coverage_topk(G))
```

The online runner still uses the generic internal field names `FRG` / `GMI`, but in aggregate mode these correspond to the selected `C_agg` / `E_agg` metrics.

In the current preferred configuration:

- `FRG := frg_shared_topk`
- `GMI := e_agg_js`

Older double-mismatch settings such as `FRG := c_agg_cos` are kept only as ablations.


## 5. Exact online FRG computation in `static_headset` mode

To match the current offline feature definition as closely as possible, the adapter runs one additional probe step on the **last prompt token**:

```text
model(
    input_ids[:, -1:],
    past_key_values=prefill.past_key_values,
    output_attentions=True,
)
```

This gives attention tensors for the exact first-answer-token decision context.

For each late layer and head:

```text
head_attn_vis_ratio =
    attn_to_image / (attn_to_image + attn_to_text)
```

The image token span is inferred from the prompt-side image placeholder and the runtime image-token count.

Then:

```text
faithful_head_attn_mean = mean head_attn_vis_ratio over faithful heads in late window
global_late_head_attn_mean = mean head_attn_vis_ratio over all late heads
FRG = faithful_head_attn_mean - global_late_head_attn_mean
```

This is implemented in:

- `compute_head_attn_vis_ratio_last_row(...)`


## 6. Exact online GMI computation in `static_headset` mode

First compute:

```text
G_top5_mass = sum of top-5 entries of VGA guidance map G
```

Then compute:

```text
harmful_head_attn_mean = mean head_attn_vis_ratio over harmful heads in late window
faithful_head_attn_mean = mean head_attn_vis_ratio over faithful heads in late window
GMI = (harmful_head_attn_mean * G_top5_mass) - (faithful_head_attn_mean * G_top5_mass)
```

This matches the **current offline scalar proxy definition** used in the hard-veto experiments.


## 7. Hard-veto decision rule

Given calibrated thresholds `tau_FRG`, `tau_GMI`:

```text
veto(x) = 1[ FRG(x) >= tau_FRG  OR  GMI(x) >= tau_GMI ]
```

Routing:

- if `veto=1`: route to baseline generation
- else: route to VGA generation

So the final generation policy is:

```text
baseline path   if FRG high or GMI high
VGA path        otherwise
```


## 8. Generation stage

The adapter reuses the prefill cache and common VGA generation arguments.

### Baseline route

```text
model.generate(..., use_add=False)
```

### VGA route

```text
model.generate(..., use_add=True)
```

The controller itself does not modify model internals. It only decides which route to use.


## 9. Threshold sources

The online runner supports three threshold sources:

1. manual thresholds
   - `--tau_frg`
   - `--tau_gmi`
2. previously saved offline controller summary
   - `--controller_summary_json`
3. on-the-fly calibration from offline per-case CSV + feature CSV
   - `--per_case_csv`
   - `--features_csv`

This preserves the training-free calibration story.


## 10. What this implementation is not

This is **not**:

- a post-hoc best-of-two oracle
- a model-internal controller baked into VGA core files
- a token-level online FRRS residual steering operator

It is specifically:

> a wrapper-level, training-free, sample-level hard-veto controller that uses exact runtime FRG/GMI probe features to choose between baseline and VGA generation.


## 11. Recommended run form

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
  --out_dir /home/kms/LLaVA_calibration/experiments/pnp_online_vga_hard_veto_1000
```


## 12. Current extension path

This implementation is designed so that the same controller core can later be connected to:

- `VISTA`
- `EAZY`

through backend-specific online adapters, while leaving the controller logic unchanged.
