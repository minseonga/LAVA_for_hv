# VGA Hard-Veto Controller: Exact Current Implementation

This document describes the **current implemented method** in this repository, based on the code that is actually running today.

The goal of this note is to be paper-ready in the sense of being:

- precise,
- reproducible,
- explicit about what is implemented,
- explicit about what is **not** implemented yet.

It should be read as a description of the **current operational method**, not the final idealized online controller.


## 1. What the current method is

The current method is a **training-free, decision-level, hard-veto controller** built on top of:

- a **baseline** path (`vanilla LLaVA-1.5`), and
- a **VGA** path (`object-centric visual guidance`).

The controller does **not** currently modify the model internals online.
Instead, it takes:

- the baseline prediction,
- the VGA prediction,
- a precomputed sample-level routing feature `C`,
- a precomputed sample-level mismatch feature `E`,

and then decides, **per sample**, whether to:

- keep the VGA decision, or
- veto VGA and fall back to the baseline decision.

In short:

`use VGA unless the sample looks D2-like; if D2-like, revert to baseline`.


## 2. What is the exact code path

### Main implementation

- [`scripts/run_vga_hard_veto_controller.py`](/home/kms/LLaVA_calibration/scripts/run_vga_hard_veto_controller.py)

### Repro wrapper

- [`scripts/run_vga_hard_veto_controller_9000.sh`](/home/kms/LLaVA_calibration/scripts/run_vga_hard_veto_controller_9000.sh)

### Upstream inputs used by the controller

- Baseline/VGA comparison table:
  - [`experiments/pope_full_9000/all_models_full_strict/vga/taxonomy/per_case_compare.csv`](/home/kms/LLaVA_calibration/experiments/pope_full_9000/all_models_full_strict/vga/taxonomy/per_case_compare.csv)
- Full-9000 feature table:
  - [`experiments/pope_feature_screen_v1_full9000/features_unified_table.csv`](/home/kms/LLaVA_calibration/experiments/pope_feature_screen_v1_full9000/features_unified_table.csv)

### Current saved result

- [`experiments/pope_full_9000/vga_hard_veto_controller_9000/summary.json`](/home/kms/LLaVA_calibration/experiments/pope_full_9000/vga_hard_veto_controller_9000/summary.json)


## 3. Inputs and labels used by the controller

The controller consumes a merged table with one row per sample.

### Prediction inputs

From `per_case_compare.csv`:

- `gt`
- `pred_baseline`
- `pred_vga`
- `case_type`

The key field is `case_type`, which is already computed by the taxonomy script:

- `both_correct`
- `both_wrong`
- `vga_improvement`
- `vga_regression`

In the paper framing:

- `vga_improvement` corresponds to **D1-like beneficial guidance**
- `vga_regression` corresponds to **D2-like harmful guidance**

### Feature inputs

From `features_unified_table.csv`, the current controller uses exactly two scalar sample-level features:

- `faithful_minus_global_attn`
- `guidance_mismatch_score`

These are interpreted as:

- `C := faithful_minus_global_attn`
- `E := guidance_mismatch_score`

Important:

- These are already **precomputed scalar sample-level features**.
- The current controller does **not** compute token-level TopK means online.
- The controller directly uses the feature columns as provided in the CSV.

So the currently implemented method is best described as:

`sample-level FRG/GMI-gated decision controller using precomputed features`.


## 4. Mathematical form of the current method

Let:

- `y_base(x)` be the baseline prediction for sample `x`
- `y_vga(x)` be the VGA prediction for sample `x`
- `C(x)` be the scalar feature `faithful_minus_global_attn`
- `E(x)` be the scalar feature `guidance_mismatch_score`

### 4.1 Hard veto rule

The controller computes a binary veto decision:

```text
d(x) = 1[ C(x) >= tau_C  OR  E(x) >= tau_E ]
```

Interpretation:

- `d(x)=1`: veto VGA, use baseline
- `d(x)=0`: keep VGA

### 4.2 Final prediction

```text
y_hat(x) =
    y_base(x), if d(x)=1
    y_vga(x),  if d(x)=0
```

This is implemented in:

- [`scripts/run_vga_hard_veto_controller.py`](/home/kms/LLaVA_calibration/scripts/run_vga_hard_veto_controller.py)

Specifically:

- `compute_veto_mask(...)` implements the rule
- `pred_controller = baseline if veto else vga`


## 5. Threshold calibration

The thresholds `tau_C` and `tau_E` are not manually fixed in the default run.
They are calibrated from the same merged table using a random calibration subset.

### 5.1 Split protocol

The implementation:

- takes all rows with valid `C` and `E`,
- shuffles them with `seed=42`,
- uses `30%` as calibration split,
- uses `70%` as held-out test split.

This is controlled by:

- `--calib_ratio 0.3`
- `--seed 42`

### 5.2 Candidate thresholds

The search grid is built from quantiles of the observed `C` and `E` distributions.

Default quantile grid:

```text
0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
```

So the candidate thresholds are:

- `tau_C in quantiles(C)`
- `tau_E in quantiles(E)`

### 5.3 Calibration objective

The objective is explicitly D2-focused:

```text
J(tau_C, tau_E) = (# D2 correctly vetoed) - lambda_D1 * (# D1 wrongly vetoed)
```

with:

- `lambda_D1 = 1.0`

There is also a hard constraint:

```text
D1_wrong_veto_rate <= max_d1_wrong_rate
```

with:

- `max_d1_wrong_rate = 0.35`

If all threshold pairs violate the constraint, the code falls back to the unconstrained maximum of the same objective.

This logic is implemented in:

- `calibrate_thresholds(...)`

inside:

- [`scripts/run_vga_hard_veto_controller.py`](/home/kms/LLaVA_calibration/scripts/run_vga_hard_veto_controller.py)


## 6. Exact default configuration currently used

The current 9000-sample run uses:

- `C = faithful_minus_global_attn`
- `E = guidance_mismatch_score`
- `fallback_when_missing_feature = vga`
- `calib_ratio = 0.3`
- `seed = 42`
- `lambda_d1 = 1.0`
- `max_d1_wrong_rate = 0.35`
- `q_grid = [0.50, ..., 0.95]`

The selected thresholds for the saved run are:

- `tau_C = -0.00934999483136276`
- `tau_E = 0.06121958185580838`

These values are recorded in:

- [`experiments/pope_full_9000/vga_hard_veto_controller_9000/summary.json`](/home/kms/LLaVA_calibration/experiments/pope_full_9000/vga_hard_veto_controller_9000/summary.json)


## 7. Current empirical result of the method

For the saved 9000-sample run:

- Baseline accuracy: `0.8522`
- VGA accuracy: `0.8661`
- Hard-veto controller accuracy: `0.8750`

So the controller improves:

- `+0.0089` over VGA
- `+0.0228` over baseline

Additional controller diagnostics:

- veto count: `4041 / 9000`
- veto rate: `0.449`
- `D1_total = 471`
- `D2_total = 346`
- `D1_wrong_veto = 55`
- `D2_correct_veto = 135`
- `D1_wrong_veto_rate = 0.1168`
- `D2_correct_veto_rate = 0.3902`

Interpretation:

- the controller removes a substantial fraction of harmful VGA cases,
- while only vetoing a smaller fraction of VGA-helpful cases.


## 8. What this method is claiming scientifically

The current implemented method supports the following claim:

> Always-on object-centric guidance is not uniformly beneficial.
> A sample-level controller can improve over always-on VGA by selectively vetoing guidance in regimes associated with higher routing usability and/or higher guidance mismatch.

Equivalently:

- low-risk samples keep VGA,
- D2-like samples revert to baseline.

The key modeling assumption in the current implementation is:

```text
high C  OR  high E  =>  higher probability that VGA is harmful
```

where:

- `C` is a faithful-routing-derived scalar,
- `E` is a guidance-mismatch-derived scalar.


## 9. What is not implemented yet

This is important for paper accuracy.

### Not currently implemented

1. Online 2-pass inference

The current method does **not** run:

- a probe pass,
- then a route decision,
- then a second full forward pass.

Instead, it selects between already-computed baseline and VGA predictions offline.

2. Internal model gating

The controller does **not** currently:

- modify attention logits,
- modify hidden states,
- modify residual streams,
- intervene inside `VGA_origin` at generation time.

3. Token-level online FRG/GMI

The current method does **not** compute token-level `C_i` / `E_i` online and then aggregate them at runtime.

It uses the **precomputed sample-level scalar columns** already stored in the features table.

### Therefore, the correct description is

The current method is:

> an offline, training-free, sample-level hard-veto controller over baseline vs VGA decisions.

It is **not yet** an online plug-and-play controller inside the model forward path.


## 10. Relationship to oracle experiments

The repo also contains oracle-style scripts:

- [`scripts/build_vga_decision_oracles.py`](/home/kms/LLaVA_calibration/scripts/build_vga_decision_oracles.py)

These compute upper bounds such as:

- best-of-two oracle,
- veto-only oracle,
- 4-way regime oracle.

Those oracle experiments are not the main method.
They are used to estimate the ceiling of the controller family.

For the VGA 9000 setup, the oracle ceiling was approximately:

- `0.9046` accuracy

This means the current hard-veto controller (`0.8750`) recovers part, but not all, of the available oracle gain.


## 11. Repro command

To reproduce the current hard-veto result:

```bash
cd /home/kms/LLaVA_calibration
bash scripts/run_vga_hard_veto_controller_9000.sh
```


## 12. Suggested paper wording

If the paper needs a faithful description of the current implementation, the safest wording is:

> We implement a training-free hard-veto controller over baseline and VGA predictions.
> The controller uses two precomputed sample-level signals, a faithful-routing score (`faithful_minus_global_attn`) and a guidance-mismatch score (`guidance_mismatch_score`), and vetoes VGA whenever either score exceeds a calibrated threshold.
> Thresholds are selected on a calibration split by maximizing D2 recovery while penalizing D1 suppression.
> The resulting controller improves over always-on VGA on POPE-9000.

If the paper wants to describe the future target method, that should be clearly labeled as:

> an online extension,

not the current implemented version.

