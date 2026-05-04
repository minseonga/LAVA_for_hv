# RaPiC Paper Draft Plan

This document is a detailed paper-facing plan for a NeurIPS-style submission.
It is not a lightweight outline. It specifies the intended claims, section
logic, table placement, result interpretation, and appendix split.

Working title:

> Calibrating Vision-Language Intervention Methods by Replay-Aware Post-hoc
> Control

Short title:

> RaPiC: Replay-Aware Post-intervention Calibration

## Core Thesis

Hallucination mitigation interventions are usually evaluated as always-on
decoding or activation-time methods. This hides a sample-level fact: an
intervention can correct some baseline errors while introducing new regressions.
RaPiC treats the intervention output as a candidate, estimates whether the
intervention is likely to be harmful using deployable replay features under the
frozen backbone, and falls back to the baseline response only for calibrated
risk cases.

The paper should make a bounded but strong claim:

> RaPiC is not a universal correction oracle. It is a calibration layer for
> intervention methods. When intervention-induced regressions are identifiable
> from replay features, RaPiC recovers harmful cases while preserving helpful
> intervention gains. When the calibration distribution is not representative,
> RaPiC exposes this mismatch and can abstain.

This framing handles the mixed results cleanly:

- VGA results are strong and stable across backbones.
- LLaVA-1.5 PAI/VAF results show broad utility beyond VGA.
- Qwen and LLaVA-NeXT cases show important boundaries.
- Generative CHAIR results give a clean validation-calibrated win.
- Pairwise replay is a stronger diagnostic/possible extension, but should only
  be promoted to the main method if the running full panel shows consistent
  gains across method/backbone settings.

## Abstract Draft

Modern vision-language hallucination mitigation methods often intervene during
decoding or internal computation and are applied uniformly to all inputs.
Although such interventions can improve average accuracy, they are not
monotonic at the instance level: the same method can correct some baseline
errors while causing new regressions. We introduce RaPiC, a replay-aware
post-intervention calibration framework that makes such interventions
selective. Given a baseline response and an intervention response, RaPiC uses
teacher-forced replay features from the frozen backbone to estimate whether the
intervention is likely to be harmful, and falls back to the baseline response
only for calibrated high-risk cases. For discriminative yes/no object-presence
benchmarks, RaPiC calibrates transition-specific controllers for `yes->no` and
`no->yes` changes. For generative captioning, RaPiC calibrates an object-token
suppression controller on validation and applies it unchanged to test. Across
POPE-style MSCOCO, AOKVQA, and GQA evaluations, RaPiC improves several
intervention methods over their always-on variants and often recovers or
exceeds the vanilla baseline when interventions regress. On COCO-CHAIR, RaPiC
reduces CHAIRi from 8.90 to 8.28 and CHAIRs from 32.2 to 30.8 over the VGA
intervention, with negligible recall loss. We further analyze failure modes
where calibration and test intervention behavior diverge, showing that replay
features can reveal when a mitigation method is not reliably calibratable.

## 1. Introduction

### Motivation

Most hallucination mitigation interventions are deployed as always-on methods:
given an input, they modify decoding, attention, logits, or internal activations
and output a single intervention response. This average-case evaluation leaves
an important question unanswered:

> If an intervention is beneficial on average, should we trust it on every
> sample?

The answer is no. On POPE-style yes/no evaluations, an intervention can:

- fix a baseline error: baseline wrong, intervention correct;
- introduce a regression: baseline correct, intervention wrong;
- leave correctness unchanged.

Therefore, the deployment problem is not only designing stronger interventions.
It is also calibrating when to accept an intervention.

### Main idea

RaPiC is post-intervention and intervention-first:

1. Generate the intervention response.
2. Compute replay-based risk features for that response under the frozen
   backbone.
3. Route to the intervention response or to the baseline response according to a
   discovery-calibrated policy.

The baseline response may be cached in offline experiments. In deployment, it
can be generated only when the calibrated risk gate fires. This distinction
should be stated clearly to avoid the impression that RaPiC is baseline-first.

### Contributions

Suggested contribution list:

1. **Instance-level analysis of intervention regressions.** We formalize
   harmful and helpful intervention changes and show that popular
   hallucination-mitigation interventions are not monotonic across samples.
2. **Replay-aware post-intervention calibration.** We propose RaPiC, a
   label-free deployment-time controller calibrated on a discovery split using
   teacher-forced replay features from the frozen backbone.
3. **Transition-specific routing for discriminative VLM decisions.** We show
   that `yes->no` and `no->yes` intervention changes have different risk
   profiles and calibrate separate controllers for each transition.
4. **Validation-calibrated generative control.** We extend the same calibration
   principle to object-token suppression for captioning, selecting the
   threshold on validation and applying it unchanged to test.
5. **Failure analysis and diagnostics.** We identify a calibration mismatch
   failure mode where the harm/help prior changes within the same transition,
   and we show how pairwise replay exposes local separability but also feature
   non-stationarity.

## 2. Analysis: Why Intervention Outputs Need Calibration

The Analysis section should come before Method. Its role is to justify the
method design from observed intervention behavior rather than presenting RaPiC
as an arbitrary router.

### 2.1 Interventions are not monotonic

Definitions for a sample `(x_i, q_i)` with ground-truth label `g_i`:

- baseline answer: `y_B`
- intervention answer: `y_I`
- baseline correctness: `b_i = 1[y_B = g_i]`
- intervention correctness: `m_i = 1[y_I = g_i]`

Effect labels:

```text
harm  = 1[b_i = 1 and m_i = 0]
help  = 1[b_i = 0 and m_i = 1]
neutral = otherwise
```

Paper-facing text:

> We call an intervention change harmful when it turns a correct baseline
> answer into an incorrect intervention answer, and helpful when it corrects a
> baseline error. These labels are used only for calibration and analysis; the
> deployment controller receives no test labels.

Recommended table:

| Method / Backbone | Dataset | Base | Intervention | Changed | Harm | Help | Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VGA / LLaVA-1.5 | MSCOCO | fill | fill | fill | fill | fill | fill |
| VAF / LLaVA-NeXT | MSCOCO | fill | fill | fill | fill | fill | fill |

Purpose:

- show that interventions create both gains and regressions;
- motivate sample-wise calibration.

### 2.2 Directional intervention behavior

For yes/no POPE-style tasks, answer changes have directions:

```text
yes->no
no->yes
```

The risk semantics differ:

- `yes->no` can suppress hallucinated yes answers, but can also remove correct
  positive detections.
- `no->yes` can recover missed objects, but can also insert hallucinated
  objects.

Paper-facing text:

> The same intervention direction is not uniformly good or bad. A `no->yes`
> change can be a helpful correction when the queried object is present, or a
> harmful hallucination when it is absent. This motivates transition-specific
> calibration rather than a single pooled changed-answer threshold.

Recommended diagnostic table:

| Method / Backbone | Split | Transition | n | Harm | Help | Harm Rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| VAF / LLaVA-NeXT | discovery | no->yes | 135 | 118 | 17 | 87.4 |
| VAF / LLaVA-NeXT | MSCOCO | no->yes | 403 | 244 | 159 | 60.5 |
| VAF / LLaVA-NeXT | AOKVQA | no->yes | 350 | 296 | 54 | 84.6 |
| VAF / LLaVA-NeXT | GQA | no->yes | 387 | 330 | 57 | 85.3 |

Interpretation:

- Discovery for LLaVA-NeXT VAF is harm-heavy within `no->yes`.
- MSCOCO test contains many more helpful `no->yes` corrections.
- This is not merely a transition-mixing problem; it is a within-transition
  prior shift.

This table belongs either in Analysis or Failure Analysis depending on space.
It is very useful for explaining why naive changed-answer rollback is not a
valid claim.

### 2.3 Replay features as deployable evidence

RaPiC uses the frozen backbone to replay candidate answers. The core intuition:

> If the intervention answer is internally unsupported by the original model's
> image-question context, the intervention is more likely to be a regression.

Single-candidate intervention replay:

```text
S_I = support(y_I | x, q; f_0)
```

Pairwise diagnostic replay:

```text
S_B = support(y_B | x, q; f_0)
S_I = support(y_I | x, q; f_0)
Delta = S_B - S_I
```

Paper-facing text:

> Single-candidate replay measures whether the intervention answer is
> self-supported under the frozen backbone. Pairwise replay is a stricter
> diagnostic: it compares baseline and intervention answers for the same
> image-question pair, partially normalizing sample-specific likelihood scale.

Current diagnostic results for LLaVA-NeXT VAF:

| Split | Pairwise Feature Type | n | H/G | Best AUROC | Oracle H/G/Net |
| --- | --- | ---: | ---: | ---: | ---: |
| discovery | pairwise delta | 135 | 118/17 | 0.722 | 116/13/103 |
| MSCOCO | pairwise delta | 403 | 244/159 | 0.664 | 181/66/115 |
| AOKVQA | pairwise delta | 350 | 296/54 | 0.649 | 296/54/242 |
| GQA | pairwise delta | 387 | 330/57 | 0.668 | 330/57/273 |

Important interpretation:

- Pairwise replay has local signal.
- Discovery-selected pairwise feature semantics do not necessarily transfer to
  MSCOCO.
- Therefore pairwise replay is promising but should not be promoted to the main
  method unless the full panel shows consistent discovery-to-test gains.

Feature-transfer diagnostic:

| Discovery Feature | Discovery AUROC | MSCOCO Same-Direction AUROC | MSCOCO Oriented AUROC |
| --- | ---: | ---: | ---: |
| object entropy delta | 0.722 | 0.470 | 0.530 |
| object target-lp delta | 0.718 | 0.452 | 0.548 |
| decision prob delta | 0.665 | 0.418 | 0.582 |
| content lp mean delta | 0.649 | 0.377 | 0.623 |

Full pairwise replay panel:

| Method / Backbone | Dataset | Base | Method | Pairwise RaPiC | dMethod | dBase | Fallback | H/G/Net | Hrec | Grec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PAI-attn / LLaVA-1.5 | MSCOCO | 85.22 | 83.99 | 85.78 | +1.79 | +0.56 | 455 | 308/147/161 | 71.63 | 46.08 |
| PAI-attn / LLaVA-1.5 | AOKVQA | 78.98 | 77.04 | 78.90 | +1.86 | -0.08 | 329 | 248/81/167 | 62.78 | 36.65 |
| PAI-attn / LLaVA-1.5 | GQA | 76.61 | 75.06 | 77.27 | +2.21 | +0.66 | 375 | 287/88/199 | 82.00 | 41.90 |
| VAF / LLaVA-1.5 | MSCOCO | 85.22 | 86.47 | 86.50 | +0.03 | +1.28 | 441 | 222/219/3 | 58.27 | 44.42 |
| VAF / LLaVA-1.5 | AOKVQA | 78.98 | 81.32 | 82.02 | +0.70 | +3.04 | 325 | 194/131/63 | 54.34 | 23.06 |
| VAF / LLaVA-1.5 | GQA | 76.61 | 80.58 | 80.82 | +0.24 | +4.21 | 256 | 139/117/22 | 47.12 | 17.94 |
| PAI-attn / LLaVA-NeXT | MSCOCO | 89.17 | 89.38 | 89.28 | -0.10 | +0.11 | 31 | 11/20/-9 | 26.83 | 33.33 |
| PAI-attn / LLaVA-NeXT | AOKVQA | 85.23 | 85.64 | 85.43 | -0.21 | +0.20 | 53 | 17/36/-19 | 38.64 | 44.44 |
| PAI-attn / LLaVA-NeXT | GQA | 83.19 | 83.67 | 83.51 | -0.16 | +0.32 | 52 | 19/33/-14 | 61.29 | 44.59 |
| VAF / LLaVA-NeXT | MSCOCO | 89.17 | 88.22 | 88.84 | +0.62 | -0.32 | 284 | 170/114/56 | 69.67 | 71.70 |
| VAF / LLaVA-NeXT | AOKVQA | 85.23 | 82.54 | 84.59 | +2.04 | -0.64 | 274 | 229/45/184 | 77.36 | 83.33 |
| VAF / LLaVA-NeXT | GQA | 83.19 | 80.16 | 82.66 | +2.50 | -0.53 | 297 | 261/36/225 | 79.09 | 63.16 |
| PAI-attn / Qwen2.5-VL-7B | MSCOCO | 83.77 | 83.79 | 83.77 | -0.02 | +0.00 | 40 | 19/21/-2 | 100.00 | 100.00 |
| PAI-attn / Qwen2.5-VL-7B | AOKVQA | 85.00 | 85.26 | 85.00 | -0.26 | +0.00 | 89 | 33/56/-23 | 100.00 | 100.00 |
| PAI-attn / Qwen2.5-VL-7B | GQA | 84.96 | 85.33 | 84.96 | -0.38 | +0.00 | 122 | 44/78/-34 | 100.00 | 100.00 |
| VAF / Qwen2.5-VL-7B | MSCOCO | 83.77 | 85.21 | 83.77 | -1.44 | +0.00 | 224 | 47/177/-130 | 100.00 | 100.00 |
| VAF / Qwen2.5-VL-7B | AOKVQA | 85.00 | 86.24 | 85.00 | -1.24 | +0.00 | 284 | 86/198/-112 | 100.00 | 100.00 |
| VAF / Qwen2.5-VL-7B | GQA | 84.96 | 87.03 | 84.96 | -2.08 | +0.00 | 345 | 79/266/-187 | 100.00 | 100.00 |
| VGA / LLaVA-1.5 | MSCOCO | 85.22 | 84.74 | 85.16 | +0.41 | -0.07 | 415 | 226/189/37 | 65.32 | 40.13 |
| VGA / LLaVA-1.5 | AOKVQA | 78.98 | 81.04 | 80.97 | -0.08 | +1.99 | 505 | 249/256/-7 | 78.06 | 50.69 |
| VGA / LLaVA-1.5 | GQA | 76.61 | 80.08 | 78.43 | -1.64 | +1.82 | 512 | 182/330/-148 | 70.82 | 58.00 |
| VGA / LLaVA-NeXT | MSCOCO | 89.17 | 89.68 | 89.84 | +0.17 | +0.68 | 119 | 67/52/15 | 47.18 | 27.66 |
| VGA / LLaVA-NeXT | AOKVQA | 85.23 | 85.96 | 86.39 | +0.43 | +1.16 | 117 | 78/39/39 | 47.56 | 17.03 |
| VGA / LLaVA-NeXT | GQA | 83.19 | 84.70 | 85.11 | +0.41 | +1.92 | 153 | 95/58/37 | 53.37 | 18.47 |
| VGA / Qwen2.5-VL-7B | MSCOCO | 83.77 | 82.89 | 83.68 | +0.79 | -0.09 | 81 | 76/5/71 | 71.03 | 17.86 |
| VGA / Qwen2.5-VL-7B | AOKVQA | 85.00 | 84.88 | 85.42 | +0.54 | +0.42 | 103 | 76/27/49 | 75.25 | 30.00 |
| VGA / Qwen2.5-VL-7B | GQA | 84.96 | 85.08 | 85.24 | +0.17 | +0.29 | 31 | 23/8/15 | 40.35 | 11.76 |

Interpretation:

- Pairwise replay is not a uniformly better main controller.
- It helps PAI-attn / LLaVA-1.5 and some VGA settings.
- It hurts PAI-attn / LLaVA-NeXT, Qwen PAI/VAF, and VGA / LLaVA-1.5 on
  AOKVQA/GQA.
- It improves VAF / LLaVA-NeXT over the intervention but still does not recover
  the baseline, so it does not solve the calibration mismatch.
- Therefore pairwise replay should be reported as an appendix diagnostic and
  possible future extension, not as the main RaPiC method.

## 3. Method: RaPiC

The Method section should be compact in the main paper and push feature details
to the appendix.

### 3.1 Problem formulation

Let:

- `f_0`: frozen baseline VLM
- `T`: intervention method, such as VGA, VAF, or PAI-attn
- `x`: image
- `q`: question
- `y_B = f_0(x, q)`: baseline answer
- `y_I = T(f_0, x, q)`: intervention answer

RaPiC outputs:

```text
y_R =
  y_B, if rho(x, q, y_B, y_I) = baseline
  y_I, otherwise.
```

The controller `rho` is calibrated on a discovery split and frozen for test.

Deployment note:

> RaPiC is intervention-first. The intervention answer is produced before the
> risk gate. In offline experiments we cache `y_B` for exact evaluation, but in
> deployment the baseline answer only needs to be generated when the fallback
> gate fires.

### 3.2 Candidate set

For discriminative yes/no experiments, RaPiC only considers fallback on changed
answers:

```text
C = {i : parse(y_B_i) != parse(y_I_i)}
```

Unchanged answers are routed to the intervention response because fallback
cannot change the parsed yes/no decision.

This should be stated because it prevents the method from looking like it
arbitrarily switches all samples.

### 3.3 Replay feature extraction

Single-candidate replay:

1. Teacher-force `y_I` under `f_0`.
2. Extract content-token likelihood, entropy, target-token margins, and binary
   yes/no decision features.
3. Build two feature groups:
   - C features: content/object support and confidence.
   - D features: decision-level yes/no support and margin.

Main text version:

> RaPiC does not train a new VLM. It reuses the frozen backbone as a replay
> model and extracts low-cost scalar features from the teacher-forced
> intervention answer.

Appendix detail:

- exact feature list;
- token selection rules;
- yes/no token parsing;
- z-score orientation and normalization;
- runtime backend differences for LLaVA-1.5, LLaVA-NeXT, and Qwen2.5-VL.

### 3.4 Feature orientation and scoring

Each scalar feature is oriented on discovery so that larger values predict
harm:

```text
z_f(i) = (orient_f(f_i) - mu_f) / sigma_f
```

Group scores:

```text
s_C(i) = mean_{f in C} z_f(i)
s_D(i) = mean_{f in D} z_f(i)
```

Policy families:

```text
c_only:    s(i) = s_C(i)
d_only:    s(i) = s_D(i)
cd_fusion: s(i) = (1 - alpha) s_C(i) + alpha s_D(i)
noop:      never fallback
```

Decision:

```text
rho(i) = baseline if s(i) >= tau
rho(i) = method otherwise
```

### 3.5 Transition-split calibration

Main method:

```text
fit separate policies for yes->no and no->yes
```

Rationale:

> The same replay score can correspond to different semantic risks depending on
> whether the intervention suppresses a positive answer or inserts a positive
> answer. We therefore calibrate separate controllers for each transition.

For each transition direction:

1. filter discovery changed rows to that direction;
2. orient features toward harm;
3. choose feature family, `alpha`, and `tau`;
4. allow `noop` if no policy improves over the intervention.

### 3.6 Calibration objective

Current main objective:

```text
maximize discovery final accuracy
```

Report additional operating statistics:

```text
Hrec = selected_harm / total_harm
Grec = selected_help / total_help
```

Paper-facing caution:

> Final accuracy alone can select aggressive fallback policies when the
> discovery changed set is highly harm-heavy. We therefore report H/G/Net,
> Hrec, and Grec for every policy, and we analyze cases where calibration
> degenerates into broad rollback.

Optional/appendix objective:

```text
J = Hrec - Grec
```

This is prior-invariant and equivalent to a ROC-style Youden criterion. Use it
for pairwise replay analysis or ablation, not necessarily the main method unless
the full panel supports it.

### 3.7 Generative RaPiC

For captioning, the intervention is VGA and the failure mode is object
hallucination. The generative RaPiC controller:

1. extracts candidate object mentions from the intervention caption;
2. scores object risk using a next-token yes/no object-presence probe;
3. suppresses risky object tokens during regeneration;
4. selects the risk threshold on validation;
5. applies the selected threshold unchanged to test.

Key paper-facing sentence:

> The threshold is selected on validation, not on test.

Current selected threshold:

```text
yp = 0.60
suppression mode = first_token
bias = -1.0
object vocabulary = COCO-80
```

## 4. Experiments

### 4.1 Experimental setup

Discriminative:

- Datasets: POPE-style MSCOCO, AOKVQA, GQA.
- Backbones: LLaVA-1.5-7B, LLaVA-NeXT-LLaMA3-8B, Qwen2.5-VL-7B.
- Interventions: VGA, VAF, PAI-attn.
- Calibration: discovery split only.
- Test: frozen policy on MSCOCO/AOKVQA/GQA.

Generative:

- Dataset/evaluation: COCO-CHAIR, 500 validation examples and 500 test examples.
- Baseline: LLaVA-1.5 vanilla captioning.
- Intervention: VGA captioning.
- RaPiC: validation-selected object-token suppression.

Mandatory protocol sentence:

> All main RaPiC policies are calibrated without test labels. Test-calibrated
> results are reported only as diagnostic upper bounds when used.

### 4.2 Main discriminative results

Recommended main table columns:

| Method / Backbone | Dataset | Base | Method | RaPiC | dMethod | dBase | Fallback | H/G/Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

Avoid putting too many weak/failure rows in the main table. Put the full matrix
in appendix. The main table should include:

- VGA across three backbones;
- LLaVA-1.5 PAI/VAF;
- selected Qwen cases if framed as conservative generalization;
- generative CHAIR table separately.

#### Current VGA main results

| Backbone | Dataset | Base | VGA | VGA+RaPiC | dVGA | dBase | Fallback | H/G/Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLaVA-1.5 | MSCOCO | 85.22 | 84.74 | 85.78 | +1.03 | +0.56 | 219 | 156/63/93 |
| LLaVA-1.5 | AOKVQA | 78.98 | 81.04 | 82.31 | +1.27 | +3.33 | 182 | 148/34/114 |
| LLaVA-1.5 | GQA | 76.61 | 80.08 | 80.83 | +0.76 | +4.22 | 146 | 107/39/68 |
| LLaVA-NeXT | MSCOCO | 89.17 | 89.68 | 89.84 | +0.17 | +0.68 | 23 | 19/4/15 |
| LLaVA-NeXT | AOKVQA | 85.23 | 85.96 | 86.54 | +0.59 | +1.31 | 69 | 61/8/53 |
| LLaVA-NeXT | GQA | 83.19 | 84.70 | 85.17 | +0.47 | +1.98 | 96 | 69/27/42 |
| Qwen2.5-VL-7B | MSCOCO | 83.77 | 82.89 | 83.79 | +0.90 | +0.02 | 125 | 103/22/81 |
| Qwen2.5-VL-7B | AOKVQA | 85.00 | 84.88 | 85.23 | +0.36 | +0.23 | 158 | 95/63/32 |
| Qwen2.5-VL-7B | GQA | 84.96 | 85.08 | 85.12 | +0.04 | +0.17 | 98 | 51/47/4 |

Interpretation:

- RaPiC consistently improves VGA over the always-on intervention.
- Gains are strongest when VGA introduces recoverable regressions.
- On Qwen, gains are smaller but mostly positive; this supports conservative
  transfer rather than a universal large-gain claim.

#### Current PAI/VAF main results

| Method / Backbone | Dataset | Base | Method | RaPiC | dMethod | dBase | Fallback | H/G/Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VAF / LLaVA-1.5 | MSCOCO | 85.22 | 86.47 | 86.59 | +0.12 | +1.37 | 37 | 24/13/11 |
| VAF / LLaVA-1.5 | AOKVQA | 78.98 | 81.32 | 81.58 | +0.26 | +2.60 | 37 | 30/7/23 |
| VAF / LLaVA-1.5 | GQA | 76.61 | 80.58 | 80.59 | +0.01 | +3.98 | 17 | 9/8/1 |
| PAI-attn / LLaVA-1.5 | MSCOCO | 85.22 | 83.99 | 85.53 | +1.54 | +0.31 | 377 | 258/119/139 |
| PAI-attn / LLaVA-1.5 | AOKVQA | 78.98 | 77.04 | 79.22 | +2.18 | +0.24 | 286 | 241/45/196 |
| PAI-attn / LLaVA-1.5 | GQA | 76.61 | 75.06 | 77.51 | +2.46 | +0.90 | 315 | 268/47/221 |
| PAI-attn / Qwen2.5-VL-7B | MSCOCO | 83.77 | 83.79 | 83.99 | +0.20 | +0.22 | 18 | 18/0/18 |
| PAI-attn / Qwen2.5-VL-7B | AOKVQA | 85.00 | 85.26 | 85.23 | -0.02 | +0.23 | 32 | 15/17/-2 |
| PAI-attn / Qwen2.5-VL-7B | GQA | 84.96 | 85.33 | 85.26 | -0.08 | +0.30 | 79 | 36/43/-7 |
| VAF / Qwen2.5-VL-7B | MSCOCO | 83.77 | 85.21 | 85.46 | +0.24 | +1.69 | 26 | 24/2/22 |
| VAF / Qwen2.5-VL-7B | AOKVQA | 85.00 | 86.24 | 86.21 | -0.03 | +1.21 | 27 | 12/15/-3 |
| VAF / Qwen2.5-VL-7B | GQA | 84.96 | 87.03 | 86.60 | -0.43 | +1.64 | 75 | 18/57/-39 |

Interpretation:

- LLaVA-1.5 PAI has large recoverable regressions; RaPiC recovers them and
  improves over baseline.
- LLaVA-1.5 VAF is already strong; RaPiC makes smaller, conservative gains.
- Qwen cases are mixed and should be framed as generalization/diagnostic rather
  than the headline claim.

#### LLaVA-NeXT PAI/VAF status

| Method / Backbone | Dataset | Base | Method | RaPiC | dMethod | dBase | Fallback | H/G/Net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VAF / LLaVA-NeXT | MSCOCO | 89.17 | 88.22 | 89.13 | +0.91 | -0.03 | 394 | 238/156/82 |
| VAF / LLaVA-NeXT | AOKVQA | 85.23 | 82.54 | 85.24 | +2.70 | +0.01 | 333 | 288/45/243 |
| VAF / LLaVA-NeXT | GQA | 83.19 | 80.16 | 83.10 | +2.94 | -0.09 | 373 | 319/54/265 |
| PAI-attn / LLaVA-NeXT | MSCOCO | 89.17 | 89.38 | 89.38 | +0.00 | +0.21 | 0 | 0/0/0 |
| PAI-attn / LLaVA-NeXT | AOKVQA | 85.23 | 85.64 | 85.64 | +0.00 | +0.41 | 0 | 0/0/0 |
| PAI-attn / LLaVA-NeXT | GQA | 83.19 | 83.67 | 83.67 | +0.00 | +0.48 | 0 | 0/0/0 |

Interpretation:

- PAI-attn / LLaVA-NeXT is a clean abstention case: the intervention already
  improves the baseline, and discovery calibration selects `noop`.
- VAF / LLaVA-NeXT recovers much of a degraded intervention, but the policy is
  broad and captures many helpful cases. This should be failure analysis unless
  pairwise panel results improve it consistently.

### 4.3 Generative CHAIR results

This is a strong main table because the threshold is validation-selected.

| Backbone | Method | CHAIRs down | CHAIRi down | Recall up | Precision up | F1 up |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-1.5-7B | Vanilla | 56.6 | 15.30 | 80.41 | 73.09 | 76.57 |
| LLaVA-1.5-7B | VGA | 32.2 | 8.90 | 72.81 | 82.15 | 77.20 |
| LLaVA-1.5-7B | RaPiC | 30.8 | 8.28 | 72.68 | 83.23 | 77.60 |

Validation selection:

| Method | Split | CHAIRs | CHAIRi | Recall | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VGA | val | 31.4 | 8.90 | 72.55 | 82.71 | 77.30 |
| RaPiC, yp=0.60 | val | 29.6 | 8.26 | 72.81 | 83.61 | 77.84 |
| VGA | test | 32.2 | 8.90 | 72.81 | 82.15 | 77.20 |
| RaPiC, yp=0.60 | test | 30.8 | 8.28 | 72.68 | 83.23 | 77.60 |

Text:

> RaPiC improves both CHAIRs and CHAIRi over VGA while preserving recall. The
> threshold is chosen on validation (`yp=0.60`) and transferred unchanged to the
> test split, avoiding test-time threshold selection.

### 4.4 Ablations

Recommended ablations:

1. Pooled changed-answer calibration.
2. Transition-split calibration.
3. Transition-split with `noop`.
4. Pairwise replay delta panel, if full results are favorable.
5. Test-calibrated oracle, clearly labeled diagnostic only.

Table:

| Variant | Dataset | RaPiC | dMethod | Fallback | H/G/Net | Grec |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pooled changed | fill | fill | fill | fill | fill | fill |
| transition split | fill | fill | fill | fill | fill | fill |
| pairwise replay | fill | fill | fill | fill | fill | fill |
| test-calib oracle | fill | fill | fill | fill | fill | fill |

The point is to show:

- transition split matters;
- `noop` prevents unnecessary fallback;
- pairwise replay may increase evidence strength, but only if it transfers;
- test-calib is an upper bound, not the main result.

### 4.5 Failure analysis

Use LLaVA-NeXT VAF and Qwen VAF/PAI as honest failure/boundary cases.

Main failure statement:

> RaPiC can fail when the discovery changed set is not representative of the
> test harm/help prior within the same transition. In this case, a policy that
> looks optimal on discovery can become a broad rollback policy on test.

Concrete LLaVA-NeXT VAF:

```text
discovery no->yes: 118 harm / 17 help
MSCOCO no->yes:    244 harm / 159 help
```

Pairwise replay diagnostic:

- Test-local pairwise features show local separability.
- Discovery-selected pairwise feature directions do not transfer to MSCOCO.
- Therefore the issue is not only score normalization; feature semantics are
  non-stationary.

Avoid saying:

```text
RaPiC solves this failure.
```

Say instead:

```text
This failure mode motivates stronger visual evidence verifiers or
effect-balanced calibration splits.
```

## 5. Appendix Plan

Appendix A: Method details

- replay feature list;
- feature orientation;
- C/D grouping;
- alpha grid;
- threshold sweep;
- `noop` selection;
- transition split implementation.

Appendix B: Dataset and calibration details

- discovery split construction;
- full POPE file paths;
- baseline and intervention prediction sources;
- LLaVA/Qwen/NeXT runtime settings;
- parsing rules.

Appendix C: Full discriminative tables

- all method/backbone/dataset rows;
- source-calib results;
- transition-split results;
- yes->no-only diagnostic;
- test-calib oracle.

Appendix D: Pairwise replay diagnostics

- pairwise feature AUROCs;
- discovery-to-test feature transfer table;
- full pairwise panel when available;
- object/category audits.

Appendix E: Generative CHAIR details

- validation sweep table;
- selected threshold justification;
- object vocabulary;
- suppression mode;
- exact CHAIR json/csv artifacts.

Appendix F: Runtime and cost

- intervention-first deployment;
- cached baseline in offline evaluation;
- expected fallback rate;
- additional replay cost;
- generative regeneration cost.

## 6. Table Selection Strategy

Main paper should not include every result. Use tables to support the story:

1. **Analysis table:** interventions create both harm and help.
2. **Main discriminative table:** RaPiC improves stable intervention/backbone
   settings.
3. **Generative table:** validation-calibrated CHAIR improvement.
4. **Failure table:** one compact table showing calibration mismatch.

Full result matrix goes to appendix.

## 7. Claims to Avoid

Avoid:

```text
RaPiC improves every intervention.
RaPiC always distinguishes harm from help.
Pairwise replay solves the LLaVA-NeXT VAF failure.
Discovery subset was chosen to match test behavior.
```

Use:

```text
RaPiC improves interventions when replay-risk features identify recoverable
regressions.
RaPiC can abstain when the intervention is already preferable.
Calibration representativeness is necessary; mismatch is detectable through
transition and help-capture diagnostics.
Pairwise replay exposes additional evidence but requires stable calibration.
```

## 8. Immediate Writing Tasks

1. Freeze the main discriminative table around transition-split RaPiC.
2. Keep pairwise replay as an appendix diagnostic rather than the main
   controller, because the full panel is mixed.
3. Write Analysis section first using harm/help/transition statistics.
4. Write Method with transition-split RaPiC as the main discriminative protocol.
5. Add generative CHAIR as a clean validation-calibrated result.
6. Move all test-calibrated, oracle, and pairwise-delta variants to appendix
   diagnostics unless a later version adds an explicit stability criterion.

## 9. Current Recommendation

Given the completed pairwise panel, the safest main method is:

```text
Transition-split RaPiC with intervention-answer replay features.
```

The safest main empirical story is:

```text
VGA is the broadest discriminative success.
LLaVA-1.5 PAI/VAF show method generality.
Generative CHAIR gives a clean validation-calibrated improvement.
NeXT/Qwen failure cases are analyzed as calibration mismatch and abstention
boundaries.
```

Pairwise only helps selected cases, so keep it in Analysis/Appendix:

```text
Pairwise replay provides evidence that stronger visual support signals can
separate harm/help locally, but current discovery-to-test feature transfer is
not stable enough for the main controller.
```
