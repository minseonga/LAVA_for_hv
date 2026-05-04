# RaPiC Method Section Draft

This document is a method-only draft for the RaPiC paper. It intentionally
excludes the full experimental narrative and non-deployable diagnostic variants.
The goal is to provide a clean, defensible method section that can be inserted
into a NeurIPS-style paper, with detailed appendix-ready implementation notes.

## Method Overview

We propose **RaPiC**, a replay-aware post-intervention calibration framework for
vision-language hallucination mitigation methods. RaPiC is designed for the
common setting in which a frozen backbone model is paired with an intervention
method, such as visual contrastive decoding, visual attention modulation, or
attention-based suppression. These interventions are usually applied uniformly
to all inputs. RaPiC instead treats the intervention output as a candidate and
selectively falls back to the baseline output only when a calibrated replay
controller predicts that the intervention has likely introduced a regression.

The central design principle is:

> The mitigation method should produce the candidate answer, but a separate
> calibration layer should decide whether that candidate is reliable for the
> current sample.

RaPiC has two instantiations:

1. **Discriminative RaPiC**, for yes/no object-presence benchmarks such as
   POPE-style MSCOCO, AOKVQA, and GQA.
2. **Generative RaPiC**, for caption generation under COCO-CHAIR evaluation,
   where the controller suppresses risky object tokens and regenerates the
   caption.

Both instantiations share the same structure:

1. generate an intervention output;
2. replay the candidate output under the frozen backbone;
3. extract scalar support features;
4. choose a calibrated operating point on a held-out calibration split;
5. freeze the policy and apply it unchanged to test.

RaPiC does not train or fine-tune a new VLM. It calibrates a lightweight
post-hoc routing or suppression policy using replay features from the original
backbone.

## 1. Problem Setup

Let `f_0` denote a frozen baseline vision-language model. Given an image `x`
and a textual query `q`, the baseline model produces:

```text
y_B = f_0(x, q).
```

Let `T` denote an intervention method applied to the same backbone. The
intervention produces:

```text
y_I = T(f_0, x, q).
```

The goal is not to replace the intervention method. Instead, RaPiC wraps the
intervention with a calibrated post-hoc decision rule:

```text
y_R =
  y_B, if rho(x, q, y_B, y_I) = baseline
  y_I, otherwise.
```

Here `rho` is a lightweight controller calibrated on a discovery split and then
held fixed for test.

For discriminative yes/no tasks, `y_B`, `y_I`, and `y_R` are parsed into binary
labels. For generative captioning, `y_I` is a caption, and RaPiC modifies the
generation process by suppressing selected object tokens before producing the
final caption.

### Harm and Help

On a calibration set with ground-truth labels `g_i`, we define:

```text
b_i = 1[parse(y_B_i) = g_i]
m_i = 1[parse(y_I_i) = g_i]
```

An intervention change is:

```text
harm: b_i = 1 and m_i = 0
help: b_i = 0 and m_i = 1
neutral: b_i = m_i
```

RaPiC falls back to `y_B` for selected samples. Therefore:

- selecting a harm sample improves final accuracy;
- selecting a help sample hurts final accuracy;
- selecting a neutral sample does not change parsed accuracy.

For any selected fallback set `S`, the net gain in number of correct examples is:

```text
Net(S) = #harm(S) - #help(S).
```

On a test set of size `N`, the final full-set accuracy is:

```text
Acc_R = Acc_I + Net(S) / N.
```

This identity is useful because it makes the role of fallback explicit: RaPiC is
valuable only when selected fallbacks are enriched for intervention-induced
harm relative to intervention-induced help.

## 2. Discriminative RaPiC

Discriminative RaPiC is used for yes/no hallucination benchmarks. It receives a
baseline prediction file, an intervention prediction file, a question file, and
a ground-truth file for calibration. At test time, the calibrated controller is
applied to replay feature rows without using test labels.

### 2.1 Candidate Set

RaPiC only considers fallback for samples where the parsed baseline and
intervention answers differ:

```text
C = {i : parse(y_B_i) != parse(y_I_i)}.
```

If the parsed answers are identical, fallback cannot change the final yes/no
decision. Such samples are routed to the intervention output by default.

This candidate restriction is important for both accuracy and interpretation:
RaPiC is not an arbitrary ensemble over all examples. It is a post-intervention
controller over samples where the intervention actually changes the decision.

### 2.2 Replay Features

For each candidate intervention answer `y_I`, RaPiC teacher-forces the answer
under the frozen baseline model `f_0` and extracts scalar features from token
logits and probabilities.

The replay operation is:

```text
R_I = Replay(f_0, x, q, y_I).
```

From `R_I`, RaPiC extracts two groups of features.

**Content-support features (`C` features):**

These measure whether the content tokens of the intervention answer are
supported by the frozen model.

Representative features include:

```text
cheap_lp_content_min
cheap_lp_content_std
cheap_entropy_content_mean
cheap_first_target_gap
cheap_target_gap_content_min
```

They summarize token likelihood, entropy, and target-token margins over content
tokens.

**Decision-support features (`D` features):**

These measure whether the binary yes/no decision itself is supported.

Representative features include:

```text
cheap_decision_candidate_minus_alt
cheap_decision_candidate_prob_binary
cheap_decision_candidate_label_lp
cheap_decision_candidate_kl_uniform
```

They summarize the probability and margin of the candidate binary decision
relative to the alternative label.

The exact feature list is fixed before test application. The calibration
procedure may select a subset of features based on discovery-set AUROC, but test
labels are never used for feature selection in the deployed protocol.

### 2.3 Feature Orientation

Each feature is oriented on the discovery split so that larger values indicate
higher predicted intervention harm.

For feature `f`, let `a_f` be an orientation sign:

```text
a_f in {+1, -1}.
```

The sign is selected on calibration data to maximize harm separation. The
oriented feature is:

```text
u_f(i) = a_f * f_i.
```

RaPiC then standardizes the oriented feature using calibration statistics:

```text
z_f(i) = (u_f(i) - mu_f) / sigma_f.
```

The calibration mean `mu_f` and standard deviation `sigma_f` are saved in the
policy bundle and reused unchanged at test time.

Features with insufficient present rate or insufficient calibration AUROC are
discarded. In the current protocol:

```text
min_present_rate = 0.8
min_feature_auroc = 0.55
```

### 2.4 C/D Policy Families

After feature orientation and standardization, RaPiC forms group scores:

```text
s_C(i) = mean_{f in F_C} z_f(i)
s_D(i) = mean_{f in F_D} z_f(i)
```

where `F_C` and `F_D` are the selected content-support and decision-support
feature subsets.

RaPiC searches over three policy families:

```text
c_only:
  s(i) = s_C(i)

d_only:
  s(i) = s_D(i)

cd_fusion:
  s(i) = (1 - alpha) * s_C(i) + alpha * s_D(i)
```

The fusion weight `alpha` is selected from a fixed grid. The dense grid used in
recent experiments is:

```text
alpha in {0.025, 0.050, ..., 0.975}
```

A threshold `tau` converts the score into a route:

```text
rho(i) = baseline, if s(i) >= tau
rho(i) = method, otherwise.
```

The policy bundle stores:

- selected family;
- selected feature subsets;
- feature orientations;
- calibration means and standard deviations;
- `alpha`;
- `tau`;
- candidate filter;
- operating-point statistics.

### 2.5 Transition-Split Calibration

The same replay score can have different semantics depending on the direction
of the intervention change. A `yes->no` change suppresses a positive answer,
while a `no->yes` change inserts a positive answer. Each direction can contain
both harm and help:

```text
yes->no:
  harm when the object is present and the intervention says no
  help when the object is absent and the intervention says no

no->yes:
  harm when the object is absent and the intervention says yes
  help when the object is present and the intervention says yes
```

RaPiC therefore calibrates separate policies for the two directions:

```text
rho_yes_to_no
rho_no_to_yes
```

For each transition direction `d`, the calibration set is:

```text
C_d = {i in C : transition(i) = d}.
```

Feature orientation, feature selection, family selection, and threshold
selection are performed independently for each `C_d`.

At test time, RaPiC applies the corresponding policy for the observed
transition:

```text
if transition(i) = yes->no:
  route with rho_yes_to_no
elif transition(i) = no->yes:
  route with rho_no_to_yes
else:
  route to method
```

The merged route is baseline if either direction-specific policy selects
fallback for the matching transition.

### 2.6 Help-Preservation Constraint

Maximizing final accuracy on a harm-heavy discovery split can select overly
aggressive fallback policies. Such policies may catch many harm samples, but
they may also route many helpful intervention corrections back to the baseline.

To prevent broad rollback, RaPiC can impose a help-preservation constraint
during calibration:

```text
Grec(S) = #selected_help / #total_help <= gamma.
```

Here `gamma` is a pre-specified maximum helpful-change recall. Candidate
thresholds that violate this constraint are rejected.

Paper-facing phrasing:

> We calibrate fallback under a help-preservation constraint, rejecting
> operating points that would route more than `gamma` of helpful intervention
> changes back to the baseline on the discovery split.

This is not a new method variant. It is a constraint on the same
transition-split calibration procedure. It is useful when an intervention
produces many helpful changes and the unconstrained controller tends to become
a broad rollback policy.

Recommended sweep values:

```text
gamma in {0.20, 0.30, 0.40}
```

The current default candidate for paper-facing constrained calibration is:

```text
gamma = 0.30
```

The constraint is applied independently within each transition direction. If no
non-noop operating point satisfies the objective and constraints, RaPiC may
select `noop`.

### 2.7 Noop Policy

RaPiC includes a `noop` policy:

```text
rho_noop(i) = method for all i.
```

This policy is selected when no calibrated fallback operating point improves
the calibration objective under the required constraints.

The `noop` option is important for two reasons:

1. It prevents unnecessary fallback when the intervention is already reliable.
2. It turns calibration failures into abstentions rather than forced routing.

In the paper, `noop` should be described as a built-in abstention mechanism,
not as a post-hoc exception.

### 2.8 Calibration Objective

For each transition direction and policy family, RaPiC sweeps thresholds over
the calibration scores. The primary objective is final calibration accuracy:

```text
maximize Acc_R on discovery
```

subject to route constraints such as:

```text
min_selected_count
min_harm_precision
min_harm_recall
max_help_recall
min_baseline_rate
max_baseline_rate
```

The current main protocol uses:

```text
tau_objective = final_acc
min_selected_count = 5
allow_noop_policy = true
```

When the help-preservation constraint is enabled:

```text
max_help_recall = gamma
```

For reporting, RaPiC always includes:

```text
selected_harm
selected_help
net = selected_harm - selected_help
Hrec = selected_harm / total_harm
Grec = selected_help / total_help
fallback_count
```

These statistics are essential because final accuracy alone can obscure broad
rollback behavior.

### 2.9 Discriminative Algorithm

Algorithm 1 gives the calibration procedure.

```text
Algorithm 1: Calibrating Discriminative RaPiC

Input:
  discovery examples D
  frozen backbone f_0
  intervention predictions y_I
  baseline predictions y_B
  ground-truth labels g
  feature groups C and D
  alpha grid A
  help-preservation cap gamma

1. Build changed candidate set:
     C_all = {i : parse(y_B_i) != parse(y_I_i)}

2. For each direction d in {yes->no, no->yes}:
     C_d = {i in C_all : transition(i) = d}

3. For each i in C_d:
     replay y_I_i under f_0
     extract replay features
     assign harm/help label using y_B_i, y_I_i, and g_i

4. For each feature:
     choose orientation toward harm
     compute calibration mean and std
     discard weak or sparse features

5. For each policy family in {c_only, d_only, cd_fusion}:
     for each alpha in A:
       compute score s(i)
       sweep tau
       reject operating points violating constraints
       evaluate final discovery accuracy

6. Compare best non-noop policies with noop.

7. Save one selected policy for direction d.

Output:
  policy bundle {rho_yes_to_no, rho_no_to_yes}
```

Algorithm 2 gives test-time application.

```text
Algorithm 2: Applying Discriminative RaPiC

Input:
  test example (x, q)
  frozen backbone f_0
  intervention method T
  calibrated policies rho_yes_to_no, rho_no_to_yes

1. Generate intervention answer:
     y_I = T(f_0, x, q)

2. Obtain baseline answer:
     y_B = f_0(x, q)
   In offline evaluation y_B may be cached.
   In deployment y_B can be generated only when fallback is needed, depending
   on system design.

3. If parse(y_B) = parse(y_I):
     return y_I

4. Replay y_I under f_0 and compute score using the matching transition policy.

5. If score exceeds tau for the matching transition:
     return y_B
   else:
     return y_I
```

Deployment note:

The experiments cache baseline predictions to evaluate exact full-set routing.
This does not require the conceptual method to be baseline-first. RaPiC is
intervention-first: the intervention candidate is generated first, and fallback
is triggered only by the calibrated risk policy.

## 3. Generative RaPiC

Generative RaPiC applies the same calibration principle to captioning. Instead
of choosing between a baseline caption and an intervention caption, it uses a
calibrated object-risk threshold to decide which object tokens in an
intervention caption should be suppressed during regeneration.

### 3.1 Generative Setup

Let:

```text
c_I = T(f_0, x)
```

be an intervention caption, such as a VGA-generated caption.

The goal is to reduce object hallucination while preserving useful visual
content. The evaluation metric is COCO-CHAIR, which reports:

```text
CHAIRs: fraction of captions with hallucinated objects
CHAIRi: fraction of object mentions that are hallucinated
Recall
Precision
F1
```

RaPiC targets object hallucination at the token level.

### 3.2 Candidate Object Extraction

For each intervention caption, RaPiC extracts object mentions and maps them to
a fixed object vocabulary. The current paper-facing setting uses:

```text
object vocabulary = COCO-80
max objects per caption = 8
```

Only vocabulary-matched object mentions are considered for suppression. This
keeps the controller deployable and avoids arbitrary free-form object rules.

### 3.3 Object-Presence Risk Probe

For each candidate object `o`, RaPiC constructs an object-presence query:

```text
Is there a {o} in the image?
```

The frozen model is probed with a next-token yes/no decision. Let:

```text
p_yes(o | x)
```

be the model's probability of answering yes to the object-presence probe.

Objects with low `p_yes` are treated as high-risk hallucination candidates. The
controller uses a threshold:

```text
mark o as risky if p_yes(o | x) <= yp.
```

The threshold `yp` is selected on validation and applied unchanged to test.

Current selected threshold:

```text
yp = 0.60
```

### 3.4 Token Suppression and Regeneration

For each risky object, RaPiC suppresses the corresponding object token during
caption regeneration. The current deployable setting is:

```text
suppression mode = first_token
bias = -1.0
use_add = true
max_gen_len = 512
```

The final caption is generated with the same intervention method, but with
logit bias applied to risky object tokens.

The method does not remove words by string post-processing. It changes the
generation distribution and lets the model produce a new caption under the
suppression constraint.

### 3.5 Validation Calibration

Generative RaPiC selects `yp` using validation CHAIR metrics. The preferred
selection objective is:

```text
minimize CHAIRi
subject to recall drop <= epsilon
and no increase in CHAIRs when required.
```

In the current validation-to-test protocol:

```text
threshold candidates = {0.20, 0.25, ..., 0.60}
selected yp = 0.60
```

Paper-facing sentence:

> The object-risk threshold is selected on validation and transferred unchanged
> to the test split.

This sentence should be repeated in the experiment setup because it separates
RaPiC from test-time threshold tuning.

### 3.6 Generative Algorithm

```text
Algorithm 3: Generative RaPiC with Object-Token Suppression

Input:
  image x
  frozen backbone f_0
  intervention captioning method T
  object vocabulary V
  validation-selected threshold yp
  suppression bias beta

1. Generate intervention caption:
     c_I = T(f_0, x)

2. Extract object mentions:
     O = ExtractObjects(c_I) intersect V

3. For each object o in O:
     query f_0 with "Is there a {o} in the image?"
     compute p_yes(o | x)

4. Risk set:
     R = {o in O : p_yes(o | x) <= yp}

5. Regenerate caption with intervention method T while applying logit bias
   beta to the first token of each object in R.

Output:
  regenerated caption c_R
```

## 4. What Is Not Part of the Main Method

The current method should be kept narrow and clean. The following variants are
diagnostics, not main-method components.

### Yes-to-No-Only Gate

Restricting fallback to `yes->no` changes can reduce helpful `no->yes`
fallbacks, but it changes the method's semantic scope and does not address
settings where harm also occurs in `no->yes`. The main method instead uses
transition-split calibration with optional help-preservation constraints.

### Rate Cap

A rate cap limits the number of fallback routes directly. It is useful as a
diagnostic, but the main method should prefer the help-preservation constraint
because it is tied to the calibration objective rather than an arbitrary route
budget.

## 5. Implementation Details for Appendix

This section can be moved to an appendix.

### 5.1 Discriminative Inputs and Outputs

Calibration inputs:

```text
question_jsonl
gt_csv
baseline_pred_jsonl
intervention_pred_jsonl
changed_q_with_object.jsonl
changed_gt.csv
online_feature_rows.csv
```

Policy outputs:

```text
selected_policy.json
summary.json
c_feature_metrics.csv
d_feature_metrics.csv
tau_sweep.csv
```

Application outputs:

```text
pcp_route_rows.csv
deployment_summary.json
```

### 5.2 Canonical Discriminative Protocol

The canonical protocol for paper tables should be:

```text
1. Generate baseline and intervention predictions.
2. Build changed-answer subset on discovery.
3. Extract intervention-answer replay features on discovery.
4. Fit transition-split RaPiC policies on discovery only.
5. Build changed-answer subsets for each test dataset.
6. Extract intervention-answer replay features on test changed subsets.
7. Apply frozen transition policies to test feature rows.
8. Summarize full 9000-example deployment accuracy using deployment_summary.json.
```

Do not mix this with:

```text
yes->no-only diagnostic
rate-cap diagnostic
changed-set-only accuracy
```

### 5.3 Current Feature Columns

Current content-support columns:

```text
cheap_lp_content_min
cheap_lp_content_std
cheap_entropy_content_mean
cheap_first_target_gap
cheap_target_gap_content_min
```

Current decision-support columns:

```text
cheap_decision_candidate_minus_alt
cheap_decision_candidate_prob_binary
cheap_decision_candidate_label_lp
cheap_decision_candidate_kl_uniform
```

Current controller search:

```text
min_present_rate = 0.8
min_feature_auroc = 0.55
top_k_c = 3
top_k_d = 4
tau_objective = final_acc
min_selected_count = 5
allow_noop_policy = true
```

Dense alpha grid:

```text
alpha = 0.025, 0.050, ..., 0.975
```

Help-preservation candidates:

```text
max_help_recall gamma in {0.20, 0.30, 0.40}
```

### 5.4 Reporting Fields

Every discriminative result table should report:

```text
Base
Method
RaPiC
dMethod = RaPiC - Method
dBase = RaPiC - Base
Fallback
H/G/Net
Hrec
Grec
```

The `Base`, `Method`, and `RaPiC` columns must be full-set accuracies, not
changed-subset accuracies.

Changed-subset diagnostic tables must be explicitly labeled as changed-set
diagnostics and kept separate from full-test deployment tables.

### 5.5 Deployment Cost

RaPiC adds replay feature extraction for candidate changed rows. In offline
evaluation, the baseline prediction is cached for all examples. In deployment,
several implementations are possible:

1. **Cached baseline mode:** generate baseline and intervention answers, then
   route.
2. **Intervention-first fallback mode:** generate intervention answer first,
   compute replay risk, and generate baseline only when the fallback gate fires.
3. **Batch auditing mode:** apply RaPiC offline to already-generated baseline
   and intervention outputs.

The paper should state that offline evaluation uses cached baseline predictions
for reproducibility, while the method itself is compatible with
intervention-first deployment.

## 6. Paper-Ready Method Text

The following text can be adapted directly into the main paper.

> We introduce RaPiC, a replay-aware post-intervention calibration framework
> that makes hallucination mitigation methods selective at the instance level.
> Given a frozen backbone `f_0`, an input image-question pair `(x, q)`, and an
> intervention method `T`, we first obtain a baseline answer `y_B = f_0(x, q)`
> and an intervention answer `y_I = T(f_0, x, q)`. RaPiC then decides whether
> to accept `y_I` or fall back to `y_B`. The controller is calibrated on a
> held-out discovery split and frozen for test.

> For discriminative yes/no tasks, RaPiC only considers fallback when the parsed
> baseline and intervention answers differ. We teacher-force the intervention
> answer under the frozen backbone and extract scalar replay features measuring
> content support and binary decision support. Each feature is oriented on
> discovery so that larger values indicate higher intervention risk, then
> standardized using discovery statistics. RaPiC searches over content-only,
> decision-only, and content-decision fusion policies and selects a threshold
> that maximizes discovery final accuracy subject to operating constraints.

> Because `yes->no` and `no->yes` changes represent different semantic failure
> modes, RaPiC calibrates separate controllers for the two transition
> directions. We further include a `noop` policy, allowing RaPiC to abstain from
> fallback when no calibrated operating point improves over the intervention.
> To prevent broad rollback of helpful intervention corrections, we optionally
> impose a help-preservation constraint that rejects thresholds whose selected
> helpful-change recall exceeds a pre-specified cap.

> For generative captioning, RaPiC applies the same calibration principle to
> object-token suppression. It extracts object mentions from the intervention
> caption, probes the frozen model with next-token yes/no object-presence
> questions, and suppresses risky object tokens during regeneration. The object
> risk threshold is selected on validation and transferred unchanged to test.

## 7. Method Claims to Keep

Use these claims:

```text
RaPiC is a calibration layer for intervention methods.
RaPiC is post-intervention and model-agnostic given prediction/replay hooks.
RaPiC uses frozen-backbone replay features, not new VLM training.
RaPiC calibrates separate controllers for yes->no and no->yes transitions.
RaPiC can abstain through noop when fallback is not beneficial.
Help-preservation constraints reduce broad rollback of helpful corrections.
Generative RaPiC uses validation-selected object-token suppression.
```

Avoid these claims:

```text
RaPiC always improves every intervention.
RaPiC perfectly distinguishes harm from help.
RaPiC uses test labels for deployment.
Changed-set accuracy is full-test accuracy.
```
