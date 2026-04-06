# Analysis-Driven Post-Hoc Control for Multimodal Interventions

Draft manuscript outline.
This draft reflects the current mainline choice:

- `Discriminative`: sample-wise post-hoc `meta` controller
- `Generative`: constrained shallow `tree` controller

It is written as a paper draft, not as a final submission-ready manuscript.
Numbers below should be treated as draft values unless explicitly replaced by held-out results.


## Title Options

1. Analysis-Driven Post-Hoc Control for Harmful Multimodal Interventions
2. Detecting and Routing Harmful Intervention Cases in Multimodal Models
3. Analysis-Driven Control of Sample-Level Intervention Susceptibility


## Abstract

Always-on multimodal interventions are unsafe: average gains hide sample-level cases in which the baseline model would have been preferable. We study intervention-first post-hoc control: after an intervention output has already been produced, can we decide whether to keep it or fall back to the baseline using only ground-truth-free signals extracted from that output?

Our framework shares four invariants across regimes: intervention-first execution, ground-truth-free post-hoc observables, discovery-time fallback supervision, and frozen held-out application. What differs is how harmful intervention manifests. In discriminative question answering, complementary verifier-style and output-side views support a sample-wise meta-controller that improves over always-on intervention on held-out data. In generative captioning, the key issue is a hallucination-coverage trade-off rather than hallucination alone, so we use a constrained fallback target and a shallow decision tree instead of the discriminative meta-controller.

The main conclusion is not that one raw feature or one controller solves every intervention. Rather, a shared analysis-driven post-hoc framework can be instantiated with regime-specific observables and controller classes, yielding strong discriminative gains and a generative controller that recovers part of the available constrained fallback headroom.


## 1. Introduction

Decoding-time interventions for multimodal large language models are attractive because they do not require retraining and can significantly improve average performance. However, these interventions are not uniformly beneficial. On some samples, the intervention helps; on others, it causes a regression relative to the baseline model.

This sample-level variability creates a practical control problem. The intervention should not be used unconditionally if it is known to hurt on a non-trivial subset of inputs. At the same time, one cannot assume access to ground truth at deployment time, so any controller must rely on test-time observable signals alone.

This paper studies that problem through an analysis-driven lens. Rather than proposing a single universal feature or a single universal controller, we first analyze how harmful intervention cases manifest in different regimes, then derive regime-appropriate post-hoc controllers from that analysis. The resulting framework is shared, but the observable signal and the best controller instantiation differ across regimes.

Our setting is intentionally practical. For each sample, we already have an intervention output. The controller decides whether to keep that output or to fall back to the baseline output. This is post-hoc control, not pre-decode gating. We focus on whether harmful intervention cases can be detected from the produced output itself.


## 2. Problem Setup

Let `x` denote an input sample, consisting of an image and either a question (`discriminative`) or an instruction to describe the image (`generative`).

Let:

- `y_base(x)` be the baseline model output
- `y_int(x)` be the intervention model output

Our post-hoc controller chooses a final route:

- keep `y_int(x)`, or
- fall back to `y_base(x)`

Formally, the controller outputs:

`route(x) in {method, baseline}`

and the final output is:

`y_final(x) = y_int(x)` if `route(x)=method`, else `y_base(x)`.

The central question is:

> Can we predict when baseline fallback is preferable using only post-hoc signals extracted from the produced output, without access to test-time ground truth?

We evaluate this question in two regimes.

### 2.1 Discriminative regime

The task is answer prediction. The natural fallback target is a harmful regression:

- `regression`: baseline correct, intervention wrong
- `improvement`: baseline wrong, intervention correct
- `neutral`: both correct or both wrong

The controller should ideally route regressions to baseline while leaving improvements untouched.

### 2.2 Generative regime

The task is image captioning. Here the notion of “better” is multi-objective. An intervention can reduce hallucination yet omit supported content, or preserve content while increasing hallucination. We therefore do not use a single custom composite metric as the main target. Instead, we report:

- `CHAIRs`
- `CHAIRi`
- `Recall`
- `Precision`
- `F1`

and define a constrained fallback target for controller learning. The key question becomes:

> Can we fall back to baseline only on samples where doing so preserves or improves hallucination metrics while improving caption completeness?


## 3. Shared Invariants

The unifying claim of the paper is not a universal raw feature and not a universal controller class. The unifying claim is a shared procedure. Across both regimes, the method is defined by four invariants.

### 3.1 Invariant 1: Intervention-first execution

We always start from an already-produced intervention output `y_int(x)`. The controller is therefore post-hoc rather than pre-decode. It decides whether to keep `y_int(x)` or replace it with `y_base(x)`.

This matters because the controller is only allowed to use information that becomes available after intervention generation.

### 3.2 Invariant 2: Ground-truth-free post-hoc observables

At test time, the controller only sees observables extracted from the produced output and the model-image context. It does not use reference answers, task labels, or any fallback supervision target at inference time.

The shared latent concept is:

`intervention-induced support mismatch`

That is, harmful intervention outputs leave traces suggesting that the produced output is weakly supported by the model-image-output relationship. What changes across regimes is the measurable manifestation of this mismatch.

### 3.3 Invariant 3: Discovery-time fallback supervision

We do not directly fit the controller against raw task labels in a naive way. Instead, on discovery data we define a fallback supervision target: a label indicating when baseline routing would have been preferable.

- In `discriminative`, this target is regression-oriented.
- In `generative`, this target is a constrained fallback teacher reflecting the desired hallucination-coverage trade-off.

This supervision target is used only during controller fitting on discovery. It is never queried at test time.

### 3.4 Invariant 4: Frozen held-out apply

After discovery-time fitting, the controller is frozen. Held-out evaluation follows a fixed protocol:

1. generate or load baseline and intervention outputs
2. compute the same post-hoc observables
3. apply the frozen controller
4. report routed performance

No held-out retuning is allowed.

### 3.5 What varies across regimes

Within these invariants, three objects are regime-specific:

- the fallback supervision target
- the measurable post-hoc observable family
- the best shallow controller instantiation

This is why the paper should claim a shared framework with regime-specific realizations, rather than a single universal feature set or a single identical controller.


## 4. Discriminative: Expert Arbitration over Complementary Post-Hoc Views

### 4.1 Signals

We use two complementary post-hoc signal families.

#### B-family: richer verifier-style risk

This family is derived from replay- or verifier-style signals that indicate whether the intervention output is weakly supported under image-conditioned evaluation. In our current implementation, this includes `Stage-A` / `Stage-B` style signals extracted from replay-based evaluation.

This family is intended to act as a coarse `high-risk` detector.

#### C-family: cheap output-side risk

This family uses cheap output-side features such as low-margin, low-gap, or instability-based indicators computed from the produced intervention answer.

This family acts as a finer-grained surface-level risk detector.

### 4.2 Expert decomposition

We define three experts:

- `B-only`
- `C-only`
- `B+C`

Their role separation is:

- `B-only`: if the sample enters the verifier-defined high-risk subset, fall back to baseline
- `C-only`: use cheap output-side risk to decide fallback directly
- `B+C`: use the verifier-defined subset as a coarse gate and the cheap signal as the finer rescue decision

### 4.3 Sample-wise meta-controller

The discriminative main controller is a sample-wise meta-controller. It does not commit globally to one expert. Instead, it selects which expert to trust on each sample based on the post-hoc score geometry.

Let:

- `s_B(x)` be the B-risk score
- `s_C(x)` be the C-risk score
- `s_F(x)` be a fused score

The controller chooses an expert:

- `B-only`
- `C-only`
- `fusion`

using score magnitudes, their relative dominance, and their agreement pattern. After expert selection, the chosen expert applies its own frozen fallback rule to decide:

- baseline fallback
- or intervention keep

The full discriminative pipeline is:

1. fit `B-only`, `C-only`, and `B+C` on discovery
2. compute sample-wise score geometry from `s_B(x)`, `s_C(x)`, and `s_F(x)`
3. fit a shallow arbitration rule that selects one expert per sample
4. freeze both the experts and the arbitration rule
5. apply them unchanged on held-out data

This yields a dynamic sample-wise post-hoc controller whose adaptivity comes from expert arbitration rather than from re-optimizing the intervention itself.

### 4.4 Why meta is the main discriminative instantiation

Empirically, the same shallow controller class need not be globally optimal across all methods. The discriminative regime benefits from arbitration among complementary experts because different interventions expose harmfulness through different observable surfaces.

In the current results:

- `VGA` is best handled by a fusion-heavy controller
- `VCD` can be handled almost entirely by output-side signals
- `PAI` remains more mixed

The meta-controller is therefore the best discriminative mainline because it preserves a single shared framework while adapting the trusted expert on a per-sample basis.


## 5. Generative: Constrained Fallback under a Hallucination-Coverage Trade-off

### 5.1 Why generative is different

A naive port of the discriminative objective is inappropriate. In captioning, the intervention can improve hallucination metrics while hurting coverage. We therefore do not define harmfulness as a single scalar label such as a custom utility. Instead, the generative regime is evaluated in terms of:

- `CHAIRs`: sentence-level hallucination indicator
- `CHAIRi`: instance-level hallucination rate
- `Recall`: supported content coverage
- `Precision`: correctness of expressed content
- `F1`: balance between correctness and coverage

### 5.2 Post-hoc signal family

The dominant post-hoc signals in generative captioning are not generic token-level fragility signals. Our analysis indicates that harmful cases are better described as:

- omission risk
- coverage collapse
- tail shutdown
- conservative under-generation

Accordingly, the main feature family includes signals such as:

- `tail tokens after last mention`
- `tail vs head entropy`
- `last mention position`
- `tail vs head gap`
- content-length and mention-structure statistics

These are all computed from the generated caption and therefore remain ground-truth-free at test time.

### 5.3 Discovery-time constrained fallback teacher

For controller fitting, we define a discovery-time fallback teacher from observed baseline/intervention trade-offs.

The strict fallback teacher marks a sample as baseline-favorable only when:

1. baseline `F1` is higher than intervention `F1`
2. baseline `CHAIRi` is no worse than intervention `CHAIRi`
3. baseline `CHAIRs` is no worse than intervention `CHAIRs`

This teacher is not used at test time. It is only a discovery-time supervision target for controller fitting.

The teacher is therefore not an inference-time oracle. It is a structured supervision signal encoding the generative behavior we actually want:

> preserve hallucination quality while recovering caption completeness

### 5.4 Constrained shallow tree controller

The generative main controller is a shallow decision tree fitted against the Pareto-style fallback teacher.

The controller pipeline is:

1. compute post-hoc caption features
2. fit a shallow tree to predict fallback probability under the teacher target
3. threshold the tree output
4. constrain policy selection so that selected policies do not worsen the relevant hallucination metrics beyond the raw intervention reference on discovery

The generative training-and-apply protocol is:

1. build the fallback teacher on discovery
2. compute ground-truth-free caption features
3. fit candidate shallow trees
4. select the best tree under explicit `CHAIR`-preservation constraints
5. freeze the selected tree and threshold
6. apply the frozen controller on held-out data

We use a shallow tree instead of the discriminative meta-controller for two reasons.

First, the generative signal is more non-linear: omission risk emerges from interactions among tail behavior, entropy shape, and mention structure. Second, the relevant objective is explicitly constrained rather than purely scalar.

### 5.5 Why tree is the main generative instantiation

In the current experiments, unconstrained linear or fusion-style controllers tended to recover `F1` by sacrificing `CHAIR` metrics. This is not the desired generative behavior. The shallow constrained tree is the first non-trivial controller that achieves the intended direction:

- preserve `CHAIRs`
- preserve or improve `CHAIRi`
- improve `F1`

Therefore the tree controller is the main generative instantiation.


## 6. Experimental Protocol

### 6.1 Discovery and held-out split

All controller selection is performed on a discovery set.

The held-out evaluation protocol is:

1. generate baseline and intervention outputs
2. compute post-hoc signals
3. apply the frozen controller
4. report final routed performance

No held-out retuning is allowed.

### 6.2 Metrics

#### Discriminative

- baseline accuracy
- intervention accuracy
- final routed accuracy
- delta vs intervention
- selected harm precision / recall

#### Generative

- `CHAIRs`
- `CHAIRi`
- `Recall`
- `Precision`
- `F1`

We do not use custom claim-utility as the main generative metric in the paper draft. If included at all, it should appear only as a secondary diagnostic.


## 7. Current Main Results Structure

### 7.1 Discriminative

The main table should compare:

- baseline
- intervention
- post-hoc meta controller

for `VGA`, `PAI`, and `VCD`.

The main message is:

- average harmful regressions are controllable
- the meta-controller is the strongest discriminative mainline

The tree controller should be reported as a controller-class ablation:

- same shallow controller class also works
- especially strong or efficient for some methods
- but not the best overall discriminative mainline

### 7.2 Generative

The main table should compare:

- baseline captioning
- raw intervention captioning
- constrained tree post-hoc routing

using:

- `CHAIRs`
- `CHAIRi`
- `Recall`
- `Precision`
- `F1`

The main message is:

- raw intervention improves hallucination metrics but can lose coverage
- the constrained tree recovers part of that coverage while preserving hallucination metrics
- the generative evidence is currently strongest as an analysis-plus-controller story and should not be written as more mature than the discriminative held-out story unless the held-out evaluation is fully closed


## 8. Draft Result Narrative

### 8.1 Discriminative narrative

The discriminative regime shows that harmful intervention cases are not random. They are detectable using post-hoc signals extracted from the intervention answer. A sample-wise meta-controller that arbitrates among richer verifier signals, cheap output-side signals, and their fusion produces the strongest overall improvement across interventions.

This supports the claim that harmful intervention susceptibility is measurable without access to ground truth, and that shallow post-hoc routing can improve accuracy over always-on intervention.

### 8.2 Generative narrative

The generative regime is more subtle. Raw intervention often moves the model toward safer, lower-hallucination captions, but at the cost of reduced coverage. This means the relevant target is not hallucination reduction alone.

Our analysis shows that the right controller target is a constrained fallback target that preserves hallucination quality while improving caption completeness. A shallow decision tree trained against this target can recover part of the hidden Pareto headroom. This demonstrates that harmful generative intervention cases also leave detectable post-hoc traces, but the right observable and the right controller differ from the discriminative case.

This part of the paper should be written honestly as the weaker of the two main regimes unless full held-out evidence is in place. The strength of the current generative story is that the analysis identifies the right trade-off, shows that a meaningful constrained fallback set exists, and then recovers part of that set with a shallow controller.


## 9. Main Claim

We do not claim a universal raw feature or a universal controller class.

We claim a shared framework with four invariants:

1. intervention-first execution
2. ground-truth-free post-hoc observables
3. discovery-time fallback supervision
4. frozen held-out apply

Within that framework:

- `discriminative` is best instantiated by a sample-wise meta-controller
- `generative` is best instantiated by a constrained shallow tree


## 10. Limitations

1. We do not have one shared raw feature across regimes. The measurable proxy is regime-specific.
2. We do not have one shared controller class across regimes. The best discriminative controller is `meta`, whereas the current generative mainline is a constrained `tree`.
3. The generative fallback target is constrained and task-structured. It is not a single generic scalar objective, and it depends on the caption trade-off we choose to preserve.
4. Some interventions may expose strong post-hoc activity signals without exposing equally strong harm-aligned signals. `PAI` is the clearest current example.
5. The final paper should only write the two regimes as equally mature if the held-out evidence is equally closed. At the moment, the discriminative story is stronger.


## 11. Planned Final Figure and Table Layout

### Figure 1

Framework overview:

- intervention output
- post-hoc signal extraction
- regime-specific controller
- baseline fallback or intervention keep

### Figure 2

Discriminative:

- B-only / C-only / fusion / meta comparison
- harmful regression routing intuition

### Figure 3

Generative:

- intervention improves `CHAIR` but hurts coverage
- constrained tree preserves `CHAIR` and improves `F1`

### Table 1

Discriminative main results:

- baseline
- intervention
- meta

for each method

### Table 2

Generative main results:

- baseline
- intervention
- constrained tree

with `CHAIRs`, `CHAIRi`, `Recall`, `Precision`, `F1`

### Table 3

Controller ablations:

- discriminative: tree vs meta
- generative: linear vs tree vs unconstrained variants


## 12. Writing Notes

Avoid these claims:

- one universal raw feature explains all interventions
- one identical controller is best in every regime
- custom claim-utility is the main generative metric

Prefer these claims:

- the framework is shared
- the manifestation of support mismatch is regime-specific
- the controller instantiation is regime-specific
- shallow post-hoc control can improve over always-on intervention
