# Method Draft: PCP C/D Router

This draft is written so that final experiment values can be filled in later.
Use `TODO_RESULT` markers for values that must come from fixed-policy held-out
experiments.

## Method

This section currently covers two regimes under one PCP principle. The
discriminative regime uses answer-level PCP and selectively falls back from the
intervention answer to the baseline answer. The generative CHAIR regime uses
object-level PCP and suppresses unsupported object mentions inside the
intervention caption. The shared idea is to replay intervention-produced content
under the same model, estimate its internal consistency, and apply the smallest
correction to unsupported content.

### Method Assumptions

The method relies on four assumptions. These assumptions should be stated
explicitly because they define the scope of the paper.

First, the intervention has already produced an output. PCP is post-hoc control,
not an alternative decoding algorithm for producing the initial answer or
caption. The method operates after \(y_I\) or \(c_I\) exists.

Second, harmful intervention cases leave measurable consistency defects under
model replay. In discriminative yes/no QA, this defect can appear at the answer
token level or in the yes/no decision distribution. In captioning, it appears at
the object-mention level: an object mentioned by the intervention caption is not
strongly supported by an image-conditioned yes/no probe.

Third, the correction target must match the output regime. In short-answer QA,
the smallest practical correction is often whole-answer fallback to \(y_B\). In
captioning, whole-caption fallback is too coarse because it can change many
supported mentions. The smaller correction is local suppression of the risky
object pathway.

Fourth, all routing and suppression thresholds are selected on discovery or
validation data and then frozen. Test labels are never used for routing.

The paper should therefore claim a shared PCP control principle, not a single
identical controller across tasks.

### Discriminative Problem Setup

We study post-hoc control for decoding-time multimodal interventions. For each
image-question pair \(x=(I,q)\), we assume two candidate answers are available:

\[
y_B = M_B(I,q), \qquad y_I = M_I(I,q),
\]

where \(y_B\) is the baseline model answer and \(y_I\) is the answer produced by
an intervention such as VGA. Standard always-on intervention always returns
\(y_I\). Our goal is different: after the intervention answer has been produced,
we decide whether to keep it or fall back to the baseline answer.

\[
y_{\mathrm{final}} =
\begin{cases}
y_B, & \text{if } r(I,q,y_I) \ge \tau,\\
y_I, & \text{otherwise.}
\end{cases}
\]

The routing score \(r\) is computed without test-time labels. It uses only the
image, question, intervention answer, and frozen model replay statistics. The
baseline answer is a candidate fallback, not a supervision signal at inference
time.

### Answer-Level PCP

The central idea is to test whether the intervention answer is internally
supported by the same model that produced or consumes it. For each sample, we
teacher-force the intervention answer \(y_I\) under the original image-question
context and extract replay statistics from the continuation tokens and from the
binary yes/no decision distribution.

This produces an answer-level PCP feature vector

\[
z(I,q,y_I)=h(I,q,y_I),
\]

where \(h\) is a label-free feature extractor. We separate these features into
two consistency views: a token-consistency view and a decision-consistency view.
The final router is trained only on a discovery split and then frozen for held-
out evaluation.

### Token-Consistency Risk Score

The token-consistency score measures whether the intervention answer is stable
as a generated string under teacher-forced replay. We use content-token
statistics rather than all tokens when possible, because punctuation and format
tokens are less informative about answer support.

The default C-feature set is

\[
\mathcal{C} =
\{
\texttt{cheap\_target\_gap\_content\_min},
\texttt{cheap\_lp\_content\_min},
\texttt{cheap\_lp\_content\_std}
\}.
\]

These features capture three related failure signatures.
`cheap_target_gap_content_min` measures the weakest target-vs-alternative logit
margin among content tokens. `cheap_lp_content_min` measures the lowest
teacher-forced log-probability among content tokens. `cheap_lp_content_std`
measures instability of token-level confidence across the answer.

Each raw feature is oriented on the discovery split so that larger values
indicate higher intervention risk. Let \(f_j\) be a raw C-feature and
\(\eta_j \in \{-1,+1\}\) its discovery-selected orientation. With discovery
mean and standard deviation \((\mu_j,\sigma_j)\), the normalized feature is

\[
\tilde f_j(x)=\frac{\eta_j f_j(x)-\mu_j}{\sigma_j+\epsilon}.
\]

The token-consistency risk score is the average normalized risk:

\[
C(x)=\frac{1}{|\mathcal{C}|}\sum_{j\in\mathcal{C}}\tilde f_j(x).
\]

We refer to \(C\) as a risk score rather than a support score: after orientation,
higher \(C\) means the intervention answer is less reliable.

### Decision-Consistency Risk Score

Token-level replay can miss failures where the answer string is locally
plausible but the answer identity is weakly supported. To capture this, we also
measure the model's binary yes/no decision distribution for the intervention
answer. If the intervention answer is "yes", the candidate label is "yes" and
the alternative label is "no"; if the intervention answer is "no", the roles are
reversed.

The default D-feature set is

\[
\mathcal{D} =
\{
\texttt{cheap\_decision\_candidate\_minus\_alt},
\texttt{cheap\_decision\_candidate\_prob\_binary},
\texttt{cheap\_decision\_candidate\_label\_lp},
\texttt{cheap\_decision\_candidate\_kl\_uniform}
\}.
\]

`cheap_decision_candidate_minus_alt` is the candidate-vs-alternative decision
margin. `cheap_decision_candidate_prob_binary` is the candidate probability in
the normalized yes/no binary distribution. `cheap_decision_candidate_label_lp`
is the log-probability of the candidate label. `cheap_decision_candidate_kl_uniform`
measures how far the binary distribution is from a uniform yes/no distribution.

As with C-features, each D-feature is oriented and standardized on discovery:

\[
\tilde g_k(x)=\frac{\eta_k g_k(x)-\mu_k}{\sigma_k+\epsilon}.
\]

The decision-consistency risk score is

\[
D(x)=\frac{1}{|\mathcal{D}|}\sum_{k\in\mathcal{D}}\tilde g_k(x).
\]

Higher \(D\) indicates weaker decision-level support for the intervention
answer after discovery-time orientation.

### PCP Policy Families

We evaluate three fixed policy families. The first uses token consistency only:

\[
r_C(x)=C(x).
\]

The second uses decision consistency only:

\[
r_D(x)=D(x).
\]

The third combines both views with a calibrated convex mixture:

\[
r_{C+D}(x;\alpha)=(1-\alpha)C(x)+\alpha D(x),
\qquad \alpha\in[0,1].
\]

All policies use the same fallback rule:

\[
\mathrm{route}(x)=
\mathbf{1}[r(x)\ge\tau],
\]

where `route=1` means fall back to \(y_B\) and `route=0` means keep \(y_I\).
The parameters \(\alpha\) and \(\tau\) are selected only on discovery data and
then frozen. In the default deployment policy, PCP is only eligible to route
samples where the baseline and intervention answers disagree at the parsed
yes/no label level. This `changed_answer` filter is label-free: it uses only
the two model outputs. If both systems already produce the same yes/no answer,
falling back cannot change the task answer and is not counted as a meaningful
PCP action.

### Discovery Calibration

On the discovery split, labels are used only to calibrate the router. We define
sample outcomes relative to the baseline and intervention answers:

\[
\text{harm}: y_B \text{ correct and } y_I \text{ wrong},
\]
\[
\text{help}: y_B \text{ wrong and } y_I \text{ correct}.
\]

Routing a harm sample to the baseline fixes an intervention-induced regression,
whereas routing a help sample to the baseline loses an intervention gain. We
therefore track the failure-sensitive utility

\[
\mathrm{net} = \mathrm{selected\_harm} - \mathrm{selected\_help}.
\]

For each policy family, we sweep the threshold \(\tau\) over route-eligible
samples. For \(C+D\), we also sweep \(\alpha\) over a fixed grid. The selected
policy is the one that maximizes the discovery objective, currently final
accuracy, with net and precision used as secondary diagnostics. After this step,
all feature orientations, normalization constants, selected family, \(\alpha\),
\(\tau\), and the route-candidate filter are frozen.

### Held-Out Application

Held-out evaluation does not use labels for routing. For each test sample, we
load or generate \(y_B\) and \(y_I\), replay \(y_I\), compute \(C(x)\) and
\(D(x)\), apply the frozen policy, and report the final answer.

The held-out protocol is:

1. Load baseline answer \(y_B\) and intervention answer \(y_I\).
2. Teacher-force \(y_I\) under the image-question context.
3. Compute C-features and D-features.
4. Apply frozen discovery statistics to compute \(C(x)\) and \(D(x)\).
5. Apply the frozen route-candidate filter, policy family, and threshold.
6. Output \(y_B\) if routed to fallback; otherwise output \(y_I\).

This makes PCP a post-hoc wrapper around the intervention. It does not change
the intervention mechanism and does not train a new answer generator.

## Generative CHAIR PCP

The CHAIR setting differs from discriminative yes/no QA. The output is a
caption, and replacing the entire intervention caption with a baseline caption
can trade one error for another: it may recover omitted supported objects while
also reintroducing hallucinated objects. Therefore, the generative branch should
not be presented as the same fallback router. It is the same post-intervention
control principle instantiated with object-level PCP and local correction.

### CHAIR Objective and Failure Mode

For captioning, we evaluate with CHAIR metrics:

\[
\mathrm{CHAIR_s}, \quad \mathrm{CHAIR_i}, \quad
\mathrm{Recall}, \quad \mathrm{Precision}, \quad \mathrm{F1}.
\]

The intervention can reduce hallucination but also reduce coverage. A useful
controller should therefore avoid claiming a single scalar improvement unless
the trade-off is explicitly defined. In the main table, CHAIR results should
report hallucination and coverage metrics together.

The practical failure mode is an unsupported object mention introduced or
preserved by the intervention caption. Let \(c_I\) be the intervention caption
and let \(\mathcal{O}(c_I)\) be the object mentions extracted from it. The
controller estimates object-level residual risk for mentions in \(c_I\), then
applies a conservative correction only to the highest-risk object pathway.

### Object-Level PCP Probe

For each object candidate \(o\in\mathcal{O}(c_I)\), we estimate whether the
image supports the object with a yes/no probe:

\[
\texttt{Is there a [object] in the image? Answer yes or no.}
\]

The intended deployable version uses only the next-token yes/no distribution:

\[
p_{\mathrm{yes}}(o\mid I)
=
\frac{\exp \ell_{\mathrm{yes}}}
{\exp \ell_{\mathrm{yes}}+\exp \ell_{\mathrm{no}}}.
\]

This is the captioning analogue of the discriminative D-score. Instead of asking
whether a sample-level answer label is internally supported, we ask whether each
caption object mention is internally supported. A simple object-level PCP risk
score is

\[
\mathrm{D}_{\mathrm{obj}}(o)=1-p_{\mathrm{yes}}(o\mid I).
\]

The fuller object-level PCP variant can mirror the discriminative D-score:

\[
\mathrm{D}_{\mathrm{obj}}(o)
=
\mathrm{mean}\_z(
-m_{\mathrm{yes/no}}(o),
-p_{\mathrm{yes}}(o),
H_{\mathrm{yes/no}}(o),
-\mathrm{KL}_{\mathrm{uniform}}(o)
),
\]

where \(m_{\mathrm{yes/no}}\) is the yes-vs-no margin and
\(H_{\mathrm{yes/no}}\) is binary decision entropy. A token-consistency variant
is also possible by teacher-forcing the caption and measuring log-probability
or margin over the object mention span:

\[
\mathrm{C}_{\mathrm{obj}}(o)
=
\mathrm{mean}\_z(
\text{object-span token risk features}
).
\]

The current cleanest deployable CHAIR branch uses a simple decision-consistency
object risk. The current `v83_fast_next_token_risk_full500` artifact is not
usable as a main result because its summary reports errors for all 500 rows.
This branch should be rerun or excluded.

### Object Selection and Local Correction

The top-risk object is

\[
o^\star=\arg\max_{o\in\mathcal{O}(c_I)}\mathrm{D}_{\mathrm{obj}}(o).
\]

Correction is triggered only if

\[
\mathrm{D}_{\mathrm{obj}}(o^\star)\ge\tau_{\mathrm{obj}},
\]

where \(\tau_{\mathrm{obj}}\) is selected on validation or discovery data and
then frozen.

### Local Object Suppression

When the top-risk object passes the risk threshold, the correction does not
fall back to a baseline caption. Instead, it reruns the intervention decoder
with a negative logit bias on surface forms of the selected object. If
\(\mathcal{T}(o^\star)\) is the token set for object \(o^\star\), decoding uses

\[
\ell_t(v) \leftarrow \ell_t(v) + b\cdot\mathbf{1}[v\in\mathcal{T}(o^\star)],
\qquad b<0.
\]

The output remains an intervention-style caption, but the highest-risk object
pathway is suppressed. This is more compatible with CHAIR than whole-caption
fallback because it targets an unsupported object mention while preserving the
rest of the caption as much as possible.

Thus the generative PCP policy is

\[
c_{\mathrm{final}} =
\begin{cases}
\mathrm{Suppress}(c_I,o^\star), &
\mathrm{D}_{\mathrm{obj}}(o^\star)\ge\tau_{\mathrm{obj}},\\
c_I, & \text{otherwise.}
\end{cases}
\]

This is parallel to discriminative PCP, but the unit of control changes from an
answer to an object mention.

### Current CHAIR Evidence

The strongest currently available deployable CHAIR-style artifact is the
object-token suppression run:

- source: `experiments/coco_chair_v82_object_token_suppression_full500_soft_fixed_0.5`
- split: test
- \(n=500\)
- correction: first-token object suppression with bias \(-0.5\), support
  threshold `yp0.40`

Current fixed-output numbers from this artifact are:

| Method | n | CHAIRs | CHAIRi | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Intervention | 500 | 0.322 | 0.0890 | 0.7281 | 0.8215 | 0.7720 |
| Object suppression | 500 | 0.318 | 0.0873 | 0.7274 | 0.8237 | 0.7726 |

This is a small but directionally clean result: hallucination decreases slightly
and F1 is roughly preserved or slightly improved. It is not yet a large main
result, so the paper should frame it as a generative extension of the control
principle unless a stronger full run is produced.

The older whole-caption fallback/distillation branch remains useful for
analysis but is not the cleanest main CHAIR method. In
`experiments/coco_chair_vga_linear_v48b_trace_cascade/distill/test_apply_diag`,
the routed policy improves recall and F1 over intervention but increases
CHAIRi/CHAIRs relative to always-on intervention. This demonstrates the
hallucination-coverage trade-off and motivates local object-level correction,
but it should not be presented as the final generative method.

### CHAIR Results to Fill

The final CHAIR table should use a fixed validation-selected policy and a held-
out test split. Fill the table below only with fixed-policy results.

| Method | n | CHAIRs | CHAIRi | Recall | Precision | F1 | Len |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline caption | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| Always-on intervention | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| CHAIR object-control | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |

The main generative claim should be conservative:

> In captioning, PCP should operate at the object-mention level rather than as
> whole-output fallback; object-level decision consistency can identify
> unsupported mentions, and local suppression can reduce hallucinated object
> pathways while preserving most intervention caption quality.

Do not claim that the current CHAIR branch solves generative hallucination
globally. The current evidence supports a scoped local-correction claim.

## Analysis: Overlap and Arbitration

### Overlap and Arbitration Analysis

The three basic policy families assume that a single global score is sufficient.
However, token-consistency and decision-consistency may relate differently
across backbones. We therefore analyze the selected sets

\[
S_C=\{x:C(x)\ge\tau_C\}, \qquad
S_D=\{x:D(x)\ge\tau_D\}.
\]

We report the utility of \(S_C\setminus S_D\), \(S_D\setminus S_C\),
\(S_C\cap S_D\), and \(S_C\cup S_D\). This diagnostic tells us whether the two
views are redundant or complementary.

If most positive net utility lies in \(S_C\cap S_D\), then the two scores are
redundant and an agreement-style policy is more appropriate. If both exclusive
regions have positive net utility and overlap is small, then the two scores
capture complementary harmful pockets, making union or sample-wise arbitration
more appropriate than a single global mixture.

Current discovery observations are consistent with this distinction:
LLaVA-1.5 shows high C/D overlap and useful mass concentrated in the
intersection, while LLaVA-NeXT changed-only diagnostics suggest lower overlap
and stronger complementarity. The LLaVA-NeXT fixed-policy result remains
`TODO_RESULT`.

### Current Results to Fill

The following values should be filled only from fixed-policy held-out runs.
Oracle or changed-only diagnostics should not be used as main method results.

| Backbone | Dataset | Baseline | VGA | PCP family | PCP final | Delta vs VGA | Selected | Harm fixed | Help lost | Net |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5 | MSCOCO | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| LLaVA-1.5 | A-OKVQA | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| LLaVA-1.5 | GQA | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| LLaVA-NeXT | MSCOCO | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| LLaVA-NeXT | A-OKVQA | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |
| LLaVA-NeXT | GQA | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT | TODO_RESULT |

### Claim Boundary

The main claim should be limited to fixed-policy PCP routing:

> PCP estimates intervention risk from token-consistency and decision-consistency
> replay signals and uses a discovery-calibrated fixed policy to selectively
> revert unsupported intervention answers to the baseline.

The following should remain analysis, not main method claims:

- changed-only oracle results
- dynamic union selected on the evaluation set
- best threshold selected separately per benchmark test set
- legacy `meta_strong` B/C controller results
