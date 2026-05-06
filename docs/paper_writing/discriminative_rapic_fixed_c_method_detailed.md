# Discriminative RAPIC With Fixed Replay-Content Calibration

This note documents the current discriminative RAPIC method after replacing the
earlier adaptive C/D controller with a fixed replay-content selector. It is
intended to be the source text for the paper method section and the experiment
protocol.

## 1. Problem Setting

For each POPE-style discriminative example, let

\[
x_i=(I_i,q_i)
\]

be an image-question pair, where \(q_i\) is a binary object-existence question.
The ground-truth answer is

\[
g_i \in \{\mathrm{yes},\mathrm{no}\}.
\]

The vanilla vision-language model produces a baseline answer

\[
y_i^{B}=M_\theta(I_i,q_i),
\]

and an intervention method \(T\), such as VGA, PAI-attn, or VAF, produces an
intervention answer

\[
y_i^{I}=T(M_\theta,I_i,q_i).
\]

Both answers are mapped to binary labels by a deterministic parser
\(\pi(\cdot)\):

\[
a_i^{B}=\pi(y_i^{B}), \qquad a_i^{I}=\pi(y_i^{I}),
\]

where \(a_i^{B},a_i^{I}\in\{\mathrm{yes},\mathrm{no}\}\) when parsing succeeds.

RAPIC is a post-hoc controller. It does not generate a new answer. It only
decides whether to keep the intervention answer or locally roll back to the
baseline answer:

\[
y_i^{R} =
\begin{cases}
y_i^{B}, & r_i=1,\\
y_i^{I}, & r_i=0,
\end{cases}
\]

where \(r_i=1\) means fallback to the baseline and \(r_i=0\) means keep the
intervention.

## 2. Changed-Answer Routing Scope

RAPIC is applied only when the intervention changes the parsed answer:

\[
\mathcal{C} =
\{i: a_i^{B},a_i^{I}\in\{\mathrm{yes},\mathrm{no}\},\; a_i^{B}\ne a_i^{I}\}.
\]

If \(a_i^{B}=a_i^{I}\), fallback cannot change the final binary answer, so
RAPIC keeps the intervention output:

\[
r_i=0 \qquad \text{if } i\notin\mathcal{C}.
\]

The changed-answer set is split by transition direction:

\[
\mathcal{C}_{Y\rightarrow N}
=
\{i: a_i^{B}=\mathrm{yes},\; a_i^{I}=\mathrm{no}\},
\]

\[
\mathcal{C}_{N\rightarrow Y}
=
\{i: a_i^{B}=\mathrm{no},\; a_i^{I}=\mathrm{yes}\}.
\]

This split is central to the current method. The two directions can have very
different harm/help distributions. Therefore RAPIC calibrates separate
thresholds

\[
\tau_{Y\rightarrow N}
\quad\text{and}\quad
\tau_{N\rightarrow Y}.
\]

At test time, the transition direction is determined only from
\((a_i^{B},a_i^{I})\). Ground-truth labels are not used during routing.

## 3. Discovery-Time Harm and Help Labels

The discovery split is used only to calibrate the post-hoc selector. For a
changed prediction, define an intervention-induced harmful flip as

\[
h_i =
\mathbf{1}
\left[
a_i^{B}=g_i
\;\land\;
a_i^{I}\ne g_i
\right],
\]

and a helpful gain as

\[
u_i =
\mathbf{1}
\left[
a_i^{B}\ne g_i
\;\land\;
a_i^{I}=g_i
\right].
\]

Here \(h_i=1\) means the intervention corrupts a baseline-correct answer, while
\(u_i=1\) means the intervention fixes a baseline-wrong answer.

If RAPIC falls back on a selected set \(S\subseteq\mathcal{C}\), the selected
harm and selected gain are

\[
H(S)=\sum_{i\in S}h_i,
\qquad
G(S)=\sum_{i\in S}u_i.
\]

The net accuracy change induced by fallback on a benchmark of size \(N\) is

\[
\Delta_{\mathrm{fallback}}(S)
=
\frac{H(S)-G(S)}{N}.
\]

Thus fallback is beneficial when it recovers more harmful flips than it removes
helpful gains:

\[
H(S) > G(S).
\]

## 4. Fixed Replay-Content Feature Set

The current main RAPIC method uses exactly three replay-content features:

\[
\mathcal{F}_{C3}
=
\{
f_{\mathrm{lp}},
f_{\mathrm{gap\_content}},
f_{\mathrm{gap\_first}}
\},
\]

where

\[
f_{\mathrm{lp}}
=
\texttt{cheap\_lp\_content\_min},
\]

\[
f_{\mathrm{gap\_content}}
=
\texttt{cheap\_target\_gap\_content\_min},
\]

\[
f_{\mathrm{gap\_first}}
=
\texttt{cheap\_first\_target\_gap}.
\]

These features are computed by replaying the intervention answer under the
original image-question context and measuring whether the frozen model supports
the intervention answer content.

| Feature | Interpretation | Expected harm signal |
| --- | --- | --- |
| `cheap_lp_content_min` | Minimum replay log-probability over content tokens in the intervention answer. | Low support for intervention content. |
| `cheap_target_gap_content_min` | Minimum replay target-vs-alternative margin over content tokens. | Weak semantic margin for intervention content. |
| `cheap_first_target_gap` | Target-vs-alternative margin at the first answer token. | Weak answer-onset support. |

The fixed-C method deliberately removes the earlier adaptive feature-family
search. It does not use D features, confidence-only features, layer trajectory
features, or a learned meta-classifier in the main discriminative protocol.

## 5. Direction-Specific Feature Orientation

For each transition direction

\[
d\in\{Y\rightarrow N,\;N\rightarrow Y\},
\]

and each replay-content feature \(f\in\mathcal{F}_{C3}\), RAPIC orients the
feature on the discovery split so that larger values indicate greater
intervention-harm risk.

Let \(\mathcal{D}_d\) be the discovery rows for direction \(d\). RAPIC compares
the AUROC of \(f\) and \(-f\) for predicting \(h_i\):

\[
A^+_{f,d}
=
\mathrm{AUROC}
\left(
\{f_i\}_{i\in\mathcal{D}_d},
\{h_i\}_{i\in\mathcal{D}_d}
\right),
\]

\[
A^-_{f,d}
=
\mathrm{AUROC}
\left(
\{-f_i\}_{i\in\mathcal{D}_d},
\{h_i\}_{i\in\mathcal{D}_d}
\right).
\]

The orientation is

\[
o_{f,d}
=
\begin{cases}
+1, & A^+_{f,d}\ge A^-_{f,d},\\
-1, & A^+_{f,d}< A^-_{f,d}.
\end{cases}
\]

The oriented feature value is

\[
\tilde{f}_{i,d}
=
o_{f,d} f_i.
\]

The orientation is fitted only on discovery and then frozen.

## 6. Discovery Normalization

For each direction \(d\) and feature \(f\), compute the discovery mean and
standard deviation of the oriented values:

\[
\mu_{f,d}
=
\frac{1}{|\mathcal{D}_{f,d}|}
\sum_{i\in\mathcal{D}_{f,d}}
\tilde{f}_{i,d},
\]

\[
\sigma_{f,d}
=
\sqrt{
\frac{1}{|\mathcal{D}_{f,d}|-1}
\sum_{i\in\mathcal{D}_{f,d}}
\left(\tilde{f}_{i,d}-\mu_{f,d}\right)^2
},
\]

where \(\mathcal{D}_{f,d}\) denotes rows where feature \(f\) is present.

The normalized oriented z-score is

\[
z_{i,f,d}
=
\frac{\tilde{f}_{i,d}-\mu_{f,d}}
{\max(\sigma_{f,d},10^{-6})}.
\]

The implementation requires feature availability on at least

\[
80\%
\]

of discovery rows for the corresponding direction. Otherwise that direction can
fall back to a disabled policy.

## 7. Median Replay-C Score

For each changed example \(i\in\mathcal{C}_d\), RAPIC aggregates the three
normalized replay-content scores by a median:

\[
s_{i,d}
=
\mathrm{median}
\left(
z_{i,f_{\mathrm{lp}},d},
z_{i,f_{\mathrm{gap\_content}},d},
z_{i,f_{\mathrm{gap\_first}},d}
\right).
\]

The median makes the selector robust to one noisy replay feature while
preserving a scalar, transparent routing score.

The routing score is direction-specific because orientation and normalization
are direction-specific:

\[
s_{i,Y\rightarrow N}
\ne
s_{i,N\rightarrow Y}
\quad
\text{in general}.
\]

## 8. Threshold Sweep and Policy Selection

For each direction \(d\), RAPIC sweeps thresholds over discovery score
quantiles:

\[
\mathcal{T}_d
=
\mathrm{Grid}
\left(
\{s_{i,d}: i\in\mathcal{D}_d\}
\right).
\]

For a threshold \(\tau\), the direction-specific fallback rule is

\[
r_i(\tau,d)
=
\mathbf{1}
\left[
i\in\mathcal{C}_d
\land
s_{i,d}\ge \tau
\right].
\]

Let

\[
S_{\tau,d}
=
\{i\in\mathcal{D}_d: r_i(\tau,d)=1\}.
\]

The routed discovery accuracy for direction \(d\) is

\[
a_i^R(\tau,d)
=
\begin{cases}
a_i^B, & r_i(\tau,d)=1,\\
a_i^I, & r_i(\tau,d)=0,
\end{cases}
\]

\[
\mathrm{Acc}_d(\tau)
=
\frac{1}{|\mathcal{D}_d|}
\sum_{i\in\mathcal{D}_d}
\mathbf{1}
\left[a_i^R(\tau,d)=g_i\right],
\]

Equivalently, because fallback only swaps intervention output back to baseline,
maximizing routed accuracy over changed examples is equivalent to maximizing

\[
H(S_{\tau,d})-G(S_{\tau,d})
\]

up to the constant intervention accuracy.

The selected threshold is

\[
\tau_d^*
=
\arg\max_{\tau\in\mathcal{T}_d}
\mathrm{Acc}_d(\tau),
\]

subject to the minimum selected-count constraint

\[
|S_{\tau,d}|\ge 5.
\]

If the disabled keep-all policy is at least as good as every valid fallback
policy, RAPIC chooses a NOOP policy:

\[
r_i=0
\qquad
\forall i\in\mathcal{C}_d.
\]

This is important for cases where an intervention is already net helpful for a
transition direction.

## 9. Held-Out Deployment

After discovery calibration, RAPIC freezes:

1. the three feature names,
2. the direction-specific feature orientations,
3. the direction-specific z-score means and standard deviations,
4. the direction-specific thresholds,
5. and any direction-specific NOOP decision.

For a held-out example, RAPIC performs:

\[
r_i =
\begin{cases}
\mathbf{1}[s_{i,Y\rightarrow N}\ge \tau^*_{Y\rightarrow N}],
& a_i^B=\mathrm{yes}, a_i^I=\mathrm{no},\\
\mathbf{1}[s_{i,N\rightarrow Y}\ge \tau^*_{N\rightarrow Y}],
& a_i^B=\mathrm{no}, a_i^I=\mathrm{yes},\\
0, & \text{otherwise}.
\end{cases}
\]

The final output is

\[
y_i^{R}
=
\begin{cases}
y_i^B, & r_i=1,\\
y_i^I, & r_i=0.
\end{cases}
\]

No held-out labels are used for threshold selection, feature orientation, or
normalization.

## 10. Metrics

The baseline, intervention, and RAPIC accuracies are

\[
\mathrm{Acc}_{B}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[a_i^B=g_i],
\]

\[
\mathrm{Acc}_{I}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[a_i^I=g_i],
\]

\[
\mathrm{Acc}_{R}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[\pi(y_i^R)=g_i].
\]

The RAPIC gain over the intervention is

\[
\Delta_I
=
\mathrm{Acc}_{R}-\mathrm{Acc}_{I}.
\]

The selected fallback set on held-out data is

\[
S_R=\{i:r_i=1\}.
\]

Selected harmful flips, selected helpful gains, and selected net recovery are

\[
\mathrm{Sel.H}
=
\sum_{i\in S_R}h_i,
\]

\[
\mathrm{Sel.G}
=
\sum_{i\in S_R}u_i,
\]

\[
\mathrm{Sel.Net}
=
\mathrm{Sel.H}-\mathrm{Sel.G}.
\]

The full-benchmark accuracy gain from fallback satisfies

\[
\Delta_I
=
\frac{\mathrm{Sel.H}-\mathrm{Sel.G}}{N},
\]

up to parsing and unchanged-answer bookkeeping.

Harm recall and gain recall are

\[
\mathrm{Hrec}
=
\frac{\mathrm{Sel.H}}{\sum_i h_i},
\]

\[
\mathrm{Grec}
=
\frac{\mathrm{Sel.G}}{\sum_i u_i}.
\]

Fallback rate is

\[
\mathrm{Fb.\%}
=
\frac{|S_R|}{N}.
\]

Fallback precision, when reported, is

\[
\mathrm{Fb.Prec.}
=
\frac{\mathrm{Sel.H}}
{\mathrm{Sel.H}+\mathrm{Sel.G}}.
\]

## 11. Why Transition-Split Calibration Is Needed

A single global threshold would assume that

\[
p(h_i=1\mid s_i, Y\rightarrow N)
\approx
p(h_i=1\mid s_i, N\rightarrow Y).
\]

The empirical calibration statistics violate this assumption. The two
directions can have different:

- score distributions,
- optimal thresholds,
- harmful/helpful flip balance,
- fallback rates,
- and selected harm/gain tradeoffs.

Therefore the current method fits

\[
\tau^*_{Y\rightarrow N}
\quad\text{and}\quad
\tau^*_{N\rightarrow Y}
\]

separately.

For VGA/LLaVA-1.5, the calibrated fixed-C policies illustrate this:

\[
\tau^*_{Y\rightarrow N}\approx 0.557,
\qquad
\tau^*_{N\rightarrow Y}\approx -2.580.
\]

The \(Y\rightarrow N\) policy is selective, while the \(N\rightarrow Y\) policy
is much closer to broad rollback. This difference is discovered on calibration
data and then applied without test-time tuning.

## 12. Direct Selector Baselines

The current analysis compares fixed-C RAPIC against:

1. single replay-C selectors,
2. random fallback at the same fallback budget,
3. and always rollback.

### Single Replay-C Selector

For a single feature \(f\), the same orientation, z-score normalization, and
transition-specific threshold sweep are used, but the score is

\[
s_{i,d}^{(f)}=z_{i,f,d}.
\]

This tests whether one replay feature is sufficient.

### Random Fallback

For random fallback, let \(K\) be the number of samples selected by fixed-C
RAPIC on the same dataset. Random fallback selects \(K\) changed-answer samples
uniformly in expectation. If the changed set contains \(H\) harmful flips and
\(G\) helpful gains among \(C\) changed samples, the expected selected counts
are

\[
\mathbb{E}[\mathrm{Sel.H}_{\mathrm{rand}}]
=
K\frac{H}{C},
\]

\[
\mathbb{E}[\mathrm{Sel.G}_{\mathrm{rand}}]
=
K\frac{G}{C}.
\]

This isolates whether RAPIC is selecting the right samples rather than merely
using more fallback compute.

### Always Rollback

Always rollback selects every changed-answer sample:

\[
r_i=1
\qquad
\forall i\in\mathcal{C}.
\]

This returns the full benchmark to baseline behavior on all changed answers.
It is useful as a stress test: if an intervention is net helpful, always
rollback destroys many helpful gains.

## 13. Current Main Method Summary

The final discriminative method can be summarized as:

\[
\boxed{
\text{RAPIC}_{C3}
=
\text{transition-split thresholding of a median over three oriented replay-C z-scores}
}
\]

or procedurally:

```text
Input:
  baseline answer y_B
  intervention answer y_I
  replay-content features for y_I

Discovery calibration:
  for each direction d in {yes->no, no->yes}:
    orient each fixed C feature toward harmful flips by AUROC
    normalize oriented features using discovery mean/std
    compute median z-score over the three features
    sweep thresholds
    select tau_d by discovery final accuracy
    allow NOOP if no fallback policy improves discovery objective

Held-out deployment:
  if answer did not change:
    keep intervention
  else:
    identify transition direction d
    compute frozen median replay-C score s_d
    fallback to baseline iff s_d >= tau_d
```

## 14. Difference From The Previous C/D Controller

The previous RAPIC variant searched over C-only, D-only, and C+D fusion
families:

\[
s_i
=
(1-\alpha)s_i^{C}
+
\alpha s_i^{D},
\]

with discovery-selected feature families, \(\alpha\), and threshold.

The current fixed-C version removes this adaptive controller search:

\[
\alpha=0,
\qquad
\mathcal{F}=\mathcal{F}_{C3},
\qquad
\mathrm{aggregate}=\mathrm{median}.
\]

The only learned quantities are simple discovery calibration statistics:

\[
\{o_{f,d},\mu_{f,d},\sigma_{f,d},\tau_d^*\}.
\]

This makes the main method easier to state and reduces concern that gains come
from a high-capacity selector.

## 15. Implementation

Main fixed-C calibration and application:

```text
scripts/build_transition_split_fixed_c_median_ensemble.py
```

Route materialization for downstream F1 and table refresh:

```text
scripts/materialize_fixed_c_median_routes.py
```

Transition-specific diagnostics and selector comparison:

```text
scripts/analyze_transition_split_selector_diagnostics.py
```

Canonical run root used in the current discriminative experiments:

```text
experiments/paper_pcp_cd_transition_split_fixed_c_median_ensemble_fullacc
```
