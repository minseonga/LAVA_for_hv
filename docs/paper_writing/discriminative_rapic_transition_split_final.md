# Final Discriminative RaPiC Method: Transition-Split Calibration

This note freezes the final discriminative RaPiC protocol for POPE-style
yes/no experiments. It replaces the earlier pooled `changed_answer` controller
as the main discriminative protocol.

The final choice is:

> Calibrate separate RaPiC controllers for `yes->no` and `no->yes` intervention
> transitions, then apply the matching controller to each transition at test
> time.

This is the most robust unified protocol among the variants tested:

- Pooled `changed_answer` calibration works well for several settings but fails
  badly for Qwen VAF, where discovery and test intervention directions differ.
- A hard `yes->no` gate works well for VGA but damages PAI because PAI can
  introduce harmful edits in both transition directions.
- Transition-split calibration keeps a single unified method while allowing each
  intervention/backbone pair to learn which transition direction is risky.

This document covers only the discriminative yes/no setting. Generative caption
or CHAIR-style routing is outside this protocol.

## 1. Setting

Each input is an image-question pair

\[
x=(I,q),
\]

where \(q\) is a yes/no object-existence question. A backbone model
\(M_\theta\) produces a baseline answer

\[
y_B = M_\theta(I,q),
\]

and a training-free intervention \(T\), such as VGA, PAI-attn, or VAF, produces

\[
y_T = T(M_\theta, I, q).
\]

Let

\[
a_B = \pi(y_B), \qquad a_T = \pi(y_T),
\]

where \(\pi(\cdot)\) parses a free-form answer into a yes/no label.

RaPiC is a post-intervention controller. The final output is

\[
y_{\mathrm{RaPiC}} =
\begin{cases}
y_B, & \rho(x, y_B, y_T)=\texttt{baseline},\\
y_T, & \rho(x, y_B, y_T)=\texttt{method}.
\end{cases}
\]

In the discriminative protocol, the router can use the baseline and
intervention yes/no labels \(a_B, a_T\) to determine the transition direction.
This is label-free with respect to ground truth. In the offline experiments,
baseline answers are cached. The reported "Fallback" count is the number of
samples whose final answer is routed to the baseline, not necessarily the number
of baseline answers computed for transition detection.

## 2. Discovery Labels

Ground-truth labels are used only on the discovery split to calibrate the fixed
router. Let \(g \in \{\texttt{yes}, \texttt{no}\}\) be the reference answer.

For changed predictions,

\[
\mathcal{C}_{\mathrm{chg}}
= \{x : a_B, a_T \in \{\texttt{yes}, \texttt{no}\},\; a_B \ne a_T\}.
\]

The intervention causes harm when the baseline was correct and the intervention
is wrong:

\[
h(x)=\mathbf{1}[a_B=g \land a_T\ne g].
\]

The intervention helps when the baseline was wrong and the intervention is
correct:

\[
b(x)=\mathbf{1}[a_B\ne g \land a_T=g].
\]

The net gain from falling back to the baseline on a selected set \(S\) is

\[
\Delta(S)=\sum_{x\in S} h(x) - \sum_{x\in S} b(x).
\]

RaPiC should select baseline fallback on high-risk harms while avoiding helpful
intervention corrections.

## 3. Transition Split

The changed set is split into two deployable transition subsets:

\[
\mathcal{C}_{Y\rightarrow N}
= \{x : a_B=\texttt{yes}, a_T=\texttt{no}\},
\]

\[
\mathcal{C}_{N\rightarrow Y}
= \{x : a_B=\texttt{no}, a_T=\texttt{yes}\}.
\]

RaPiC calibrates one controller for each direction:

\[
\rho_{Y\rightarrow N}(x), \qquad \rho_{N\rightarrow Y}(x).
\]

At deployment:

\[
\rho(x) =
\begin{cases}
\rho_{Y\rightarrow N}(x), & a_B=\texttt{yes}, a_T=\texttt{no},\\
\rho_{N\rightarrow Y}(x), & a_B=\texttt{no}, a_T=\texttt{yes},\\
\texttt{method}, & a_B=a_T \text{ or either label is invalid}.
\end{cases}
\]

This avoids hard-coding a global transition direction. VGA is mostly a
hallucination suppressor and often benefits from `yes->no` fallback, while PAI
and VAF can introduce harmful or helpful edits in either direction depending on
the backbone and dataset.

## 4. Replay Features

For each changed-answer candidate, the frozen backbone is replayed under the
original image-question context and the intervention answer. RaPiC uses two
feature families.

### C Features: Content Support

C features measure whether the intervention answer is internally supported by
the model under the image-question context. The final default feature pool is:

```text
cheap_lp_content_min
cheap_lp_content_std
cheap_entropy_content_mean
cheap_first_target_gap
cheap_target_gap_content_min
```

### D Features: Decision Confidence

D features measure the yes/no decision margin and confidence for the candidate
answer:

```text
cheap_decision_candidate_minus_alt
cheap_decision_candidate_prob_binary
cheap_decision_candidate_label_lp
cheap_decision_candidate_kl_uniform
```

`cheap_decision_candidate_kl_uniform` is derived from the binary candidate
probability when needed.

## 5. Feature Orientation and Scoring

On the discovery split, each feature is oriented toward intervention harm. For
feature \(f\), RaPiC chooses the orientation with higher AUROC for predicting
\(h(x)=1\). The oriented feature is standardized into a z-score.

Feature selection defaults:

```text
min_present_rate = 0.8
min_feature_auroc = 0.55
top_k_c = 3
top_k_d = 4
```

For a row \(x\), define

\[
s_C(x)=\mathrm{mean}_{f\in C} z_f(x),
\]

\[
s_D(x)=\mathrm{mean}_{f\in D} z_f(x).
\]

RaPiC searches three controller families:

\[
s_{\mathrm{C}}(x)=s_C(x),
\]

\[
s_{\mathrm{D}}(x)=s_D(x),
\]

\[
s_{\mathrm{CD}}(x;\alpha)=(1-\alpha)s_C(x)+\alpha s_D(x).
\]

The fusion alpha grid is

```text
0.025, 0.050, ..., 0.975
```

The fallback rule for each direction is

\[
\rho_d(x)=
\begin{cases}
\texttt{baseline}, & s_d(x)\ge \tau_d,\\
\texttt{method}, & s_d(x)<\tau_d.
\end{cases}
\]

where \(d \in \{Y\rightarrow N, N\rightarrow Y\}\).

## 6. Calibration Objective

For each transition direction, RaPiC sweeps over family, alpha, and threshold
on the discovery split. The main objective is final accuracy after routing:

\[
\mathrm{Acc}_{\mathrm{RaPiC}}(\rho_d).
\]

Equivalently, on the changed subset this maximizes the net recovered count
\(\sum h - \sum b\), while preserving all method outputs outside the selected
fallback set.

Default constraints:

```text
tau_objective = final_acc
min_selected_count = 5
min_baseline_rate = 0.0
max_baseline_rate = 1.0
allow_noop_policy = true
max_help_recall = 1.0
```

The `noop` policy routes every sample to the intervention output. It is included
as a safety option when a transition direction has no reliable fallback signal.
This matters for methods such as VAF, where one direction may be mostly helpful
or absent in discovery.

## 7. Deployment and Merge

The deployed split controller is the union of two direction-specific routes:

1. Apply the `yes->no` policy only to `yes->no` changed samples.
2. Apply the `no->yes` policy only to `no->yes` changed samples.
3. If either direction policy selects baseline, route to baseline.
4. Otherwise route to the intervention output.

The two transition subsets are disjoint, so no conflict resolution is needed in
practice.

## 8. Why Split Calibration

The failure mode motivating the split is Qwen VAF. Under pooled
`changed_answer` calibration, the discovery split and test splits have
substantially different intervention behavior:

```text
Discovery: mostly yes->no changes.
Test:      mostly no->yes changes for Qwen VAF.
```

Pooled calibration learns a threshold from discovery harms, then applies it to a
test distribution whose dominant changed examples are helpful `no->yes`
corrections. This can route nearly the entire changed set back to the baseline.

Observed symptom:

```text
Qwen VAF / MSCOCO pooled calibration:
tau = 0.3692
test score range = [1.5186, 3.8656]
selected = 224 / 224 changed candidates
```

This is not an implementation error. It is a violation of the calibration
representativeness assumption. Transition-split calibration reduces this risk
by calibrating each direction separately and allowing unsupported directions to
abstain.

## 9. Final Split-Calibration Results

The table below is the current final discriminative split-calibration result.
All values are percentages except counts and `H/G/Net`.

| Method / Backbone | Dataset | Policies | Base | Method | Split-Calib RaPiC | dMethod | dBase | Fallback | H/G/Net | Hrec | Grec |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VGA / LLaVA-1.5 | mscoco | Y:d_only@1.387 / N:c_only@-0.653 | 85.22 | 84.74 | 85.78 | +1.03 | +0.56 | 219 | 156/63/93 | 44.57 | 20.52 |
| VGA / LLaVA-1.5 | aokvqa | Y:d_only@1.387 / N:c_only@-0.653 | 78.98 | 81.04 | 82.31 | +1.27 | +3.33 | 182 | 148/34/114 | 46.39 | 6.73 |
| VGA / LLaVA-1.5 | gqa | Y:d_only@1.387 / N:c_only@-0.653 | 76.61 | 80.08 | 80.83 | +0.76 | +4.22 | 146 | 107/39/68 | 41.63 | 6.85 |
| VGA / LLaVA-NeXT | mscoco | Y:c_only@1.124 / N:c_only@-1.578 | 89.17 | 89.68 | 89.84 | +0.17 | +0.68 | 23 | 19/4/15 | 13.38 | 2.13 |
| VGA / LLaVA-NeXT | aokvqa | Y:c_only@1.124 / N:c_only@-1.578 | 85.23 | 85.96 | 86.54 | +0.59 | +1.31 | 69 | 61/8/53 | 37.20 | 3.49 |
| VGA / LLaVA-NeXT | gqa | Y:c_only@1.124 / N:c_only@-1.578 | 83.19 | 84.70 | 85.17 | +0.47 | +1.98 | 96 | 69/27/42 | 38.76 | 8.60 |
| VGA / Qwen2.5-VL-7B | mscoco | Y:c_only@-0.947 / N:d_only@-0.465 | 83.77 | 82.89 | 83.79 | +0.90 | +0.02 | 125 | 103/22/81 | 96.26 | 78.57 |
| VGA / Qwen2.5-VL-7B | aokvqa | Y:c_only@-0.947 / N:d_only@-0.465 | 85.00 | 84.88 | 85.23 | +0.36 | +0.23 | 158 | 95/63/32 | 94.06 | 70.00 |
| VGA / Qwen2.5-VL-7B | gqa | Y:c_only@-0.947 / N:d_only@-0.465 | 84.96 | 85.08 | 85.12 | +0.04 | +0.17 | 98 | 51/47/4 | 89.47 | 69.12 |
| PAI-attn / LLaVA-1.5 | mscoco | Y:cd_fusion@1.748 / N:c_only@-2.209 | 85.22 | 83.99 | 85.53 | +1.54 | +0.31 | 377 | 258/119/139 | 60.00 | 37.30 |
| PAI-attn / LLaVA-1.5 | aokvqa | Y:cd_fusion@1.748 / N:c_only@-2.209 | 78.98 | 77.04 | 79.22 | +2.18 | +0.24 | 286 | 241/45/196 | 61.01 | 20.36 |
| PAI-attn / LLaVA-1.5 | gqa | Y:cd_fusion@1.748 / N:c_only@-2.209 | 76.61 | 75.06 | 77.51 | +2.46 | +0.90 | 315 | 268/47/221 | 76.57 | 22.38 |
| VAF / LLaVA-1.5 | mscoco | Y:c_only@1.919 / N:noop | 85.22 | 86.47 | 86.59 | +0.12 | +1.37 | 37 | 24/13/11 | 6.30 | 2.64 |
| VAF / LLaVA-1.5 | aokvqa | Y:c_only@1.919 / N:noop | 78.98 | 81.32 | 81.58 | +0.26 | +2.60 | 37 | 30/7/23 | 8.40 | 1.23 |
| VAF / LLaVA-1.5 | gqa | Y:c_only@1.919 / N:noop | 76.61 | 80.58 | 80.59 | +0.01 | +3.98 | 17 | 9/8/1 | 3.05 | 1.23 |
| PAI-attn / Qwen2.5-VL-7B | mscoco | Y:cd_fusion@0.386 / N:noop | 83.77 | 83.79 | 83.99 | +0.20 | +0.22 | 18 | 18/0/18 | 94.74 | 0.00 |
| PAI-attn / Qwen2.5-VL-7B | aokvqa | Y:cd_fusion@0.386 / N:noop | 85.00 | 85.26 | 85.23 | -0.02 | +0.23 | 32 | 15/17/-2 | 45.45 | 30.36 |
| PAI-attn / Qwen2.5-VL-7B | gqa | Y:cd_fusion@0.386 / N:noop | 84.96 | 85.33 | 85.26 | -0.08 | +0.30 | 79 | 36/43/-7 | 81.82 | 55.13 |
| VAF / Qwen2.5-VL-7B | mscoco | Y:cd_fusion@0.374 / N:noop | 83.77 | 85.21 | 85.46 | +0.24 | +1.69 | 26 | 24/2/22 | 51.06 | 1.13 |
| VAF / Qwen2.5-VL-7B | aokvqa | Y:cd_fusion@0.374 / N:noop | 85.00 | 86.24 | 86.21 | -0.03 | +1.21 | 27 | 12/15/-3 | 13.95 | 7.58 |
| VAF / Qwen2.5-VL-7B | gqa | Y:cd_fusion@0.374 / N:noop | 84.96 | 87.03 | 86.60 | -0.43 | +1.64 | 75 | 18/57/-39 | 22.78 | 21.43 |

## 10. Interpretation

The split protocol is not always the highest-scoring variant for every
method/backbone pair, but it is the most stable unified discriminative protocol.

Main observations:

1. VGA benefits consistently across all three backbones.
2. LLaVA-1.5 PAI-attn benefits strongly. This confirms that harmful PAI edits
   are not confined to a single transition direction.
3. LLaVA-1.5 VAF remains positive but with smaller gains than pooled
   calibration, because the split protocol abstains more often.
4. Qwen VAF no longer catastrophically fails on MSCOCO, but AOKVQA is near
   no-op and GQA remains negative. This is the remaining hard case and should be
   discussed as a transition/score-shift failure mode.
5. Qwen PAI-attn is mostly conservative: one positive case and two small
   negatives, all with much smaller damage than pooled calibration.

The paper should therefore avoid the claim that RaPiC improves every
intervention/backbone/dataset. The safer claim is:

> Transition-aware RaPiC provides a unified post-hoc controller that improves
> many intervention settings and exposes when the calibration split does not
> support safe fallback.

## 11. Reproduction

The split-calibration diagnostic was generated from existing discovery/test
feature rows and apply outputs:

```bash
cd /home/kms/LLaVA_calibration
git pull

export CAL=/home/kms/LLaVA_calibration
export CAL_PY=/home/kms/miniconda3/envs/vga_base/bin/python

$CAL_PY scripts/run_transition_split_calibration_from_existing_features.py \
  2>&1 | tee "$CAL/experiments/paper_pcp_cd_transition_split_calib_existing/run.log"
```

Output table:

```bash
cat /home/kms/LLaVA_calibration/experiments/paper_pcp_cd_transition_split_calib_existing/transition_split_calib_summary.md
```

Relevant scripts:

```text
scripts/build_pcp_c_d_controller.py
scripts/apply_pcp_c_d_controller.py
scripts/run_transition_split_calibration_from_existing_features.py
scripts/summarize_pcp_deployment_from_routes.py
```

## 12. Paper Placement

Recommended paper structure:

1. Method: define RaPiC as post-intervention routing.
2. Method: introduce transition-split calibration as the final discriminative
   controller.
3. Main discriminative table: report split-calibration results.
4. Ablation: compare pooled `changed_answer`, hard `yes->no`, and split
   calibration.
5. Failure analysis: discuss Qwen VAF GQA as a calibration mismatch case, using
   transition distribution and score saturation as label-free diagnostics.

