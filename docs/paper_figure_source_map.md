# Paper Figure Source Map

This note maps each proposed paper figure to the artifacts already present in the
repo or to the minimum additional export needed.

## Figure 1. Framework Overview

Status: ready, but should be drawn manually as a schematic.

Message:
- shared framework
- intervention-first
- GT-free post-hoc observable
- discovery-time supervision
- frozen held-out apply
- discriminative -> meta
- generative -> constrained tree

Source material:
- [method_section_draft_v2.md](/Users/gangminseong/LAVA_for_hv/docs/method_section_draft_v2.md)
- [algorithm1_shared_discovery_to_heldout_protocol.md](/Users/gangminseong/LAVA_for_hv/docs/algorithm1_shared_discovery_to_heldout_protocol.md)
- [analysis_driven_posthoc_control_paper_draft.md](/Users/gangminseong/LAVA_for_hv/docs/analysis_driven_posthoc_control_paper_draft.md)

Notes:
- This should be a clean diagram, not a data plot.

## Figure 2. Why Always-On Intervention Is Unsafe

Status: ready for discriminative, ready for generative if we use the existing
claim-aware table.

Recommended discriminative data:
- discovery raw outcome composition by method
- held-out raw outcome composition by method

Available sources:
- VGA discovery/test stage-B summaries:
  - [summary.json](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_b_c_v1_full/discovery_stageb/summary.json)
  - [summary.json](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_b_c_v1_full/test_stageb/summary.json)
- PAI discovery/test stage-B summaries:
  - [summary.json](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_pai_full/discovery_stageb/summary.json)
  - [summary.json](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_pai_full/test_stageb/summary.json)
- Pairwise harm/help separation files:
  - [pairwise_metrics.csv](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_b_c_v1_full/discovery_stageb/stage_b_validation/pairwise_metrics.csv)
  - [pairwise_metrics.csv](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_pai_full/discovery_stageb/stage_b_validation/pairwise_metrics.csv)

Available generative source:
- [vga_claim_aware_table.csv](/Users/gangminseong/LAVA_for_hv/experiments/vga_generative_coverage_probe_v1/vga_claim_aware_table.csv)

Recommended plot:
- stacked bars: help / harm / neutral

## Figure 3. Discriminative Harmful Signature

Status: ready.

Message:
- harmful cases appear as answer-critical token-local collapse

Available sources:
- VGA discovery score/features:
  - [sample_scores.csv](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_b_c_v1_full/discovery_stageb/sample_scores.csv)
  - [cheap_online_features.csv](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_b_c_v1_full/discovery/cheap_online_features.csv)
- PAI discovery score/features:
  - [sample_scores.csv](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_pai_full/discovery_stageb/sample_scores.csv)
  - [cheap_online_features.csv](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_pai_full/discovery/cheap_online_features.csv)

Best panel candidates:
- `cheap_target_gap_content_min`
- `cheap_lp_content_q10` or `cheap_lp_content_min`
- `cheap_conflict_lp_minus_entropy` if VCD stays in appendix

Recommended plot:
- score distribution by regression / improvement / neutral
- or regression rate by score bin

## Figure 4. Meta-Controller Intuition / Expert Arbitration

Status: ready for VGA and PAI.

Message:
- expert arbitration is sample-wise
- no single detector is uniformly best

Available sources:
- VGA meta route rows:
  - strong held-out route rows should be taken from the server artifact
  - canonical path on server:
    `/home/kms/LLaVA_calibration/experiments/paper_main_meta_vga_full_strong/test/meta_fixed_eval/meta_route_rows.csv`
- PAI meta route rows:
  - [meta_route_rows.csv](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_pai_full/test/meta_fixed_eval/meta_route_rows.csv)

Recommended plot:
- 2D scatter: `b_score` vs `c_score`
- point color: chosen expert
- point marker or alpha: harm / help / neutral

Backup plot:
- expert usage stacked bar by method

## Figure 5. Generative Harmful Signature

Status: ready.

Message:
- harmful generative cases are driven by supported-claim loss / omission risk,
  not just weak hallucinated additions

Available source:
- [vga_claim_aware_table.csv](/Users/gangminseong/LAVA_for_hv/experiments/vga_generative_coverage_probe_v1/vga_claim_aware_table.csv)
- [vga_claim_aware_table.summary.json](/Users/gangminseong/LAVA_for_hv/experiments/vga_generative_coverage_probe_v1/vga_claim_aware_table.summary.json)

Recommended plot:
- 2D scatter
  - x: `delta_hall_rate` or CHAIR-style hallucination change
  - y: `delta_supported_recall` or `delta_f1`
  - color: help / harm / neutral

Backup plot:
- grouped bars:
  - supported claim dropped
  - supported claim gained
  - wrong claim removed
  - wrong claim added

## Figure 6. Generative Controller Behavior

Status: yellow. Conceptually ready, but the canonical constrained-tree export
should be re-saved from the current run before plotting.

Message:
- generative control should preserve hallucination quality while recovering F1
- constrained tree is better aligned than unconstrained ranking

Needed points:
- baseline
- intervention
- unconstrained meta or linear
- constrained tree

Expected source artifacts:
- constrained tree summary from the VGA generative run
- discovery meta summary from the VGA generative run
- claim-aware table for baseline/intervention coordinates

Current repo state:
- the main claim-aware table is present:
  - [vga_claim_aware_table.csv](/Users/gangminseong/LAVA_for_hv/experiments/vga_generative_coverage_probe_v1/vga_claim_aware_table.csv)
- the constrained tree and generative meta result files were previously saved to
  temporary paths and should be re-exported into `experiments/` before final plotting

Recommended plot:
- Pareto scatter with four points:
  - baseline
  - intervention
  - unconstrained controller
  - constrained tree

## Figure 7. Discovery-to-Held-Out Protocol Sanity

Status: optional. Likely unnecessary if Algorithm 1 stays.

Available source:
- [algorithm1_shared_discovery_to_heldout_protocol.md](/Users/gangminseong/LAVA_for_hv/docs/algorithm1_shared_discovery_to_heldout_protocol.md)

Recommendation:
- keep as Algorithm 1, not a figure

## Figure 8. Calibration / Robustness

Status: appendix only.

Available sources:
- discovery/test threshold sweeps:
  - [threshold_sweep.csv](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_pai_full/test_stageb/threshold_sweep.csv)
  - [threshold_sweep.csv](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_b_c_v1_full/test_stageb/threshold_sweep.csv)
- discovery/test operating sweeps:
  - [operating_sweep.csv](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_meta_pai_full/test_stageb/stage_b_validation/operating_sweep.csv)
  - [operating_sweep.csv](/Users/gangminseong/LAVA_for_hv/experiments/paper_main_b_c_v1_full/test_stageb/stage_b_validation/operating_sweep.csv)

Recommendation:
- keep this in appendix unless the main paper needs an anti-brittleness defense

## Recommended Main Set

If we keep the paper compact, the safest main set is:
- Figure 1: framework schematic
- Figure 2: always-on intervention is unsafe
- Figure 3: discriminative harmful signature
- Figure 4: meta-controller arbitration
- Figure 5: generative harmful signature
- Figure 6: generative constrained-tree behavior

Immediate blocker:
- Figure 6 should be re-exported from the current constrained-tree run into a
  stable `experiments/...` directory before plotting.
