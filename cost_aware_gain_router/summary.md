# Cost-Aware Gain Router

Detailed explainer: [README.md](./README.md)
Server execution guide: [SERVER_ONLINE_EXPERIMENT.md](./SERVER_ONLINE_EXPERIMENT.md)

## Setup
- Runtime goal: run only one expensive branch (`baseline` or `method`) after a cheap probe.
- Training target: signed utility of choosing method over baseline (`+1` gain / `-1` harm / `0` neutral).
- Offline tau in probe log: -0.00684115

## Reference policies
- Baseline only: accuracy=0.8522, method_rate=0.0000
- VGA only: accuracy=0.8661, method_rate=1.0000
- Offline controller: accuracy=0.8764, method_rate=0.5942
- Strict gain oracle: accuracy=0.9046, method_rate=0.0523

## Efficient operating points
- HGB utility router @20% method budget: accuracy=0.8691, selected_gain_rate=0.1294, selected_harm_rate=0.0450
- HGB utility router @30% method budget: accuracy=0.8726, selected_gain_rate=0.1081, selected_harm_rate=0.0404
- Best HGB point on tested grid: budget=0.50, accuracy=0.8751
- Best depth-3 tree point on tested grid: budget=0.30, accuracy=0.8662

## Interpretation
- Training does require both branches offline once, because the gain/harm label comes from comparing their correctness.
- Inference does not require both branches: compute the cheap probe features, score utility, then run exactly one branch.
- If efficiency matters more than the last few points of accuracy, a budgeted router is the right framing, not route imitation.

## Output files
- `reference_policies.csv`
- `budget_sweep.csv`
- `oof_scores.csv`
- `tree_utility_rules.txt`
- `deployment_rule_30_budget.md`
- `deployment_score_bands_30_budget.csv`
- `deployment_tree_leaf_table.csv`
