from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Optional, Tuple

from pnp_deploy.discriminative_meta import FeatureSpec, write_json


def read_json(path: str) -> Any:
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        return json.load(f)


def parse_yes_no(text: object) -> str:
    s = str(text or "").strip().lower()
    if not s:
        return ""
    first = s.split(".", 1)[0].replace(",", " ")
    words = {w.strip() for w in first.split()}
    if "no" in words or "not" in words:
        return "no"
    if "yes" in words:
        return "yes"
    if s.startswith("no"):
        return "no"
    if s.startswith("yes"):
        return "yes"
    return ""


@dataclass(frozen=True)
class DirectionPolicy:
    direction: str
    tau: float
    features: Tuple[FeatureSpec, ...]
    disabled: bool = False

    @classmethod
    def from_mapping(cls, direction: str, obj: Mapping[str, Any]) -> "DirectionPolicy":
        disabled = bool(obj.get("disabled")) or str(obj.get("family", "")) == "noop"
        features = tuple(FeatureSpec.from_mapping(x) for x in (obj.get("selected_c_features") or []))
        return cls(
            direction=str(direction),
            tau=float(obj.get("tau", 0.0) or 0.0),
            features=features,
            disabled=disabled,
        )

    def score(self, row: Mapping[str, Any]) -> Optional[float]:
        if self.disabled or not self.features:
            return None
        vals = []
        for feature in self.features:
            value = feature.oriented_z(row)
            if value is None or not math.isfinite(float(value)):
                return None
            vals.append(float(value))
        return float(median(vals)) if vals else None

    def reaches_tau(self, score: Optional[float]) -> bool:
        return score is not None and float(score) >= float(self.tau)


@dataclass(frozen=True)
class TransitionScores:
    yes_to_no_score: Optional[float]
    no_to_yes_score: Optional[float]
    yes_to_no_tau: float
    no_to_yes_tau: float

    @property
    def may_need_baseline(self) -> bool:
        return (
            self.yes_to_no_score is not None
            and float(self.yes_to_no_score) >= float(self.yes_to_no_tau)
        ) or (
            self.no_to_yes_score is not None
            and float(self.no_to_yes_score) >= float(self.no_to_yes_tau)
        )


@dataclass(frozen=True)
class LazyTransitionDecision:
    route: str
    final_text: str
    final_source: str
    method_label: str
    baseline_label: str
    actual_direction: str
    selected_score: Optional[float]
    selected_tau: Optional[float]
    baseline_generated: bool
    reason: str


class FixedCTransitionController:
    """Deployment controller for transition-split fixed-C median policies.

    The controller scores the method answer first. Baseline generation can be
    skipped whenever neither directional score reaches its calibrated threshold.
    """

    def __init__(self, *, yes_to_no: DirectionPolicy, no_to_yes: DirectionPolicy) -> None:
        self.yes_to_no = yes_to_no
        self.no_to_yes = no_to_yes

    @classmethod
    def from_policy_mappings(
        cls,
        *,
        yes_policy: Mapping[str, Any],
        no_policy: Mapping[str, Any],
    ) -> "FixedCTransitionController":
        return cls(
            yes_to_no=DirectionPolicy.from_mapping("yes_to_no", yes_policy),
            no_to_yes=DirectionPolicy.from_mapping("no_to_yes", no_policy),
        )

    @classmethod
    def from_fixed_json(cls, path: str, *, target: str, dataset: str) -> "FixedCTransitionController":
        payload = read_json(path)
        for item in payload.get("per_dataset") or []:
            if str(item.get("target", "")).strip() != str(target):
                continue
            if str(item.get("dataset", "")).strip() != str(dataset):
                continue
            yes_policy = item.get("yes_policy_json") or {}
            no_policy = item.get("no_policy_json") or {}
            if not yes_policy or not no_policy:
                raise ValueError(f"Missing yes/no policies for {target}/{dataset} in {path}")
            return cls.from_policy_mappings(yes_policy=yes_policy, no_policy=no_policy)
        raise ValueError(f"No fixed-C policy for target={target!r}, dataset={dataset!r} in {path}")

    def score(self, row: Mapping[str, Any]) -> TransitionScores:
        return TransitionScores(
            yes_to_no_score=self.yes_to_no.score(row),
            no_to_yes_score=self.no_to_yes.score(row),
            yes_to_no_tau=float(self.yes_to_no.tau),
            no_to_yes_tau=float(self.no_to_yes.tau),
        )

    def decide(
        self,
        *,
        method_text: str,
        baseline_text: str,
        scores: TransitionScores,
        baseline_generated: bool,
    ) -> LazyTransitionDecision:
        method_label = parse_yes_no(method_text)
        baseline_label = parse_yes_no(baseline_text)
        if not baseline_generated:
            return LazyTransitionDecision(
                route="method",
                final_text=method_text,
                final_source="method_early_keep",
                method_label=method_label,
                baseline_label="",
                actual_direction="",
                selected_score=None,
                selected_tau=None,
                baseline_generated=False,
                reason="both_direction_scores_below_tau",
            )

        if baseline_label not in {"yes", "no"} or method_label not in {"yes", "no"}:
            return LazyTransitionDecision(
                route="method",
                final_text=method_text,
                final_source="method_parse_failure",
                method_label=method_label,
                baseline_label=baseline_label,
                actual_direction="",
                selected_score=None,
                selected_tau=None,
                baseline_generated=True,
                reason="unparseable_yesno",
            )
        if baseline_label == method_label:
            return LazyTransitionDecision(
                route="method",
                final_text=method_text,
                final_source="method_unchanged",
                method_label=method_label,
                baseline_label=baseline_label,
                actual_direction="unchanged",
                selected_score=None,
                selected_tau=None,
                baseline_generated=True,
                reason="baseline_and_method_same_label",
            )

        if baseline_label == "yes" and method_label == "no":
            direction = "yes_to_no"
            selected_score = scores.yes_to_no_score
            selected_tau = scores.yes_to_no_tau
        elif baseline_label == "no" and method_label == "yes":
            direction = "no_to_yes"
            selected_score = scores.no_to_yes_score
            selected_tau = scores.no_to_yes_tau
        else:
            direction = ""
            selected_score = None
            selected_tau = None

        use_baseline = selected_score is not None and selected_tau is not None and float(selected_score) >= float(selected_tau)
        return LazyTransitionDecision(
            route="baseline" if use_baseline else "method",
            final_text=baseline_text if use_baseline else method_text,
            final_source="baseline_lazy" if use_baseline else "method_direction_below_tau",
            method_label=method_label,
            baseline_label=baseline_label,
            actual_direction=direction,
            selected_score=selected_score,
            selected_tau=selected_tau,
            baseline_generated=True,
            reason="direction_score_reaches_tau" if use_baseline else "actual_direction_score_below_tau",
        )

    def write_bundle(self, path: str) -> None:
        write_json(
            path,
            {
                "mode": "fixed_c_transition_lazy",
                "yes_to_no": {
                    "tau": self.yes_to_no.tau,
                    "disabled": self.yes_to_no.disabled,
                    "features": [feature.__dict__ for feature in self.yes_to_no.features],
                },
                "no_to_yes": {
                    "tau": self.no_to_yes.tau,
                    "disabled": self.no_to_yes.disabled,
                    "features": [feature.__dict__ for feature in self.no_to_yes.features],
                },
            },
        )


def score_or_blank(value: Optional[float]) -> object:
    return "" if value is None else float(value)
