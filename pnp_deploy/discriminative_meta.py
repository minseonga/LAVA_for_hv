from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def maybe_float(value: object) -> Optional[float]:
    s = str(value if value is not None else "").strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(s)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def maybe_int(value: object) -> Optional[int]:
    value_f = maybe_float(value)
    if value_f is None:
        return None
    return int(round(value_f))


def safe_div(num: float, den: float) -> float:
    if float(den) == 0.0:
        return 0.0
    return float(num / den)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(str(key))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def infer_case_type(base_correct: Optional[int], int_correct: Optional[int]) -> str:
    if base_correct is None or int_correct is None:
        return "unknown"
    if int(base_correct) == 1 and int(int_correct) == 0:
        return "regression"
    if int(base_correct) == 0 and int(int_correct) == 1:
        return "improvement"
    if int(base_correct) == 1 and int(int_correct) == 1:
        return "both_correct"
    if int(base_correct) == 0 and int(int_correct) == 0:
        return "both_wrong"
    return "unknown"


def merge_score_feature_rows(scores_csv: str, features_csv: str) -> List[Dict[str, Any]]:
    score_rows = read_csv_rows(scores_csv)
    feat_rows = read_csv_rows(features_csv)
    feat_map = {str(row.get("id", "")).strip(): row for row in feat_rows}
    merged: List[Dict[str, Any]] = []
    for row in score_rows:
        sid = str(row.get("id", "")).strip()
        feat = feat_map.get(sid, {})
        baseline_correct = maybe_int(row.get("baseline_correct"))
        intervention_correct = maybe_int(row.get("intervention_correct"))
        case_type = infer_case_type(baseline_correct, intervention_correct)
        merged_row: Dict[str, Any] = dict(row)
        merged_row["id"] = sid
        merged_row["baseline_correct"] = baseline_correct
        merged_row["intervention_correct"] = intervention_correct
        merged_row["case_type"] = case_type
        merged_row["harm"] = int(case_type == "regression")
        merged_row["help"] = int(case_type == "improvement")
        for key, value in feat.items():
            if key == "id":
                continue
            merged_row[key] = value
        merged.append(merged_row)
    return merged


@dataclass(frozen=True)
class FeatureSpec:
    feature: str
    direction: str
    mu: float
    sd: float

    @classmethod
    def from_mapping(cls, obj: Mapping[str, Any]) -> "FeatureSpec":
        return cls(
            feature=str(obj["feature"]),
            direction=str(obj.get("direction", "high")),
            mu=float(obj.get("mu", 0.0)),
            sd=max(float(obj.get("sd", 1.0)), 1e-6),
        )

    def oriented_z(self, row: Mapping[str, Any]) -> Optional[float]:
        raw = maybe_float(row.get(self.feature))
        if raw is None:
            return None
        oriented = float(raw) if self.direction == "high" else -float(raw)
        return float((oriented - self.mu) / self.sd)


@dataclass(frozen=True)
class ExpertSpec:
    tau: float
    w_b: float = 0.0
    w_c: float = 0.0

    @classmethod
    def from_mapping(cls, obj: Optional[Mapping[str, Any]]) -> Optional["ExpertSpec"]:
        if not obj:
            return None
        return cls(
            tau=float(obj["tau"]),
            w_b=float(obj.get("w_b", 0.0)),
            w_c=float(obj.get("w_c", 0.0)),
        )


def sign(value: Optional[float]) -> int:
    if value is None:
        return 0
    if float(value) > 0:
        return 1
    if float(value) < 0:
        return -1
    return 0


def expert_route(score: Optional[float], tau: Optional[float]) -> Optional[str]:
    if score is None or tau is None:
        return None
    return "baseline" if float(score) >= float(tau) else "method"


@dataclass(frozen=True)
class MetaStrongDecision:
    expert: str
    route: str
    b_score: Optional[float]
    c_score: Optional[float]
    f_score: Optional[float]
    score: Optional[float]
    tau: Optional[float]


class MetaStrongController:
    """Pure routing core for the paper `meta_strong` discriminative controller."""

    def __init__(
        self,
        *,
        b_feature: Optional[FeatureSpec],
        c_features: Sequence[FeatureSpec],
        b_expert: Optional[ExpertSpec],
        c_expert: Optional[ExpertSpec],
        fusion_expert: Optional[ExpertSpec],
        mode: str,
        delta: float,
    ) -> None:
        self.b_feature = b_feature
        self.c_features = list(c_features)
        self.b_expert = b_expert
        self.c_expert = c_expert
        self.fusion_expert = fusion_expert
        self.mode = str(mode)
        self.delta = float(delta)

    @classmethod
    def from_bundle(cls, bundle: Mapping[str, Any]) -> "MetaStrongController":
        best_experts = bundle.get("best_experts") or {}
        best_meta_policy = bundle.get("best_meta_policy") or {}
        if not best_experts or not best_meta_policy:
            raise ValueError("Policy bundle is missing best_experts or best_meta_policy.")
        b_raw = bundle.get("best_b_feature")
        return cls(
            b_feature=None if not b_raw else FeatureSpec.from_mapping(b_raw),
            c_features=[FeatureSpec.from_mapping(x) for x in (bundle.get("selected_c_features") or [])],
            b_expert=ExpertSpec.from_mapping(best_experts.get("b_only")),
            c_expert=ExpertSpec.from_mapping(best_experts.get("c_only")),
            fusion_expert=ExpertSpec.from_mapping(best_experts.get("fusion")),
            mode=str(best_meta_policy["mode"]),
            delta=float(best_meta_policy["delta"]),
        )

    @classmethod
    def from_bundle_path(cls, path: str) -> "MetaStrongController":
        return cls.from_bundle(read_json(path))

    def score_components(self, row: Mapping[str, Any]) -> Dict[str, Optional[float]]:
        b_score = None if self.b_feature is None else self.b_feature.oriented_z(row)
        c_vals = [feat.oriented_z(row) for feat in self.c_features]
        if any(value is None for value in c_vals):
            c_score = None
        else:
            c_score = None if not c_vals else float(sum(float(v) for v in c_vals if v is not None) / len(c_vals))

        f_score = None
        if b_score is not None or c_score is not None:
            w_b = 0.0 if self.fusion_expert is None else float(self.fusion_expert.w_b)
            w_c = 0.0 if self.fusion_expert is None else float(self.fusion_expert.w_c)
            f_score = float(w_b * float(b_score or 0.0) + w_c * float(c_score or 0.0))
        return {"b_score": b_score, "c_score": c_score, "f_score": f_score}

    def choose_expert(self, b_score: Optional[float], c_score: Optional[float]) -> str:
        b_ok = b_score is not None
        c_ok = c_score is not None
        if b_ok and not c_ok:
            return "b_only"
        if c_ok and not b_ok:
            return "c_only"
        if not b_ok and not c_ok:
            return "none"

        assert b_score is not None and c_score is not None
        abs_b = abs(float(b_score))
        abs_c = abs(float(c_score))
        if abs_b - abs_c >= self.delta:
            return "b_only"
        if abs_c - abs_b >= self.delta:
            return "c_only"
        if self.mode == "delta_then_fusion":
            return "fusion"
        if self.mode == "delta_then_stronger":
            return "b_only" if abs_b >= abs_c else "c_only"
        if self.mode == "agree_fusion_else_stronger":
            return "fusion" if sign(b_score) == sign(c_score) else ("b_only" if abs_b >= abs_c else "c_only")
        return "fusion"

    def decide(self, row: Mapping[str, Any]) -> MetaStrongDecision:
        scores = self.score_components(row)
        expert = self.choose_expert(scores["b_score"], scores["c_score"])
        score: Optional[float]
        tau: Optional[float]
        if expert == "b_only":
            score = scores["b_score"]
            tau = None if self.b_expert is None else self.b_expert.tau
        elif expert == "c_only":
            score = scores["c_score"]
            tau = None if self.c_expert is None else self.c_expert.tau
        elif expert == "fusion":
            score = scores["f_score"]
            tau = None if self.fusion_expert is None else self.fusion_expert.tau
        else:
            score = None
            tau = None
        route = expert_route(score, tau)
        if route is None:
            route = "method"
        return MetaStrongDecision(
            expert=expert,
            route=route,
            b_score=scores["b_score"],
            c_score=scores["c_score"],
            f_score=scores["f_score"],
            score=score,
            tau=tau,
        )

    def route_row(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        decision = self.decide(row)
        baseline_correct = maybe_int(row.get("baseline_correct"))
        intervention_correct = maybe_int(row.get("intervention_correct"))
        final_correct: Optional[int] = None
        if baseline_correct is not None and intervention_correct is not None:
            final_correct = baseline_correct if decision.route == "baseline" else intervention_correct
        return {
            "id": str(row.get("id", "")).strip(),
            "expert": decision.expert,
            "route": decision.route,
            "b_score": decision.b_score,
            "c_score": decision.c_score,
            "f_score": decision.f_score,
            "score": decision.score,
            "tau": decision.tau,
            "harm": int(maybe_int(row.get("harm")) or 0),
            "help": int(maybe_int(row.get("help")) or 0),
            "baseline_correct": baseline_correct,
            "intervention_correct": intervention_correct,
            "final_correct": final_correct,
        }

    def evaluate(self, rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        route_rows = [self.route_row(row) for row in rows]
        n = 0
        baseline_correct_total = 0
        intervention_correct_total = 0
        final_correct_total = 0
        total_harm = 0
        total_help = 0
        selected_harm = 0
        selected_help = 0
        selected_neutral = 0
        selected_count = 0
        expert_counts = {"b_only": 0, "c_only": 0, "fusion": 0, "none": 0}

        for row in route_rows:
            baseline_correct = maybe_int(row.get("baseline_correct"))
            intervention_correct = maybe_int(row.get("intervention_correct"))
            if baseline_correct is None or intervention_correct is None:
                continue
            n += 1
            harm = int(maybe_int(row.get("harm")) or 0)
            help_ = int(maybe_int(row.get("help")) or 0)
            total_harm += harm
            total_help += help_
            baseline_correct_total += baseline_correct
            intervention_correct_total += intervention_correct
            expert = str(row.get("expert", "none"))
            expert_counts[expert] = int(expert_counts.get(expert, 0)) + 1
            if str(row.get("route")) == "baseline":
                selected_count += 1
                selected_harm += harm
                selected_help += help_
                selected_neutral += int((harm == 0) and (help_ == 0))
                final_correct_total += baseline_correct
            else:
                final_correct_total += intervention_correct

        baseline_rate = safe_div(float(selected_count), float(max(1, n)))
        precision = safe_div(float(selected_harm), float(max(1, selected_count)))
        recall = safe_div(float(selected_harm), float(max(1, total_harm)))
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        return route_rows, {
            "mode": self.mode,
            "delta": self.delta,
            "n_eval": int(n),
            "baseline_rate": baseline_rate,
            "method_rate": float(1.0 - baseline_rate),
            "final_acc": safe_div(float(final_correct_total), float(max(1, n))),
            "baseline_acc": safe_div(float(baseline_correct_total), float(max(1, n))),
            "intervention_acc": safe_div(float(intervention_correct_total), float(max(1, n))),
            "delta_vs_intervention": safe_div(float(final_correct_total - intervention_correct_total), float(max(1, n))),
            "selected_count": int(selected_count),
            "selected_harm": int(selected_harm),
            "selected_help": int(selected_help),
            "selected_neutral": int(selected_neutral),
            "selected_harm_precision": precision,
            "selected_harm_recall": recall,
            "selected_harm_f1": f1,
            "expert_b_only_rate": safe_div(float(expert_counts["b_only"]), float(max(1, n))),
            "expert_c_only_rate": safe_div(float(expert_counts["c_only"]), float(max(1, n))),
            "expert_fusion_rate": safe_div(float(expert_counts["fusion"]), float(max(1, n))),
        }


def compare_route_rows(current_rows: Sequence[Mapping[str, Any]], reference_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ref_by_id = {str(row.get("id", "")).strip(): row for row in reference_rows}
    n = 0
    route_mismatch = 0
    expert_mismatch = 0
    final_mismatch = 0
    score_max_abs_diff = {"b_score": 0.0, "c_score": 0.0, "f_score": 0.0}
    examples: List[Dict[str, Any]] = []
    for row in current_rows:
        sid = str(row.get("id", "")).strip()
        ref = ref_by_id.get(sid)
        if ref is None:
            continue
        n += 1
        cur_route = str(row.get("route", ""))
        ref_route = str(ref.get("route", ""))
        cur_expert = str(row.get("expert", ""))
        ref_expert = str(ref.get("expert", ""))
        cur_final = str(row.get("final_correct", ""))
        ref_final = str(ref.get("final_correct", ""))
        mismatch = False
        if cur_route != ref_route:
            route_mismatch += 1
            mismatch = True
        if cur_expert != ref_expert:
            expert_mismatch += 1
            mismatch = True
        if cur_final != ref_final:
            final_mismatch += 1
            mismatch = True
        for key in score_max_abs_diff:
            cur = maybe_float(row.get(key))
            old = maybe_float(ref.get(key))
            if cur is not None and old is not None:
                score_max_abs_diff[key] = max(score_max_abs_diff[key], abs(float(cur) - float(old)))
        if mismatch and len(examples) < 10:
            examples.append(
                {
                    "id": sid,
                    "route": cur_route,
                    "reference_route": ref_route,
                    "expert": cur_expert,
                    "reference_expert": ref_expert,
                    "final_correct": cur_final,
                    "reference_final_correct": ref_final,
                }
            )
    return {
        "n_compared": int(n),
        "route_mismatch": int(route_mismatch),
        "expert_mismatch": int(expert_mismatch),
        "final_correct_mismatch": int(final_mismatch),
        "score_max_abs_diff": score_max_abs_diff,
        "examples": examples,
    }

