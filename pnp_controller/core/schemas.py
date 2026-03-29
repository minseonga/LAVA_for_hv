from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ThresholdCalibrationConfig:
    calib_ratio: float = 0.3
    seed: int = 42
    lambda_improvement: float = 1.0
    max_wrong_veto_rate: float = 0.35
    q_grid: List[float] = field(
        default_factory=lambda: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )


@dataclass
class HardVetoConfig:
    frg_col: str = "faithful_minus_global_attn"
    gmi_col: str = "guidance_mismatch_score"
    improvement_case_value: str = "vga_improvement"
    regression_case_value: str = "vga_regression"
    fallback_when_missing_feature: str = "method"
    tau_frg: Optional[float] = None
    tau_gmi: Optional[float] = None
    calibration: ThresholdCalibrationConfig = field(default_factory=ThresholdCalibrationConfig)


@dataclass
class OfflineTableSchema:
    id_col_per_case: str = "id"
    id_col_feature: str = "id"
    gt_col: str = "gt"
    baseline_col: str = "pred_baseline"
    method_col: str = "pred_vga"
    case_col: str = "case_type"


@dataclass
class CalibrationResult:
    tau_frg: float
    tau_gmi: float
    objective: float
    n_cal: int
    wrong_veto_count: int
    correct_veto_count: int
    veto_rate: float
    mode: str = "calibrated"


@dataclass
class ProbeState:
    sample_id: str
    frg: float
    gmi: float
    extras: Dict[str, Any] = field(default_factory=dict)
