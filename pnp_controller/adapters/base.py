from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from pnp_controller.core.schemas import HardVetoConfig, OfflineTableSchema, ProbeState


class OfflineComparisonAdapter(ABC):
    backend_name = "offline"

    def __init__(self, schema: OfflineTableSchema | None = None) -> None:
        self.schema = schema or OfflineTableSchema()

    @abstractmethod
    def load_per_case(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def load_features(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def load_merged(self, controller_cfg: HardVetoConfig) -> pd.DataFrame:
        raise NotImplementedError


class OnlineMethodAdapter(ABC):
    backend_name = "online"

    @abstractmethod
    def probe(self, sample: Any, branch_text: str | None = None) -> ProbeState:
        raise NotImplementedError

    @abstractmethod
    def predict_base(self, sample: Any, probe_state: ProbeState | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict_method(self, sample: Any, probe_state: ProbeState | None = None) -> Any:
        raise NotImplementedError
