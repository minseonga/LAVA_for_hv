from __future__ import annotations

import os

import pandas as pd

from pnp_controller.core.features import merge_feature_table
from pnp_controller.core.schemas import HardVetoConfig, OfflineTableSchema

from .base import OfflineComparisonAdapter


class GenericOfflineCsvAdapter(OfflineComparisonAdapter):
    backend_name = "generic_csv"

    def __init__(self, per_case_csv: str, features_csv: str, schema: OfflineTableSchema | None = None) -> None:
        super().__init__(schema=schema)
        self.per_case_csv = os.path.abspath(per_case_csv)
        self.features_csv = os.path.abspath(features_csv)

    def load_per_case(self) -> pd.DataFrame:
        return pd.read_csv(self.per_case_csv)

    def load_features(self) -> pd.DataFrame:
        return pd.read_csv(self.features_csv)

    def load_merged(self, controller_cfg: HardVetoConfig) -> pd.DataFrame:
        per_case = self.load_per_case()
        features = self.load_features()
        return merge_feature_table(
            per_case=per_case,
            features=features,
            schema=self.schema,
            controller_cfg=controller_cfg,
        )
