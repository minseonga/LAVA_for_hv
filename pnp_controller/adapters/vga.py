from __future__ import annotations

from pnp_controller.core.schemas import OfflineTableSchema

from .offline_csv import GenericOfflineCsvAdapter


class VGAOfflineAdapter(GenericOfflineCsvAdapter):
    backend_name = "vga"

    def __init__(self, per_case_csv: str, features_csv: str) -> None:
        super().__init__(
            per_case_csv=per_case_csv,
            features_csv=features_csv,
            schema=OfflineTableSchema(
                id_col_per_case="id",
                id_col_feature="id",
                gt_col="gt",
                baseline_col="pred_baseline",
                method_col="pred_vga",
                case_col="case_type",
            ),
        )
