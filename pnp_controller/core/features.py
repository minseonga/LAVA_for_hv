from __future__ import annotations

import pandas as pd

from .schemas import HardVetoConfig, OfflineTableSchema


def merge_feature_table(
    per_case: pd.DataFrame,
    features: pd.DataFrame,
    schema: OfflineTableSchema,
    controller_cfg: HardVetoConfig,
) -> pd.DataFrame:
    cols = [schema.id_col_feature, controller_cfg.frg_col, controller_cfg.gmi_col]
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    feat = features[cols].copy()
    feat = feat.rename(
        columns={
            schema.id_col_feature: "__id__",
            controller_cfg.frg_col: "__FRG__",
            controller_cfg.gmi_col: "__GMI__",
        }
    )

    merged = per_case.copy()
    if schema.id_col_per_case not in merged.columns:
        raise ValueError(f"Missing per-case id column: {schema.id_col_per_case}")
    merged["__id__"] = merged[schema.id_col_per_case]
    merged = merged.merge(feat, on="__id__", how="left")
    return merged
