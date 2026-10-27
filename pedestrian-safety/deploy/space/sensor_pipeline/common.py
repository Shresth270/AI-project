"""
Shared helpers for the sensor models.
Used by Phase 2 (training) and later by the agent's forecast tool, so the
features are built in exactly the same way everywhere.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR, FOOTFALL_LEVELS

EVENT_TYPES = ["none", "celebration", "sport", "festival", "shopping", "other"]

# Counts are divided by the sensor's own "busy" level (75th percentile), so a busy
# CBD sensor and a quiet side-street sensor speak the same language:
#   lag_1_rel = 1.0  ->  last hour was as busy as this place's usual busy hour
FEATURES = [
    "hour", "weekday", "month", "is_weekend", "is_holiday", "is_event", "event_code",
    "lag_1_rel", "lag_2_rel", "lag_3_rel", "lag_24_rel", "lag_168_rel",
    "trend_rel", "baseline_rel", "surge_so_far",
]
# columns SMOTENC must treat as categories (it must not invent "weekday 3.4")
CATEGORICAL = ["weekday", "is_weekend", "is_holiday", "is_event", "event_code"]


def load_thresholds() -> pd.DataFrame:
    """Per-sensor p25 / p75 / p95 from the TRAIN period (made in Phase 1)."""
    return pd.read_csv(MODELS_DIR / "level_thresholds.csv", index_col=0)


def add_features(df: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """Add the scale-free model features to a Phase-1 style table."""
    df = df.copy()
    busy = df["sensor_id"].map(thresholds["p75"]).astype("float32") + 1.0
    for col in ["lag_1", "lag_2", "lag_3", "lag_24", "lag_168", "trend", "baseline"]:
        df[f"{col}_rel"] = (df[col] / busy).astype("float32")
    df["event_code"] = df["event_type"].map({e: i for i, e in enumerate(EVENT_TYPES)}).fillna(5).astype("int8")
    return df


def level_from_count(counts, sensor_ids, thresholds: pd.DataFrame) -> np.ndarray:
    """Turn raw counts into level 0-3 using each sensor's percentiles (same rule as the target)."""
    t = thresholds.reindex(np.asarray(sensor_ids))
    c = np.asarray(counts, dtype=float)
    return np.select([c <= t["p25"].values, c <= t["p75"].values, c <= t["p95"].values], [0, 1, 2], 3)


class SoftVote:
    """Soft-voting ensemble over models that are ALREADY trained: averages their probabilities."""

    def __init__(self, models: dict):
        self.models = models
        self.classes_ = np.arange(len(FOOTFALL_LEVELS))

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models.values()], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
