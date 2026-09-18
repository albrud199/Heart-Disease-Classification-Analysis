'''Stable inference API for the locked Phase 5 heart-disease artifact.

This module performs input validation before calling the serialized production
pipeline. It deliberately does not fit, calibrate, tune, or alter the model.
Research use only; not a clinical diagnostic device.
'''
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import phase5_calibration_final_eval as p5  # noqa: E402
MANIFEST_PATH = ROOT / "models" / "phase5" / "lock_manifest.json"
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def _resolve_artifact_path(manifest: dict[str, Any]) -> Path:
    raw = Path(str(manifest["artifact_path"]))
    return raw if raw.is_absolute() and raw.exists() else ROOT / "models" / "phase5" / raw.name

def load_model() -> tuple[Any, dict[str, Any]]:
    '''Load the exact locked artifact and manifest; never retrain.'''
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    artifact_path = _resolve_artifact_path(manifest)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Locked production artifact not found: {artifact_path}")
    return p5.load_production_artifact(artifact_path), manifest

def validate_input(values: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    '''Validate and normalize one row or a dataframe with the production schema.'''
    frame = values.copy() if isinstance(values, pd.DataFrame) else pd.DataFrame([dict(values)])
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in FEATURE_COLUMNS]
    if missing or extra:
        raise ValueError(f"Invalid feature schema; missing={missing}, extra={extra}")
    frame = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Input contains missing or non-finite feature values")
    if len(frame) == 0:
        raise ValueError("At least one input row is required")
    return frame

def predict(values: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    '''Return calibrated probability and locked-threshold decision for each row.'''
    model, manifest = load_model()
    frame = validate_input(values)
    probabilities = np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
    decisions = np.asarray(model.predict(frame), dtype=int)
    threshold = float(manifest["locked_pipeline"]["decision_threshold"])
    return pd.DataFrame({"probability": probabilities, "prediction": decisions, "threshold": threshold})


def predict_one(values: Mapping[str, Any]) -> dict[str, Any]:
    '''Predict one validated feature mapping and return JSON-friendly values.'''
    if not isinstance(values, Mapping):
        raise TypeError("predict_one expects a mapping of feature names to values")
    row = predict(values).iloc[0].to_dict()
    result: dict[str, Any] = {}
    for key, value in row.items():
        result[str(key)] = value.item() if hasattr(value, "item") else value
    return result
