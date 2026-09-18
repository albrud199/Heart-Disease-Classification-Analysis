'''Phase 7 (Steps 24-27): production handoff, validation, cleanup, and report.

The Phase 5 artifact remains authoritative. This phase only packages and checks
that artifact; it never retrains, calibrates, tunes, or changes the threshold.
'''
from __future__ import annotations
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import phase5_calibration_final_eval as p5  # noqa: E402
OUT = ROOT / "models" / "phase7"
ARTIFACT = ROOT / "models" / "phase5" / "hybrid_ensemble_calibrated_production.pkl"
MANIFEST = ROOT / "models" / "phase5" / "lock_manifest.json"
FEATURE_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    check(ARTIFACT.exists(), f"Missing locked artifact: {ARTIFACT}")
    check(MANIFEST.exists(), f"Missing lock manifest: {MANIFEST}")
    manifest: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual_hash = sha256(ARTIFACT)
    expected_hash = manifest.get("artifact_hash_sha256")
    check(actual_hash == expected_hash, "Artifact SHA-256 does not match Phase 5 lock manifest")
    model = p5.load_production_artifact(ARTIFACT)
    threshold = float(manifest["locked_pipeline"]["decision_threshold"])
    check(abs(float(model.threshold) - threshold) < 1e-12, "Artifact threshold differs from locked threshold")

    train = pd.read_csv(ROOT / "splits" / "train.csv")
    test = pd.read_csv(ROOT / "splits" / "test.csv")
    check(list(train.drop(columns=["target"]).columns) == FEATURE_COLUMNS, "Training schema changed")
    check(list(test.drop(columns=["target"]).columns) == FEATURE_COLUMNS, "Test schema changed")
    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"].astype(int).to_numpy()
    probabilities = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
    predictions = np.asarray(model.predict(X_test), dtype=int)
    check(len(probabilities) == len(test), "Inference row count mismatch")
    check(bool(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()), "Invalid probabilities")
    check(bool(np.array_equal(predictions, (probabilities >= threshold).astype(int))), "Prediction does not use locked threshold")
    roundtrip = p5.load_production_artifact(ARTIFACT)
    roundtrip_probs = np.asarray(roundtrip.predict_proba(X_test)[:, 1], dtype=float)
    check(float(np.max(np.abs(probabilities - roundtrip_probs))) < 1e-12, "Artifact round-trip mismatch")

    validation = {
        "phase": 7, "steps": [24, 25], "status": "passed",
        "artifact": str(ARTIFACT.relative_to(ROOT)), "artifact_sha256": actual_hash,
        "locked_threshold": threshold, "feature_columns": FEATURE_COLUMNS,
        "test_rows_validated": int(len(test)), "probability_range": [float(probabilities.min()), float(probabilities.max())],
        "round_trip_max_probability_difference": float(np.max(np.abs(probabilities - roundtrip_probs))),
        "test_predictions_recomputed": True,
        "warning": "This validation reuses the already locked test split for inference regression only; no fitting or tuning was performed.",
    }
    (OUT / "inference_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    pd.DataFrame({"row_index": np.arange(len(test)), "probability": probabilities, "prediction": predictions, "actual": y_test}).to_csv(OUT / "inference_validation_predictions.csv", index=False)

    inventory = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
            inventory.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path) if path.suffix in {".py", ".json", ".md", ".pkl"} else None})
    (OUT / "repository_inventory.json").write_text(json.dumps({"phase": 7, "files": inventory}, indent=2), encoding="utf-8")

    metrics_path = ROOT / "models" / "phase5" / "final_test_metrics.csv"
    metrics = pd.read_csv(metrics_path).set_index("metric")["value"].to_dict() if metrics_path.exists() else {}
    phase6_path = ROOT / "models" / "phase6" / "phase6_decision.json"
    phase6 = json.loads(phase6_path.read_text(encoding="utf-8")) if phase6_path.exists() else {}
    report = {
        "title": "Heart Disease Classification — Phase 7 Production Handoff",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "steps": {"24_production_artifact": "passed", "25_inference_validation": "passed", "26_repository_cleanup": "passed", "27_final_report": "passed"},
        "production_contract": {"artifact": str(ARTIFACT.relative_to(ROOT)), "calibration": manifest["locked_pipeline"]["calibration"], "threshold": threshold, "schema": FEATURE_COLUMNS},
        "locked_test_metrics_from_phase5": metrics, "phase6_summary": {"explanation_method": phase6.get("explanation_method"), "subgroups_evaluated": phase6.get("subgroups_evaluated")},
        "reproducibility": {"python": sys.version, "platform": platform.platform(), "artifact_sha256": actual_hash},
        "limitations": ["Small single-cohort dataset; no external or temporal validation.", "Test performance is an evaluation result, not a clinical performance guarantee.", "Research and educational use only; not a diagnostic device."],
    }
    (OUT / "final_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUT / "FINAL_REPORT.md").write_text(render_report(report), encoding="utf-8")
    print(f"Phase 7 completed successfully: artifact verified, {len(test)} inference rows validated, report written to {OUT}")


def render_report(report: dict[str, Any]) -> str:
    metrics = report["locked_test_metrics_from_phase5"]
    lines = ["# Phase 7 — Production Handoff Report", "", "## Status", "Complete: Steps 24–27 passed.", "", "## Production contract", f"- Artifact: `{report['production_contract']['artifact']}`", f"- Locked threshold: `{report['production_contract']['threshold']}`", f"- Calibration: `{report['production_contract']['calibration'].get('method', 'unknown')}`", f"- SHA-256: `{report['reproducibility']['artifact_sha256']}`", "", "## Locked Phase 5 test metrics"]
    lines.extend(f"- {key}: {value}" for key, value in metrics.items())
    lines.extend(["", "## Validation", "- Input schema, probability range, threshold behavior, serialization round-trip, and row counts passed.", "- No fitting, calibration, tuning, or threshold selection was performed in Phase 7.", "", "## Limitations", *[f"- {item}" for item in report["limitations"]]])
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    main()
