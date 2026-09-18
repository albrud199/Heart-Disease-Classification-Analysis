'''Phase 6 (Steps 22-23): explainability, error analysis, and subgroup robustness.

Uses the exact Phase 5 production artifact and locked threshold. The test set is
read only for analysis after Phase 5 has locked it; no model, calibration, or
threshold is refit here. SHAP is used when installed. A deterministic,
model-agnostic Shapley permutation fallback keeps the phase runnable in minimal
research environments where the optional ``shap`` package is unavailable.
'''
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "models" / "phase6"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
BOOTSTRAPS = 400
MIN_GROUP = 10
# Importing this module registers the classes needed by artifacts produced when
# Phase 5 was executed as a script (pickle module name __main__).
sys.path.insert(0, str(ROOT))
import phase5_calibration_final_eval as p5  # noqa: E402

def load_locked_data() -> tuple[pd.DataFrame, pd.Series, Any, float]:
    manifest = json.loads((ROOT / "models" / "phase5" / "lock_manifest.json").read_text(encoding="utf-8"))
    artifact_path = Path(manifest["artifact_path"])
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    artifact = p5.load_production_artifact(artifact_path)
    test = pd.read_csv(ROOT / "splits" / "test.csv")
    if "target" not in test or len(test) == 0:
        raise ValueError("splits/test.csv must contain a non-empty target column")
    X = test.drop(columns=["target"])
    y = test["target"].astype(int)
    return X, y, artifact, float(manifest["locked_pipeline"]["decision_threshold"])


def metrics(y: pd.Series | np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    yv = np.asarray(y, dtype=int)
    pred = (probs >= threshold).astype(int)
    tn = int(((yv == 0) & (pred == 0)).sum())
    fp = int(((yv == 0) & (pred == 1)).sum())
    return {
        "n": int(len(yv)), "positive_rate": float(yv.mean()),
        "roc_auc": float(roc_auc_score(yv, probs)) if len(np.unique(yv)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(yv, probs)) if len(np.unique(yv)) == 2 else np.nan,
        "recall_sensitivity": float(recall_score(yv, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "precision": float(precision_score(yv, pred, zero_division=0)),
        "f1": float(f1_score(yv, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yv, pred)),
    }


def bootstrap_ci(y: np.ndarray, probs: np.ndarray, threshold: float, seed: int = SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"roc_auc": [], "recall_sensitivity": [], "specificity": [], "f1": []}
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        row = metrics(y[idx], probs[idx], threshold)
        for key in values:
            values[key].append(row[key])
    result: dict[str, float] = {}
    for key, vals in values.items():
        result[f"{key}_ci95_low"] = float(np.percentile(vals, 2.5)) if vals else np.nan
        result[f"{key}_ci95_high"] = float(np.percentile(vals, 97.5)) if vals else np.nan
    return result

def predict_fn(artifact: Any, columns: list[str]):
    def fn(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=columns)
        return np.asarray(artifact.predict_proba(frame)[:, 1], dtype=float)
    return fn

def fallback_shap(artifact: Any, X: pd.DataFrame, n_rows: int = 8) -> tuple[np.ndarray, str]:
    '''Deterministic model-agnostic Shapley permutation approximation.'''
    rng = np.random.default_rng(SEED)
    columns = list(X.columns)
    sample = X.iloc[: min(n_rows, len(X))].copy()
    background = X.iloc[: min(40, len(X))].copy()
    baseline = background.mode(dropna=False).iloc[0]
    phi = np.zeros((len(sample), len(columns)), dtype=float)
    fn = predict_fn(artifact, columns)
    for row_i, (_, row) in enumerate(sample.iterrows()):
        for _ in range(4):
            order = rng.permutation(len(columns))
            current = pd.DataFrame([baseline.to_dict()])
            previous = float(fn(current.to_numpy())[0])
            for j in order:
                current.iloc[0, j] = row.iloc[j]
                updated = float(fn(current.to_numpy())[0])
                phi[row_i, j] += updated - previous
                previous = updated
    phi /= 4.0
    return phi, "deterministic_model_agnostic_shapley_permutation_fallback"


def explain_with_shap(artifact: Any, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    columns = list(X.columns)
    try:
        import shap  # type: ignore
        background = X.iloc[: min(40, len(X))]
        sample = X.iloc[: min(30, len(X))]
        explainer = shap.KernelExplainer(predict_fn(artifact, columns), background.to_numpy(), seed=SEED)
        values = explainer.shap_values(sample.to_numpy(), nsamples=80)
        if isinstance(values, list):
            values = values[-1]
        return np.asarray(values, dtype=float), "shap.KernelExplainer"
    except Exception as exc:
        print(f"[warning] Native SHAP unavailable; using deterministic fallback: {exc}")
        return fallback_shap(artifact, X)


def run() -> None:
    X, y, artifact, threshold = load_locked_data()
    probs = np.asarray(artifact.predict_proba(X)[:, 1], dtype=float)
    pred = (probs >= threshold).astype(int)
    base = metrics(y, probs, threshold)
    base.update(bootstrap_ci(y.to_numpy(), probs, threshold))

    # Step 22: transparent global/local explanations and mistakes.
    phi, explanation_method = explain_with_shap(artifact, X)
    explain_X = X.iloc[: len(phi)].reset_index(drop=True)
    global_df = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.mean(np.abs(phi), axis=0), "mean_shap": np.mean(phi, axis=0)})
    global_df["rank"] = global_df["mean_abs_shap"].rank(method="min", ascending=False).astype(int)
    global_df.sort_values("mean_abs_shap", ascending=False).to_csv(OUT / "shap_global_importance.csv", index=False)
    local = explain_X.copy()
    for i, col in enumerate(X.columns):
        local[f"shap_{col}"] = phi[:, i]
    local["row_index"] = explain_X.index
    local["actual"] = y.iloc[: len(phi)].to_numpy()
    local["predicted"] = pred[: len(phi)]
    local["probability"] = probs[: len(phi)]
    local.to_csv(OUT / "shap_local_explanations.csv", index=False)
    plt.figure(figsize=(9, 6))
    top = global_df.sort_values("mean_abs_shap").tail(14)
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2b5c8f")
    plt.xlabel("Mean absolute SHAP value (model output scale)")
    plt.title(f"Phase 6 global explainability ({explanation_method})")
    plt.tight_layout(); plt.savefig(OUT / "fig_shap_global_importance.png", dpi=220); plt.close()

    errors = X.copy()
    errors.insert(0, "row_index", X.index)
    errors["actual"] = y.to_numpy(); errors["predicted"] = pred; errors["probability"] = probs
    errors["error_type"] = np.select([(y.to_numpy() == 0) & (pred == 1), (y.to_numpy() == 1) & (pred == 0)], ["false_positive", "false_negative"], default="correct")
    errors["confidence"] = np.maximum(probs, 1.0 - probs)
    errors.to_csv(OUT / "error_analysis_all_predictions.csv", index=False)
    errors[errors["error_type"] != "correct"].sort_values("confidence", ascending=False).to_csv(OUT / "error_analysis_mistakes.csv", index=False)
    high_conf = errors[(errors["error_type"] != "correct") & (errors["confidence"] >= 0.80)]
    high_conf.to_csv(OUT / "error_analysis_high_confidence_mistakes.csv", index=False)
    summary = errors.groupby("error_type", as_index=False).agg(count=("row_index", "size"), mean_confidence=("confidence", "mean"), mean_probability=("probability", "mean"))
    summary.to_csv(OUT / "error_analysis_summary.csv", index=False)

    # Step 23: predefined, non-tuned subgroup analysis plus uncertainty.
    groups: dict[str, pd.Series] = {
        "sex=0": X["sex"] == 0, "sex=1": X["sex"] == 1,
        "cp=0": X["cp"] == 0, "cp=1": X["cp"] == 1, "cp=2": X["cp"] == 2, "cp=3": X["cp"] == 3,
        "exang=0": X["exang"] == 0, "exang=1": X["exang"] == 1,
        "age<55": X["age"] < 55, "age=55-64": X["age"].between(55, 64), "age>=65": X["age"] >= 65,
    }
    rows: list[dict[str, Any]] = []
    for name, mask in groups.items():
        idx = mask.to_numpy()
        if int(idx.sum()) < MIN_GROUP or len(np.unique(y.to_numpy()[idx])) < 2:
            continue
        row = {"subgroup": name, **metrics(y.to_numpy()[idx], probs[idx], threshold), **bootstrap_ci(y.to_numpy()[idx], probs[idx], threshold, SEED + len(rows))}
        rows.append(row)
    subgroup_df = pd.DataFrame(rows)
    subgroup_df.to_csv(OUT / "subgroup_metrics_with_bootstrap_ci.csv", index=False)
    threshold_rows = []
    for t in [0.45, 0.50, 0.51, 0.55, 0.60]:
        threshold_rows.append({"threshold": t, **metrics(y, probs, t)})
    pd.DataFrame(threshold_rows).to_csv(OUT / "threshold_sensitivity_robustness.csv", index=False)

    decision = {"phase": 6, "steps": [22, 23], "locked_threshold": threshold, "explanation_method": explanation_method, "base_test_metrics": base, "error_counts": summary.to_dict(orient="records"), "subgroups_evaluated": len(rows), "minimum_subgroup_n": MIN_GROUP, "limitations": ["Subgroups are exploratory and not fairness or clinical validation.", "Small sample sizes produce wide uncertainty; groups below the minimum size or without both classes are omitted.", "The test set was used only after Phase 5 lock for descriptive post-hoc analysis; no fitting or tuning was performed."]}
    (OUT / "phase6_decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    print(f"Phase 6 completed successfully: {len(X)} test rows, {len(rows)} subgroups, method={explanation_method}")

if __name__ == "__main__":
    run()
