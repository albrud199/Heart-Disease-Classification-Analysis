# Phase 5 (Steps 19–21): Calibration, threshold analysis, final locked test evaluation
# Strict anti-leakage protocol:
#   * Steps 19–20 operate exclusively on splits/train.csv (Out-Of-Fold) and splits/val.csv.
#   * Calibration methods are compared via internal cross-validation; final calibrators are
#     fitted on validation data only (never on the test set).
#   * The decision threshold is selected exclusively from validation evidence.
#   * splits/test.csv is read EXACTLY ONCE, in Step 21, after the pipeline is locked.
# Outputs are written to models/phase5/.

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Literal, cast

# Ensure stdout uses UTF-8 or safe encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase4_ensemble_ablation as p4  # noqa: E402  (reuses Phase 4 building blocks)

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from numpy.typing import ArrayLike, NDArray  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration and Reproducibility Setup
# -----------------------------------------------------------------------------
SEED = 42
CV_FOLDS = 5
N_BINS = 10
BOOTSTRAP_ITERATIONS = 2000
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
CANDIDATES = ["KNN", "SVM", "Random Forest", "Hybrid Ensemble (Weighted)"]
ENSEMBLE_NAME = "Hybrid Ensemble (Weighted)"
CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")
CalibrationMethod = Literal["none", "sigmoid", "isotonic"]

ROOT = Path(__file__).resolve().parent
PHASE4_DIR = ROOT / "models" / "phase4"
OUTPUT_DIR = ROOT / "models" / "phase5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Module-level guard proving the locked test split is consumed exactly once.
_TEST_SPLIT_ACCESSED = False


# -----------------------------------------------------------------------------
# Calibrators (public scikit-learn primitives only; fully serializable)
# -----------------------------------------------------------------------------
class IdentityCalibrator:
    """No-op calibrator (raw model probabilities)."""

    name = "none"

    def fit(self, probs: ArrayLike, y_true: ArrayLike) -> "IdentityCalibrator":
        return self

    def transform(self, probs: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(probs, dtype=float)

    def params(self) -> dict[str, Any]:
        return {"method": "none"}


class PlattCalibrator:
    """Platt scaling (sigmoid): logistic regression on the logit of raw scores."""

    name = "sigmoid"

    def __init__(self) -> None:
        self._lr: LogisticRegression | None = None

    def fit(self, probs: ArrayLike, y_true: ArrayLike) -> "PlattCalibrator":
        z = self._logit(probs)
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self._lr.fit(z, np.asarray(y_true, dtype=int))
        return self

    def transform(self, probs: ArrayLike) -> NDArray[np.float64]:
        if self._lr is None:
            raise RuntimeError("PlattCalibrator must be fitted before transform().")
        z = self._logit(probs)
        return self._lr.predict_proba(z)[:, 1]

    @staticmethod
    def _logit(probs: ArrayLike) -> NDArray[np.float64]:
        p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def params(self) -> dict[str, Any]:
        if self._lr is None:
            raise RuntimeError("PlattCalibrator must be fitted before params().")
        return {
            "method": "sigmoid",
            "coef": float(self._lr.coef_[0, 0]),
            "intercept": float(self._lr.intercept_[0]),
        }


class IsotonicCalibrator:
    """Isotonic regression calibration (non-parametric monotonic mapping)."""

    name = "isotonic"

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, probs: ArrayLike, y_true: ArrayLike) -> "IsotonicCalibrator":
        self._iso.fit(np.asarray(probs, dtype=float), np.asarray(y_true, dtype=float))
        return self

    def transform(self, probs: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self._iso.predict(np.asarray(probs, dtype=float)), dtype=float)

    def params(self) -> dict[str, Any]:
        f = cast(Any, self._iso).f_
        return {
            "method": "isotonic",
            "thresholds": [float(v) for v in np.asarray(f.x)],
            "values": [float(v) for v in np.asarray(f.y)],
        }


def build_calibrator(method: CalibrationMethod) -> IdentityCalibrator | PlattCalibrator | IsotonicCalibrator:
    """Factory returning the requested calibrator instance."""
    if method == "none":
        return IdentityCalibrator()
    if method == "sigmoid":
        return PlattCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibration method: {method}")


# -----------------------------------------------------------------------------
# Calibration & Metric Helpers
# -----------------------------------------------------------------------------
def expected_calibration_error(
    y_true: ArrayLike,
    probs: ArrayLike,
    n_bins: int = N_BINS,
) -> float:
    """Equal-width binned Expected Calibration Error (ECE)."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probs, dtype=float)
    bin_idx = np.clip((scores * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        ece += (count / len(labels)) * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return float(ece)


def calibration_metrics(y_true: ArrayLike, probs: ArrayLike) -> dict[str, float]:
    """Brier score, ECE, and log loss for a probability vector."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "brier": float(brier_score_loss(labels, scores)),
        "ece": expected_calibration_error(labels, scores),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
    }


def reliability_curve(
    y_true: ArrayLike,
    probs: ArrayLike,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """Binned reliability curve data (mean predicted probability vs observed frequency)."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probs, dtype=float)
    bin_idx = np.clip((scores * n_bins).astype(int), 0, n_bins - 1)
    rows: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = bin_idx == b
        if int(np.count_nonzero(mask)) == 0:
            continue
        rows.append({
            "bin_low": b / n_bins,
            "bin_high": (b + 1) / n_bins,
            "mean_predicted": float(scores[mask].mean()),
            "observed_frequency": float(labels[mask].mean()),
            "count": int(np.count_nonzero(mask)),
        })
    return pd.DataFrame(rows)


def classification_metrics_at_threshold(
    y_true: ArrayLike,
    probs: ArrayLike,
    threshold: float,
) -> dict[str, float]:
    """Full confusion-matrix-based metric set at an explicit decision threshold."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probs, dtype=float)
    predictions = (scores >= threshold).astype(int)

    tp = int(np.count_nonzero((predictions == 1) & (labels == 1)))
    fp = int(np.count_nonzero((predictions == 1) & (labels == 0)))
    tn = int(np.count_nonzero((predictions == 0) & (labels == 0)))
    fn = int(np.count_nonzero((predictions == 0) & (labels == 1)))
    total = tp + fp + tn + fn

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "youden_j": recall + specificity - 1.0,
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "predicted_positive_rate": (tp + fp) / total if total > 0 else 0.0,
    }


def bootstrap_ci(
    y_true: NDArray[np.int_],
    probs: NDArray[np.float64],
    metric_fn: Any,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile bootstrap 95% confidence interval for an arbitrary probability metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: list[float] = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(float(metric_fn(y_true[idx], probs[idx])))
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


# -----------------------------------------------------------------------------
# Data Loading & Probability Generation (train/val only until Step 21)
# -----------------------------------------------------------------------------
def load_training_and_val_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Loads training and validation splits. Test split is NOT touched here."""
    train = pd.read_csv(ROOT / "splits" / "train.csv")
    val = pd.read_csv(ROOT / "splits" / "val.csv")
    if "target" not in train.columns or "target" not in val.columns:
        raise ValueError("Both train.csv and val.csv must contain a 'target' column")
    X_train = train.drop(columns=["target"])
    y_train = train["target"].astype(int)
    X_val = val.drop(columns=["target"])
    y_val = val["target"].astype(int)
    return X_train, y_train, X_val, y_val


def build_probability_sets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    best_configs: dict[str, Any],
    weights: dict[str, float],
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]], Any, bool]:
    """
    Produces honest probability sets for every calibration candidate:
      * oof_probs: Out-Of-Fold probabilities on the training data (Phase 4 protocol,
        seed 42, identical to the probabilities saved by Phase 4).
      * val_probs: probabilities on the validation set from candidates fitted on the
        FULL training data (the Phase 4 authoritative checkpoint protocol).
    Returns (oof_probs, val_probs, fitted_ensemble, oof_reproduction_ok).
    """
    # --- Out-of-fold probabilities on training data (Phase 4 protocol, seed 42) ---
    oof_components, _ = p4.generate_oof_predictions(X_train, y_train, best_configs, cv_seed=SEED)
    oof_probs: dict[str, NDArray[np.float64]] = {
        "KNN": oof_components["KNN"],
        "SVM": oof_components["SVM"],
        "Random Forest": oof_components["Random Forest"],
        ENSEMBLE_NAME: (
            weights["KNN"] * oof_components["KNN"]
            + weights["SVM"] * oof_components["SVM"]
            + weights["Random Forest"] * oof_components["Random Forest"]
        ),
    }

    # --- Integrity check against Phase 4 saved OOF probabilities ---
    oof_reproduction_ok = True
    phase4_oof_path = PHASE4_DIR / "oof_probabilities.csv"
    if phase4_oof_path.exists():
        saved = pd.read_csv(phase4_oof_path)
        col_map = {"KNN": "prob_knn", "SVM": "prob_svm", "Random Forest": "prob_random_forest"}
        for cand, col in col_map.items():
            if col in saved.columns:
                diff = float(np.max(np.abs(saved[col].to_numpy() - oof_probs[cand])))
                oof_reproduction_ok &= diff < 1e-6
                print(f"  [integrity] OOF reproduction {cand}: max|diff| = {diff:.2e}")

    # --- Validation probabilities from candidates fitted on the full training set ---
    val_probs: dict[str, NDArray[np.float64]] = {}
    for cand in ["KNN", "SVM", "Random Forest"]:
        pipeline = p4.build_pipeline(X_train, cast(p4.ModelName, cand), best_configs)
        pipeline.fit(X_train, y_train)
        val_probs[cand] = pipeline.predict_proba(X_val)[:, 1]

    # Authoritative hybrid ensemble: refit deterministically (RF/SVC use fixed seeds)
    # with the exact Phase 4 configuration and optimized weights.
    ensemble = p4.HybridEnsembleClassifier(weights=weights, best_configs=best_configs)
    ensemble.fit(X_train, y_train)
    val_probs[ENSEMBLE_NAME] = ensemble.predict_proba(X_val)[:, 1]
    return oof_probs, val_probs, ensemble, bool(oof_reproduction_ok)


# -----------------------------------------------------------------------------
# Step 19: Probability Calibration
# -----------------------------------------------------------------------------
def cross_validated_calibration_comparison(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int_],
    cv_seed: int = SEED,
) -> pd.DataFrame:
    """
    Honest, internal-CV comparison of calibration methods for one candidate.
    The calibrator is fitted on 4/5 of the (prob, y) pairs and scored on the held-out
    fifth; scores are averaged across stratified folds. No other data is touched.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=cv_seed)
    rows: list[dict[str, Any]] = []
    for method in CALIBRATION_METHODS:
        fold_brier: list[float] = []
        fold_ece: list[float] = []
        fold_logloss: list[float] = []
        for train_idx, test_idx in cv.split(np.zeros(len(y_true)), y_true):
            calibrator = build_calibrator(cast(CalibrationMethod, method))
            calibrator.fit(probs[train_idx], y_true[train_idx])
            calibrated = calibrator.transform(probs[test_idx])
            m = calibration_metrics(y_true[test_idx], calibrated)
            fold_brier.append(m["brier"])
            fold_ece.append(m["ece"])
            fold_logloss.append(m["log_loss"])
        rows.append({
            "method": method,
            "brier_mean": float(np.mean(fold_brier)),
            "brier_std": float(np.std(fold_brier)),
            "ece_mean": float(np.mean(fold_ece)),
            "log_loss_mean": float(np.mean(fold_logloss)),
        })
    return pd.DataFrame(rows).sort_values("brier_mean").reset_index(drop=True)


def run_step19_calibration_part1(
    oof_probs: dict[str, NDArray[np.float64]],
    val_probs: dict[str, NDArray[np.float64]],
    y_train_arr: NDArray[np.int_],
    y_val_arr: NDArray[np.int_],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """
    Step 19 (part 1): assess uncalibrated calibration of best individual models and
    the ensemble, then compare calibration methods via internal stratified CV on both
    OOF-train and validation probability sets.
    """
    # --- 1. Uncalibrated calibration assessment ---
    assessment_rows: list[dict[str, Any]] = []
    for cand in CANDIDATES:
        for split, y_ref, probs in [
            ("train_oof", y_train_arr, oof_probs[cand]),
            ("validation", y_val_arr, val_probs[cand]),
        ]:
            m = calibration_metrics(y_ref, probs)
            assessment_rows.append({"candidate": cand, "split": split, "calibration": "none", **m})
    assessment_df = pd.DataFrame(assessment_rows)
    assessment_df.to_csv(OUTPUT_DIR / "calibration_assessment_uncalibrated.csv", index=False)
    print("\n--- Uncalibrated Calibration Assessment (Brier / ECE / LogLoss) ---")
    print(assessment_df.to_string(index=False))

    # --- 2. Cross-validated comparison of calibration methods ---
    comparison_rows: list[pd.DataFrame] = []
    for cand in CANDIDATES:
        for split, y_ref, probs in [
            ("train_oof", y_train_arr, oof_probs[cand]),
            ("validation", y_val_arr, val_probs[cand]),
        ]:
            cmp_df = cross_validated_calibration_comparison(probs, y_ref)
            cmp_df.insert(0, "candidate", cand)
            cmp_df.insert(1, "split", split)
            comparison_rows.append(cmp_df)
    comparison_df = pd.concat(comparison_rows, ignore_index=True)
    comparison_df.to_csv(OUTPUT_DIR / "calibration_method_cv_comparison.csv", index=False)
    print("\n--- Cross-Validated Calibration Method Comparison (5-fold, per candidate) ---")
    print(comparison_df[comparison_df["split"] == "validation"].to_string(index=False))
    return assessment_df, comparison_df, {}


def run_step19_calibration_part2(
    val_probs: dict[str, NDArray[np.float64]],
    y_val_arr: NDArray[np.int_],
    comparison_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Step 19 (part 2): select the final calibration method per candidate using
    validation evidence, refit final calibrators on the FULL validation set
    (roadmap protocol), and export reliability-curve data.
    """
    chosen: dict[str, str] = {}
    val_only = comparison_df[comparison_df["split"] == "validation"]
    for cand in CANDIDATES:
        cand_rows = val_only[val_only["candidate"] == cand]
        best_row = cand_rows.sort_values("brier_mean").iloc[0]
        chosen[cand] = str(best_row["method"])

    with open(OUTPUT_DIR / "chosen_calibration.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_metric": "cross-validated Brier score (5-fold stratified, within-set)",
                "selection_data": "validation probabilities (calibrators refit on the full validation set)",
                "chosen_method_per_candidate": chosen,
                "final_procedure": {
                    "candidate": ENSEMBLE_NAME,
                    "calibrator": chosen[ENSEMBLE_NAME],
                    "fitted_on": "splits/val.csv (probabilities from the Phase 4 ensemble trained on splits/train.csv)",
                },
            },
            f,
            indent=2,
        )
    print("\n--- Chosen Calibration Method Per Candidate (by CV Brier on validation) ---")
    for cand in CANDIDATES:
        print(f"  * {cand:28s}: {chosen[cand]}")

    # --- Fit final calibrators on the FULL validation set; report calibrated metrics ---
    final_calibrators: dict[str, Any] = {}
    final_rows: list[dict[str, Any]] = []
    for cand in CANDIDATES:
        method = cast(CalibrationMethod, chosen[cand])
        calibrator = build_calibrator(method)
        calibrator.fit(val_probs[cand], y_val_arr)
        final_calibrators[cand] = calibrator
        calibrated = calibrator.transform(val_probs[cand])
        m = calibration_metrics(y_val_arr, calibrated)
        final_rows.append({"candidate": cand, "calibration": method, "fitted_on": "validation", **m})
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(OUTPUT_DIR / "calibration_final_comparison.csv", index=False)
    print("\n--- Final Calibrated Validation Metrics (calibrators fitted on validation) ---")
    print(final_df.to_string(index=False))

    # --- Reliability curve data (uncalibrated vs final calibrated, validation) ---
    reliability_frames: list[pd.DataFrame] = []
    for cand in CANDIDATES:
        calibrated_probs = final_calibrators[cand].transform(val_probs[cand])
        for label, probs in [("uncalibrated", val_probs[cand]), ("calibrated", calibrated_probs)]:
            rc = reliability_curve(y_val_arr, probs)
            rc.insert(0, "curve", label)
            rc.insert(0, "candidate", cand)
            reliability_frames.append(rc)
    reliability_df = pd.concat(reliability_frames, ignore_index=True)
    reliability_df.to_csv(OUTPUT_DIR / "calibration_reliability_curves.csv", index=False)

    return {"chosen": chosen, "final_calibrators": final_calibrators, "final_df": final_df}


# -----------------------------------------------------------------------------
# Step 20: Decision-Threshold Analysis (validation evidence only)
# -----------------------------------------------------------------------------
def run_step20_threshold_analysis(
    val_probs: NDArray[np.float64],
    y_val_arr: NDArray[np.int_],
) -> dict[str, Any]:
    """
    Step 20 — Decision-threshold analysis on the CALIBRATED validation probabilities.
    Sweeps an explicit threshold grid, builds the threshold-performance table and
    trade-off visualization, and locks a documented threshold rule.
    CRITICAL: the test set is NOT used anywhere in this step.
    """
    sweep_rows: list[dict[str, float]] = []
    for t in THRESHOLD_GRID:
        row = classification_metrics_at_threshold(y_val_arr, val_probs, float(t))
        sweep_rows.append(row)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUTPUT_DIR / "threshold_performance.csv", index=False)

    # --- Threshold rule: maximize Youden's J; ties (within 1e-3) resolved toward
    # higher sensitivity, reflecting the screening context where false negatives
    # (missed disease) are costlier than false positives (extra follow-up tests). ---
    max_j = float(sweep_df["youden_j"].max())
    tie_mask = sweep_df["youden_j"] >= max_j - 1e-3
    tied = sweep_df[tie_mask].sort_values("recall_sensitivity", ascending=False)
    best_row = tied.iloc[0]
    final_threshold = float(best_row["threshold"])

    baseline_row = sweep_df[np.isclose(sweep_df["threshold"], 0.5)].iloc[0]

    print("\n--- Threshold Sweep Highlights (calibrated validation probabilities) ---")
    display_cols = [
        "threshold", "recall_sensitivity", "specificity", "precision",
        "f1", "balanced_accuracy", "youden_j", "predicted_positive_rate",
    ]
    show_idx = list(np.linspace(0, len(sweep_df) - 1, 10).astype(int))
    print(sweep_df.iloc[show_idx][display_cols].to_string(index=False))
    print(f"\n  Default threshold 0.50 -> recall={baseline_row['recall_sensitivity']:.3f}, "
          f"specificity={baseline_row['specificity']:.3f}, F1={baseline_row['f1']:.3f}")
    print(f"  LOCKED threshold {final_threshold:.2f} -> recall={best_row['recall_sensitivity']:.3f}, "
          f"specificity={best_row['specificity']:.3f}, F1={best_row['f1']:.3f}, "
          f"balanced_acc={best_row['balanced_accuracy']:.3f}, J={best_row['youden_j']:.3f}")

    threshold_decision = {
        "locked_threshold": final_threshold,
        "selection_rule": "Maximize Youden's J (sensitivity + specificity - 1); ties within 1e-3 resolved toward higher sensitivity.",
        "operational_rationale": (
            "Heart-disease screening context: a missed case (false negative) is costlier than an "
            "unnecessary follow-up test (false positive). Youden's J balances sensitivity and "
            "specificity without assuming 0.50 is optimal; near-ties favor higher sensitivity."
        ),
        "evidence_source": "calibrated validation probabilities only (splits/val.csv); test set NOT accessed",
        "validation_operating_point": {
            k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in best_row.items()
        },
        "default_threshold_050_comparison": {
            k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in baseline_row.items()
        },
    }
    with open(OUTPUT_DIR / "threshold_decision.json", "w", encoding="utf-8") as f:
        json.dump(threshold_decision, f, indent=2)

    return {"sweep_df": sweep_df, "best_row": best_row, "final_threshold": final_threshold}


# -----------------------------------------------------------------------------
# Step 21: Locked Production Artifact & Final Untouched Test Evaluation
# -----------------------------------------------------------------------------
class CalibratedHybridEnsemble:
    """
    Production inference artifact for the locked final system (roadmap Step 21 gate:
    'the calibration procedure itself must be part of the final serialized pipeline').

    Bundles, in exact operating order:
      1. The Phase 4 hybrid soft-voting ensemble (KNN + SVM + RF, fitted on train).
      2. The chosen probability calibrator (fitted on validation data).
      3. The locked decision threshold (selected on validation evidence).
    """

    def __init__(self, ensemble: Any, calibrator: Any, threshold: float):
        self.ensemble = ensemble
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        raw = self.ensemble.predict_proba(X)[:, 1]
        calibrated = np.asarray(self.calibrator.transform(raw), dtype=float)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: pd.DataFrame) -> NDArray[np.int_]:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def load_production_artifact(path: Path) -> "CalibratedHybridEnsemble":
    """
    Loads the serialized production artifact from any process/module.
    Registers Phase 5 classes on __main__ if needed, because pickle references
    classes by the module in which the producing script was executed.
    """
    import __main__

    for cls in (CalibratedHybridEnsemble, IdentityCalibrator, PlattCalibrator, IsotonicCalibrator):
        if not hasattr(__main__, cls.__name__):
            setattr(__main__, cls.__name__, cls)
    return cast(CalibratedHybridEnsemble, joblib.load(path))


def run_step21_locked_test_evaluation(
    ensemble: Any,
    ensemble_calibrator: Any,
    final_threshold: float,
    calibration_method: str,
    X_val: pd.DataFrame,
    y_val_arr: NDArray[np.int_],
    val_probs_calibrated: NDArray[np.float64],
) -> dict[str, Any]:
    """
    Step 21 — Final untouched test evaluation.
    1. Freezes the locked pipeline (writes the production artifact + lock manifest).
    2. Reads splits/test.csv EXACTLY ONCE (the single authorized access in Phase 5).
    3. Applies the locked pipeline unchanged and reports the final generalization result.
    """
    global _TEST_SPLIT_ACCESSED
    if _TEST_SPLIT_ACCESSED:
        raise RuntimeError("Test split may only be accessed once per evaluation cycle.")

    # --- 1. Freeze the locked pipeline ---
    artifact_path = OUTPUT_DIR / "hybrid_ensemble_calibrated_production.pkl"
    locked_artifact = CalibratedHybridEnsemble(ensemble, ensemble_calibrator, final_threshold)
    joblib.dump(locked_artifact, artifact_path)

    calibrator_params = ensemble_calibrator.params()
    lock_manifest = {
        "locked_pipeline": {
            "model": "Hybrid Ensemble (Weighted) — Phase 4 (fitted on splits/train.csv)",
            "model_source": str(PHASE4_DIR / "hybrid_ensemble_model.pkl"),
            "calibration": calibrator_params,
            "calibrator_fitted_on": "splits/val.csv",
            "decision_threshold": final_threshold,
            "threshold_rule": "Youden's J on calibrated validation probabilities",
        },
        "input_hashes": {
            "phase4_ensemble_model.pkl": sha256_of_file(PHASE4_DIR / "hybrid_ensemble_model.pkl"),
            "ensemble_weights.json": sha256_of_file(PHASE4_DIR / "ensemble_weights.json"),
            "best_configurations.json": sha256_of_file(ROOT / "models" / "phase3" / "best_configurations.json"),
        },
        "artifact_hash_sha256": None,  # filled after artifact is written
        "artifact_path": str(artifact_path),
    }
    lock_manifest["artifact_hash_sha256"] = sha256_of_file(artifact_path)
    with open(OUTPUT_DIR / "lock_manifest.json", "w", encoding="utf-8") as f:
        json.dump(lock_manifest, f, indent=2)
    print(f"\n[lock] Production artifact written : {artifact_path}")
    print(f"[lock] Artifact SHA-256            : {lock_manifest['artifact_hash_sha256']}")

    # Artifact round-trip regression check (validates serialization integrity)
    reloaded = load_production_artifact(artifact_path)
    reload_diff = float(np.max(np.abs(reloaded.predict_proba(X_val)[:, 1] - val_probs_calibrated)))
    assert reload_diff < 1e-9, "Production artifact round-trip mismatch!"
    print(f"[lock] Artifact round-trip check   : OK (max|diff| = {reload_diff:.2e})")

    # --- 2. THE single authorized read of the untouched test split ---
    _TEST_SPLIT_ACCESSED = True
    test = pd.read_csv(ROOT / "splits" / "test.csv")
    X_test = test.drop(columns=["target"])
    y_test_arr = test["target"].to_numpy(dtype=np.int_)
    print(f"\n[test] splits/test.csv accessed EXACTLY ONCE: {len(X_test)} samples "
          f"(positive rate {y_test_arr.mean():.3f})")

    # --- 3. Apply the locked pipeline UNCHANGED ---
    raw_test_probs = ensemble.predict_proba(X_test)[:, 1]
    calibrated_test_probs = np.asarray(ensemble_calibrator.transform(raw_test_probs), dtype=float)
    test_predictions = (calibrated_test_probs >= final_threshold).astype(int)
    return _step21_collect_results(
        y_test_arr, calibrated_test_probs, raw_test_probs, test_predictions,
        ensemble, ensemble_calibrator, final_threshold, calibration_method,
        artifact_path, X_val, val_probs_calibrated,
    )


def _step21_collect_results(
    y_test_arr: NDArray[np.int_],
    calibrated_test_probs: NDArray[np.float64],
    raw_test_probs: NDArray[np.float64],
    test_predictions: NDArray[np.int_],
    ensemble: Any,
    ensemble_calibrator: Any,
    final_threshold: float,
    calibration_method: str,
    artifact_path: Path,
    X_val: pd.DataFrame,
    val_probs_calibrated: NDArray[np.float64],
) -> dict[str, Any]:
    """Computes final test metrics, exports curve data, and locks the decision record."""
    # Discrimination metrics with bootstrap 95% CIs
    roc_auc = float(roc_auc_score(y_test_arr, calibrated_test_probs))
    pr_auc = float(average_precision_score(y_test_arr, calibrated_test_probs))
    roc_lo, roc_hi = bootstrap_ci(y_test_arr, calibrated_test_probs, roc_auc_score)
    pr_lo, pr_hi = bootstrap_ci(y_test_arr, calibrated_test_probs, average_precision_score)

    # Threshold-based operating metrics and confusion matrix
    op = classification_metrics_at_threshold(y_test_arr, calibrated_test_probs, final_threshold)
    cm = confusion_matrix(y_test_arr, test_predictions, labels=[0, 1])

    # Probability-quality metrics
    q = calibration_metrics(y_test_arr, calibrated_test_probs)
    raw_q = calibration_metrics(y_test_arr, raw_test_probs)

    final_metrics: dict[str, float] = {
        "roc_auc": roc_auc,
        "roc_auc_ci95_low": roc_lo,
        "roc_auc_ci95_high": roc_hi,
        "pr_auc": pr_auc,
        "pr_auc_ci95_low": pr_lo,
        "pr_auc_ci95_high": pr_hi,
        "accuracy": op["accuracy"],
        "recall_sensitivity": op["recall_sensitivity"],
        "specificity": op["specificity"],
        "precision": op["precision"],
        "f1": op["f1"],
        "balanced_accuracy": op["balanced_accuracy"],
        "brier_calibrated": q["brier"],
        "ece_calibrated": q["ece"],
        "log_loss_calibrated": q["log_loss"],
        "brier_uncalibrated_reference": raw_q["brier"],
        "ece_uncalibrated_reference": raw_q["ece"],
    }
    metrics_df = pd.DataFrame(
        [{"metric": k, "value": float(v)} for k, v in final_metrics.items()]
    )
    metrics_df.to_csv(OUTPUT_DIR / "final_test_metrics.csv", index=False)

    # Export full curve data
    fpr, tpr, _ = roc_curve(y_test_arr, calibrated_test_probs)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(OUTPUT_DIR / "test_roc_curve.csv", index=False)
    prec, rec, _ = precision_recall_curve(y_test_arr, calibrated_test_probs)
    pd.DataFrame({"recall": rec, "precision": prec}).to_csv(OUTPUT_DIR / "test_pr_curve.csv", index=False)
    reliability_curve(y_test_arr, calibrated_test_probs).to_csv(
        OUTPUT_DIR / "test_calibration_curve.csv", index=False
    )
    pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"]).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    return {
        "y_test": y_test_arr,
        "probs": calibrated_test_probs,
        "raw_probs": raw_test_probs,
        "predictions": test_predictions,
        "metrics": final_metrics,
        "confusion_matrix": cm,
        "artifact_path": artifact_path,
        "threshold": final_threshold,
        "calibration_method": calibration_method,
        "ensemble": ensemble,
        "ensemble_calibrator": ensemble_calibrator,
        "X_val": X_val,
        "val_probs_calibrated": val_probs_calibrated,
    }


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
def generate_calibration_figures(
    val_probs: dict[str, NDArray[np.float64]],
    y_val_arr: NDArray[np.int_],
    final_calibrators: dict[str, Any],
) -> None:
    """Reliability diagrams (uncalibrated vs calibrated) + Brier score comparison."""
    # Panel figure: one reliability diagram per candidate
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, cand in zip(axes.ravel(), CANDIDATES):
        calibrated = final_calibrators[cand].transform(val_probs[cand])
        rc_raw = reliability_curve(y_val_arr, val_probs[cand])
        rc_cal = reliability_curve(y_val_arr, calibrated)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1, label="Perfect calibration")
        ax.plot(rc_raw["mean_predicted"], rc_raw["observed_frequency"], "o-", color="#c0392b",
                label="Uncalibrated", markersize=5)
        ax.plot(rc_cal["mean_predicted"], rc_cal["observed_frequency"], "s-", color="#27ae60",
                label=f"Calibrated ({final_calibrators[cand].name})", markersize=5)
        ax.set_title(cand, fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive frequency")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.suptitle("Step 19: Calibration Curves on Validation Data (Uncalibrated vs Calibrated)",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_calibration_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Brier score comparison (uncalibrated vs final calibrated, validation)
    uncal = [calibration_metrics(y_val_arr, val_probs[cand])["brier"] for cand in CANDIDATES]
    cal = [calibration_metrics(y_val_arr, final_calibrators[cand].transform(val_probs[cand]))["brier"]
           for cand in CANDIDATES]
    x = np.arange(len(CANDIDATES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, uncal, width, label="Uncalibrated", color="#c0392b")
    ax.bar(x + width / 2, cal, width, label="Calibrated", color="#27ae60")
    for xi, (u, c) in enumerate(zip(uncal, cal)):
        ax.text(xi - width / 2, u + 0.001, f"{u:.4f}", ha="center", fontsize=9)
        ax.text(xi + width / 2, c + 0.001, f"{c:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(CANDIDATES, rotation=15, ha="right")
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title("Step 19: Brier Score Comparison on Validation Data", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_brier_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_threshold_figure(
    sweep_df: pd.DataFrame,
    final_threshold: float,
) -> None:
    """Two-panel threshold trade-off visualization with the locked operating point."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    ax1 = axes[0]
    ax1.plot(sweep_df["threshold"], sweep_df["recall_sensitivity"], "-", color="#439775",
             linewidth=2, label="Recall / Sensitivity")
    ax1.plot(sweep_df["threshold"], sweep_df["specificity"], "-", color="#6c5b7b",
             linewidth=2, label="Specificity")
    ax1.plot(sweep_df["threshold"], sweep_df["precision"], "--", color="#2b5c8f",
             linewidth=1.8, label="Precision")
    ax1.axvline(final_threshold, color="#c0392b", linestyle=":", linewidth=2,
                label=f"Locked threshold = {final_threshold:.2f}")
    ax1.axvline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Default 0.50")
    ax1.set_xlabel("Decision Threshold", fontsize=11)
    ax1.set_ylabel("Metric Value", fontsize=11)
    ax1.set_title("Sensitivity / Specificity / Precision vs Threshold", fontsize=12, fontweight="bold")
    ax1.legend(loc="center left", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = axes[1]
    ax2.plot(sweep_df["threshold"], sweep_df["youden_j"], "-", color="#d65f5f",
             linewidth=2, label="Youden's J")
    ax2.plot(sweep_df["threshold"], sweep_df["balanced_accuracy"], "-", color="#2b5c8f",
             linewidth=2, label="Balanced Accuracy")
    ax2.plot(sweep_df["threshold"], sweep_df["f1"], "--", color="#e27c38",
             linewidth=1.8, label="F1 Score")
    ax2.axvline(final_threshold, color="#c0392b", linestyle=":", linewidth=2,
                label=f"Locked threshold = {final_threshold:.2f}")
    best_j = float(sweep_df["youden_j"].max())
    ax2.annotate(
        f"Max J = {best_j:.3f}",
        xy=(final_threshold, best_j),
        xytext=(final_threshold + 0.08, best_j - 0.06),
        arrowprops={"arrowstyle": "->", "color": "#c0392b"},
        fontsize=10,
        color="#c0392b",
        fontweight="bold",
    )
    ax2.set_xlabel("Decision Threshold", fontsize=11)
    ax2.set_ylabel("Metric Value", fontsize=11)
    ax2.set_title("Composite Selection Criteria vs Threshold", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower center", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Step 20: Threshold Trade-off Analysis (Calibrated Validation Probabilities)",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_threshold_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_test_figures(results: dict[str, Any]) -> None:
    """Final test figures: confusion matrix, ROC + PR curves, calibration curve."""
    y_test = results["y_test"]
    probs = results["probs"]
    cm = results["confusion_matrix"]
    threshold = results["threshold"]
    m = results["metrics"]

    # Panel 1: Confusion matrix heatmap
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = [f"{v}" for v in cm.flatten()]
    labels = [f"{name}\n{count}" for name, count in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=True, square=True,
                linewidths=1.5, annot_kws={"size": 14, "fontweight": "bold"},
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"], ax=axes[0])
    axes[0].set_title(f"Confusion Matrix @ threshold {threshold:.2f}", fontsize=12, fontweight="bold")

    # Panel 2: ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    axes[1].plot(fpr, tpr, color="#2b5c8f", linewidth=2.2,
                 label=f"ROC-AUC = {m['roc_auc']:.4f} (95% CI [{m['roc_auc_ci95_low']:.3f}, {m['roc_auc_ci95_high']:.3f}])")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1, label="Chance")
    axes[1].set_xlabel("False Positive Rate", fontsize=11)
    axes[1].set_ylabel("True Positive Rate", fontsize=11)
    axes[1].set_title("Final Test ROC Curve (Locked Pipeline)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, probs)
    axes[2].plot(rec, prec, color="#e27c38", linewidth=2.2,
                 label=f"PR-AUC = {m['pr_auc']:.4f} (95% CI [{m['pr_auc_ci95_low']:.3f}, {m['pr_auc_ci95_high']:.3f}])")
    axes[2].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[2].set_ylabel("Precision", fontsize=11)
    axes[2].set_title("Final Test Precision-Recall Curve (Locked Pipeline)", fontsize=12, fontweight="bold")
    axes[2].legend(loc="lower left", fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Step 21: Final Locked Test Evaluation", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_test_confusion_roc_pr.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Test calibration curve (reliability diagram)
    rc = reliability_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1, label="Perfect calibration")
    ax.plot(rc["mean_predicted"], rc["observed_frequency"], "s-", color="#27ae60", markersize=6,
            label=f"Calibrated ({results['calibration_method']})")
    ax2 = ax.twinx()
    ax2.hist(probs, bins=20, alpha=0.25, color="#2b5c8f", label="Prediction density")
    ax2.set_ylabel("Count", fontsize=10)
    ax2.legend(loc="upper center", fontsize=9)
    ax.set_xlabel("Mean predicted probability (calibrated)", fontsize=11)
    ax.set_ylabel("Observed positive frequency", fontsize=11)
    ax.set_title(
        f"Final Test Calibration Curve — Brier = {m['brier_calibrated']:.4f}, ECE = {m['ece_calibrated']:.4f}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_test_calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    start = time.time()
    print("=" * 80)
    print("PHASE 5: CALIBRATION, THRESHOLD ANALYSIS, FINAL LOCKED TEST EVALUATION")
    print("Steps 19-21 | Test split untouched until the final locked evaluation")
    print("=" * 80)

    # 1. Load data & locked Phase 4 configuration
    X_train, y_train, X_val, y_val = load_training_and_val_data()
    best_configs = p4.load_shortlisted_configs()
    with open(PHASE4_DIR / "ensemble_weights.json", "r", encoding="utf-8") as f:
        weights = json.load(f)
    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()
    print(f"\n[1/6] Data loaded: train n={len(X_train)} | val n={len(X_val)} | test n=LOCKED (not read)")
    print(f"  - Hybrid ensemble weights: { {k: round(v, 4) for k, v in weights.items()} }")

    # 2. Generate honest probability sets (OOF train + validation)
    print(f"\n[2/6] Generating OOF probabilities (Phase 4 protocol, seed {SEED}) + validation probabilities...")
    oof_probs, val_probs, ensemble, oof_ok = build_probability_sets(
        X_train, y_train, X_val, best_configs, weights
    )
    print(f"  - OOF reproduction matches Phase 4 saved probabilities: {oof_ok}")
    print(f"  - Candidates: {', '.join(CANDIDATES)}")

    # 3. Step 19 — Probability calibration
    print(f"\n[3/6] STEP 19: Probability Calibration (E07)...")
    _, comparison_df, _ = run_step19_calibration_part1(oof_probs, val_probs, y_train_arr, y_val_arr)
    step19 = run_step19_calibration_part2(val_probs, y_val_arr, comparison_df)
    final_calibrators = step19["final_calibrators"]
    generate_calibration_figures(val_probs, y_val_arr, final_calibrators)
    print("  [OK] fig_calibration_curves.png, fig_brier_comparison.png")

    # 4. Step 20 — Decision-threshold analysis (calibrated validation probabilities)
    print(f"\n[4/6] STEP 20: Decision-Threshold Analysis (E08)...")
    ensemble_calibrator = final_calibrators[ENSEMBLE_NAME]
    calibrated_val_probs = np.asarray(
        ensemble_calibrator.transform(val_probs[ENSEMBLE_NAME]), dtype=float
    )
    step20 = run_step20_threshold_analysis(calibrated_val_probs, y_val_arr)
    final_threshold = step20["final_threshold"]
    generate_threshold_figure(step20["sweep_df"], final_threshold)
    print("  [OK] fig_threshold_tradeoff.png")

    # 5. Step 21 — Final locked test evaluation (single authorized test read)
    print(f"\n[5/6] STEP 21: Final Locked Test Evaluation (E10)...")
    results = run_step21_locked_test_evaluation(
        ensemble, ensemble_calibrator, final_threshold, ensemble_calibrator.name,
        X_val, y_val_arr, calibrated_val_probs,
    )
    generate_test_figures(results)
    print("  [OK] fig_test_confusion_roc_pr.png, fig_test_calibration_curve.png")

    # 6. Final report & locked decision record
    print(f"\n[6/6] Final Test Metrics (LOCKED PIPELINE — these are the paper numbers):")
    for k, v in results["metrics"].items():
        print(f"  * {k:32s}: {v:.4f}")
    cm = results["confusion_matrix"]
    print(f"\n  Confusion Matrix @ threshold {final_threshold:.2f}:")
    print(f"    TN={cm[0, 0]:3d}  FP={cm[0, 1]:3d}")
    print(f"    FN={cm[1, 0]:3d}  TP={cm[1, 1]:3d}")

    decision_payload = {
        "phase": 5,
        "steps": [19, 20, 21],
        "experiment_ids": ["E07", "E08", "E10"],
        "primary_metric": "ROC-AUC",
        "step19_calibration": {
            "chosen_method_per_candidate": step19["chosen"],
            "final_procedure": {
                "candidate": ENSEMBLE_NAME,
                "calibrator": ensemble_calibrator.name,
                "fitted_on": "splits/val.csv",
            },
        },
        "step20_threshold": {
            "locked_threshold": final_threshold,
            "selection_rule": "Maximize Youden's J; ties resolved toward higher sensitivity",
            "validation_operating_point": {
                "recall_sensitivity": float(step20["best_row"]["recall_sensitivity"]),
                "specificity": float(step20["best_row"]["specificity"]),
                "f1": float(step20["best_row"]["f1"]),
            },
        },
        "step21_final_test": results["metrics"],
        "production_artifact": {
            "path": str(results["artifact_path"]),
            "contents": "Phase 4 hybrid ensemble + calibration + locked threshold",
        },
        "integrity": {
            "oof_reproduction_matches_phase4": oof_ok,
            "calibration_fitted_without_test_data": True,
            "threshold_selected_without_test_data": True,
            "test_split_accessed_exactly_once": _TEST_SPLIT_ACCESSED,
        },
        "decision": (
            "LOCKED. Final pipeline (ensemble + calibration + threshold) evaluated exactly once on "
            "the untouched test set. Results are final; no model, feature, calibration, or threshold "
            "changes are permitted without reopening the evaluation protocol."
        ),
    }
    with open(OUTPUT_DIR / "phase5_decision.json", "w", encoding="utf-8") as f:
        json.dump(decision_payload, f, indent=2)

    elapsed = time.time() - start
    print(f"\nPhase 5 (Steps 19-21) completed successfully in {elapsed:.1f}s!")
    print(f"All outputs and figures stored in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()













