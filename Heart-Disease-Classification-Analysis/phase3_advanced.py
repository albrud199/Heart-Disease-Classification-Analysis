# Phase 3 (Steps 13-16): imbalance, tuning, stability, and model shortlist.
# The final test split is never read by this script. CV contains all learned operations.
# Outputs are written to models/phase3.

from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Any, Literal, TypedDict, cast
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
warnings.filterwarnings("ignore")

SEED = 42
CV_FOLDS = 5
STABILITY_SEEDS = (11, 22, 33, 44, 55)
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "models" / "phase3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ModelConfig(TypedDict):
    strategy: str
    params: dict[str, Any]

ModelName = Literal["Logistic Regression", "KNN", "SVM", "Random Forest"]


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    # Load only the Phase 1 training split; validation/test remain untouched.
    train = pd.read_csv(ROOT / "splits" / "train.csv")
    if "target" not in train.columns:
        raise ValueError("splits/train.csv must contain a 'target' column")
    return train.drop(columns=["target"]), train["target"].astype(int)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"] if c in X]
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )


def model_specs() -> dict[ModelName, BaseEstimator]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=SEED, n_jobs=-1),
    }


def build_pipeline(
    X: pd.DataFrame,
    model_name: ModelName,
    strategy: str,
    params: dict[str, Any] | None = None,
) -> ImbPipeline:
    model: Any = clone(model_specs()[model_name])
    if params:
        model_params: dict[str, Any] = {
            key.removeprefix("model__"): value for key, value in params.items()
        }
        model.set_params(**model_params)
    if strategy == "class_weight" and model_name in {"Logistic Regression", "SVM", "Random Forest"}:
        model.set_params(class_weight="balanced")
    preprocessor: Any = make_preprocessor(X)
    model_step: Any = model
    sampler: Any = SMOTE(random_state=SEED, k_neighbors=3)
    if strategy == "smote":
        steps: Any = [
            ("preprocess", preprocessor),
            ("smote", sampler),
            ("model", model_step),
        ]
    else:
        steps: Any = [("preprocess", preprocessor), ("model", model_step)]
    return ImbPipeline(steps)


def scoring() -> dict[str, str]:
    return {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1", "recall": "recall"}


def safe_mean(values: ArrayLike) -> float:
    return float(np.nanmean(np.asarray(values, dtype=float)))


def imbalance_experiments(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    strategies_by_model: dict[ModelName, tuple[str, ...]] = {
        "Logistic Regression": ("none", "class_weight", "smote"),
        "KNN": ("none", "smote"),
        "SVM": ("none", "class_weight", "smote"),
        "Random Forest": ("none", "class_weight", "smote"),
    }
    for model_name, strategies in strategies_by_model.items():
        for strategy in strategies:
            pipeline: Any = build_pipeline(X, model_name, strategy)
            result = cast(dict[str, NDArray[np.float64]], cross_validate(
                pipeline, X, y, cv=cv, scoring=scoring(), n_jobs=-1, error_score="raise",
            ))
            test_roc_auc = np.asarray(result["test_roc_auc"], dtype=float)
            test_pr_auc = np.asarray(result["test_pr_auc"], dtype=float)
            test_f1 = np.asarray(result["test_f1"], dtype=float)
            test_recall = np.asarray(result["test_recall"], dtype=float)
            rows.append({
                "model": model_name,
                "strategy": strategy,
                "roc_auc_mean": safe_mean(test_roc_auc),
                "roc_auc_std": float(np.std(test_roc_auc, ddof=1)),
                "pr_auc_mean": safe_mean(test_pr_auc),
                "f1_mean": safe_mean(test_f1),
                "recall_mean": safe_mean(test_recall),
            })
    frame = pd.DataFrame(rows).sort_values(["model", "roc_auc_mean"], ascending=[True, False])
    frame.to_csv(OUTPUT_DIR / "imbalance_comparison.csv", index=False)
    best = frame.groupby("model", as_index=False).first()
    best.to_csv(OUTPUT_DIR / "selected_imbalance_strategy.csv", index=False)
    return frame

def tuning(
    X: pd.DataFrame,
    y: pd.Series,
    imbalance: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[ModelName, ModelConfig]]:
    search_spaces: dict[ModelName, dict[str, list[Any]]] = {
        "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
        "KNN": {"model__n_neighbors": [3, 5, 7, 11], "model__weights": ["uniform", "distance"]},
        "SVM": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
        "Random Forest": {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
    }
    rows: list[dict[str, Any]] = []
    best_configs: dict[ModelName, ModelConfig] = {}
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    for model_name, grid in search_spaces.items():
        strategy = str(
            imbalance.loc[imbalance["model"] == model_name]
            .sort_values("roc_auc_mean", ascending=False)
            .iloc[0]["strategy"]
        )
        search: Any = GridSearchCV(
            build_pipeline(X, model_name, strategy), grid, scoring="roc_auc",
            cv=cv, n_jobs=-1, refit=True, return_train_score=False, error_score="raise",
        )
        search.fit(X, y)
        raw_params = cast(list[Any], cast(dict[str, Any], search.cv_results_)["params"])
        candidate_params: list[dict[str, Any]] = [cast(dict[str, Any], item) for item in raw_params]
        best_params: dict[str, Any] = dict(cast(dict[str, Any], search.best_params_))
        rows.append({
            "model": model_name,
            "strategy": strategy,
            "best_cv_roc_auc": float(search.best_score_),
            "best_params": json.dumps(best_params, sort_keys=True),
            "evaluated_candidates": len(candidate_params),
        })
        best_configs[model_name] = ModelConfig(strategy=strategy, params=best_params)
    frame = pd.DataFrame(rows).sort_values("best_cv_roc_auc", ascending=False)
    frame.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)
    with open(OUTPUT_DIR / "best_configurations.json", "w", encoding="utf-8") as handle:
        json.dump(best_configs, handle, indent=2)
    return frame, best_configs

def metrics(y_true: ArrayLike, probabilities: ArrayLike) -> dict[str, float]:
    labels: NDArray[np.int_] = np.asarray(y_true, dtype=int)
    scores: NDArray[np.float64] = np.asarray(probabilities, dtype=float)
    predictions: NDArray[np.int_] = np.asarray(scores >= 0.5, dtype=int)
    negatives: NDArray[np.bool_] = np.asarray(labels == 0, dtype=bool)
    true_negatives = int(np.count_nonzero((predictions == 0) & negatives))
    negative_total = int(np.count_nonzero(negatives))
    specificity = true_negatives / negative_total if negative_total > 0 else 0.0
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": specificity,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, scores)),
    }


def stability(
    X: pd.DataFrame,
    y: pd.Series,
    configs: dict[ModelName, ModelConfig],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, config in configs.items():
        for seed in STABILITY_SEEDS:
            cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
            estimator = build_pipeline(X, model_name, config["strategy"], config["params"])
            result: dict[str, NDArray[np.float64]] = cast(
                dict[str, NDArray[np.float64]],
                cross_validate(estimator, X, y, cv=cv, scoring=scoring(), n_jobs=-1, error_score="raise"),
            )
            test_roc_auc = np.asarray(result["test_roc_auc"], dtype=float)
            test_pr_auc = np.asarray(result["test_pr_auc"], dtype=float)
            test_f1 = np.asarray(result["test_f1"], dtype=float)
            test_recall = np.asarray(result["test_recall"], dtype=float)
            rows.append({
                "model": model_name,
                "seed": seed,
                "roc_auc_mean": safe_mean(test_roc_auc),
                "roc_auc_std_folds": float(np.std(test_roc_auc, ddof=1)),
                "pr_auc_mean": safe_mean(test_pr_auc),
                "f1_mean": safe_mean(test_f1),
                "recall_mean": safe_mean(test_recall),
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(OUTPUT_DIR / "stability_by_seed.csv", index=False)
    summary = detail.groupby("model", as_index=False).agg(
        roc_auc_mean=("roc_auc_mean", "mean"), roc_auc_std=("roc_auc_mean", "std"),
        pr_auc_mean=("pr_auc_mean", "mean"), f1_mean=("f1_mean", "mean"),
        recall_mean=("recall_mean", "mean"),
    )
    summary["stability_score"] = summary["roc_auc_mean"] - summary["roc_auc_std"].fillna(0)
    summary.sort_values("stability_score", ascending=False).to_csv(OUTPUT_DIR / "stability_summary.csv", index=False)
    return summary

def shortlist(
    tuned: pd.DataFrame,
    stable: pd.DataFrame,
) -> pd.DataFrame:
    leaderboard = tuned.merge(stable, on="model", how="left")
    leaderboard["shortlist_rank"] = leaderboard["stability_score"].rank(method="min", ascending=False).astype(int)
    leaderboard["selected_for_phase4"] = leaderboard["shortlist_rank"] <= 3
    leaderboard = leaderboard.sort_values(["selected_for_phase4", "stability_score"], ascending=[False, False])
    leaderboard.to_csv(OUTPUT_DIR / "model_shortlist_leaderboard.csv", index=False)
    selected = leaderboard.loc[leaderboard["selected_for_phase4"], "model"].tolist()
    with open(OUTPUT_DIR / "phase3_decision.json", "w", encoding="utf-8") as handle:
        json.dump({
            "primary_metric": "ROC-AUC",
            "secondary_metrics": ["PR-AUC", "F1", "recall", "specificity", "precision", "Brier score"],
            "test_set_used": False,
            "shortlisted_models": selected,
            "selection_rule": "top three by stability score = mean ROC-AUC minus across-seed standard deviation",
            "phase4_note": "Use these CV-selected candidates for ensemble research; do not use final test results for selection.",
        }, handle, indent=2)
    return leaderboard

def main() -> None:
    print("Phase 3: advanced modeling (steps 13-16)")
    X, y = load_training_data()
    print(f"Training rows: {len(X)}; features: {X.shape[1]}; positive rate: {y.mean():.3f}")
    imbalance = imbalance_experiments(X, y)
    tuned, configs = tuning(X, y, imbalance)
    stable = stability(X, y, configs)
    board = shortlist(tuned, stable)
    print("\nPhase 3 completed successfully.")
    print(board[["model", "best_cv_roc_auc", "roc_auc_mean", "roc_auc_std", "selected_for_phase4"]].to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
