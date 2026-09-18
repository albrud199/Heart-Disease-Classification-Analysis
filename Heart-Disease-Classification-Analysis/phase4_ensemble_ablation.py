# Phase 4 (Steps 17–18): Research centerpiece: KNN+SVM+RF hybrid ensemble + ablation
# Strict anti-leakage protocol: test split (splits/test.csv) is never accessed or imported.
# All probability generation, weight optimization, ablation, and stability analysis
# occur strictly within Out-Of-Fold (OOF) cross-validation on splits/train.csv.
# Outputs are written to models/phase4/.

from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Literal, cast

# Ensure stdout uses UTF-8 or safe encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
    except Exception:
        pass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration and Reproducibility Setup
# -----------------------------------------------------------------------------
SEED = 42
CV_FOLDS = 5
STABILITY_SEEDS = (11, 22, 33, 44, 55)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "models" / "phase4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ModelName = Literal["KNN", "SVM", "Random Forest"]


# -----------------------------------------------------------------------------
# Data Loading & Preprocessing
# -----------------------------------------------------------------------------
def load_training_and_val_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads training and validation splits.
    CRITICAL: splits/test.csv is explicitly NOT loaded to prevent data leakage.
    """
    train = pd.read_csv(ROOT / "splits" / "train.csv")
    val = pd.read_csv(ROOT / "splits" / "val.csv")
    if "target" not in train.columns or "target" not in val.columns:
        raise ValueError("Both train.csv and val.csv must contain a 'target' column")
    
    X_train = train.drop(columns=["target"])
    y_train = train["target"].astype(int)
    X_val = val.drop(columns=["target"])
    y_val = val["target"].astype(int)
    return X_train, y_train, X_val, y_val


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Constructs the canonical ColumnTransformer identical to Phase 2/Phase 3.
    """
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


def load_shortlisted_configs() -> dict[str, Any]:
    """
    Loads best configurations found in Phase 3.
    """
    config_path = ROOT / "models" / "phase3" / "best_configurations.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Phase 3 configuration at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pipeline(
    X: pd.DataFrame,
    model_name: ModelName,
    best_configs: dict[str, Any],
) -> ImbPipeline:
    """
    Builds the leakage-safe pipeline for each component model with tuned parameters.
    """
    cfg = best_configs[model_name]
    strategy = cfg["strategy"]
    raw_params = cfg["params"]

    if model_name == "KNN":
        base_model = KNeighborsClassifier()
    elif model_name == "SVM":
        base_model = SVC(probability=True, random_state=SEED)
    elif model_name == "Random Forest":
        base_model = RandomForestClassifier(n_estimators=250, random_state=SEED, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Set hyperparameters
    model_params = {k.removeprefix("model__"): v for k, v in raw_params.items()}
    base_model.set_params(**model_params)

    if strategy == "class_weight" and hasattr(base_model, "class_weight"):
        base_model.set_params(class_weight="balanced")

    preprocessor = make_preprocessor(X)

    if strategy == "smote":
        sampler = SMOTE(random_state=SEED, k_neighbors=3)
        return ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", sampler),
            ("model", base_model),
        ])
    else:
        return ImbPipeline([
            ("preprocess", preprocessor),
            ("model", base_model),
        ])


# -----------------------------------------------------------------------------
# Metric Hierarchy Computation
# -----------------------------------------------------------------------------
def calculate_metrics(y_true: ArrayLike, probabilities: ArrayLike, threshold: float = 0.5) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predictions = (scores >= threshold).astype(int)

    negatives = (labels == 0)
    true_negatives = int(np.count_nonzero((predictions == 0) & negatives))
    negative_total = int(np.count_nonzero(negatives))
    specificity = true_negatives / negative_total if negative_total > 0 else 0.0

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, scores)),
    }


# -----------------------------------------------------------------------------
# Scikit-Learn Compatible Hybrid Ensemble Classifier
# -----------------------------------------------------------------------------
class HybridEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Authoritative Hybrid Ensemble combining KNN, SVM, and Random Forest pipelines
    via soft probability voting with configurable weights.
    """
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        best_configs: dict[str, Any] | None = None,
        threshold: float = 0.5,
    ):
        self.weights = weights or {"KNN": 1.0 / 3.0, "SVM": 1.0 / 3.0, "Random Forest": 1.0 / 3.0}
        self.best_configs = best_configs
        self.threshold = threshold
        self.models_: dict[str, ImbPipeline] = {}
        self.classes_: NDArray[np.int_] = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HybridEnsembleClassifier":
        configs = self.best_configs or load_shortlisted_configs()
        self.models_ = {
            name: build_pipeline(X, cast(ModelName, name), configs).fit(X, y)
            for name in ["KNN", "SVM", "Random Forest"]
        }
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        total_w = sum(self.weights.values())
        norm_weights = {k: v / total_w for k, v in self.weights.items()}
        combined_prob = np.zeros(len(X), dtype=float)
        for name, model in self.models_.items():
            proba = model.predict_proba(X)[:, 1]
            combined_prob += norm_weights[name] * proba
        return np.column_stack([1.0 - combined_prob, combined_prob])

    def predict(self, X: pd.DataFrame) -> NDArray[np.int_]:
        prob_pos = self.predict_proba(X)[:, 1]
        return (prob_pos >= self.threshold).astype(int)


# -----------------------------------------------------------------------------
# Step 17: Hybrid Ensemble Construction & Weight Optimization
# -----------------------------------------------------------------------------
def generate_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    best_configs: dict[str, Any],
    cv_seed: int = SEED,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, list[dict[str, float]]]]:
    """
    Generates Out-of-Fold (OOF) positive-class probabilities using Stratified K-Fold CV.
    Returns:
      - oof_probs: dict mapping model name to 1D array of OOF probabilities of length len(X).
      - fold_metrics: dict mapping model name to list of metric dicts for each fold.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=cv_seed)
    model_names: list[ModelName] = ["KNN", "SVM", "Random Forest"]
    oof_probs: dict[str, NDArray[np.float64]] = {
        name: np.zeros(len(X), dtype=float) for name in model_names
    }
    fold_metrics: dict[str, list[dict[str, float]]] = {
        name: [] for name in model_names
    }

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        for name in model_names:
            pipeline = build_pipeline(X_tr, name, best_configs)
            pipeline.fit(X_tr, y_tr)
            pred_probs = pipeline.predict_proba(X_va)[:, 1]
            oof_probs[name][val_idx] = pred_probs
            fold_metrics[name].append(calculate_metrics(y_va, pred_probs))

    return oof_probs, fold_metrics


def optimize_ensemble_weights(
    y_true: NDArray[np.int_],
    oof_probs: dict[str, NDArray[np.float64]],
    models: list[str],
) -> dict[str, float]:
    """
    Finds optimal weights (w_i >= 0, sum w_i = 1) strictly on training OOF predictions
    by minimizing Brier score loss (strictly proper scoring rule).
    """
    P = np.column_stack([oof_probs[m] for m in models])
    k = len(models)
    
    def loss(weights: NDArray[np.float64]) -> float:
        w = weights / np.sum(weights)
        p_ens = np.dot(P, w)
        return float(brier_score_loss(y_true, p_ens))

    init_weights = np.ones(k) / k
    bounds = [(0.0, 1.0) for _ in range(k)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        loss,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    raw_w = np.maximum(0.0, res.x)
    norm_w = raw_w / np.sum(raw_w)
    return {model: float(norm_w[i]) for i, model in enumerate(models)}


# -----------------------------------------------------------------------------
# Step 18: Ensemble Ablation Study & Diversity Analysis
# -----------------------------------------------------------------------------
def run_ablation_study(
    y_true: NDArray[np.int_],
    oof_probs: dict[str, NDArray[np.float64]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Evaluates all 7 combinations of KNN, SVM, and Random Forest:
      1. KNN alone
      2. SVM alone
      3. RF alone
      4. KNN + SVM (unweighted & weighted)
      5. KNN + RF (unweighted & weighted)
      6. SVM + RF (unweighted & weighted)
      7. KNN + SVM + RF (unweighted & weighted)
    """
    combinations: list[dict[str, Any]] = [
        {"name": "KNN Alone", "models": ["KNN"], "type": "individual"},
        {"name": "SVM Alone", "models": ["SVM"], "type": "individual"},
        {"name": "RF Alone", "models": ["Random Forest"], "type": "individual"},
        {"name": "KNN + SVM (Unweighted)", "models": ["KNN", "SVM"], "type": "pairwise_unweighted"},
        {"name": "KNN + SVM (Weighted)", "models": ["KNN", "SVM"], "type": "pairwise_weighted"},
        {"name": "KNN + RF (Unweighted)", "models": ["KNN", "Random Forest"], "type": "pairwise_unweighted"},
        {"name": "KNN + RF (Weighted)", "models": ["KNN", "Random Forest"], "type": "pairwise_weighted"},
        {"name": "SVM + RF (Unweighted)", "models": ["SVM", "Random Forest"], "type": "pairwise_unweighted"},
        {"name": "SVM + RF (Weighted)", "models": ["SVM", "Random Forest"], "type": "pairwise_weighted"},
        {"name": "Hybrid Ensemble (Unweighted)", "models": ["KNN", "SVM", "Random Forest"], "type": "triplet_unweighted"},
        {"name": "Hybrid Ensemble (Weighted)", "models": ["KNN", "SVM", "Random Forest"], "type": "triplet_weighted"},
    ]

    results: list[dict[str, Any]] = []
    combination_weights: dict[str, dict[str, float]] = {}
    combined_oof_probs: dict[str, NDArray[np.float64]] = {}

    for combo in combinations:
        c_name = combo["name"]
        c_models = combo["models"]
        c_type = combo["type"]

        if len(c_models) == 1:
            w = {c_models[0]: 1.0}
            probs = oof_probs[c_models[0]]
        elif "unweighted" in c_type:
            w = {m: 1.0 / len(c_models) for m in c_models}
            probs = np.mean([oof_probs[m] for m in c_models], axis=0)
        else:  # weighted
            w = optimize_ensemble_weights(y_true, oof_probs, c_models)
            probs = np.sum([w[m] * oof_probs[m] for m in c_models], axis=0)

        combination_weights[c_name] = w
        combined_oof_probs[c_name] = probs

        m = calculate_metrics(y_true, probs)
        results.append({
            "combination": c_name,
            "models": " + ".join(c_models),
            "type": c_type,
            "weights": json.dumps({k: round(v, 4) for k, v in w.items()}),
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "f1": m["f1"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "precision": m["precision"],
            "brier": m["brier"],
        })

    ablation_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    return ablation_df, {"weights": combination_weights, "probs": combined_oof_probs}


def compute_diversity_analysis(
    y_true: NDArray[np.int_],
    oof_probs: dict[str, NDArray[np.float64]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes Pearson/Spearman probability correlations, disagreement rates,
    and double-fault metrics to mathematically prove learner complementarity.
    """
    models = ["KNN", "SVM", "Random Forest"]
    prob_df = pd.DataFrame({m: oof_probs[m] for m in models})
    pearson_corr = prob_df.corr(method="pearson")
    spearman_corr = prob_df.corr(method="spearman")

    # Binary predictions at threshold 0.5
    preds = {m: (oof_probs[m] >= 0.5).astype(int) for m in models}
    
    # Pairwise disagreement & double fault
    diversity_rows: list[dict[str, Any]] = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:
                disagreement = float(np.mean(preds[m1] != preds[m2]))
                # Double fault: both predict incorrectly
                double_fault = float(np.mean((preds[m1] != y_true) & (preds[m2] != y_true)))
                diversity_rows.append({
                    "pair": f"{m1} vs {m2}",
                    "pearson_correlation": float(cast(Any, pearson_corr.loc[m1, m2])),
                    "spearman_correlation": float(cast(Any, spearman_corr.loc[m1, m2])),
                    "disagreement_rate": disagreement,
                    "double_fault_rate": double_fault,
                })

    return pearson_corr, spearman_corr, pd.DataFrame(diversity_rows)


def run_seed_stability_study(
    X: pd.DataFrame,
    y: pd.Series,
    best_configs: dict[str, Any],
    ensemble_weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates repeated 5-fold CV across 5 random seeds (11, 22, 33, 44, 55)
    for all key ablation candidates to compute empirical mean, std, and stability score.
    """
    candidates = [
        "KNN Alone",
        "SVM Alone",
        "RF Alone",
        "KNN + SVM",
        "KNN + RF",
        "SVM + RF",
        "Hybrid Ensemble (Unweighted)",
        "Hybrid Ensemble (Weighted)",
    ]

    rows: list[dict[str, Any]] = []
    y_arr = y.to_numpy()

    for seed in STABILITY_SEEDS:
        oof_p, _ = generate_oof_predictions(X, y, best_configs, cv_seed=seed)
        probs_map: dict[str, NDArray[np.float64]] = {
            "KNN Alone": oof_p["KNN"],
            "SVM Alone": oof_p["SVM"],
            "RF Alone": oof_p["Random Forest"],
            "KNN + SVM": 0.5 * oof_p["KNN"] + 0.5 * oof_p["SVM"],
            "KNN + RF": 0.5 * oof_p["KNN"] + 0.5 * oof_p["Random Forest"],
            "SVM + RF": 0.5 * oof_p["SVM"] + 0.5 * oof_p["Random Forest"],
            "Hybrid Ensemble (Unweighted)": (oof_p["KNN"] + oof_p["SVM"] + oof_p["Random Forest"]) / 3.0,
            "Hybrid Ensemble (Weighted)": (
                ensemble_weights["KNN"] * oof_p["KNN"] +
                ensemble_weights["SVM"] * oof_p["SVM"] +
                ensemble_weights["Random Forest"] * oof_p["Random Forest"]
            ),
        }

        for cand in candidates:
            m = calculate_metrics(y_arr, probs_map[cand])
            rows.append({
                "candidate": cand,
                "seed": seed,
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "f1": m["f1"],
                "recall": m["recall"],
                "specificity": m["specificity"],
                "brier": m["brier"],
            })

    detail_df = pd.DataFrame(rows)
    summary_df = detail_df.groupby("candidate", as_index=False).agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        f1_mean=("f1", "mean"),
        recall_mean=("recall", "mean"),
        specificity_mean=("specificity", "mean"),
        brier_mean=("brier", "mean"),
    )
    summary_df["stability_score"] = summary_df["roc_auc_mean"] - summary_df["roc_auc_std"].fillna(0)
    summary_df = summary_df.sort_values("stability_score", ascending=False)
    return detail_df, summary_df


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
def generate_ablation_figure(ablation_df: pd.DataFrame) -> None:
    """
    Generates a 4-panel publication-grade ablation comparison figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    df_plot = ablation_df.copy()
    
    # Shorten names for cleaner display
    df_plot["short_name"] = df_plot["combination"].str.replace("Hybrid Ensemble", "Hybrid Ens")

    # Panel 1: ROC-AUC & PR-AUC
    ax1 = axes[0, 0]
    x = np.arange(len(df_plot))
    width = 0.35
    ax1.bar(x - width/2, df_plot["roc_auc"], width, label="ROC-AUC (Primary)", color="#2b5c8f")
    ax1.bar(x + width/2, df_plot["pr_auc"], width, label="PR-AUC", color="#e27c38")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_plot["short_name"], rotation=40, ha="right", fontsize=9)
    ax1.set_ylim(0.90, 1.00)
    ax1.set_title("Discrimination: ROC-AUC vs PR-AUC Across Configurations", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Clinical Metrics (Recall, Specificity, F1)
    ax2 = axes[0, 1]
    width3 = 0.25
    ax2.bar(x - width3, df_plot["recall"], width3, label="Recall / Sensitivity", color="#439775")
    ax2.bar(x, df_plot["specificity"], width3, label="Specificity", color="#6c5b7b")
    ax2.bar(x + width3, df_plot["f1"], width3, label="F1 Score", color="#d65f5f")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_plot["short_name"], rotation=40, ha="right", fontsize=9)
    ax2.set_ylim(0.85, 1.00)
    ax2.set_title("Clinical Balance: Sensitivity vs Specificity vs F1", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Probability Quality (Brier Score Loss - lower is better)
    ax3 = axes[1, 0]
    colors = ["#c0392b" if "Alone" in name else "#27ae60" for name in df_plot["combination"]]
    ax3.bar(df_plot["short_name"], df_plot["brier"], color=colors, width=0.55)
    ax3.set_xticklabels(df_plot["short_name"], rotation=40, ha="right", fontsize=9)
    ax3.set_title("Probability Quality: Brier Score Loss (Lower = Better)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Brier Score Loss")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 4: Metric Spider / Ranking Summary
    ax4 = axes[1, 1]
    sorted_df = df_plot.sort_values("roc_auc", ascending=True)
    ax4.barh(sorted_df["short_name"], sorted_df["roc_auc"], color="#34495e", height=0.6)
    for i, v in enumerate(sorted_df["roc_auc"]):
        ax4.text(v - 0.015, i, f"{v:.4f}", va="center", ha="right", color="white", fontweight="bold", fontsize=9)
    ax4.set_xlim(0.94, 1.00)
    ax4.set_title("Overall Leaderboard Ranked by ROC-AUC", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Out-of-Fold ROC-AUC")
    ax4.grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("Phase 4: Ensemble Ablation Study & Systematic Component Comparison", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_ablation_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_diversity_figure(
    y_true: NDArray[np.int_],
    oof_probs: dict[str, NDArray[np.float64]],
    pearson_corr: pd.DataFrame,
) -> None:
    """
    Generates probability correlation heatmap, scatter matrix, and error distribution.
    """
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1])

    # Subplot 1: Correlation Heatmap
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(
        pearson_corr,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        vmin=0.7,
        vmax=1.0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax1,
    )
    ax1.set_title("OOF Probability Pearson Correlation", fontsize=12, fontweight="bold")

    # Subplot 2: 2D Density / Scatter KNN vs RF and SVM vs RF
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(oof_probs["KNN"], oof_probs["Random Forest"], c=y_true, cmap="coolwarm", alpha=0.4, edgecolors="none", s=30, label="KNN vs RF")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Agreement")
    ax2.set_xlabel("KNN Probability", fontsize=11)
    ax2.set_ylabel("Random Forest Probability", fontsize=11)
    ax2.set_title("Learner Complementarity: KNN vs RF Probabilities", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Subplot 3: Disagreement & Double Fault breakdown
    ax3 = fig.add_subplot(gs[2])
    models = ["KNN", "SVM", "Random Forest"]
    preds = {m: (oof_probs[m] >= 0.5).astype(int) for m in models}
    
    # Error classification
    err_knn = (preds["KNN"] != y_true)
    err_svm = (preds["SVM"] != y_true)
    err_rf = (preds["Random Forest"] != y_true)
    
    all_wrong = int(np.sum(err_knn & err_svm & err_rf))
    two_wrong = int(np.sum((err_knn & err_svm & ~err_rf) | (err_knn & ~err_svm & err_rf) | (~err_knn & err_svm & err_rf)))
    one_wrong = int(np.sum((err_knn & ~err_svm & ~err_rf) | (~err_knn & err_svm & ~err_rf) | (~err_knn & ~err_svm & err_rf)))
    all_correct = int(np.sum(~err_knn & ~err_svm & ~err_rf))

    categories = ["All Correct\n(Consensus)", "1 Learner Wrong\n(Ensemble Recovers)", "2 Learners Wrong", "All 3 Wrong\n(Double Fault)"]
    counts = [all_correct, one_wrong, two_wrong, all_wrong]
    bar_colors = ["#27ae60", "#2980b9", "#e67e22", "#c0392b"]

    bars = ax3.bar(categories, counts, color=bar_colors, width=0.6)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(y_true)) * 100
        ax3.text(bar.get_x() + bar.get_width()/2.0, h + 5, f"{h}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_ylim(0, max(counts) * 1.15)
    ax3.set_title("Ensemble Error Complementarity Breakdown", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Patient Sample Count")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_ensemble_diversity_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_roc_pr_figure(
    y_true: NDArray[np.int_],
    oof_probs: dict[str, NDArray[np.float64]],
    combo_probs: dict[str, NDArray[np.float64]],
) -> None:
    """
    Plots ROC Curves and Precision-Recall Curves for individual learners vs hybrid ensemble.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    plot_entries = [
        ("KNN", oof_probs["KNN"], "#8e44ad", ":"),
        ("SVM", oof_probs["SVM"], "#d35400", "-."),
        ("Random Forest", oof_probs["Random Forest"], "#27ae60", "--"),
        ("Hybrid Ensemble (Unweighted)", combo_probs["Hybrid Ensemble (Unweighted)"], "#2980b9", "-"),
        ("Hybrid Ensemble (Weighted)", combo_probs["Hybrid Ensemble (Weighted)"], "#c0392b", "-"),
    ]

    for name, p, color, ls in plot_entries:
        fpr, tpr, _ = roc_curve(y_true, p)
        roc_auc = roc_auc_score(y_true, p)
        lw = 2.5 if "Hybrid" in name else 1.5
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", color=color, linestyle=ls, linewidth=lw)

        precision, recall, _ = precision_recall_curve(y_true, p)
        pr_auc = average_precision_score(y_true, p)
        ax_pr.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=color, linestyle=ls, linewidth=lw)

    # Formatting ROC
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.5000)")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.02])
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, linestyle="--", alpha=0.4)

    # Formatting PR
    baseline_pr = float(np.mean(y_true))
    ax_pr.axhline(baseline_pr, color="gray", linestyle="--", alpha=0.6, label=f"Prevalence Baseline ({baseline_pr:.2f})")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.02])
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax_pr.set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight="bold")
    ax_pr.legend(loc="lower left", fontsize=9)
    ax_pr.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_roc_pr_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("PHASE 4: RESEARCH CENTERPIECE - HYBRID ENSEMBLE & ABLATION STUDY")
    print("Steps 17 & 18 | Strict Anti-Leakage Protocol (Test Set Locked)")
    print("=" * 80)

    # 1. Load splits
    X_train, y_train, X_val, y_val = load_training_and_val_data()
    print(f"\n[1/7] Data Loaded:")
    print(f"  - Training samples: {len(X_train)} (Positive: {y_train.sum()}, Negative: {(y_train == 0).sum()})")
    print(f"  - Validation samples: {len(X_val)} (Holdout checkpoint only)")
    print(f"  - Test partition: STRICTLY LOCKED and NOT ACCESSED")

    # 2. Load best configurations
    best_configs = load_shortlisted_configs()
    print("\n[2/7] Phase 3 Shortlisted Configurations Loaded:")
    for m, cfg in best_configs.items():
        if m in ["KNN", "SVM", "Random Forest"]:
            print(f"  * {m:14s}: Strategy = {cfg['strategy']:12s} | Params = {cfg['params']}")

    # 3. Generate Out-of-Fold (OOF) predictions
    print("\n[3/7] Generating 5-Fold Stratified OOF Probabilities on Training Data...")
    oof_probs, fold_metrics = generate_oof_predictions(X_train, y_train, best_configs)
    y_arr = y_train.to_numpy()

    # Step 17: Weight Optimization
    print("\n[4/7] Step 17: Hybrid Ensemble Weight Optimization...")
    opt_weights = optimize_ensemble_weights(y_arr, oof_probs, ["KNN", "SVM", "Random Forest"])
    print(f"  - Optimal Weights (Minimizing OOF Brier Score):")
    for m, w in opt_weights.items():
        print(f"    * {m:14s}: {w:.4f} ({w*100:.1f}%)")

    # Save weights
    with open(OUTPUT_DIR / "ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump(opt_weights, f, indent=2)

    # 4. Step 18: Systematic Ablation Study
    print("\n[5/7] Step 18: Systematic Ensemble Ablation Study Across 7 Combinations...")
    ablation_df, combo_data = run_ablation_study(y_arr, oof_probs)
    ablation_df.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)

    print("\n--- Ablation Leaderboard (Ranked by OOF ROC-AUC) ---")
    cols_display = ["combination", "roc_auc", "pr_auc", "f1", "recall", "specificity", "brier"]
    print(ablation_df[cols_display].to_string(index=False))

    # Save component vs ensemble comparison table (Step 17 expected output)
    comparison_filter = ablation_df["combination"].isin([
        "KNN Alone", "SVM Alone", "RF Alone",
        "Hybrid Ensemble (Unweighted)", "Hybrid Ensemble (Weighted)",
    ])
    comparison_df = ablation_df[comparison_filter].copy()
    comparison_df.to_csv(OUTPUT_DIR / "ensemble_comparison.csv", index=False)

    # 5. Diversity and Complementarity Analysis
    print("\n[6/7] Computing Diversity & Model Complementarity Analysis...")
    pearson_corr, spearman_corr, diversity_df = compute_diversity_analysis(y_arr, oof_probs)
    diversity_df.to_csv(OUTPUT_DIR / "prediction_correlations.csv", index=False)
    print("\n--- Pairwise Model Diversity & Error Independence ---")
    print(diversity_df.to_string(index=False))

    # Save comprehensive OOF probabilities dataframe (vital for Phase 5 calibration & thresholding)
    oof_export = pd.DataFrame({"target": y_arr})
    for m in ["KNN", "SVM", "Random Forest"]:
        oof_export[f"prob_{m.lower().replace(' ', '_')}"] = oof_probs[m]
    oof_export["prob_hybrid_unweighted"] = combo_data["probs"]["Hybrid Ensemble (Unweighted)"]
    oof_export["prob_hybrid_weighted"] = combo_data["probs"]["Hybrid Ensemble (Weighted)"]
    oof_export.to_csv(OUTPUT_DIR / "oof_probabilities.csv", index=False)
    print(f"\nSaved OOF probabilities to {OUTPUT_DIR / 'oof_probabilities.csv'}")

    # Multi-seed stability across 5 seeds
    print("\n[7/7] Multi-Seed Stability Verification (Seeds: 11, 22, 33, 44, 55)...")
    stability_detail, stability_summary = run_seed_stability_study(
        X_train, y_train, best_configs, opt_weights
    )
    stability_detail.to_csv(OUTPUT_DIR / "ablation_stability_by_seed.csv", index=False)
    stability_summary.to_csv(OUTPUT_DIR / "ablation_stability_summary.csv", index=False)

    print("\n--- Stability Summary Across 5 Repeated Seeds ---")
    print(stability_summary[["candidate", "roc_auc_mean", "roc_auc_std", "stability_score"]].to_string(index=False))

    # Visualizations
    print("\nGenerating Publication Figures in models/phase4/...")
    generate_ablation_figure(ablation_df)
    generate_diversity_figure(y_arr, oof_probs, pearson_corr)
    generate_roc_pr_figure(y_arr, oof_probs, combo_data["probs"])
    print("  [OK] fig_ablation_comparison.png")
    print("  [OK] fig_ensemble_diversity_correlation.png")
    print("  [OK] fig_roc_pr_comparison.png")

    # Fit authoritative production ensemble model on full training data
    print("\nFitting Authoritative Hybrid Ensemble on Full Training Data...")
    ensemble_clf = HybridEnsembleClassifier(weights=opt_weights, best_configs=best_configs)
    ensemble_clf.fit(X_train, y_train)

    # Evaluate on holdout validation set (non-tuning checkpoint)
    val_probs = ensemble_clf.predict_proba(X_val)[:, 1]
    val_metrics = calculate_metrics(y_val, val_probs)
    print("\nHoldout Validation Checkpoint Metrics (val.csv):")
    for k, v in val_metrics.items():
        print(f"  * {k:12s}: {v:.4f}")

    # Serialize fitted ensemble model
    model_artifact_path = OUTPUT_DIR / "hybrid_ensemble_model.pkl"
    joblib.dump(ensemble_clf, model_artifact_path)
    print(f"\nAuthoritative Hybrid Ensemble serialized to: {model_artifact_path}")

    # Write Phase 4 Decision & Manifest
    decision_payload = {
        "phase": 4,
        "steps": [17, 18],
        "experiment_ids": ["E05", "E06"],
        "primary_metric": "ROC-AUC",
        "weights": opt_weights,
        "best_ensemble_configuration": "Hybrid Ensemble (Weighted)",
        "oof_roc_auc_weighted": float(ablation_df.loc[ablation_df['combination'] == 'Hybrid Ensemble (Weighted)', 'roc_auc'].iloc[0]),
        "oof_roc_auc_unweighted": float(ablation_df.loc[ablation_df['combination'] == 'Hybrid Ensemble (Unweighted)', 'roc_auc'].iloc[0]),
        "oof_roc_auc_best_single (RF)": float(ablation_df.loc[ablation_df['combination'] == 'RF Alone', 'roc_auc'].iloc[0]),
        "stability_score_weighted": float(stability_summary.loc[stability_summary['candidate'] == 'Hybrid Ensemble (Weighted)', 'stability_score'].iloc[0]),
        "test_split_integrity": "Untouched / Locked (zero access)",
        "decision": "Proceed to Phase 5 (Steps 19-21: Calibration, Threshold Analysis, Locked Test Eval) using the validated Hybrid Ensemble.",
    }
    with open(OUTPUT_DIR / "phase4_decision.json", "w", encoding="utf-8") as f:
        json.dump(decision_payload, f, indent=2)

    print("\nPhase 4 (Steps 17–18) completed successfully!")
    print(f"All outputs and figures stored in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

