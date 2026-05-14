"""Train a simple RandomForest on `heart.csv` and save `models/model.pkl`.
Also logs a run to MLflow and optionally to Weights & Biases if `WANDB_API_KEY` is set.

Usage:
    python scripts/prepare_model.py
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main(csv_path="heart.csv"):
    df = pd.read_csv(csv_path)
    if "target" not in df.columns:
        raise ValueError("Expected 'target' column in CSV")

    X = df.drop(columns=["target"]) 
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "model.pkl")
    joblib.dump(clf, model_path)

    # Optional MLflow logging (local file store)
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_experiment("heart-mlops")
        with mlflow.start_run():
            mlflow.log_param("model", "RandomForest")
            mlflow.log_metric("accuracy", float(acc))
            mlflow.sklearn.log_model(clf, "model")
    except Exception:
        pass

    # Optional W&B logging if key present
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb.init(project="heart-mlops", reinit=True)
            wandb.log({"accuracy": float(acc)})
            wandb.sklearn.plot_classifier(clf, X_test, y_test)
            wandb.finish()
        except Exception:
            pass

    print(f"Saved model to {model_path} — test accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
