import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

def load_model():
    local_path = os.path.join("models", "model.pkl")
    if os.path.exists(local_path):
        return joblib.load(local_path)

    heart_csv = "heart.csv"
    if os.path.exists(heart_csv):
        data = pd.read_csv(heart_csv)
        if "target" in data.columns:
            features = data.drop(columns=["target"])
            target = data["target"]
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(features, target)
            return model

    repo = os.environ.get("HF_MODEL_REPO")
    if repo:
        try:
            from huggingface_hub import hf_hub_download
            fname = hf_hub_download(repo_id=repo, filename="model.pkl")
            return joblib.load(fname)
        except Exception:
            pass

    raise FileNotFoundError("Model not found. Place `models/model.pkl` or set `HF_MODEL_REPO`.")


MODEL = None
try:
    MODEL = load_model()
except Exception:
    MODEL = None


def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    if MODEL is None:
        return "Model not loaded. Add `models/model.pkl` or set `HF_MODEL_REPO`.", ""

    row = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
        columns=FEATURE_COLUMNS,
    )

    try:
        pred = MODEL.predict(row)[0]
    except Exception as e:
        return f"Prediction error: {e}", ""

    prob = ""
    if hasattr(MODEL, "predict_proba"):
        try:
            proba = MODEL.predict_proba(row)[0]
            if len(proba) == 2:
                prob = f"P(no disease)={proba[0]:.3f}, P(disease)={proba[1]:.3f}"
            else:
                prob = ", ".join([f"c{i}={p:.3f}" for i, p in enumerate(proba)])
        except Exception:
            prob = ""

    return int(pred), prob


with gr.Blocks(title="Heart Disease Prediction") as demo:
    gr.Markdown(
        "# Heart Disease Prediction\n"
        "Enter each medical feature by name. Categorical values are shown as code values so the model input is explicit."
    )

    with gr.Row():
        age = gr.Number(label="age", value=63, precision=0)
        sex = gr.Dropdown(
            choices=[("0 - female", 0), ("1 - male", 1)],
            value=1,
            label="sex",
        )
        cp = gr.Dropdown(
            choices=[
                ("0 - typical angina", 0),
                ("1 - atypical angina", 1),
                ("2 - non-anginal pain", 2),
                ("3 - asymptomatic", 3),
            ],
            value=3,
            label="cp",
        )

    with gr.Row():
        trestbps = gr.Number(label="trestbps", value=145, precision=0)
        chol = gr.Number(label="chol", value=233, precision=0)
        fbs = gr.Dropdown(choices=[("0 - false", 0), ("1 - true", 1)], value=1, label="fbs")
        restecg = gr.Dropdown(
            choices=[
                ("0 - normal", 0),
                ("1 - ST-T wave abnormality", 1),
                ("2 - left ventricular hypertrophy", 2),
            ],
            value=0,
            label="restecg",
        )

    with gr.Row():
        thalach = gr.Number(label="thalach", value=150, precision=0)
        exang = gr.Dropdown(choices=[("0 - no", 0), ("1 - yes", 1)], value=0, label="exang")
        oldpeak = gr.Number(label="oldpeak", value=2.3, precision=1)
        slope = gr.Dropdown(
            choices=[
                ("0 - upsloping", 0),
                ("1 - flat", 1),
                ("2 - downsloping", 2),
            ],
            value=0,
            label="slope",
        )

    with gr.Row():
        ca = gr.Number(label="ca", value=0, precision=0)
        thal = gr.Number(label="thal", value=1, precision=0)

    submit = gr.Button("Predict")
    prediction = gr.Label(label="Prediction")
    probability = gr.Textbox(label="Probability", interactive=False)

    submit.click(
        fn=predict,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
        outputs=[prediction, probability],
    )

    gr.Markdown(
        "## Feature guide\n"
        "- `age`: age in years\n"
        "- `sex`: 0 = female, 1 = male\n"
        "- `cp`: chest pain type code\n"
        "- `trestbps`: resting blood pressure\n"
        "- `chol`: serum cholesterol\n"
        "- `fbs`: fasting blood sugar > 120 mg/dl\n"
        "- `restecg`: resting ECG code\n"
        "- `thalach`: maximum heart rate achieved\n"
        "- `exang`: exercise-induced angina\n"
        "- `oldpeak`: ST depression\n"
        "- `slope`: slope of peak exercise ST segment\n"
        "- `ca`: number of major vessels\n"
        "- `thal`: thalassemia code\n"
    )


if __name__ == "__main__":
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860")))
    demo.launch(server_name="0.0.0.0", server_port=server_port, ssr_mode=False)
