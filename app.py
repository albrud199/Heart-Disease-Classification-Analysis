import os
import joblib
import numpy as np
import gradio as gr

def load_model():
    local_path = os.path.join("models", "model.pkl")
    if os.path.exists(local_path):
        return joblib.load(local_path)

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


def parse_features(text: str):
    try:
        parts = [p.strip() for p in text.replace(',', ' ').split() if p.strip()]
        arr = [float(x) for x in parts]
        return np.array(arr).reshape(1, -1)
    except Exception:
        return None


def predict(text: str):
    if MODEL is None:
        return "Model not loaded. Add `models/model.pkl` or set `HF_MODEL_REPO`.", ""

    X = parse_features(text)
    if X is None:
        return "Could not parse input. Provide comma-separated numeric features.", ""

    try:
        pred = MODEL.predict(X)[0]
    except Exception as e:
        return f"Prediction error: {e}", ""

    prob = ""
    if hasattr(MODEL, "predict_proba"):
        try:
            proba = MODEL.predict_proba(X)[0]
            if len(proba) == 2:
                prob = f"P(no disease)={proba[0]:.3f}, P(disease)={proba[1]:.3f}"
            else:
                prob = ", ".join([f"c{i}={p:.3f}" for i, p in enumerate(proba)])
        except Exception:
            prob = ""

    return int(pred), prob


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Enter features as comma-separated numbers (e.g. 63, 1, 3, ...)"),
    outputs=[gr.Label(num_top_classes=2), gr.Textbox()],
    examples=[["63,1,3,145,233,1,0,150,0,2.3,0,0,1"]],
    title="Heart Disease Prediction",
    description=(
        "Provide numeric features as comma-separated values. "
        "The model should be stored at `models/model.pkl` or set `HF_MODEL_REPO` env var to a Hugging Face repo with `model.pkl`."
    ),
)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
