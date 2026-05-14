"""Upload `models/model.pkl` to a Hugging Face Hub repo.

Requires env var `HUGGINGFACE_HUB_TOKEN` and `HF_MODEL_REPO` (repo id like 'username/repo').

Usage:
    python scripts/push_hf_model.py
"""
import os
from huggingface_hub import HfApi

def main():
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    repo_id = os.environ.get("HF_MODEL_REPO")
    if not token or not repo_id:
        raise RuntimeError("Set HUGGINGFACE_HUB_TOKEN and HF_MODEL_REPO environment variables.")

    path = "models/model.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError("models/model.pkl not found — run scripts/prepare_model.py first")

    api = HfApi()
    print(f"Uploading {path} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo="model.pkl",
        repo_id=repo_id,
        token=token,
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
