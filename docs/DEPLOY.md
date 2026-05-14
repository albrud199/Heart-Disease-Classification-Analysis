# Deploy to Hugging Face Spaces (Free for public repos)

1. Push your repo to GitHub (make repo public):

```bash
git add .
git commit -m "prepare for HF Spaces"
git push origin main
```

2. Create a new Space on https://huggingface.co/spaces — choose `Gradio`.

3. Connect your GitHub repo to the Space or push directly to the Space repo.

4. HF Spaces will run `app.py` and provide a public URL.

Notes:
- If your `models/model.pkl` is larger than 50MB, use Git LFS or upload the model to the Hugging Face Hub and set `HF_MODEL_REPO=your-user/your-repo` in the Space settings.
- To upload model to HF Hub: set `HUGGINGFACE_HUB_TOKEN` locally and run:

```bash
python scripts/push_hf_model.py
```
