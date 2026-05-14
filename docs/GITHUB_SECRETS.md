# GitHub Actions: Auto-Upload Model on Push

This workflow automatically uploads `models/model.pkl` to Hugging Face Hub whenever you push a new model.

## Setup (one-time)

### 1. Create Hugging Face Hub Token

- Go to https://huggingface.co/settings/tokens
- Click "New token"
- Choose "Write" access (to upload models)
- Copy the token (e.g., `hf_xxx...`)

### 2. Create Hugging Face Model Repo

- Go to https://huggingface.co/new
- Choose model type (or leave blank)
- Create repo (e.g., `heart-disease-model`)
- Note the repo ID: `your-username/heart-disease-model`

### 3. Add GitHub Secrets

Go to your GitHub repo:
- Settings → Secrets and variables → Actions
- Click "New repository secret"
- Add these two secrets:

**Secret 1:**
- Name: `HUGGINGFACE_HUB_TOKEN`
- Value: paste the token from step 1

**Secret 2:**
- Name: `HF_MODEL_REPO`
- Value: your repo ID from step 2 (e.g., `myusername/heart-disease-model`)

### 4. Test It

```bash
git add .
git commit -m "test model upload"
git push origin main
```

Then go to GitHub → Actions tab and watch the workflow run. It will upload `models/model.pkl` to your HF repo automatically.

## How It Works

The workflow in `.github/workflows/upload_model.yml`:
- Triggers when you push changes to `models/` or `scripts/prepare_model.py`
- Runs `python scripts/push_hf_model.py`
- Uses the secrets to authenticate to HF Hub
- Uploads `models/model.pkl` to your repo

## Troubleshooting

- **Workflow doesn't run?** Check that your repo is public or GitHub Actions is enabled.
- **Upload fails?** Check that the secrets are set correctly in Settings → Secrets.
- **Token expired?** Create a new token on HF and update the secret.
