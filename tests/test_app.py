import os
import importlib.util
import subprocess
import sys
from pathlib import Path


def test_prepare_model_and_predict():
    # Ensure model exists
    if not os.path.exists("models/model.pkl"):
        subprocess.check_call([sys.executable, "scripts/prepare_model.py"]) 

    # Import app.py from repository root even if pytest cwd changes.
    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "app.py"
    spec = importlib.util.spec_from_file_location("app", app_path)
    app = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(app)

    # sample from README defaults
    pred, prob = app.predict(63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1)

    assert isinstance(pred, int)
