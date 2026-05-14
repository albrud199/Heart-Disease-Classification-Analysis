import os
import importlib
import subprocess
import sys


def test_prepare_model_and_predict():
    # Ensure model exists
    if not os.path.exists("models/model.pkl"):
        subprocess.check_call([sys.executable, "scripts/prepare_model.py"]) 

    # Import app and call predict
    app = importlib.import_module("app")

    # simple sample from README/example
    sample = "63,1,3,145,233,1,0,150,0,2.3,0,0,1"
    pred, prob = app.predict(sample)

    assert isinstance(pred, int)
