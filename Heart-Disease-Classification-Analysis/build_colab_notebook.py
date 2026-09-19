import json
import os

phases = [
    "phase1_foundation.py",
    "phase2_modeling_foundation.py",
    "phase3_advanced.py",
    "phase4_ensemble_ablation.py",
    "phase5_calibration_final_eval.py",
    "phase6_explainability_robustness.py",
    "phase7_production_handoff.py",
    "production_inference.py"
]

cells = []

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Heart Disease Classification Analysis - Colab Pipeline\n",
        "This notebook contains the full Heart Disease classification pipeline.\n",
        "\n",
        "### ⚠️ Instructions\n",
        "1. Please **upload `heart.csv`** to the default `/content` directory in this Colab environment before running the cells below.\n",
        "2. Run all cells sequentially. The first cell will install any dependencies. The next few cells will write the python scripts into the Colab environment, and the subsequent cells will execute the pipeline."
    ]
})

# Dependencies
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["!pip install shap imbalanced-learn scikit-learn pandas numpy matplotlib seaborn joblib"]
})

# Writefile cells
for phase in phases:
    if not os.path.exists(phase):
        print(f"File {phase} not found.")
        continue
        
    with open(phase, "r", encoding="utf-8") as f:
        code = f.read()
    
    # Prepend %%writefile phase.py
    code = f"%%writefile {phase}\n" + code
    
    lines = [line + "\n" for line in code.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Execute Pipeline\n",
        "The following cells will execute each phase of the pipeline sequentially."
    ]
})

# Execution cells
for phase in phases[:-1]:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"!python {phase}"]
    })

notebook = {
    "cells": cells,
    "metadata": {
      "language_info": {
        "name": "python"
      }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("Heart_Disease_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated: Heart_Disease_Colab.ipynb")

