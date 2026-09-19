import json
import glob
import os

phases = [
    "phase1_foundation.py",
    "phase2_modeling_foundation.py",
    "phase3_advanced.py",
    "phase4_ensemble_ablation.py",
    "phase5_calibration_final_eval.py",
    "phase6_explainability_robustness.py",
    "phase7_production_handoff.py"
]

cells = []

# Add intro markdown
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Heart Disease Classification Pipeline\n",
        "This notebook combines all the phases of the ML pipeline for execution in Google Colab.\n",
        "\n",
        "**Note**: Since these were originally separate scripts, ensure that the data `heart.csv` is uploaded to the Colab environment."
    ]
})

for phase in phases:
    if not os.path.exists(phase):
        print(f"File {phase} not found.")
        continue
        
    with open(phase, "r", encoding="utf-8") as f:
        code = f.read()
    
    # Add a markdown cell for the phase
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {phase}"]
    })
    
    # Add the code cell
    # Split code by lines and append \n to each except the last
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

with open("Heart_Disease_Pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated: Heart_Disease_Pipeline.ipynb")

