# Heart Disease Classification Analysis

Machine learning pipeline for heart disease classification using structured clinical data. Includes EDA, modeling phases (baseline through ensemble), calibration, explainability, and production handoff.

## Project Structure
- `phase1_foundation.py` to `phase7_production_handoff.py`: Full ML pipeline
- `production_inference.py`: Inference script
- `models/`: Trained model artifacts and evaluation results
- `splits/`: Train/val/test CSVs
- `eda/`: Exploratory data analysis

## Setup
```bash
pip install -r requirements.txt
```

## Usage
Run phases sequentially:
```bash
python phase1_foundation.py
python phase2_modeling_foundation.py
# ... through phase7
```

## Data
`heart.csv` — clinical features for heart disease prediction.
