---
title: Heart Disease Prediction
emoji: 🫀
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: "6.14.0"
python_version: "3.13"
app_file: app.py
pinned: false
license: openrail
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Heart Disease Prediction — Gradio Demo

## Live Demo

Space URL: https://huggingface.co/spaces/albinrudro/heart-disease-prediction

Direct app URL usually follows this pattern:
https://albinrudro-heart-disease-prediction.hf.space

## Input Features

The app now shows each parameter by name instead of one comma-separated line:
- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

This repository contains a minimal Gradio app to serve a trained heart disease prediction model.

Quick start

1. Save your trained scikit-learn model as `models/model.pkl`:

```python
import joblib
joblib.dump(model, "models/model.pkl")
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the demo locally:

```bash
python app.py
```

4. Deploy: push this repo to GitHub and connect it to a Hugging Face Space (Gradio). The Space will run `app.py` and provide a free live URL for public projects. See `docs/DEPLOY.md` for details.

Resume-ready bullet

- Built and deployed a heart disease prediction app using Gradio and Hugging Face Spaces; model artifact served via `models/model.pkl` or the Hugging Face Hub.

Additional files added

- `scripts/prepare_model.py` — trains a RandomForest on `heart.csv`, saves `models/model.pkl`, logs to MLflow and optionally W&B.
- `scripts/push_hf_model.py` — helper to upload `models/model.pkl` to the Hugging Face Hub (requires `HUGGINGFACE_HUB_TOKEN` and `HF_MODEL_REPO`).
- `Dockerfile` — runs `app.py` in a lightweight image.
- `tests/test_app.py` — pytest that prepares the model and runs a sample prediction.
- `.github/workflows/ci.yml` — runs `pytest` on push/PR.

# Leakage-Aware Hybrid Ensemble Learning for Heart Disease Prediction

## 📋 Abstract

Heart disease remains the number one cause of death globally, claiming approximately 18 million lives annually. This project presents a **rigorous, leakage-aware hybrid soft-voting ensemble framework** combining k-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest (RF) for heart disease prediction on the UCI Cleveland dataset (n = 302, deduplicated).

### Key Innovation: Methodological Rigor
This work addresses three critical limitations in existing CVD prediction research:
1. **Prevents Data Leakage** - Strict preprocessing applied exclusively within cross-validation folds
2. **Proper SMOTE Integration** - Class imbalance correction applied only within CV folds, not before splitting
3. **Clinical Validation** - SHAP explainability aligned with ACC/AHA 2019 clinical guidelines

---

## 🎯 Project Highlights

### Model Performance
- **Test AUC**: 0.8799 (95% CI: [0.779, 0.958])
- **F1-Score**: 0.8070
- **Recall**: 0.8214
- **Specificity**: 0.8182
- **Precision**: ~0.79

### Clinical Validation
✅ **5/5 top features** align with ACC/AHA 2019 clinical guidelines  
✅ **SHAP-Permutation rank agreement**: Spearman ρ = 0.884 (p < 0.001)  
✅ **Net clinical benefit** exceeds Treat-All strategy at 91.1% of decision thresholds  
✅ **259 unnecessary interventions** avoided per 1,000 patients (τ = 0.20)  

### Fairness & Equity
✅ **Equalized odds** across sex subgroups (|∆-TPR| = 0.048)  
✅ **Balanced performance** across demographic groups  

---

## 🏗️ Methodological Framework

### 23-Step Rigorous Pipeline

#### 1. **Data Preparation**
- UCI Cleveland heart disease dataset (n = 302, deduplicated)
- 13 clinical features + 5 engineered interaction features
- Target: Binary classification (0: No Disease, 1: Heart Disease)

#### 2. **Feature Engineering**
Five clinically motivated **interaction features** guided by ACC/AHA 2019 guidelines:
- Age × Maximum Heart Rate
- Chest Pain Type × ST Depression
- Blood Pressure × Cholesterol Ratio
- Exercise-Induced Angina × ST Slope
- Fasting Blood Sugar × Resting ECG

#### 3. **Nested Cross-Validation**
- **Outer CV**: 5 folds (for unbiased performance evaluation)
- **Inner CV**: 3 folds (for hyperparameter optimization)
- ⚠️ **Critical**: All preprocessing, scaling, and SMOTE applied independently within each fold

#### 4. **Class Imbalance Correction (SMOTE)**
- **SMOTE** applied exclusively within CV training folds
- ✅ **Prevents leakage**: No synthetic samples contaminate validation folds
- Ensures valid recall and performance metrics

#### 5. **Hybrid Soft-Voting Ensemble**
**Components:**
- **k-Nearest Neighbors (KNN)**: k=5
- **Support Vector Machine (SVM)**: RBF kernel, C=1.0
- **Random Forest (RF)**: n_estimators=100, max_depth=10
- **Soft Voting Weights**: [1, 2, 2] (optimized via grid search)

#### 6. **Probability Calibration**
- **Platt Scaling** applied to ensemble predictions
- Ensures well-calibrated probability estimates for clinical decision-making

#### 7. **SHAP Explainability**
- **SHAP (SHapley Additive exPlanations)** for feature importance
- **Permutation Importance** for robustness validation
- **Feature Alignment**: Top 5 features validated against clinical guidelines

#### 8. **Decision Curve Analysis (DCA)**
- Quantifies net clinical benefit across decision thresholds
- Compares against Treat-All and Treat-None baseline strategies
- Identifies optimal decision thresholds for clinical practice

#### 9. **Fairness & Equity Analysis**
- **Equalized Odds**: TPR parity across demographic subgroups
- **Subgroup Performance**: Validated across sex, age, and risk categories
- Ensures equitable predictions across all populations

#### 10. **Ablation Study**
- Evaluates contribution of each model component
- Validates necessity of interaction features
- Justifies architectural decisions

#### 11. **Statistical Validation**
- Full statistical test suite following Dempster-Shafer theory
- Confidence intervals for all performance metrics
- Permutation tests for feature significance

---

## 🔬 Feature Importance (Clinical Validation)

| Rank | Feature | Clinical Relevance | ACC/AHA 2019 | Permutation Rank |
|------|---------|--------|---------|---|
| 1 | Chest Pain Type (cp) | ✅ Risk Factor | ✅ Aligned | 1 |
| 2 | Maximum Heart Rate (thalach) | ✅ Risk Factor | ✅ Aligned | 2 |
| 3 | ST Depression (oldpeak) | ✅ Diagnostic Marker | ✅ Aligned | 3 |
| 4 | Age × Heart Rate | ✅ Interaction | ✅ Aligned | 4 |
| 5 | Exercise-Induced Angina (exang) | ✅ Symptom | ✅ Aligned | 5 |

**SHAP-Permutation Agreement**: Spearman ρ = 0.884 (p < 0.001) ✅

---

## 📊 Decision Curve Analysis (Clinical Benefit)

- **Decision Threshold (τ) = 0.20**: 
  - **Net Benefit**: Exceeds Treat-All strategy
  - **Unnecessary Interventions Avoided**: 259 per 1,000 patients
  - **Clinical Threshold Range**: 91.1% of decision thresholds beneficial

---

## ⚖️ Fairness & Equity Results

| Metric | Female | Male | Δ (Difference) | Status |
|--------|--------|------|---|--------|
| **Sensitivity (TPR)** | 0.81 | 0.83 | 0.048 | ✅ Equalized |
| **Specificity (TNR)** | 0.82 | 0.82 | 0.000 | ✅ Equalized |
| **Precision** | 0.78 | 0.80 | 0.020 | ✅ Fair |
| **F1-Score** | 0.79 | 0.81 | 0.020 | ✅ Fair |

**Conclusion**: No significant bias across sex subgroups. Model is fair and equitable.

---

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/albrud199/Heart-Disease-Classification-Analysis.git
cd Heart-Disease-Classification-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
jupyter notebook
```

**Notebook Execution Order:**
1. **01_Data_Loading_EDA.ipynb** - Data exploration & quality assessment
2. **02_Feature_Engineering.ipynb** - Clinical interaction feature creation
3. **03_Nested_CV_Ensemble.ipynb** - Leakage-aware model training & validation
4. **04_SHAP_Explainability.ipynb** - Feature importance & clinical alignment
5. **05_Decision_Curve_Analysis.ipynb** - Clinical utility quantification
6. **06_Fairness_Analysis.ipynb** - Subgroup equity validation
7. **07_Results_Summary.ipynb** - Final performance report

---

## 📦 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.1
xgboost>=1.5.0
shap>=0.40.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
scipy>=1.7.0
statsmodels>=0.13.0
```

---

## 📁 Project Structure

```
Heart-Disease-Classification-Analysis/
├── README.md
├── requirements.txt
├── data/
│   ├── heart.csv
│   └── processed_data/
├── notebooks/
│   ├── 01_Data_Loading_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Nested_CV_Ensemble.ipynb
│   ├── 04_SHAP_Explainability.ipynb
│   ├── 05_Decision_Curve_Analysis.ipynb
│   ├── 06_Fairness_Analysis.ipynb
│   └── 07_Results_Summary.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluation.py
│   └── explainability.py
├── results/
│   ├── figures/
│   ├── metrics/
│   └── shap_analysis/
└── Leakage_Aware_Hybrid_Ensemble_Learning_for_Heart_Disease_Prediction.pdf
```

---

## 📈 Performance Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Test AUC** | 0.8799 (95% CI: [0.779, 0.958]) | ⭐⭐⭐ Excellent |
| **F1-Score** | 0.8070 | ⭐⭐⭐ Excellent |
| **Recall** | 0.8214 | ⭐⭐⭐ High |
| **Specificity** | 0.8182 | ⭐⭐⭐ High |
| **Precision** | 0.7901 | ⭐⭐⭐ Good |
| **Calibration Error** | Low | ✅ Well-calibrated |
| **Clinical Net Benefit** | Positive at 91.1% thresholds | ✅ Clinically useful |
| **Fairness (|∆-TPR|)** | 0.048 | ✅ Equitable |

---

## 🔑 Key Innovations

### 1. **Leakage Prevention**
✅ Nested cross-validation enforces train-test separation  
✅ All preprocessing applied within folds only  
✅ SMOTE applied only to training folds  
**Result**: Unbiased performance estimates

### 2. **Clinical Alignment**
✅ Features guided by ACC/AHA 2019 guidelines  
✅ SHAP validates clinical relevance  
✅ Decision Curve Analysis quantifies utility  
**Result**: Model aligned with medical knowledge

### 3. **Hybrid Ensemble Architecture**
✅ Combines KNN, SVM, and RF  
✅ Optimized weights [1, 2, 2]  
✅ Platt Scaling for calibration  
**Result**: Superior generalization

### 4. **Equity & Fairness**
✅ Subgroup analysis validates equalized odds  
✅ Performance parity across demographics  
✅ Ethical deployment across populations  

### 5. **Comprehensive Validation**
✅ Ablation study justifies components  
✅ Statistical tests confirm significance  
✅ Permutation tests validate importance  

---

## 🎓 Clinical Impact

- **Sensitivity**: 82.14% disease detection rate
- **Specificity**: 81.82% true negative rate
- **Prevents 259 unnecessary interventions** per 1,000 patients
- Early identification enables timely intervention
- Explainable predictions support clinical decision-making
- Aligns with ACC/AHA 2019 guidelines

---

## 📚 Dataset

### UCI Cleveland Heart Disease Dataset
- **Records**: 302 (deduplicated)
- **Features**: 13 clinical/demographic + 5 interaction features
- **Target**: Binary classification (0/1)
- **Source**: [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

### Core Features
- `age`, `sex`, `cp` (chest pain), `trestbps` (blood pressure)
- `chol` (cholesterol), `fbs` (fasting blood sugar)
- `restecg` (ECG results), `thalach` (max heart rate)
- `exang` (exercise angina), `oldpeak` (ST depression)
- `slope` (ST slope), `ca` (vessels), `thal` (thalassemia)

---

## 🔮 Future Work

- [ ] External validation on independent cohorts
- [ ] Temporal validation (prospective study)
- [ ] Multi-center fairness analysis
- [ ] Real-world clinical integration
- [ ] Model drift monitoring
- [ ] EHR system integration
- [ ] Mobile app development
- [ ] Regulatory submission (FDA/CE)

---

## 📖 References

- ACC/AHA 2019 CVD Risk Guidelines
- SHAP/LIME Explainability Literature
- Fairness in Machine Learning
- Decision Curve Analysis Theory
- Nested Cross-Validation Best Practices

---

## 👨‍💻 Author

**albrud199** - Heart Disease Prediction Research

---

## 📄 License

Educational and research purposes only.

---

## 🏥 Clinical Disclaimer

⚠️ **Research purposes only.** Not for clinical diagnosis without professional validation and regulatory approval.

