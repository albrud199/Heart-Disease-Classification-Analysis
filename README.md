<!--
Implemented an end-to-end heart disease prediction system using Logistic Regression with automated data loading, EDA, and hyperparameter optimization. Achieved 85% accuracy on test data with strong F1-scores (~0.87), confirming chest pain type and maximum heart rate as dominant clinical indicators.
-->

##  Heart Disease Classification using Logistic Regression

This project implements a **Logistic Regression** model to predict the presence of heart disease using clinical features from the [Heart Disease UCI dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). The pipeline includes automatic data loading, exploratory data analysis (EDA), model training, hyperparameter tuning, and performance evaluation—all in a single, reproducible script.

---

## 📊 Dataset Overview

- **Source**: [Kaggle – Heart Disease Dataset by johnsmith88](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **File**: `heart.csv`
- **Initial Records**: 303  
- **Target Distribution**:
  - `1` (Heart Disease): 165 patients  
  - `0` (No Disease): 137 patients  
- **Features (13)**:
  - `age`, `sex`, `cp` (chest pain type)
  - `trestbps` (resting blood pressure)
  - `chol` (serum cholesterol)
  - `fbs` (fasting blood sugar > 120 mg/dl)
  - `restecg` (resting ECG results)
  - `thalach` (maximum heart rate achieved)
  - `exang` (exercise-induced angina)
  - `oldpeak` (ST depression induced by exercise)
  - `slope`, `ca`, `thal`

✅ **Data Quality**:  
- **No missing values**  
- **Balanced enough** for binary classification  
- Clean and ready for modeling

---

## 🔍 Exploratory Data Analysis (EDA)

The following visualizations were generated:

1. **Correlation Heatmap**  
   - Strong **positive correlation** between `cp` (chest pain) and `target`  
   - Strong **negative correlation** between `exang`, `oldpeak`, and `target`

2. **Age vs. Max Heart Rate (`thalach`)**  
   - Clear **inverse relationship**: as age increases, max heart rate tends to decrease  
   - Confirmed via regression and line plots

3. **Age vs. Cholesterol (`chol`)**  
   - No significant linear trend observed

4. **Age Distribution by Sex**  
   - Boxplot comparing age ranges for males (`sex=1`) and females (`sex=0`)

5. **Confusion Matrix**  
   - Visualized test set performance with true/false positives/negatives

---

## 🤖 Model Development

### Baseline Logistic Regression
- **Algorithm**: `LogisticRegression(max_iter=1000)`
- **Train/Test Split**: 80/20, stratified by `target`
- **Results**:
  - **Training Accuracy**: ~87%
  - **Test Accuracy**: ~85%
  - Minimal overfitting → good generalization

### Classification Report (Test Set)
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0     | ~0.82     | ~0.83  | ~0.82    |
| 1     | ~0.88     | ~0.87  | ~0.87    |

### Feature Importance (by |coefficient|)
Top predictive features:
1. `cp` (chest pain type)  
2. `thalach` (max heart rate)  
3. `exang` (exercise-induced angina)  
4. `oldpeak` (ST depression)

> These align with clinical understanding of heart disease risk factors.

---

## ⚙️ Hyperparameter Tuning

Used **`GridSearchCV`** (5-fold cross-validation) to optimize:
- Regularization strength `C`: `[0.001, 0.01, 0.1, 1, 10, 100]`
- Solver: `['liblinear', 'lbfgs']`
- Increased `max_iter=5000` to ensure convergence

### Best Parameters
```python
{'C': 1, 'solver': 'liblinear'}
