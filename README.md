# Heart Disease Classification Analysis

## Project Overview
This project aims to analyze heart disease data and develop predictive models using a Logistic Regression pipeline. The analysis walks through data preprocessing, exploratory data analysis (EDA), model training, evaluation, and systematic improvements.

---

## 📊 Dataset Overview

The project uses the **Heart Disease UCI Dataset** from Kaggle, containing clinical features for heart disease prediction.

- **Source**: [Kaggle – Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Records**: 303 patients
- **Features**: 13 clinical and demographic attributes
- **Target**: Binary classification (0: No Disease, 1: Heart Disease)

### Key Features
- `age` - Patient age
- `sex` - Gender (1=Male, 0=Female)
- `cp` - Chest pain type
- `trestbps` - Resting blood pressure
- `chol` - Serum cholesterol levels
- `fbs` - Fasting blood sugar > 120 mg/dl
- `restecg` - Resting electrocardiographic results
- `thalach` - Maximum heart rate achieved
- `exang` - Exercise-induced angina
- `oldpeak` - ST depression induced by exercise
- `slope` - ST segment slope
- `ca` - Number of major vessels colored by fluoroscopy
- `thal` - Thalassemia

---

## 🔍 Exploratory Data Analysis (EDA)

Comprehensive EDA is performed to understand the data distribution and relationships:

### Key Findings
1. **Data Quality**: No missing values; clean dataset ready for modeling
2. **Class Distribution**: Balanced classes (165 with disease, 137 without)
3. **Feature Relationships**:
   - Strong correlation between chest pain type (`cp`) and heart disease presence
   - Inverse relationship between age and maximum heart rate achieved
   - Exercise-induced angina (`exang`) is a strong predictor

### Visualizations Generated
- Correlation heatmap showing feature dependencies
- Age vs. maximum heart rate relationship
- Cholesterol distribution analysis
- Age distribution by gender
- Confusion matrices for model evaluation

---

## 🤖 Model Development

### Logistic Regression Baseline
- **Algorithm**: `LogisticRegression(max_iter=1000)`
- **Train/Test Split**: 80/20 (stratified by target)
- **Performance**:
  - Training Accuracy: ~87%
  - Test Accuracy: ~85%
  - Indicates good generalization with minimal overfitting

### Classification Metrics (Test Set)
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 (No Disease) | ~0.82 | ~0.83 | ~0.82 |
| 1 (Disease) | ~0.88 | ~0.87 | ~0.87 |

### Feature Importance
Top predictive features (by coefficient magnitude):
1. **Chest pain type (`cp`)** - Most influential factor
2. **Maximum heart rate (`thalach`)** - Strong predictor
3. **Exercise-induced angina (`exang`)** - Clinical significance
4. **ST depression (`oldpeak`)** - Important diagnostic marker

> These findings align with established clinical understanding of heart disease risk factors.

---

## ⚙️ Hyperparameter Tuning

Optimization performed using **GridSearchCV** with 5-fold cross-validation:

### Tuned Parameters
- Regularization strength `C`: [0.001, 0.01, 0.1, 1, 10, 100]
- Solver algorithm: ['liblinear', 'lbfgs']
- `max_iter`: 5000 (to ensure convergence)

### Best Configuration
```python
{'C': 1, 'solver': 'liblinear'}
```

---

## 📁 Project Structure

```
Heart-Disease-Classification-Analysis/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── heart.csv                          # Dataset file
└── notebooks/
    ├── 01_Data_Loading_EDA.ipynb     # Data exploration & analysis
    ├── 02_Model_Training.ipynb       # Baseline model development
    └── 03_Hyperparameter_Tuning.ipynb # Model optimization
```

---

## 💻 Technology Stack

### Languages Used
- **Jupyter Notebooks**: 95.7% - Primary analysis and modeling environment
- **Python**: 4.3% - Supporting utility scripts

### Key Libraries
```python
pandas          # Data manipulation and analysis
numpy           # Numerical computing
scikit-learn    # Machine learning library
matplotlib      # Data visualization
seaborn         # Statistical plotting
```

---

## 🚀 Getting Started

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/albrud199/Heart-Disease-Classification-Analysis.git
   cd Heart-Disease-Classification-Analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebooks**:
   ```bash
   jupyter notebook
   ```

### Running the Analysis
Execute notebooks in order:
1. `01_Data_Loading_EDA.ipynb` - Load and explore the dataset
2. `02_Model_Training.ipynb` - Train the baseline Logistic Regression model
3. `03_Hyperparameter_Tuning.ipynb` - Optimize model performance

---

## 📈 Results & Performance

✅ **Test Accuracy**: 85%  
✅ **Balanced Performance**: High precision and recall across both classes  
✅ **Good Generalization**: Minimal gap between training and test accuracy  
✅ **Clinical Alignment**: Key features match medical knowledge of heart disease  

---

## 🔮 Future Enhancements

- Implement advanced algorithms (Random Forest, Gradient Boosting, Neural Networks)
- Feature engineering with domain-specific transformations
- Advanced validation techniques (stratified k-fold, time series split)
- Model interpretability using SHAP values and LIME
- Deployment pipeline (FastAPI/Flask REST API)
- Performance monitoring and drift detection

---

## 📝 License

This project is provided for educational and research purposes.

---
