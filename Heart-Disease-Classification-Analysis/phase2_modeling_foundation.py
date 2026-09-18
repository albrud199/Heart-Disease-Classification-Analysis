# Phase 2: Modeling foundation: preprocessing, feature engineering, baselines, CV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')

# Create directories for outputs
os.makedirs('models', exist_ok=True)
os.makedirs('eda', exist_ok=True)  # for any additional plots

print('Phase 2: Modeling foundation')
print('='*50)

# Load the splits
print('\n1. Loading train, validation, and test splits...')
train_df = pd.read_csv('splits/train.csv')
val_df = pd.read_csv('splits/val.csv')
test_df = pd.read_csv('splits/test.csv')

print(f'Train shape: {train_df.shape}')
print(f'Validation shape: {val_df.shape}')
print(f'Test shape: {test_df.shape}')

# Separate features and target
target_col = 'target'
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_val = val_df.drop(columns=[target_col])
y_val = val_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

print(f'\nFeature shapes:')
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_val: {X_val.shape}, y_val: {y_val.shape}')
print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')

# Identify feature types
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Since all columns are int64 or float64, we need to decide which are categorical
# Based on earlier analysis: sex, cp, fbs, restecg, exang, slope, thal are categorical (though encoded as int)
# We'll define categorical columns based on domain knowledge
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
# Ensure they exist in the dataframe
categorical_cols = [col for col in categorical_cols if col in X_train.columns]
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

print(f'\nNumeric features ({len(numeric_cols)}): {numeric_cols}')
print(f'Categorical features ({len(categorical_cols)}): {categorical_cols}')

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # though no missing values
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Fit preprocessor on training data
print('\n2. Fitting preprocessor on training data...')
preprocessor.fit(X_train)

# Transform the data
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print(f'Processed feature shapes:')
print(f'X_train_processed: {X_train_processed.shape}')
print(f'X_val_processed: {X_val_processed.shape}')
print(f'X_test_processed: {X_test_processed.shape}')

# Feature engineering: Polynomial features for numeric components?
# We'll add polynomial features (degree=2) for the numeric part after scaling?
# But note: after one-hot encoding, we have many features. We'll apply polynomial only to the original numeric features.
# We'll create a separate pipeline for polynomial features.
from sklearn.preprocessing import PolynomialFeatures

# Create a transformer that applies polynomial features to numeric features
poly = PolynomialFeatures(degree=2, include_bias=False)
# We'll fit on the numeric training data (after scaling? Usually polynomial features are applied before scaling? 
# Common practice: scale then polynomial? Actually, polynomial features can be sensitive to scale, so we should scale first.
# But we already scaled in the numeric_transformer. However, the ColumnTransformer outputs the scaled numeric features.
# We'll extract the numeric part after preprocessing? That's messy.
# Instead, let's create a preprocessing pipeline that includes polynomial features as an option.
# For simplicity, we'll skip polynomial features for now and just use the original features.
# We can add them later if needed.
print('\n3. Skipping polynomial feature engineering for baseline models.')

# We'll use the processed data as is for baselines.
X_train_final = X_train_processed
X_val_final = X_val_processed
X_test_final = X_test_processed

# Save the preprocessor and processed data for later use
print('\n3.5. Saving preprocessor and processed data...')
preprocessor_path = 'models/preprocessor.pkl'
joblib.dump(preprocessor, preprocessor_path)
print(f'Preprocessor saved to {preprocessor_path}')

# Optionally save the processed data
train_processed_df = pd.DataFrame(X_train_processed, index=X_train.index)
train_processed_df[target_col] = y_train.values
train_processed_df.to_csv('splits/train_processed.csv', index=False)

val_processed_df = pd.DataFrame(X_val_processed, index=X_val.index)
val_processed_df[target_col] = y_val.values
val_processed_df.to_csv('splits/val_processed.csv', index=False)

test_processed_df = pd.DataFrame(X_test_processed, index=X_test.index)
test_processed_df[target_col] = y_test.values
test_processed_df.to_csv('splits/test_processed.csv', index=False)

print('Processed data saved to splits/ directory.')

# Baseline models
print('\n4. Training baseline models...')
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

# Train and evaluate each model
results = []
for name, model in models.items():
    print(f'\nTraining {name}...')
    model.fit(X_train_final, y_train)
    
    # Predict on validation set
    y_val_pred = model.predict(X_val_final)
    y_val_pred_proba = model.predict_proba(X_val_final)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_pred_proba) if y_val_pred_proba is not None else None
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC_AUC': roc_auc
    })
    
    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation Precision: {precision:.4f}')
    print(f'Validation Recall: {recall:.4f}')
    print(f'Validation F1: {f1:.4f}')
    if roc_auc:
        print(f'Validation ROC-AUC: {roc_auc:.4f}')
    
    # Save the model
    model_path = f'models/{name.replace(' ', '_').lower()}_baseline.pkl'
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')

# Cross-validation on training set
print('\n5. Performing cross-validation on training set...')
cv_results = []
for name, model in models.items():
    print(f'\nCross-validating {name}...')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_final, y_train, cv=cv, scoring='accuracy')
    cv_results.append({
        'Model': name,
        'CV Accuracy Mean': cv_scores.mean(),
        'CV Accuracy Std': cv_scores.std()
    })
    print(f'CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})')

# Save results to a CSV file
results_df = pd.DataFrame(results)
cv_df = pd.DataFrame(cv_results)

results_df.to_csv('models/baseline_validation_results.csv', index=False)
cv_df.to_csv('models/baseline_cv_results.csv', index=False)

print('\n6. Baseline results saved to models/ directory.')
print('\nPhase 2 completed successfully!')