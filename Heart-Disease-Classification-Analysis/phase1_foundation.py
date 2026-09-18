import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Create directories for outputs
os.makedirs('eda', exist_ok=True)
os.makedirs('splits', exist_ok=True)

# Step 1: Load data
print("Step 1: Loading data...")
df = pd.read_csv('heart.csv')
print(f"Data shape: {df.shape}")

# Step 2: Data audit
print("\nStep 2: Data audit...")
print("\nColumns and data types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nBasic statistics:")
print(df.describe(include='all'))

# Step 3: Target variable analysis
print("\nStep 3: Target variable analysis...")
target_col = 'target'  # Assuming the target column is named 'target'
if target_col not in df.columns:
    # Try to find target column
    possible_targets = ['target', 'heart_disease', 'diagnosis', 'class']
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    else:
        target_col = df.columns[-1]  # Default to last column
        print(f"No obvious target column found, using '{target_col}' as target")

print(f"Target column: '{target_col}'")
print(f"Target distribution:\n{df[target_col].value_counts()}")
print(f"Target distribution (%):\n{df[target_col].value_counts(normalize=True) * 100}")

# Step 4: EDA - Univariate analysis
print("\nStep 4: EDA - Univariate analysis...")
# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify feature types
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# Plot numeric features distribution
if numeric_cols:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(4, 4, i)
        sns.histplot(data=X, x=col, kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('eda/numeric_features_distribution.png')
    plt.close()

# Plot categorical features distribution
if categorical_cols:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(3, 3, i)
        sns.countplot(data=X, x=col)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda/categorical_features_distribution.png')
    plt.close()

# Step 5: EDA - Bivariate analysis with target
print("\nStep 5: EDA - Bivariate analysis with target...")
# Numeric features vs target
if numeric_cols:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(4, 4, i)
        sns.boxplot(data=df, x=target_col, y=col)
        plt.title(f'{col} vs {target_col}')
    plt.tight_layout()
    plt.savefig('eda/numeric_vs_target.png')
    plt.close()

# Categorical features vs target
if categorical_cols:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(3, 3, i)
        sns.countplot(data=df, x=col, hue=target_col)
        plt.title(f'{col} vs {target_col}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda/categorical_vs_target.png')
    plt.close()

# Step 6: Correlation analysis
print("\nStep 6: Correlation analysis...")
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('eda/correlation_matrix.png')
plt.close()

# Check for high correlation with target (potential leakage indicators)
target_correlations = correlation_matrix[target_col].abs().sort_values(ascending=False)
print(f"\nTop 5 features correlated with target:\n{target_correlations.head()}")

# Step 7: Data leakage check
print("\nStep 7: Data leakage check...")
# Check for any feature that is almost perfectly correlated with target (could be leakage)
high_corr_features = target_correlations[target_correlations > 0.9].index.tolist()
high_corr_features = [f for f in high_corr_features if f != target_col]
if high_corr_features:
    print(f"WARNING: Features with >0.9 correlation with target (potential leakage): {high_corr_features}")
else:
    print("No features with >0.9 correlation with target found.")

# Check for identifier-like columns
id_like_cols = [col for col in df.columns if 'id' in col.lower() or 'index' in col.lower()]
if id_like_cols:
    print(f"Identifier-like columns found: {id_like_cols}")
    print("Consider removing these from features if they are not predictive.")
else:
    print("No obvious identifier-like columns found.")

# Step 8: Train/Validation/Test split
print("\nStep 8: Train/Validation/Test split...")
# First split: train+val vs test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val  # 0.25 * 0.8 = 0.2
)

print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Validation set size: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

# Save splits
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv('splits/train.csv', index=False)
val_df.to_csv('splits/val.csv', index=False)
test_df.to_csv('splits/test.csv', index=False)

print("\nSplits saved to 'splits/' directory:")
print("- train.csv")
print("- val.csv")
print("- test.csv")

print("\nEDA plots saved to 'eda/' directory.")

print("\nPhase 1 (Steps 1-7) completed successfully!!")

