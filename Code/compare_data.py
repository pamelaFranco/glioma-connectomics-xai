import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer

# =============================================================================
# 1. LOAD AND ALIGN DATA STRICLY BY CLEANED INTEGER IDENTIFIERS
# =============================================================================
df_connectomics = pd.read_csv('dataset_conectomicas_with_labels.csv')
df_patients = pd.read_excel('Patients_Project - Copy.xlsx')

# Clean Patient file IDs: extract numbers only (e.g., 'P01' -> 1)
df_patients['clean_id'] = df_patients['id'].astype(str).str.extract(r'(\d+)').astype(float)
df_connectomics['clean_id'] = df_connectomics['Patient_ID'].astype(str).str.extract(r'(\d+)').astype(float)

# Drop rows with no valid ID mapping
df_patients = df_patients.dropna(subset=['clean_id'])
df_connectomics = df_connectomics.dropna(subset=['clean_id'])

# Merge datasets using the synchronized numeric identifiers
df = pd.merge(df_connectomics, df_patients, on='clean_id', how='inner')

# Safely drop rows where target 'grado' is missing 
df = df.dropna(subset=['grado'])

print(f"Successfully matched and merged {len(df)} patient records.")

# Define binary target 'y' matching your ML_pipeline logic (High-Grade vs Low-Grade)
y = df['grado'].apply(lambda x: 1 if x > 2 else 0).values

# =============================================================================
# 2. FEATURE ENGINEERING & CATEGORICAL ENCODING (One-Hot Encoding)
# =============================================================================
# Administrative metadata and targets to exclude from the features matrices
metadata_cols = ['id', 'Patient_ID', 'clean_id', 'grado', 'histologia', 'area_tumor']
clinical_cols = ['edad', 'sexo', 'hemisferio']

# Scenario A: Clinical baseline features dataset
X_clinical_raw = df[clinical_cols].copy()
X_clinical = pd.get_dummies(X_clinical_raw, columns=['sexo', 'hemisferio'], drop_first=True).astype(float)

# Scenario B: Pure Structural Connectomics dataset (Drop all clinical and metadata info)
X_connectomics = df.drop(columns=metadata_cols + clinical_cols, errors='ignore').astype(float)

# Scenario C: Combined integrated dataset (Connectomics + Clinical features)
X_all_raw = df.drop(columns=metadata_cols, errors='ignore')
X_all = pd.get_dummies(X_all_raw, columns=['sexo', 'hemisferio'], drop_first=True).astype(float)

# =============================================================================
# 3. CROSS-VALIDATION SETTINGS
# =============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()
imputer = SimpleImputer(strategy='median')

# Standard Logistic Regression Models for each scenario
model_clinical = LogisticRegression(max_iter=1000, random_state=42)
model_connectomics = LogisticRegression(max_iter=1000, random_state=42)
model_all = LogisticRegression(max_iter=1000, random_state=42)

# =============================================================================
# 4. ROC CURVE GENERATION AND EVALUATION WITH PUBLICATION FORMATTING
# =============================================================================
# Reset mathtext settings to default to avoid rendering crashes
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

plt.figure(figsize=(7.5, 6))

def compute_and_plot_roc(model, X_data, y_true, label, color):
    tprs = []
    base_fpr = np.linspace(0, 1, 100)
    
    # Evaluate across cross-validation folds to avoid overfitting metrics
    for train_idx, test_idx in cv.split(X_data, y_true):
        X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
        y_train, y_test = y_true[train_idx], y_true[test_idx]
        
        # 1. Fit imputer on training fold and transform both sets
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # 2. Scale features based ONLY on training fold data to prevent data leakage
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # Train model and predict probabilities
        model.fit(X_train_scaled, y_train)
        y_probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate ROC metrics
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        tprs.append(np.interp(base_fpr, fpr, tpr))
        
    mean_tprs = np.mean(tprs, axis=0)
    mean_auc = auc(base_fpr, mean_tprs)
    
    plt.plot(base_fpr, mean_tprs, color=color, 
             label=f'{label} (Mean AUC = {mean_auc:.2f})', lw=2)

# Plot the three evaluation scenarios
compute_and_plot_roc(model_clinical, X_clinical, y, 'Clinical Baseline (Age, Sex, Hemisphere)', '#E63946')
compute_and_plot_roc(model_connectomics, X_connectomics, y, 'Pure Structural Connectomics', '#4EA8DE')
compute_and_plot_roc(model_all, X_all, y, 'Combined (Clinical + Connectomics)', '#1D3557')

# Plot visual reference line for random chance classification
plt.plot([0, 1], [0, 1], linestyle='--', color='#8D99AE', label='Random Guessing (AUC = 0.50)')

# Graph formatting matching clean scientific publication standards
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Save the final figure for your report
plt.savefig('supplementary_analysis_roc.png', dpi=300)
plt.show()