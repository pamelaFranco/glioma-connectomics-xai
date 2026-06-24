import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer

# =============================================================================
# 1. SETUP PATHS AND AUTOMATIC DATA ALIGNMENT
# =============================================================================
working_dir = r"C:\Users\pfran\Desktop\Connectomic Github\Dataset"

# File paths
combined_path = os.path.join(working_dir, "combined_patient_data.csv")
patients_excel_path = os.path.join(working_dir, "Patients_Project - Copy.xlsx")
radiomics_path = os.path.join(working_dir, "radiomics_with_classes_cleaned.csv")

print("Loading datasets...")
df_features = pd.read_csv(combined_path)
df_patients = pd.read_excel(patients_excel_path)
df_radiomics = pd.read_csv(radiomics_path)

# Normalize clinical column names to lowercase to avoid casing mismatches
df_patients.columns = df_patients.columns.str.lower()

# Extract and synchronize clean numeric IDs
df_features['clean_id'] = df_features['Patient_ID'].astype(str).str.extract(r'(\d+)').astype(float)
df_patients['clean_id'] = df_patients['id'].astype(str).str.extract(r'(\d+)').astype(float)
df_radiomics['clean_id'] = df_radiomics['Subjects'].astype(str).str.extract(r'(\d+)').astype(float)

# Drop entries missing the ID mapping key
df_features = df_features.dropna(subset=['clean_id'])
df_patients = df_patients.dropna(subset=['clean_id'])
df_radiomics = df_radiomics.dropna(subset=['clean_id'])

# Extract exact Tumor Volume from the raw radiomics file
volume_cols = [col for col in df_radiomics.columns if 'volume' in col.lower() or 'mesh' in col.lower()]
selected_vol_col = volume_cols[0]
print(f"-> Extracting volume column from radiomics: '{selected_vol_col}'")
df_vol = df_radiomics[['clean_id', selected_vol_col]].rename(columns={selected_vol_col: 'volumen'})

# Drop metadata from df_features that could cause naming collisions before merge
cols_to_drop_pre_merge = [c for c in ['edad', 'sexo', 'hemisferio', 'grado', 'histologia'] if c in df_features.columns]
df_features_clean = df_features.drop(columns=cols_to_drop_pre_merge, errors='ignore')

# Merge datasets sequentially using the clean numeric identifier
df = pd.merge(df_features_clean, df_patients, on='clean_id', how='inner')
df = pd.merge(df, df_vol, on='clean_id', how='inner')

# Safely drop rows where target 'grado' is missing 
df = df.dropna(subset=['grado'])
print(f"Successfully loaded and synchronized {len(df)} patient records with Volume tracking.")

# Define binary target 'y' (High-Grade vs Low-Grade)
y = df['grado'].apply(lambda x: 1 if x > 2 else 0).values

# =============================================================================
# 2. FEATURE ENGINEERING & CATEGORICAL ENCODING
# =============================================================================
# Set target variables to preserve or isolate
clinical_cols = ['edad', 'sexo', 'hemisferio', 'volumen']
metadata_cols = ['id', 'Patient_ID', 'clean_id', 'grado', 'histologia', 'area_tumor', 'Subjects']

# Scenario A: Clinical baseline features dataset (Control Model: Now includes Volume)
X_clinical_raw = df[clinical_cols].copy()
X_clinical = pd.get_dummies(X_clinical_raw, columns=['sexo', 'hemisferio'], drop_first=True).astype(float)

# Scenario B: Pure Structural Connectomics dataset (Excludes baseline clinical and size properties)
all_dropped_cols = metadata_cols + clinical_cols
X_connectomics = df.drop(columns=all_dropped_cols, errors='ignore').astype(float)
# Keep only numeric features from the network processing matrices
X_connectomics = X_connectomics.select_dtypes(include=[np.number])

# Scenario C: Combined integrated dataset (Connectomics + Clinical features + Volume)
X_all_raw = df.drop(columns=metadata_cols, errors='ignore')
X_all = pd.get_dummies(X_all_raw, columns=['sexo', 'hemisferio'], drop_first=True).astype(float)
X_all = X_all.select_dtypes(include=[np.number])

# Ensure exact dimensional structural matching across samples
assert len(X_clinical) == len(y), "Dimension mismatch in Clinical matrices"
assert len(X_connectomics) == len(y), "Dimension mismatch in Connectomics matrices"
assert len(X_all) == len(y), "Dimension mismatch in Combined matrices"

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
# Publication quality font render setup via LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

plt.figure(figsize=(8, 6.5))

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
compute_and_plot_roc(model_clinical, X_clinical, y, 'Baseline Profile (Age, Sex, Hemisphere, Volume)', '#E63946')
compute_and_plot_roc(model_connectomics, X_connectomics, y, 'Pure Structural Connectomics', '#4EA8DE')
compute_and_plot_roc(model_all, X_all, y, 'Combined Analysis (Baseline + Connectomics)', '#1D3557')

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

# Save the final figure back to your folder
output_fig_path = os.path.join(working_dir, 'supplementary_analysis_roc_with_volume.png')
plt.savefig(output_fig_path, dpi=300)
print(f"Analysis complete. Figure saved to: {output_fig_path}")
plt.show()