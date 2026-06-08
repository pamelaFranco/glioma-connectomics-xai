###############################################################################
# This code was developed by Dr. Pamela Franco as part of the project 
# "Feasibility of Tumor-Masked Structural Connectomics and Explainable Machine 
# Learning for Assessing White Matter Disruption in Gliomas: A Pilot Study". 
#
# Implements Strict Repeated/Nested Cross-Validation while
# preserving the Post-Hoc Data Leakage Analysis and Comparative ROC Curves.
#
#   Author:     Dr. Pamela Franco (Modified for Rigorous Nested CV & Leakage)
#   Time-stamp: 2026-06-03
#   Repository: https://github.com/pamelaFranco/glioma-ml-tractography
#   E-mail:     pamela.franco@unab.cl / pafranco@uc.cl
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import os 
import shap

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

###############################################################################
# CUSTOM DATA AND RESULTS PATH CONFIGURATION
DATA_PATH = r"C:\Users\pfran\Desktop\Connectomics Github\Dataset\dataset_conectomica_with_labels.csv"
RESULTS_PATH = r"C:\Users\pfran\Desktop\Connectomics Github\Results"

# Ensure results directory exists
os.makedirs(RESULTS_PATH, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print(f"Generating mock synthetic data at {DATA_PATH} to simulate dataset constraints (n=35, p=307)...")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    np.random.seed(42)
    features = [f"edge_{i}" for i in range(307)]
    df_mock = pd.DataFrame(np.random.randn(35, 307), columns=features)
    df_mock['edge_constant_1'] = 0.0
    df_mock['edge_constant_2'] = 1.5
    df_mock['target'] = np.random.choice([0, 1], size=35, p=[0.57, 0.43]) # approx 20 LGG, 15 HGG
    df_mock.to_csv(DATA_PATH, index=False)

###############################################################################
# 1. DATA LOADING AND CURATION
print(" Loading and sanitizing dataset...")
df = pd.read_csv(DATA_PATH)

# Sanitize column headers (remove special characters and spaces)
df.columns = [re.sub(r'[^\w\s]', '', col).strip().replace(' ', '_') for col in df.columns]

# Separate features and target labels
X = df.drop(columns=['target'])
y = df['target']

# Dynamic imputation of missing/infinite values using column-specific means
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

# Constant feature pruning (Zero-Variance Filter)
constant_filter = X.var() == 0
X_clean = X.loc[:, ~constant_filter]
print(f"Original features: {X.shape[1]} -> Remaining after constant pruning: {X_clean.shape[1]}")

###############################################################################
# 2. GLOBAL FEATURE SELECTION (For Biased Pipeline / Leakage Analysis)
print("\n Computing Global Feature Selection to simulate Data Leakage...")
scaler_global = StandardScaler()
X_scaled_global = pd.DataFrame(scaler_global.fit_transform(X_clean), columns=X_clean.columns)

# Global SFS over the entire dataset (This creates data leakage because it sees the test folds later)
global_sfs = SequentialFeatureSelector(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
    n_features_to_select=10,
    direction='forward',
    scoring='accuracy',
    cv=3,
    n_jobs=-1
)
global_sfs.fit(X_scaled_global, y)
global_chosen_features = X_clean.columns[global_sfs.get_support()].tolist()
print(f"Globally selected features (Biased): {global_chosen_features}")

###############################################################################
# 3. CROSS-VALIDATION ARCHITECTURE (BIASED VS. STRICTLY ISOLATED NESTED CV)
print("\n Initializing Parallel Validation Loops for Data Leakage Analysis...")

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Metrics storage
biased_f1_scores = []
unbiased_f1_scores = []

# Arrays for ROC curve tracking
biased_tprs = []
unbiased_tprs = []
mean_fpr = np.linspace(0, 1, 100)

# Track isolated architectures for Post-Hoc SHAP
best_isolated_estimators = []
isolated_features_per_fold = []

# Data structures for feature optimization curve (Fold 1)
max_features_to_evaluate = min(30, X_clean.shape[1])
fold1_feature_accuracies = [] 
fold1_ranked_features = []

print(" Launching Outer Cross-Validation Loop...")
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_clean, y)):
    X_train_fold, X_test_fold = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
    
    # Scale locally
    scaler = StandardScaler()
    X_train_fold_scaled = pd.DataFrame(scaler.fit_transform(X_train_fold), columns=X_clean.columns)
    X_test_fold_scaled = pd.DataFrame(scaler.transform(X_test_fold), columns=X_clean.columns)
    
    # --- PIPELINE A: BIASED EVALUATION (Uses globally selected features) ---
    X_train_biased = X_train_fold_scaled[global_chosen_features]
    X_test_biased = X_test_fold_scaled[global_chosen_features]
    
    clf_biased = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf_biased.fit(X_train_biased, y_train_fold)
    
    biased_preds = clf_biased.predict(X_test_biased)
    biased_probs = clf_biased.predict_proba(X_test_biased)[:, 1]
    biased_f1_scores.append(f1_score(y_test_fold, biased_preds, average='macro', zero_division=0))
    
    # Track Biased ROC
    fpr_b, tpr_b, _ = roc_curve(y_test_fold, biased_probs)
    biased_tprs.append(np.interp(mean_fpr, fpr_b, tpr_b))
    biased_tprs[-1][0] = 0.0

    # --- PIPELINE B: STRICTLY ISOLATED EVALUATION (Nested CV / No Leakage) ---
    # Extract intrinsic importance scores from the local RF classifier inside the Fold
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_selector.fit(X_train_fold_scaled, y_train_fold)
    
    importances = rf_selector.feature_importances_
    indices_ranking = np.argsort(importances)[::-1]
    ranked_features_fold = X_clean.columns[indices_ranking].tolist()
    
    # Leakage-free sequential validation loop
    feature_scores_cv = []
    for k in range(1, max_features_to_evaluate + 1):
        top_k_features = ranked_features_fold[:k]
        X_train_k = X_train_fold_scaled[top_k_features]
        
        score = np.mean(cross_val_score(
            RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
            X_train_k, y_train_fold, cv=inner_cv, scoring='accuracy', n_jobs=-1
        ))
        feature_scores_cv.append(score)
    
    # Save Fold 1 data metrics for subsequent plotting
    if fold == 0:
        fold1_feature_accuracies = feature_scores_cv
        fold1_ranked_features = ranked_features_fold[:max_features_to_evaluate]
    
    # Optimal subset selection based on the peak performance of the internal CV
    optimal_k = np.argmax(feature_scores_cv) + 1
    local_chosen_features = ranked_features_fold[:optimal_k]
    isolated_features_per_fold.append(local_chosen_features)
    
    X_train_unbiased = X_train_fold_scaled[local_chosen_features]
    X_test_unbiased = X_test_fold_scaled[local_chosen_features]
    
    # Isolated hyperparameter tuning via internal grid search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid=param_grid, cv=inner_cv, scoring='f1_macro', n_jobs=-1
    )
    grid_search.fit(X_train_unbiased, y_train_fold)
    clf_unbiased = grid_search.best_estimator_
    best_isolated_estimators.append(clf_unbiased)
    
    unbiased_preds = clf_unbiased.predict(X_test_unbiased)
    unbiased_probs = clf_unbiased.predict_proba(X_test_unbiased)[:, 1]
    unbiased_f1_scores.append(f1_score(y_test_fold, unbiased_preds, average='macro', zero_division=0))
    
    # Track Unbiased ROC
    fpr_u, tpr_u, _ = roc_curve(y_test_fold, unbiased_probs)
    unbiased_tprs.append(np.interp(mean_fpr, fpr_u, tpr_u))
    unbiased_tprs[-1][0] = 0.0
    
    print(f"    Fold {fold+1} Completed | Biased F1: {biased_f1_scores[-1]:.4f} | Isolated F1: {unbiased_f1_scores[-1]:.4f} | Features: {optimal_k}")


###############################################################################
# 4. POST-HOC DATA LEAKAGE ANALYSIS & STATISTICAL REPORT
print("\n=======================================================")
print("        POST-HOC DATA LEAKAGE ANALYSIS REPORT")
print("=======================================================")
mean_biased = np.mean(biased_f1_scores)
mean_unbiased = np.mean(unbiased_f1_scores)
leakage_delta = mean_biased - mean_unbiased

print(f"Mean Biased F1-Score (With Global Leakage): {mean_biased:.4f} +/- {np.std(biased_f1_scores):.4f}")
print(f"Mean Unbiased F1-Score (Strict Isolation):  {mean_unbiased:.4f} +/- {np.std(unbiased_f1_scores):.4f}")
print(f"Discrepancy Delta (Performance Inflation): {leakage_delta:.4f}")

if leakage_delta > 0.05:
    print(" WARNING: Significant data leakage detected! Global feature selection artificially inflates model performance.")
else:
    print(" NOTICE: Performance variations are within stable statistical margins.")

# 5. GENERATING COMPARATIVE ROC CURVES
print("\n Plotting Comparative ROC Curves (Biased vs. Isolated)...")
plt.figure(figsize=(8, 6))

mean_tpr_biased = np.mean(biased_tprs, axis=0)
mean_tpr_biased[-1] = 1.0
mean_auc_biased = auc(mean_fpr, mean_tpr_biased)

mean_tpr_unbiased = np.mean(unbiased_tprs, axis=0)
mean_tpr_unbiased[-1] = 1.0
mean_auc_unbiased = auc(mean_fpr, mean_tpr_unbiased)

plt.plot(mean_fpr, mean_tpr_biased, color='red', linestyle='--',
         label=r'Biased Pipeline (Mean AUC = %0.2f)' % mean_auc_biased, lw=2)
plt.plot(mean_fpr, mean_tpr_unbiased, color='blue', linestyle='-',
         label=r'Strictly Isolated Pipeline (Mean AUC = %0.2f)' % mean_auc_unbiased, lw=2)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance level (AUC = 0.50)')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Comparison')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'comparative_roc_curve.png'))
plt.close()


###############################################################################
# 5.B GENERATING FEATURE SELECTION ACCURACY CURVE (N of Features vs Accuracy)
print("\n Plotting Feature Selection Performance Curve (Fold 1)...")
plt.figure(figsize=(10, 5))

features_count = np.arange(1, len(fold1_feature_accuracies) + 1)
max_acc_idx = np.argmax(fold1_feature_accuracies)
optimal_num_features = features_count[max_acc_idx]
max_accuracy = fold1_feature_accuracies[max_acc_idx]

# Sequential validation performance curve
plt.plot(features_count, fold1_feature_accuracies, color='darkblue', marker='o', markersize=4, linestyle='-', alpha=0.7, label='Internal CV Accuracy')

# Red Dot on the global maximum score
plt.plot(optimal_num_features, max_accuracy, color='red', marker='o', markersize=9, linestyle='', label=f'Global Max (N={optimal_num_features}, Acc={max_accuracy:.3f})')

plt.xlabel('Number of Features Selected (by RF Importance Ranking)')
plt.ylabel('Validation Accuracy (Internal CV)')
plt.title('Feature Selection Process Optimization (Fold 1 Isolation)')
plt.xticks(features_count, rotation=45 if len(features_count) > 20 else 0)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'sfs_feature_accuracy_curve.png'))
plt.close()

print(f"Top {optimal_num_features} Features Ranked by RF in Fold 1: {fold1_ranked_features[:optimal_num_features]}")

###############################################################################
# 6. POST-HOC EXPLAINABILITY VIA TREE SHAP VALUES (Using Isolated Pipeline Fold 1)
print("\n Extracting game-theoretic feature attributions via TreeExplainer...")
target_model = best_isolated_estimators[0]
target_features = isolated_features_per_fold[0]

explainer = shap.TreeExplainer(target_model)
X_shap_input = X_scaled_global[target_features]
shap_values = explainer.shap_values(X_shap_input)

plt.figure(figsize=(10, 6))
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_shap_input, show=False)
else:
    shap.summary_plot(shap_values, X_shap_input, show=False)

plt.title('Structural Attributions Map within Isolated Parsimonious Sub-space (SHAP)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'shap_summary_pipeline.png'))
plt.close()

print(f"\n Updated Machine Learning Pipeline executed successfully. Outputs and plots exported to: {RESULTS_PATH}")