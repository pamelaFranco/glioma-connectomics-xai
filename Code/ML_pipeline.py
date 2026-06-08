###############################################################################
# This code was developed by Dr. Pamela Franco as part of the project 
# "Quantifying White Matter Disruption in Gliomas via Tumor-Masked Structural 
# Connectomics and Explainable Machine Learning: A Pilot Study ". 
#
# This code performs a comprehensive machine learning pipeline tailored for 
# CLASSIFICATION analysis on glioma tractography and radiomics data.
#
#   Author:     Dr. Pamela Franco
#   Time-stamp:  2026-05-25
#   Repository:  https://github.com/pamelaFranco/glioma-ml-tractography
#   E-mail:      pamela.franco@unab.cl / pafranco@uc.cl
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import os 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
import shap

# Plot settings - Scientific configuration for Matplotlib using LaTeX
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

###############################################################################
# Load data
path = r'C:\Users\pfran\Desktop\Connectomics Github\Dataset'  
data_filename = 'dataset_conectomica_with_labels.csv'
newData = os.path.join(path, data_filename)  

RESULTS_PATH = r'C:\Users\pfran\Desktop\Connectomics Github\Results'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# If the absolute path does not exist locally, search in the current working directory
if not os.path.exists(newData):
    newData = data_filename

# Reading the dataset
df_raw = pd.read_csv(newData)

# SAFE TARGET AND FEATURES EXTRACTION (ERROR FIXED) ---
# Extract the target array directly using the correct column name
label = df_raw['labels'].values

# Drop target and metadata safely using errors='ignore' to avoid KeyError exceptions
features = df_raw.drop(columns=['labels', 'Patient_ID'], errors='ignore')

# Encode labels to integers if they are imported as objects or strings
if label.dtype == 'object':
    label, _ = pd.factorize(label)

# 1. Clean feature names by removing special characters immediately
features = features.rename(columns=lambda x: re.sub('[^*A-Za-z0-9_ ]+', '', x))
feature_names = list(features.columns)

# 2. Check and Impute missing values upfront
if features.isnull().sum().any():
    print("Warning: Missing values detected upfront. Imputing with the column mean.")
    features = features.fillna(features.mean())

# 3. Handle infinite values just in case they exist in tractography metrics
if np.isinf(features).values.any():
    print("Warning: Infinite values detected. Replacing with NaNs and imputing.")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.mean())

# 4. Remove constant features (zero variance) to prevent NaNs or division by zero in selection
constant_features = [col for col in features.columns if features[col].std() == 0]
if constant_features:
    print(f"Warning: Detected {len(constant_features)} constant features. Removing them.")
    features = features.drop(columns=constant_features)
    feature_names = list(features.columns)


X_clean = features
y = pd.Series(label)

# Scale all features globally for algorithms like t-SNE / Logistic Regression
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

###############################################################################
# Hierarchical Clustering to visualize correlations between features
correlation_matrix = features_scaled.corr()

if correlation_matrix.isnull().any().any():
    correlation_matrix = correlation_matrix.fillna(0)

distance_threshold = 6.5 

model = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=distance_threshold)
cluster_cols = model.fit_predict(correlation_matrix.T)  
cluster_rows = model.fit_predict(correlation_matrix)  

n_clusters = len(np.unique(cluster_cols)) 

# CORRECCIÓN: Uso de la nueva sintaxis de colormaps de Matplotlib para evitar DeprecationWarning
colors = mpl.colormaps['plasma'].resampled(n_clusters)

col_colors = [colors(i) for i in cluster_cols]
row_colors = [colors(i) for i in cluster_rows]

cluster_features = [sum(np.array(cluster_cols) == i) for i in range(n_clusters)]
cluster_texts = [f"Cluster {i+1}: {cluster_features[i]:02d} features" for i in range(n_clusters)]

Z = linkage(correlation_matrix, method='average', metric='euclidean')

plt.figure(figsize=(20, 8))
dendrogram(Z, labels=correlation_matrix.columns, color_threshold=distance_threshold)
plt.axhline(y=distance_threshold, color='r', linestyle='--', label=f'Distance Threshold = {distance_threshold}')  
plt.xlabel('Features')
plt.ylabel('Distance')
# Guardado adaptado para usar el directorio de resultados estructurado
plt.savefig(os.path.join(RESULTS_PATH, 'dendogram.png'), format='png')
plt.show()

g = sns.clustermap(correlation_matrix, cmap='YlGnBu',
                   figsize=(30, 30),  
                   annot=False,  
                   xticklabels=True,  
                   yticklabels=True,  
                   row_cluster=True,  
                   col_cluster=True,  
                   tree_kws={'linewidths': 2},  
                   row_colors=row_colors,  
                   col_colors=col_colors  
                   )

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8)
g.ax_cbar.set_position((0.9, .02, .03, .1))
g.ax_cbar.set_ylabel('Correlation (R)')

y_position = 0.94  
for i, cluster_text in enumerate(cluster_texts):
    if i == len(cluster_texts) - 1:  
        plt.text(0.9, y_position, cluster_text, horizontalalignment='left', verticalalignment='top',
                 transform=g.fig.transFigure, fontsize=10, bbox=dict(facecolor=colors(i), edgecolor='none', boxstyle='square,pad=1'),
                 color='black')  
    else:
        plt.text(0.9, y_position, cluster_text, horizontalalignment='left', verticalalignment='top',
                 transform=g.fig.transFigure, fontsize=10, bbox=dict(facecolor=colors(i), edgecolor='none', boxstyle='square,pad=1'),
                 color='white')  
    y_position -= 0.012  

plt.savefig(os.path.join(RESULTS_PATH, 'clustermap.png'), format='png', dpi=300)  
plt.show()
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