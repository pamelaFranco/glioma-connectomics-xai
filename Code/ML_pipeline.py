###############################################################################
# This code was developed by Dr. Pamela Franco as part of the project 
# "Quantifying White Matter Disruption in Gliomas via Tumor-Masked Structural 
# Connectomics and Explainable Machine Learning: A Pilot Study ".
#
# This code performs a comprehensive machine learning pipeline tailored for 
# CLASSIFICATION analysis on glioma tractography and radiomics data.
#
#   Author:      Dr. Pamela Franco
#   Time-stamp:  2026-06-08
#   Repository:  https://github.com/pamelaFranco/glioma-ml-tractography
#   E-mail:      pamela.franco@unab.cl / pafranco@uc.cl
###############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import os 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.tree import export_graphviz
import graphviz
import shap

# Reset mathtext settings to default to avoid rendering crashes
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

###############################################################################
# SECTION 2.5.1: LOAD DATA & PREPROCESSING
###############################################################################
path = r'C:\Users\pfran\Desktop\Connectomics Github\Dataset'  
data_filename = 'dataset_conectomica_with_labels.csv'
newData = os.path.join(path, data_filename)  

RESULTS_PATH = r'C:\Users\pfran\Desktop\Connectomics Github\Results'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

if not os.path.exists(newData):
    newData = data_filename

# Reading the dataset (n = 35 patients)
df_raw = pd.read_csv(newData)

# Safe target and features extraction
label = df_raw['labels'].values
features = df_raw.drop(columns=['labels', 'Patient_ID'], errors='ignore')

if label.dtype == 'object':
    label, _ = pd.factorize(label)

# Sanitize feature headers by eliminating special characters using regex
features = features.rename(columns=lambda x: re.sub('[^*A-Za-z0-9_ ]+', '', x))
feature_names = list(features.columns)

# Handle missing, infinite and constant values upfront (Data Curation Routine)
if features.isnull().sum().any():
    print("Warning: Missing values detected upfront. Imputing with the column mean.")
    features = features.fillna(features.mean())

if np.isinf(features).values.any():
    print("Warning: Infinite values detected. Replacing with NaNs and imputing.")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(features.mean())

constant_features = [col for col in features.columns if features[col].std() == 0]
if constant_features:
    print(f"Warning: Detected {len(constant_features)} constant features with zero variance. Removing them.")
    features = features.drop(columns=constant_features)
    feature_names = list(features.columns)

X_clean = features
y = pd.Series(label)

# Scale specifically for global exploratory visualizations (Clustermap/Univariate)
scaler_exploratory = StandardScaler()
X_scaled_exploratory = pd.DataFrame(scaler_exploratory.fit_transform(X_clean), columns=X_clean.columns)

###############################################################################
# HIERARCHICAL CLUSTERING BLOCK
###############################################################################
correlation_matrix = X_scaled_exploratory.corr()

if correlation_matrix.isnull().any().any():
    correlation_matrix = correlation_matrix.fillna(0)

distance_threshold = 6.5 

model = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=distance_threshold)
cluster_cols = model.fit_predict(correlation_matrix.T)  
cluster_rows = model.fit_predict(correlation_matrix)  

n_clusters = len(np.unique(cluster_cols)) 
colors = plt.cm.get_cmap('plasma', n_clusters) if hasattr(plt.cm, 'get_cmap') else plt.colormaps['plasma']

col_colors = [colors(i) for i in cluster_cols]
row_colors = [colors(i) for i in cluster_rows]

cluster_features = [sum(np.array(cluster_cols) == i) for i in range(n_clusters)]
cluster_texts = [f"Cluster {i+1}: {cluster_features[i]:02d} features" for i in range(n_clusters)]

Z = linkage(correlation_matrix, method='average', metric='euclidean')

# Dendrogram Plot
plt.figure(figsize=(20, 8))
dendrogram(Z, labels=correlation_matrix.columns, color_threshold=distance_threshold)
plt.axhline(y=distance_threshold, color='r', linestyle='--', label=f'Distance Threshold = {distance_threshold}')  
plt.xlabel('Features')
plt.ylabel('Distance')
plt.savefig(os.path.join(RESULTS_PATH, 'dendogram.png'), format='png')
plt.close()

# Clustermap Plot
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
plt.close()

###############################################################################
# BASELINE UNIVARIATE SCREENING: L2-REGULARIZED LOGISTIC REGRESSION (PARETO)
###############################################################################
print("\n Running Baseline Univariate Screening (L2 Logistic Regression)...")
lr_baseline = LogisticRegression(penalty='l2', solver='liblinear', random_state=42, max_iter=1000)
lr_baseline.fit(X_scaled_exploratory, y)

# Extract absolute weights and build a Pareto Distribution Ranking
absolute_weights = np.abs(lr_baseline.coef_[0])
sorted_indices = np.argsort(absolute_weights)[::-1]
top_20_indices = sorted_indices[:20]

top_20_features = [X_clean.columns[i] for i in top_20_indices]
top_20_weights = absolute_weights[top_20_indices]

plt.figure(figsize=(12, 6))
plt.rcParams['text.usetex'] = True
bars = plt.bar(range(20), top_20_weights, color='teal', edgecolor='teal', alpha=0.85)
plt.xticks(range(20), top_20_features, rotation=45, ha='right', fontsize=9)
#plt.xlim(-0.5, 20.5)
plt.ylabel('Absolute Log-Odds Coefficients Weight')
plt.xlabel('Graph-Theoretic Node Metric / Connectomic Edge Vector')
plt.rcParams['text.usetex'] = True
#plt.title('Pareto Distribution Weight Ranking: Top 20 Baseline Components', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'logistic_regression_pareto_ranking.png'), dpi=300)
plt.close()

###############################################################################
# SECTION 2.5.2 & 2.5.3: FEATURE SELECTION & VALIDATION LOOP ARCHITECTURES
###############################################################################
print("\n Computing Global Feature Selection to simulate Data Leakage...")
scaler_leakage = StandardScaler()
X_scaled_leakage = pd.DataFrame(scaler_leakage.fit_transform(X_clean), columns=X_clean.columns)

global_sfs = SequentialFeatureSelector(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
    n_features_to_select=10, direction='forward', scoring='accuracy', cv=3, n_jobs=-1
)
global_sfs.fit(X_scaled_leakage, y)
global_chosen_features = X_clean.columns[global_sfs.get_support()].tolist()
print(f"Globally selected features (Biased Sub-space): {global_chosen_features}")

print("\n Initializing Parallel Validation Loops for Data Leakage Analysis...")
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

fold_metrics_records = []
biased_tprs, unbiased_tprs = [], []
mean_fpr = np.linspace(0, 1, 100)

biased_cumulative_cm = np.zeros((2, 2))
unbiased_cumulative_cm = np.zeros((2, 2))

best_isolated_estimators = []
isolated_features_per_fold = []

max_features_to_evaluate = min(30, X_clean.shape[1])
fold1_feature_accuracies_mean = [] 
fold1_feature_accuracies_std = []
fold1_ranked_features = []

print(" Launching Outer Cross-Validation Loop...")
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_clean, y)):
    X_train_fold, X_test_fold = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
    
    # --- METRIC CORRECTION: PLIEGUE ISOLATION ---
    scaler = StandardScaler()
    X_train_fold_scaled = pd.DataFrame(scaler.fit_transform(X_train_fold), columns=X_clean.columns)
    X_test_fold_scaled = pd.DataFrame(scaler.transform(X_test_fold), columns=X_clean.columns)
    
    # --- PATHWAY A: BIASED VALIDATION LOOP ---
    X_train_biased = X_train_fold_scaled[global_chosen_features]
    X_test_biased = X_test_fold_scaled[global_chosen_features]
    
    clf_biased = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf_biased.fit(X_train_biased, y_train_fold)
    
    biased_preds = clf_biased.predict(X_test_biased)
    biased_probs = clf_biased.predict_proba(X_test_biased)[:, 1]
    
    biased_cumulative_cm += confusion_matrix(y_test_fold, biased_preds)
    
    b_acc = accuracy_score(y_test_fold, biased_preds)
    b_prec = precision_score(y_test_fold, biased_preds, average='macro', zero_division=0)
    b_rec = recall_score(y_test_fold, biased_preds, average='macro', zero_division=0) 
    b_f1 = f1_score(y_test_fold, biased_preds, average='macro', zero_division=0)
    
    fpr_b, tpr_b, _ = roc_curve(y_test_fold, biased_probs)
    biased_tprs.append(np.interp(mean_fpr, fpr_b, tpr_b))
    biased_tprs[-1][0] = 0.0

    # --- PATHWAY B: STRICTLY ISOLATED NESTED PATHWAY ---
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_selector.fit(X_train_fold_scaled, y_train_fold)
    
    importances = rf_selector.feature_importances_
    indices_ranking = np.argsort(importances)[::-1]
    ranked_features_fold = X_clean.columns[indices_ranking].tolist()
    
    feature_means_cv = []
    feature_stds_cv = []
    
    for k in range(1, max_features_to_evaluate + 1):
        top_k_features = ranked_features_fold[:k]
        X_train_k = X_train_fold_scaled[top_k_features]
        
        cv_results = cross_validate(
            RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
            X_train_k, y_train_fold, cv=inner_cv, scoring='accuracy', n_jobs=-1
        )
        feature_means_cv.append(np.mean(cv_results['test_score']))
        feature_stds_cv.append(np.std(cv_results['test_score']))
    
    if fold == 0:
        fold1_feature_accuracies_mean = feature_means_cv
        fold1_feature_accuracies_std = feature_stds_cv
        fold1_ranked_features = ranked_features_fold[:max_features_to_evaluate]
    
    optimal_k = np.argmax(feature_means_cv) + 1
    local_chosen_features = ranked_features_fold[:optimal_k]
    isolated_features_per_fold.append(local_chosen_features)
    
    X_train_unbiased = X_train_fold_scaled[local_chosen_features]
    X_test_unbiased = X_test_fold_scaled[local_chosen_features]
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid=param_grid, cv=inner_cv, scoring='f1_macro', n_jobs=-1
    )
    grid_search.fit(X_train_unbiased, y_train_fold)
    clf_unbiased = grid_search.best_estimator_
    best_isolated_estimators.append(clf_unbiased)
    
    best_params = grid_search.best_params_
    
    unbiased_preds = clf_unbiased.predict(X_test_unbiased)
    unbiased_probs = clf_unbiased.predict_proba(X_test_unbiased)[:, 1]
    
    unbiased_cumulative_cm += confusion_matrix(y_test_fold, unbiased_preds)
    
    u_acc = accuracy_score(y_test_fold, unbiased_preds)
    u_prec = precision_score(y_test_fold, unbiased_preds, average='macro', zero_division=0)
    u_rec = recall_score(y_test_fold, unbiased_preds, average='macro', zero_division=0) 
    u_f1 = f1_score(y_test_fold, unbiased_preds, average='macro', zero_division=0)
    
    fpr_u, tpr_u, _ = roc_curve(y_test_fold, unbiased_probs)
    unbiased_tprs.append(np.interp(mean_fpr, fpr_u, tpr_u))
    unbiased_tprs[-1][0] = 0.0
    
    fold_metrics_records.append({
        'Fold': fold + 1,
        'Biased_Accuracy': b_acc, 'Biased_Precision': b_prec, 'Biased_Sensitivity_Recall': b_rec, 'Biased_F1_Score': b_f1,
        'Unbiased_Accuracy': u_acc, 'Unbiased_Precision': u_prec, 'Unbiased_Sensitivity_Recall': u_rec, 'Unbiased_F1_Score': u_f1,
        'Unbiased_Optimal_Features_Count': optimal_k, 'Unbiased_Params': str(best_params)
    })
    
    print(f"    Fold {fold+1} Completed | Biased F1: {b_f1:.4f} | Isolated F1: {u_f1:.4f} | Features: {optimal_k}")

df_fold_metrics = pd.DataFrame(fold_metrics_records)
df_fold_metrics.to_csv(os.path.join(RESULTS_PATH, 'pipeline_fold_metrics_and_hyperparameters.csv'), index=False)

###############################################################################
# SECTION 2.5.4: POST-HOC STATISTICAL EVALUATION & EXPORT
###############################################################################
mean_biased_f1 = df_fold_metrics['Biased_F1_Score'].mean()
mean_unbiased_f1 = df_fold_metrics['Unbiased_F1_Score'].mean()
leakage_delta = mean_biased_f1 - mean_unbiased_f1

leakage_summary = {
    'Metric': ['Accuracy', 'Precision', 'Sensitivity_Recall', 'F1_Score'],
    'Biased_Mean': [df_fold_metrics['Biased_Accuracy'].mean(), df_fold_metrics['Biased_Precision'].mean(), df_fold_metrics['Biased_Sensitivity_Recall'].mean(), mean_biased_f1],
    'Biased_Std': [df_fold_metrics['Biased_Accuracy'].std(), df_fold_metrics['Biased_Precision'].std(), df_fold_metrics['Biased_Sensitivity_Recall'].std(), df_fold_metrics['Biased_F1_Score'].std()],
    'Unbiased_Mean': [df_fold_metrics['Unbiased_Accuracy'].mean(), df_fold_metrics['Unbiased_Precision'].mean(), df_fold_metrics['Unbiased_Sensitivity_Recall'].mean(), mean_unbiased_f1],
    'Unbiased_Std': [df_fold_metrics['Unbiased_Accuracy'].std(), df_fold_metrics['Unbiased_Precision'].std(), df_fold_metrics['Unbiased_Sensitivity_Recall'].std(), df_fold_metrics['Unbiased_F1_Score'].std()],
    'Inflation_Delta': [
        df_fold_metrics['Biased_Accuracy'].mean() - df_fold_metrics['Unbiased_Accuracy'].mean(),
        df_fold_metrics['Biased_Precision'].mean() - df_fold_metrics['Unbiased_Precision'].mean(),
        df_fold_metrics['Biased_Sensitivity_Recall'].mean() - df_fold_metrics['Unbiased_Sensitivity_Recall'].mean(),
        leakage_delta
    ]
}
pd.DataFrame(leakage_summary).to_csv(os.path.join(RESULTS_PATH, 'data_leakage_statistical_analysis.csv'), index=False)

# PLOT COMPARATIVE ROC CURVES
plt.figure(figsize=(8, 6))
mean_tpr_biased = np.mean(biased_tprs, axis=0); mean_tpr_biased[-1] = 1.0
mean_tpr_unbiased = np.mean(unbiased_tprs, axis=0); mean_tpr_unbiased[-1] = 1.0

plt.plot(mean_fpr, mean_tpr_biased, color='red', linestyle='--', label='Biased Pipeline (Mean AUC = %0.2f)' % auc(mean_fpr, mean_tpr_biased), lw=2)
plt.plot(mean_fpr, mean_tpr_unbiased, color='blue', linestyle='-', label='Strictly Isolated Pipeline (Mean AUC = %0.2f)' % auc(mean_fpr, mean_tpr_unbiased), lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance level (AUC = 0.50)')
plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) Comparison', fontweight='bold')
plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'comparative_roc_curve.png'))
plt.close()

# PLOT COMPREHENSIVE EMPIRICAL CONFUSION MATRICES
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(biased_cumulative_cm, annot=True, fmt='.0f', cmap='Reds', ax=axes[0], cbar=False,
            xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'])
axes[0].set_title('Biased Pipeline: Cumulative CM (Data Leakage)', fontweight='bold')
axes[0].set_ylabel('True Label'); axes[0].set_xlabel('Predicted Label')

sns.heatmap(unbiased_cumulative_cm, annot=True, fmt='.0f', cmap='Blues', ax=axes[1], cbar=False,
            xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'])
axes[1].set_title('Isolated Nested Pipeline: Cumulative CM', fontweight='bold')
axes[1].set_ylabel('True Label'); axes[1].set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'validation_confusion_matrices.png'), dpi=300)
plt.close()

# PLOT FEATURE SELECTION ACCURACY CURVE
plt.figure(figsize=(10, 5))
features_count = np.arange(1, len(fold1_feature_accuracies_mean) + 1)
means = np.array(fold1_feature_accuracies_mean)
stds = np.array(fold1_feature_accuracies_std)
max_acc_idx = np.argmax(means)
optimal_num_features = features_count[max_acc_idx]
max_accuracy = means[max_acc_idx]

plt.plot(features_count, means, color='darkblue', marker='o', markersize=4, linestyle='-', alpha=0.8, label='Internal CV Mean Accuracy')
plt.fill_between(features_count, means - stds, means + stds, color='teal', alpha=0.15, label=r'Variance Shading ($\pm$ SD)')
plt.plot(optimal_num_features, max_accuracy, color='red', marker='o', markersize=9, linestyle='', label=f'Global Max (N={optimal_num_features}, Acc={max_accuracy:.3f})')
plt.xlabel('Number of Features Selected (by RF Importance Ranking)')
plt.ylabel('Validation Accuracy (Internal CV)')
#plt.title('Feature Selection Process Optimization (Fold 1 Isolation)', fontweight='bold')
plt.xticks(features_count, rotation=90 if len(features_count) > 20 else 0)
plt.grid(True, alpha=0.3); plt.legend(loc="lower right"); plt.xlim(1,30)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'sfs_feature_accuracy_curve.png'))
plt.close()

###############################################################################
# EXTRACTION OF ALL SHAP INTERPRETABILITY PLOTS (EXPANDED FRAMEWORK)
###############################################################################
print("\n Computing All Possible Additive Feature Attributions using TreeExplainer...")
target_model = best_isolated_estimators[0]
target_features = isolated_features_per_fold[0]

# Generate fold-isolated evaluation scale for localized SHAP tracking
scaler_shap = StandardScaler()
X_train_scaled_expl = pd.DataFrame(scaler_shap.fit_transform(X_clean), columns=X_clean.columns)
X_shap_input = X_train_scaled_expl[target_features]

plt.rcParams['text.usetex'] = False 

explainer = shap.TreeExplainer(target_model)
shap_values_raw = explainer.shap_values(X_shap_input)

if isinstance(shap_values_raw, list):
    shap_matrix = shap_values_raw[1]
elif len(shap_values_raw.shape) == 3:
    shap_matrix = shap_values_raw[:, :, 1]
else:
    shap_matrix = shap_values_raw

# 1. SHAP Summary Density Scatter Plot
plt.figure(figsize=(11, 7))
shap.summary_plot(shap_matrix, X_shap_input, show=False)
plt.rcParams['text.usetex'] = True
#plt.title('Structural Attributions Map within Isolated Parsimonious Sub-space (SHAP Scatter)', fontweight='bold')
plt.xlabel(r'SHAP interaction value (impact on model output)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'shap_1_summary_scatter.png'), dpi=300)
plt.close()

# 2. SHAP Global Feature Importance Bar Graph
plt.rcParams['text.usetex'] = False
plt.figure(figsize=(11, 7))
shap.summary_plot(shap_matrix, X_shap_input, plot_type="bar", show=False)
plt.rcParams['text.usetex'] = True
#plt.title('Global Feature Importance Framework via Mean Absolute SHAP', fontweight='bold')
plt.xlabel(r'mean(|SHAP value|) (average impact magnitude)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'shap_2_summary_bar.png'), dpi=300)
plt.close()

# 3. SHAP Feature Dependence Plots (Top 2 Predictive Features)
for idx, feature_name in enumerate(target_features[:2]):
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(feature_name, shap_matrix, X_shap_input, show=False)
    plt.rcParams['text.usetex'] = True
   # plt.title(f'SHAP Dependence Evaluation Plot: {feature_name}', pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, f'shap_3_dependence_{idx+1}_{feature_name}.png'), dpi=300)
    plt.close()

# 4. SHAP Local Force Plot (Patient 1 Sample Profiling)
plt.rcParams['text.usetex'] = False
plt.figure(figsize=(14, 4))
if hasattr(explainer, 'expected_value'):
    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    shap.force_plot(base_val, shap_matrix[0, :], X_shap_input.iloc[0, :], matplotlib=True, show=False)
    plt.rcParams['text.usetex'] = True
    #plt.title('Patient 1 Single-Sample Clinical Prediction Breakdown (SHAP Force Plot)', pad=25, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'shap_4_local_patient_force.png'), dpi=300)
plt.close()

# 5. SHAP Decision Plot
plt.rcParams['text.usetex'] = False
plt.figure(figsize=(10, 7))
if hasattr(explainer, 'expected_value'):
    shap.decision_plot(base_val, shap_matrix, X_shap_input, show=False)
    plt.rcParams['text.usetex'] = True
    #plt.title('Model Attributions Accumulation Paths (SHAP Decision Plot)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'shap_5_decision_trajectory.png'), dpi=300)
plt.close()

###############################################################################
# RANDOM FOREST TOPOLOGY: SINGLE DECISION TREE EXPORT
###############################################################################
print("\n Exporting single decision tree structural layout...")
try:
    # Temporarily clean font defaults so they don't break box constraints inside graphviz rendering
    old_usetex = plt.rcParams['text.usetex']
    plt.rcParams['text.usetex'] = False
    
    single_tree = target_model.estimators_[0]
    
    # Map 'Class_0' to 'LGG' and 'Class_1' to 'HGG' with consistent formatting
    dot_data = export_graphviz(
        single_tree, out_file=None, max_depth=3,
        feature_names=target_features, class_names=['LGG', 'HGG'],
        filled=True, rounded=True, special_characters=True
    )
    
    # Standardize fonts and ensure text constraints dynamically fit inside the rendering blocks
    dot_data = dot_data.replace('fontname="helvetica"', 'fontname="Arial"')
    # Add a global graph attribute to force nodes to fit text smoothly if needed
    dot_data = dot_data.replace('node [', 'node [fontname="Arial", fontsize=6, ')
    
    graph = graphviz.Source(dot_data)
    graph.render(os.path.join(RESULTS_PATH, 'random_forest_individual_tree'), format='png', cleanup=True)
    print(" Single decision tree exported successfully as PNG.")
    
    # Revert to normal structural configuration safely
    plt.rcParams['text.usetex'] = old_usetex
except Exception as e:
    print(f" Notice: Local Graphviz engine configuration could not build tree layout. Error caught safely: {e}")
###############################################################################
# DECISION BOUNDARY SURFACE PLOT (Top 2 Isolated Features)
###############################################################################
if len(target_features) >= 2:
    feat1, feat2 = target_features[0], target_features[1]
    X_boundary = X_train_scaled_expl[[feat1, feat2]]
    
    clf_surface = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf_surface.fit(X_boundary, y)
    
    x_min, x_max = X_boundary[feat1].min() - 1, X_boundary[feat1].max() + 1
    y_min, y_max = X_boundary[feat2].min() - 1, X_boundary[feat2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    Z_mesh = clf_surface.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(9, 7))
    
    plt.contourf(xx, yy, Z_mesh, alpha=0.3, cmap='coolwarm')
    
    scatter = plt.scatter(X_boundary[feat1], X_boundary[feat2], c=y, edgecolor='k', alpha=0.8, cmap='coolwarm')
    
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    
    plt.rcParams['text.usetex'] = True
    
    class_labels = ['LGG', 'HGG']
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_labels, title="Tumor Grade", loc="upper right", frameon=True)
    
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'decision_boundary_surface.png'), dpi=300)
    plt.close()

###############################################################################
# HIGH-DIMENSIONAL VISUAL SEGREGATION: t-SNE COORDINATE EMBEDDINGS
###############################################################################
print("\n Projecting High-Dimensional Space Transformations using t-SNE Projections...")
plt.rcParams['text.usetex'] = False 

# 1. t-SNE before feature selection (All Curated Features Space)
tsne_pre = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
X_tsne_pre = tsne_pre.fit_transform(X_train_scaled_expl)

# 2. t-SNE after feature selection (Isolated Parsimonious Sub-space from Fold 1)
X_scaled_isolated = X_train_scaled_expl[target_features]
tsne_post = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
X_tsne_post = tsne_post.fit_transform(X_scaled_isolated)

# Plot side-by-side t-SNE manifolds to track cohort segregation boundaries transformation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Define clinical class labels mapping Class 0 -> LGG and Class 1 -> HGG
class_labels = ['LGG', 'HGG']

# Panel A: Pre-selection
scatter_pre = axes[0].scatter(X_tsne_pre[:, 0], X_tsne_pre[:, 1], c=y, cmap='bwr', edgecolor='k', alpha=0.8, s=60)
axes[0].set_xlabel('t-SNE Dimension 1')
axes[0].set_ylabel('t-SNE Dimension 2')
axes[0].grid(True, alpha=0.2)

# Re-mapping Legend Labels for Panel A
handles_pre, _ = scatter_pre.legend_elements()
axes[0].legend(handles_pre, class_labels, loc="upper right", title="Tumor Grade")

# Panel B: Post-selection
scatter_post = axes[1].scatter(X_tsne_post[:, 0], X_tsne_post[:, 1], c=y, cmap='bwr', edgecolor='k', alpha=0.8, s=60)
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].grid(True, alpha=0.2)

# Re-mapping Legend Labels for Panel B
handles_post, _ = scatter_post.legend_elements()
axes[1].legend(handles_post, class_labels, loc="upper right", title="Tumor Grade")

# Reactivate LaTeX formatting for the text/titles if required by your environment
plt.rcParams['text.usetex'] = True

axes[0].set_title('A: High-Dimensional Curation Space (Pre-Selection, p=307)', fontweight='bold')
axes[1].set_title('B: Isolated Parsimonious Sub-space (Post-Selection, p=Optimal)', fontweight='bold')
fig.suptitle('Connectomic Topological Space Modification vs. Class Segregation (t-SNE Projection)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'tsne_class_segregation_comparison.png'), dpi=300)
plt.close()

print(f"\n Pipeline finalized. All comparative figures and CSV datasets exported to: {RESULTS_PATH}")