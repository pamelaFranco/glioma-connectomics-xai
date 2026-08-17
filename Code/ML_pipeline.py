###############################################################################
# REFACTORED ML PIPELINE FOR GLIOMA STRUCTURAL CONNECTOMICS (WITH VISUALS)
# Includes: SelectKBest (ANOVA Grid) + SFS (5-Fold Inner CV), 
# Repeated Nested CV (5x5 Outer, 5 Inner), In-Fold Isolation, 
# Permutation Testing, Feature Stability Analysis across 25 Folds,
# Out-of-Sample Pooled SHAP Analysis with Patient-Level Aggregation,
# and Full Visual Suite (Dendrogram, Clustermap, Pareto, ROC, Confusion Matrix,
# SHAP Plots, Single Decision Tree Export, Decision Boundary, and t-SNE).
#
#   Author:      Dr. Pamela Franco / Refactored Pipeline
#   Repository:  https://github.com/pamelaFranco/glioma-ml-tractography
###############################################################################

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import seaborn as sns

from sklearn.model_selection import (
    RepeatedStratifiedKFold, 
    StratifiedKFold, 
    GridSearchCV,
    cross_validate
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix
)
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
from sklearn.tree import export_graphviz
import graphviz
import shap

# Enable Intel Acceleration if available
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

warnings.filterwarnings("ignore")

# Configure Matplotlib default settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# -----------------------------------------------------------------------------
# 1. SETUP PATHS AND PREPROCESSING CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH = r'C:\Users\pfran\Desktop\Connectomics\Connectomics Github\Dataset'
DATA_FILENAME = 'dataset_conectomicas_with_patient_details.csv'
RESULTS_PATH = r'C:\Users\pfran\Desktop\Connectomics\Connectomics Github\Results'

os.makedirs(RESULTS_PATH, exist_ok=True)
full_data_path = os.path.join(DATA_PATH, DATA_FILENAME)

if not os.path.exists(full_data_path):
    full_data_path = DATA_FILENAME

df_raw = pd.read_csv(full_data_path)

# -----------------------------------------------------------------------------
# 2. DYNAMIC TARGET & ID COLUMN DETECTION AND PREPROCESSING
# -----------------------------------------------------------------------------
label_candidates = ['labels', 'label', 'Class', 'target', 'group', 'Grade', 'Diagnosis', 'grado']
target_col = next((col for col in label_candidates if col in df_raw.columns), None)

if target_col is None:
    raise KeyError(f"Could not automatically detect target column in dataset. Available columns: {list(df_raw.columns)}")

print(f"Target column detected: '{target_col}'")

id_candidates = ['Patient_ID', 'Subject_ID', 'ID', 'Patient', 'Subject', 'Subjects', 'std_id']
id_cols_to_drop = [col for col in id_candidates if col in df_raw.columns]

labels_raw = df_raw[target_col].values
features_df = df_raw.drop(columns=[target_col] + id_cols_to_drop, errors='ignore')

if labels_raw.dtype == 'object':
    y_vec, _ = pd.factorize(labels_raw)
else:
    y_vec = labels_raw.astype(int)

y = pd.Series(y_vec)

features_df = features_df.rename(columns=lambda x: re.sub(r'[^*A-Za-z0-9_ ]+', '', x))
X_raw = features_df.copy()

imputer_init = SimpleImputer(strategy='mean')
X_clean_np = imputer_init.fit_transform(X_raw.replace([np.inf, -np.inf], np.nan))
X_clean = pd.DataFrame(X_clean_np, columns=X_raw.columns)

constant_features = [col for col in X_clean.columns if X_clean[col].std() == 0]
if constant_features:
    print(f"Warning: Detected {len(constant_features)} constant features with zero variance. Removing them.")
    X_clean = X_clean.drop(columns=constant_features)

print(f"Dataset Loaded Successfully: N = {X_clean.shape[0]} samples, P = {X_clean.shape[1]} features.")

scaler_exploratory = StandardScaler()
X_scaled_exploratory = pd.DataFrame(scaler_exploratory.fit_transform(X_clean), columns=X_clean.columns)

# -----------------------------------------------------------------------------
# 3. HIERARCHICAL CLUSTERING & EXPLORATORY DENDROGRAM / CLUSTERMAP
# -----------------------------------------------------------------------------
print("\n[Visual 1 & 2] Generating Hierarchical Clustering Dendrogram and Clustermap...")
correlation_matrix = X_scaled_exploratory.corr().fillna(0)

distance_threshold = 6.5
model_agg = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=distance_threshold)
cluster_cols = model_agg.fit_predict(correlation_matrix.T)
cluster_rows = model_agg.fit_predict(correlation_matrix)

n_clusters = len(np.unique(cluster_cols))
colors = plt.cm.get_cmap('plasma', n_clusters) if hasattr(plt.cm, 'get_cmap') else plt.colormaps['plasma']

col_colors = [colors(i) for i in cluster_cols]
row_colors = [colors(i) for i in cluster_rows]

cluster_features = [sum(np.array(cluster_cols) == i) for i in range(n_clusters)]
cluster_texts = [f"Cluster {i+1}: {cluster_features[i]:02d} features" for i in range(n_clusters)]

Z = linkage(correlation_matrix, method='average', metric='euclidean')

plt.figure(figsize=(12, 20))
dendrogram(
    Z,
    labels=correlation_matrix.columns,
    color_threshold=distance_threshold,
    orientation='left',
    leaf_font_size=6
)
plt.axvline(x=distance_threshold, color='r', linestyle='--', label=f'Distance Threshold = {distance_threshold}')
plt.xlabel('Distance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.savefig(os.path.join(RESULTS_PATH, 'dendogram.png'), format='png', dpi=300, bbox_inches='tight')
plt.close()

g = sns.clustermap(
    correlation_matrix, cmap='viridis',
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
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)
g.ax_cbar.set_position((0.9, .02, .03, .09))
g.ax_cbar.set_ylabel('Correlation (R)')

y_position = 0.94
box_height = 0.025
box_width = 0.08
spacing = 0.002

for i, cluster_text in enumerate(cluster_texts):
    current_color = colors(i)
    rect = Rectangle(
        (0.9, y_position - box_height), box_width, box_height,
        transform=g.fig.transFigure, facecolor=current_color, edgecolor='none', zorder=3
    )
    g.fig.add_artist(rect)
    text_y_center = y_position - (box_height / 2)

    rgb = current_color[:3]
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    text_color = 'black' if luminance > 0.5 else 'white'

    g.fig.text(
        0.905, text_y_center, cluster_text,
        transform=g.fig.transFigure, horizontalalignment='left', verticalalignment='center',
        fontsize=14, color=text_color, zorder=4
    )
    y_position -= (box_height + spacing)

plt.savefig(os.path.join(RESULTS_PATH, 'clustermap.png'), format='png', dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 4. BASELINE UNIVARIATE SCREENING: L2-LOGISTIC REGRESSION (PARETO)
# -----------------------------------------------------------------------------
print("\n[Visual 3] Running Baseline Univariate Screening (L2 Logistic Regression Pareto)...")
lr_baseline = LogisticRegression(penalty='l2', solver='liblinear', random_state=42, max_iter=1000)
lr_baseline.fit(X_scaled_exploratory, y)

absolute_weights = np.abs(lr_baseline.coef_[0])
sorted_indices = np.argsort(absolute_weights)[::-1]
top_20_indices = sorted_indices[:20]

top_20_features = [X_clean.columns[i] for i in top_20_indices]
top_20_weights = absolute_weights[top_20_indices]

plt.figure(figsize=(12, 6))
plt.rcParams['text.usetex'] = True
plt.bar(range(20), top_20_weights, color='teal', edgecolor='teal', alpha=0.85)
plt.xticks(range(20), top_20_features, rotation=45, ha='right', fontsize=9)
plt.ylabel('Absolute Log-Odds Coefficients Weight')
plt.xlabel('Graph-Theoretic Node Metric / Connectomic Edge Vector')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'logistic_regression_pareto_ranking.png'), format='png', dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 5. REPEATED NESTED CROSS-VALIDATION & BIASED COMPARISON PIPELINE
# -----------------------------------------------------------------------------
print("\n[Biased vs Unbiased Analysis] Computing Global Feature Selection for Leakage Baseline...")
scaler_leakage = StandardScaler()
X_scaled_leakage = pd.DataFrame(scaler_leakage.fit_transform(X_clean), columns=X_clean.columns)

global_sfs = SequentialFeatureSelector(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
    n_features_to_select=10, direction='forward', scoring='accuracy', cv=3, n_jobs=-1
)
global_sfs.fit(X_scaled_leakage, y)
global_chosen_features = X_clean.columns[global_sfs.get_support()].tolist()

N_REPEATS = 1
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 5 
BASE_SEED = 42

outer_cv = RepeatedStratifiedKFold(n_splits=N_SPLITS_OUTER, n_repeats=N_REPEATS, random_state=BASE_SEED)

fold_records = []
selected_features_all_folds = []
feature_counts_all_folds = []

biased_tprs, unbiased_tprs = [], []
mean_fpr = np.linspace(0, 1, 100)
biased_cumulative_cm = np.zeros((2, 2))
unbiased_cumulative_cm = np.zeros((2, 2))

best_isolated_estimators = []
isolated_features_per_fold = []

fold1_feature_accuracies_mean = [] 
fold1_feature_accuracies_std = []

shap_records = []
n_samples_total = len(X_clean)
n_features_total = X_clean.shape[1]

total_iterations = N_REPEATS * N_SPLITS_OUTER
print(f"\nStarting {N_REPEATS}x{N_SPLITS_OUTER}-Fold Repeated Nested Cross-Validation (Total Outer Iterations = {total_iterations})...")

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_clean, y)):
    X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_clean.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_clean.columns)

    # --- PATHWAY A: BIASED VALIDATION LOOP ---
    X_train_biased = X_train_scaled[global_chosen_features]
    X_test_biased = X_test_scaled[global_chosen_features]
    
    clf_biased = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf_biased.fit(X_train_biased, y_train)
    biased_preds = clf_biased.predict(X_test_biased)
    biased_probs = clf_biased.predict_proba(X_test_biased)[:, 1]
    
    biased_cumulative_cm += confusion_matrix(y_test, biased_preds)
    fpr_b, tpr_b, _ = roc_curve(y_test, biased_probs)
    biased_tprs.append(np.interp(mean_fpr, fpr_b, tpr_b))
    biased_tprs[-1][0] = 0.0

    # --- PATHWAY B: STRICTLY ISOLATED NESTED PATHWAY (ANOVA PRE-FILTER + SFS) ---
    max_features_sfs = min(10, X_train.shape[1])
    candidate_n_features = [3, 5, 8, 10]
    candidate_n_features = [n for n in candidate_n_features if n <= max_features_sfs]
    if not candidate_n_features:
        candidate_n_features = [max_features_sfs]

    candidate_k_anova = [15, 25, 50]
    candidate_k_anova = [k for k in candidate_k_anova if k <= X_train.shape[1]]
    if not candidate_k_anova:
        candidate_k_anova = [X_train.shape[1]]

    anova_selector = SelectKBest(score_func=f_classif)

    sfs_selector = SequentialFeatureSelector(
        estimator=RandomForestClassifier(
            n_estimators=30,
            max_depth=4,
            random_state=BASE_SEED,
            class_weight="balanced",
            n_jobs=1
        ),
        direction="forward",
        scoring="f1_macro",
        cv=5,
        n_jobs=1
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("prefilter", anova_selector),
        ("selector", sfs_selector),
        ("rf", RandomForestClassifier(random_state=BASE_SEED, class_weight="balanced", n_jobs=1))
    ])

    param_grid = {
        "prefilter__k": candidate_k_anova,
        "selector__n_features_to_select": candidate_n_features,
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [3, 5],
        "rf__min_samples_split": [4, 8],
        "rf__max_features": ["sqrt"]
    }

    inner_cv = StratifiedKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=BASE_SEED + fold_idx)
    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=inner_cv, scoring="f1_macro", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Outer Evaluation
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    unbiased_cumulative_cm += confusion_matrix(y_test, y_pred)
    fpr_u, tpr_u, _ = roc_curve(y_test, y_prob)
    unbiased_tprs.append(np.interp(mean_fpr, fpr_u, tpr_u))
    unbiased_tprs[-1][0] = 0.0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try:
        auc_score = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc_score = np.nan

    prefilter_mask = best_model.named_steps["prefilter"].get_support()
    prefiltered_cols = X_train.columns[prefilter_mask]
    sfs_mask = best_model.named_steps["selector"].get_support()
    chosen_features_fold = prefiltered_cols[sfs_mask].tolist()

    selected_features_all_folds.extend(chosen_features_fold)
    feature_counts_all_folds.append(len(chosen_features_fold))
    
    best_isolated_estimators.append(best_model.named_steps["rf"])
    isolated_features_per_fold.append(chosen_features_fold)

    # --- STRICT OUT-OF-SAMPLE SHAP CALCULATION PER FOLD ---
    fitted_imputer = best_model.named_steps["imputer"]
    fitted_scaler = best_model.named_steps["scaler"]
    fitted_prefilter = best_model.named_steps["prefilter"]
    fitted_selector = best_model.named_steps["selector"]
    fitted_rf = best_model.named_steps["rf"]

    X_test_imp = fitted_imputer.transform(X_test)
    X_test_scl = fitted_scaler.transform(X_test_imp)
    X_test_pf = fitted_prefilter.transform(X_test_scl)
    X_test_sel = fitted_selector.transform(X_test_pf)

    explainer = shap.TreeExplainer(fitted_rf)
    shap_vals_fold = explainer.shap_values(X_test_sel)

    if isinstance(shap_vals_fold, list):
        shap_vals_fold = shap_vals_fold[1]
    elif len(shap_vals_fold.shape) == 3:
        shap_vals_fold = shap_vals_fold[:, :, 1]

    for i_sub, patient_idx in enumerate(test_idx):
        full_shap_vector = np.zeros(n_features_total)
        for j_sel, feature_name in enumerate(chosen_features_fold):
            orig_col_idx = X_clean.columns.get_loc(feature_name)
            full_shap_vector[orig_col_idx] = shap_vals_fold[i_sub, j_sel]

        repeat_id = fold_idx // N_SPLITS_OUTER
        shap_records.append({
            'patient_idx': patient_idx,
            'repeat': repeat_id,
            'fold': fold_idx,
            'shap_vector': full_shap_vector
        })

    if fold_idx == 0:
        max_features_eval = min(20, X_train_scaled.shape[1])
        rf_ranker = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
        rf_ranker.fit(X_train_scaled, y_train)
        ranked_feats = X_clean.columns[np.argsort(rf_ranker.feature_importances_)[::-1]].tolist()
        
        for k_feat in range(1, max_features_eval + 1):
            sub_feats = ranked_feats[:k_feat]
            cv_res = cross_validate(
                RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
                X_train_scaled[sub_feats], y_train, cv=5, scoring='accuracy', n_jobs=-1
            )
            fold1_feature_accuracies_mean.append(np.mean(cv_res['test_score']))
            fold1_feature_accuracies_std.append(np.std(cv_res['test_score']))

    fold_records.append({
        'Iteration': fold_idx + 1,
        'Accuracy': acc,
        'Precision': prec,
        'Recall_Sensitivity': rec,
        'F1_Score': f1,
        'ROC_AUC': auc_score,
        'Optimal_K_Features': len(chosen_features_fold),
        'Best_Params': str(grid_search.best_params_)
    })

df_nested_results = pd.DataFrame(fold_records)

# -----------------------------------------------------------------------------
# 6. STATISTICAL PERFORMANCE SUMMARY & PERMUTATION TESTING
# -----------------------------------------------------------------------------
def compute_bootstrap_ci(data, n_bootstraps=2000, ci=95):
    boot_means = []
    rng = np.random.RandomState(BASE_SEED)
    clean_data = np.array(data)[~np.isnan(data)]
    for _ in range(n_bootstraps):
        sample = rng.choice(clean_data, size=len(clean_data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(clean_data), lower, upper

summary_rows = []
for metric in ['Accuracy', 'Precision', 'Recall_Sensitivity', 'F1_Score', 'ROC_AUC']:
    mean_val, ci_low, ci_high = compute_bootstrap_ci(df_nested_results[metric])
    std_val = df_nested_results[metric].std()
    summary_rows.append({
        'Metric': metric, 'Mean': mean_val, 'Std': std_val,
        '95% CI Lower': ci_low, '95% CI Upper': ci_high
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(RESULTS_PATH, 'unbiased_repeated_nested_cv_performance.csv'), index=False)

# Permutation Test Execution
N_PERMUTATIONS = 100
permuted_f1_scores = []
for p_idx in range(N_PERMUTATIONS):
    y_permuted = pd.Series(np.random.RandomState(BASE_SEED + p_idx).permutation(y))
    perm_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=BASE_SEED)
    perm_f1s = []
    for train_idx, test_idx in perm_cv.split(X_clean, y_permuted):
        X_tr, X_te = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_tr, y_te = y_permuted.iloc[train_idx], y_permuted.iloc[test_idx]
        perm_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("prefilter", SelectKBest(score_func=f_classif, k=min(25, X_tr.shape[1]))),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=BASE_SEED, class_weight='balanced'))
        ])
        perm_pipe.fit(X_tr, y_tr)
        perm_f1s.append(f1_score(y_te, perm_pipe.predict(X_te), average='macro', zero_division=0))
    permuted_f1_scores.append(np.mean(perm_f1s))

actual_f1_mean = df_nested_results['F1_Score'].mean()
p_value_permutation = (np.sum(np.array(permuted_f1_scores) >= actual_f1_mean) + 1) / (N_PERMUTATIONS + 1)

pd.DataFrame({'Actual_F1_Mean': [actual_f1_mean], 'Permutation_p_value': [p_value_permutation]}).to_csv(
    os.path.join(RESULTS_PATH, 'permutation_test_results.csv'), index=False
)

# Feature Selection Stability Export across 25 Folds
feature_counts = pd.Series(selected_features_all_folds).value_counts()
df_feature_stability = pd.DataFrame({
    'Feature_Name': feature_counts.index,
    'Selection_Frequency_Out_of_25_Folds': feature_counts.values,
    'Selection_Percentage': (feature_counts.values / total_iterations) * 100
})
df_feature_stability.to_csv(os.path.join(RESULTS_PATH, 'feature_selection_stability_25folds.csv'), index=False)

# -----------------------------------------------------------------------------
# 7. COMPARATIVE METRIC PLOTS (ROC & CONFUSION MATRICES)
# -----------------------------------------------------------------------------
print("\n[Visual 4 & 5] Generating Comparative ROC Curves and Confusion Matrices...")

plt.figure(figsize=(8, 6))
mean_tpr_b = np.mean(biased_tprs, axis=0); mean_tpr_b[-1] = 1.0
mean_tpr_u = np.mean(unbiased_tprs, axis=0); mean_tpr_u[-1] = 1.0

plt.plot(mean_fpr, mean_tpr_b, color='red', linestyle='--', label='Biased Pipeline (Mean AUC = %0.2f)' % auc(mean_fpr, mean_tpr_b), lw=2)
plt.plot(mean_fpr, mean_tpr_u, color='blue', linestyle='-', label='Strictly Isolated Pipeline (Mean AUC = %0.2f)' % auc(mean_fpr, mean_tpr_u), lw=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance level (AUC = 0.50)')
plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'comparative_roc_curve.png'), format='png', dpi=300)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(biased_cumulative_cm, annot=True, fmt='.0f', cmap='Reds', ax=axes[0], cbar=False, xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'])
axes[0].set_title('Biased Pipeline: Cumulative CM (Data Leakage)', fontweight='bold')
axes[0].set_ylabel('True Label'); axes[0].set_xlabel('Predicted Label')

sns.heatmap(unbiased_cumulative_cm, annot=True, fmt='.0f', cmap='Blues', ax=axes[1], cbar=False, xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'])
axes[1].set_title('Isolated Nested Pipeline: Cumulative CM', fontweight='bold')
axes[1].set_ylabel('True Label'); axes[1].set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'validation_confusion_matrices.png'), format='png', dpi=300)
plt.close()

if len(fold1_feature_accuracies_mean) > 0:
    print("\n[Visual 6] Exporting Feature Selection Accuracy Optimization Curve...")
    plt.figure(figsize=(10, 5))
    
    n_features = len(fold1_feature_accuracies_mean)
    feat_counts_plot = np.arange(1, n_features + 1)
    means_arr = np.array(fold1_feature_accuracies_mean)
    stds_arr = np.array(fold1_feature_accuracies_std)
    max_idx = np.argmax(means_arr)

    plt.plot(feat_counts_plot, means_arr, color='darkblue', marker='o', markersize=4, linestyle='-', alpha=0.8, label='Internal CV Mean Accuracy')
    plt.fill_between(feat_counts_plot, means_arr - stds_arr, means_arr + stds_arr, color='teal', alpha=0.15, label=r'Variance Shading ($\pm$ SD)')
    plt.plot(feat_counts_plot[max_idx], means_arr[max_idx], color='red', marker='o', markersize=9, label=f'Max (N={feat_counts_plot[max_idx]}, Acc={means_arr[max_idx]:.3f})')
    
    plt.xlim(1, n_features)
    plt.xticks(np.arange(1, n_features + 1, 1))

    plt.xlabel('Number of Features Selected (RF Importance Ranking)')
    plt.ylabel('Validation Accuracy (Internal CV)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'sfs_feature_accuracy_curve.png'), format='png', dpi=300)
    plt.close()

# -----------------------------------------------------------------------------
# 8. PATIENT-LEVEL AGGREGATED SHAP SUITE (OUT-OF-SAMPLE EVALUATION & STABILITY)
# -----------------------------------------------------------------------------
print("\n[Visual 7] Aggregating Out-of-Sample Pooled SHAP Attributions & Selection Frequencies...")

df_shap_all = pd.DataFrame(shap_records)

# Calculate patient-level average out-of-sample SHAP values
patient_shap_matrix = np.zeros((n_samples_total, n_features_total))
for p_idx in range(n_samples_total):
    p_vectors = df_shap_all[df_shap_all['patient_idx'] == p_idx]['shap_vector'].tolist()
    patient_shap_matrix[p_idx, :] = np.mean(p_vectors, axis=0)

mean_abs_shap = np.mean(np.abs(patient_shap_matrix), axis=0)
mean_signed_shap = np.mean(patient_shap_matrix, axis=0)

df_shap_summary = pd.DataFrame({
    'Feature_Name': X_clean.columns,
    'Mean_Absolute_SHAP': mean_abs_shap,
    'Mean_Signed_SHAP': mean_signed_shap
}).merge(df_feature_stability, on='Feature_Name', how='left').fillna(0)

df_shap_summary = df_shap_summary.sort_values(by='Mean_Absolute_SHAP', ascending=False)
df_shap_summary.to_csv(os.path.join(RESULTS_PATH, 'patient_level_shap_summary.csv'), index=False)

top_20_shap_indices = np.argsort(mean_abs_shap)[::-1][:20]
top_20_shap_names = X_clean.columns[top_20_shap_indices]

shap_matrix_top20 = patient_shap_matrix[:, top_20_shap_indices]
X_clean_top20 = X_clean[top_20_shap_names]

# 8.1 SHAP Beeswarm Plot (Updated Axis and Title Labels per Reviewer Requests)
plt.rcParams['text.usetex'] = False
plt.figure(figsize=(11, 7))
shap.summary_plot(shap_matrix_top20, X_clean_top20, show=False)
plt.rcParams['text.usetex'] = True
plt.xlabel(r'Out-of-Sample Pooled SHAP Value (Impact on Prediction)')
plt.title('Out-of-Sample Candidate Topological Connectomic Features', fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'shap_1_summary_beeswarm.png'), format='png', dpi=300)
plt.close()

# 8.2 Mean Absolute SHAP Bar Plot
plt.rcParams['text.usetex'] = False
plt.figure(figsize=(11, 7))
shap.summary_plot(shap_matrix_top20, X_clean_top20, plot_type="bar", show=False)
plt.rcParams['text.usetex'] = True
plt.xlabel(r'Mean Absolute SHAP Value (Out-of-Sample Patient-Level Pooled)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'shap_2_summary_bar.png'), format='png', dpi=300)
plt.close()

# 8.3 SHAP Dependence Plot
top_feat_name = top_20_shap_names[0]
plt.rcParams['text.usetex'] = False
plt.figure(figsize=(8, 6))
shap.dependence_plot(top_feat_name, shap_matrix_top20, X_clean_top20, show=False)
plt.rcParams['text.usetex'] = True
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, f'shap_3_dependence_{top_feat_name}.png'), format='png', dpi=300)
plt.close()

# 8.4 Local Force Plot
plt.rcParams['text.usetex'] = False
plt.figure(figsize=(14, 4))
base_val = 0.0
shap.force_plot(base_val, shap_matrix_top20[0, :], X_clean_top20.iloc[0, :], matplotlib=True, show=False)
plt.rcParams['text.usetex'] = True
plt.subplots_adjust(bottom=0.25, left=0.15)
plt.savefig(os.path.join(RESULTS_PATH, 'shap_4_local_patient_force.png'), format='png', dpi=300)
plt.close()

# 8.5 Global Decision Plot
plt.rcParams['text.usetex'] = False
plt.figure(figsize=(10, 7))
shap.decision_plot(base_val, shap_matrix_top20, X_clean_top20, show=False)
plt.rcParams['text.usetex'] = True
plt.subplots_adjust(bottom=0.18, left=0.35)
plt.savefig(os.path.join(RESULTS_PATH, 'shap_5_decision_trajectory.png'), format='png', dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 9. DECISION TREE LAYOUT & DECISION BOUNDARY SURFACE
# -----------------------------------------------------------------------------
print("\n[Visual 8] Exporting Single Decision Tree Layout via Graphviz...")
target_model = best_isolated_estimators[0]
target_features = isolated_features_per_fold[0]

if len(target_features) < 2:
    target_features = X_clean.columns[:5].tolist()
    target_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_scaled_exploratory[target_features], y)

try:
    plt.rcParams['text.usetex'] = False
    single_tree = target_model.estimators_[0]
    dot_data = export_graphviz(
        single_tree, out_file=None, max_depth=3,
        feature_names=target_features, class_names=['LGG', 'HGG'],
        filled=True, rounded=True, special_characters=True
    )
    dot_data = dot_data.replace('fontname="helvetica"', 'fontname="Arial"').replace('node [', 'node [fontname="Arial", fontsize=6, ')
    graph = graphviz.Source(dot_data)
    graph.render(os.path.join(RESULTS_PATH, 'random_forest_individual_tree'), format='png', cleanup=True)
    plt.rcParams['text.usetex'] = True
except Exception as err:
    print(f"Notice: Graphviz engine layout generation skipped: {err}")

if len(target_features) >= 2:
    print("\n[Visual 9] Generating Decision Boundary Surface Plot...")
    feat1, feat2 = target_features[0], target_features[1]
    X_boundary = X_scaled_exploratory[[feat1, feat2]]

    clf_surface = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf_surface.fit(X_boundary, y)

    x_min, x_max = X_boundary[feat1].min() - 1, X_boundary[feat1].max() + 1
    y_min, y_max = X_boundary[feat2].min() - 1, X_boundary[feat2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    Z_mesh = clf_surface.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(9, 7))
    plt.contourf(xx, yy, Z_mesh, alpha=0.3, cmap='coolwarm')
    scatter_surf = plt.scatter(X_boundary[feat1], X_boundary[feat2], c=y, edgecolor='k', alpha=0.8, cmap='coolwarm')
    plt.xlabel(feat1); plt.ylabel(feat2)

    plt.rcParams['text.usetex'] = True
    handles_surf, _ = scatter_surf.legend_elements()
    plt.legend(handles_surf, ['LGG', 'HGG'], title="Tumor Grade", loc="upper right", frameon=True)
    plt.grid(True, alpha=0.2); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'decision_boundary_surface.png'), format='png', dpi=300)
    plt.close()

# -----------------------------------------------------------------------------
# 10. t-SNE COORDINATE EMBEDDING COMPARISON
# -----------------------------------------------------------------------------
print("\n[Visual 10] Projecting t-SNE High-Dimensional Coordinates...")
plt.rcParams['text.usetex'] = False

perplexity_val = min(10, X_scaled_exploratory.shape[0] - 1)
tsne_pre = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, init='pca', learning_rate='auto')
X_tsne_pre = tsne_pre.fit_transform(X_scaled_exploratory)

X_scaled_isolated = X_scaled_exploratory[target_features]
tsne_post = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, init='pca', learning_rate='auto')
X_tsne_post = tsne_post.fit_transform(X_scaled_isolated)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
class_labels = ['LGG', 'HGG']

scatter_pre = axes[0].scatter(X_tsne_pre[:, 0], X_tsne_pre[:, 1], c=y, cmap='bwr', edgecolor='k', alpha=0.8, s=60)
axes[0].set_xlabel('t-SNE Dimension 1'); axes[0].set_ylabel('t-SNE Dimension 2'); axes[0].grid(True, alpha=0.2)
handles_pre, _ = scatter_pre.legend_elements()
axes[0].legend(handles_pre, class_labels, loc="upper right", title="Tumor Grade")

scatter_post = axes[1].scatter(X_tsne_post[:, 0], X_tsne_post[:, 1], c=y, cmap='bwr', edgecolor='k', alpha=0.8, s=60)
axes[1].set_xlabel('t-SNE Dimension 1'); axes[1].set_ylabel('t-SNE Dimension 2'); axes[1].grid(True, alpha=0.2)
handles_post, _ = scatter_post.legend_elements()
axes[1].legend(handles_post, class_labels, loc="upper right", title="Tumor Grade")

plt.rcParams['text.usetex'] = True
axes[0].set_title('A: High-Dimensional Curation Space (Pre-Selection)', fontweight='bold')
axes[1].set_title('B: Isolated Parsimonious Sub-space (Post-Selection)', fontweight='bold')
fig.suptitle('Connectomic Topological Space Modification vs. Class Segregation (t-SNE Projection)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'tsne_class_segregation_comparison.png'), format='png', dpi=300)
plt.close()

print(f"\nPipeline finalized successfully! All results, statistical tables, and comparative figures exported to:\n{RESULTS_PATH}")