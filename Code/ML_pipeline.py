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

# Configure Matplotlib default settings (Strict Native LaTeX with Computer Modern Roman)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Computer Modern', 'cmr10']
plt.rcParams['mathtext.fontset'] = 'cm'

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

features_df = features_df.rename(columns=lambda x: re.sub(r'[^a-zA-Z0-9_]+', '_', x))
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
biased_records = []
selected_features_all_folds = []
feature_counts_all_folds = []

biased_tprs, unbiased_tprs = [], []
biased_aucs, unbiased_aucs = [], []
mean_fpr = np.linspace(0, 1, 100)
biased_cumulative_cm = np.zeros((2, 2))
unbiased_cumulative_cm = np.zeros((2, 2))

best_isolated_estimators = []
isolated_features_per_fold = []

shap_records = []
n_samples_total = len(X_clean)
n_features_total = X_clean.shape[1]

total_iterations = N_REPEATS * N_SPLITS_OUTER
print(f"\nStarting {N_REPEATS}x{N_SPLITS_OUTER}-Fold Repeated Nested Cross-Validation...")

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
    
    b_acc = accuracy_score(y_test, biased_preds)
    b_prec = precision_score(y_test, biased_preds, average="macro", zero_division=0)
    b_rec = recall_score(y_test, biased_preds, average="macro", zero_division=0)
    b_f1 = f1_score(y_test, biased_preds, average="macro", zero_division=0)
    
    try:
        biased_auc_score = roc_auc_score(y_test, biased_probs)
    except ValueError:
        biased_auc_score = np.nan
    biased_aucs.append(biased_auc_score)

    biased_records.append({
        'Accuracy': b_acc, 'Precision': b_prec, 'Sensitivity_Recall': b_rec, 'F1_Score': b_f1, 'ROC_AUC': biased_auc_score
    })

    # --- PATHWAY B: STRICTLY ISOLATED NESTED PATHWAY ---
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
            n_estimators=30, max_depth=4, random_state=BASE_SEED, class_weight="balanced", n_jobs=1
        ),
        direction="forward", scoring="f1_macro", cv=5, n_jobs=1
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
    interp_tpr = np.interp(mean_fpr, fpr_u, tpr_u)
    interp_tpr[0] = 0.0
    unbiased_tprs.append(interp_tpr)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try:
        auc_score = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc_score = np.nan
    
    unbiased_aucs.append(auc_score)

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

    fold_records.append({
        'Fold': fold_idx + 1,
        'Accuracy': acc,
        'Precision': prec,
        'Sensitivity_Recall': rec,
        'F1_Score': f1,
        'ROC_AUC': auc_score,
        'Optimal_K_Features': len(chosen_features_fold),
        'Selected_Features': ", ".join(chosen_features_fold),
        'Best_Params': str(grid_search.best_params_)
    })

df_nested_results = pd.DataFrame(fold_records)
df_biased_results = pd.DataFrame(biased_records)

# Export the per-fold metrics and hyperparameters table directly to CSV
df_nested_results.to_csv(os.path.join(RESULTS_PATH, 'pipeline_fold_metrics_and_hyperparameters.csv'), index=False)
print("\n[Table Export] Per-fold metrics and hyperparameters exported successfully.")

# -----------------------------------------------------------------------------
# 6. STATISTICAL PERFORMANCE COMPARISON TABLE & BOOTSTRAP 95% CIs
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

comparative_rows = []
for metric in ['Accuracy', 'Precision', 'Sensitivity_Recall', 'F1_Score', 'ROC_AUC']:
    b_mean, b_ci_low, b_ci_high = compute_bootstrap_ci(df_biased_results[metric])
    b_std = df_biased_results[metric].std()
    
    u_mean, u_ci_low, u_ci_high = compute_bootstrap_ci(df_nested_results[metric])
    u_std = df_nested_results[metric].std()
    
    delta = b_mean - u_mean
    
    comparative_rows.append({
        'Metric': metric,
        'Biased_Mean': b_mean,
        'Biased_Std': b_std,
        'Biased_95_CI_Lower': b_ci_low,
        'Biased_95_CI_Upper': b_ci_high,
        'Unbiased_Mean': u_mean,
        'Unbiased_Std': u_std,
        'Unbiased_95_CI_Lower': u_ci_low,
        'Unbiased_95_CI_Upper': u_ci_high,
        'Inflation_Delta': delta
    })

df_comparison = pd.DataFrame(comparative_rows)
df_comparison.to_csv(os.path.join(RESULTS_PATH, 'comparative_performance_summary_ci.csv'), index=False)
print("[Table Export] Comparative statistical table with 95% CIs exported successfully.")

# Feature Selection Stability Export across Folds
feature_counts = pd.Series(selected_features_all_folds).value_counts()
df_feature_stability = pd.DataFrame({
    'Feature_Name': feature_counts.index,
    'Selection_Frequency_Out_of_Folds': feature_counts.values,
    'Selection_Percentage': (feature_counts.values / total_iterations) * 100
})
df_feature_stability.to_csv(os.path.join(RESULTS_PATH, 'feature_selection_stability.csv'), index=False)

# -----------------------------------------------------------------------------
# 7. COMPARATIVE METRIC PLOTS
# -----------------------------------------------------------------------------
print("\n[Visual 4 & 5] Generating Comparative ROC Curves and Confusion Matrices...")

plt.figure(figsize=(9, 7))

mean_tpr_b = np.mean(biased_tprs, axis=0)
mean_tpr_b[-1] = 1.0

mean_tpr_u = np.mean(unbiased_tprs, axis=0)
mean_tpr_u[-1] = 1.0

mean_auc_b = auc(mean_fpr, mean_tpr_b)
std_auc_b = np.std(biased_aucs)

mean_auc_u = auc(mean_fpr, mean_tpr_u)
std_auc_u = np.std(unbiased_aucs)

b_auc_mean_bs, b_auc_ci_low, b_auc_ci_high = compute_bootstrap_ci(biased_aucs, ci=95)
u_auc_mean_bs, u_auc_ci_low, u_auc_ci_high = compute_bootstrap_ci(unbiased_aucs, ci=95)

legend_label_biased = (
    r'Biased Pipeline (Mean AUC = ' f'{mean_auc_b:.3g}' 
    r' $\pm$ ' f'{std_auc_b:.3g}, '
    f'95\\% CI [{b_auc_ci_low:.3g}, {b_auc_ci_high:.3g}])'
)

plt.plot(
    mean_fpr, mean_tpr_b, color='red', linestyle='--',
    label=legend_label_biased, lw=2
)

legend_label_unbiased = (
    r'\textbf{Isolated Pipeline} (Mean AUC = ' f'{mean_auc_u:.3g}' 
    r' $\pm$ ' f'{std_auc_u:.3g}, '
    f'95\\% CI [{u_auc_ci_low:.3g}, {u_auc_ci_high:.3g}])'
)

plt.plot(
    mean_fpr, mean_tpr_u, color='blue', linestyle='-',
    label=legend_label_unbiased, lw=2.5
)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label=r'Chance level (AUC = 0.500)')

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel(r'\textbf{False Positive Rate} ($1 - \text{Specificity}$)', fontsize=11)
plt.ylabel(r'\textbf{True Positive Rate} ($\text{Sensitivity}$)', fontsize=11)
plt.title(r'\textbf{Receiver Operating Characteristic (ROC) Curve}', fontsize=12, pad=10)
plt.legend(loc="lower right", fontsize=8.5, frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'comparative_roc_curve.png'), format='png', dpi=300)
plt.close()

# Visual 5: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(biased_cumulative_cm, annot=True, fmt='.0f', cmap='Reds', ax=axes[0], cbar=False, xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'])
axes[0].set_title(r'\textbf{Biased Pipeline: Cumulative CM (Data Leakage)}', fontweight='bold')
axes[0].set_ylabel(r'\textbf{True Label}')
axes[0].set_xlabel(r'\textbf{Predicted Label}')

sns.heatmap(unbiased_cumulative_cm, annot=True, fmt='.0f', cmap='Blues', ax=axes[1], cbar=False, xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'])
axes[1].set_title(r'\textbf{Isolated Nested Pipeline: Cumulative CM}', fontweight='bold')
axes[1].set_ylabel(r'\textbf{True Label}')
axes[1].set_xlabel(r'\textbf{Predicted Label}')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'validation_confusion_matrices.png'), format='png', dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# 8. PATIENT-LEVEL SHAP ANALYSIS
# -----------------------------------------------------------------------------
print("\n[Visual 7] Aggregating Out-of-Sample Pooled SHAP Attributions...")
df_shap_all = pd.DataFrame(shap_records)

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

plt.rcParams['text.usetex'] = False
plt.figure(figsize=(11, 7))
shap.summary_plot(shap_matrix_top20, X_clean_top20, show=False)
plt.rcParams['text.usetex'] = True
plt.xlabel(r'Out-of-Sample Pooled SHAP Value (Impact on Prediction)')
plt.title(r'\textbf{Out-of-Sample Candidate Connectomic Features}', fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'shap_summary_beeswarm.png'), format='png', dpi=300)
plt.close()

print(f"\nPipeline finalized successfully! All outputs exported to:\n{RESULTS_PATH}")