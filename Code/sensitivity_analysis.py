###############################################################################
# SKLEARN ACCELERATION & DEPENDENCIES
###############################################################################
import os
import re
import warnings
import zipfile
import xml.etree.ElementTree as ET

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats  # Imported for exact 95% CI calculation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# PUBLICATION-QUALITY GRAPHICAL SETUP WITH LATEX & SERIF TYPOGRAPHY
# =============================================================================
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 7.0

# =============================================================================
# FUNCTIONS AND MAIN CONFIGURATION
# =============================================================================
BASE_SEED = 42

def calculate_stats(metric_values, confidence=0.95):
    """
    Calculates Mean, Standard Deviation (SD), and 95% Confidence Interval (CI).
    Returns a dictionary with compressed and formatted numerical values.
    """
    metric_values = np.array(metric_values)
    mean_val = np.mean(metric_values)
    sd_val = np.std(metric_values, ddof=1)
    n = len(metric_values)
    
    # Standard Error of the Mean (SEM) and Margin of Error
    sem = sd_val / np.sqrt(n) if n > 1 else 0.0
    margin_err = sem * stats.t.ppf((1 + confidence) / 2., n - 1) if n > 1 else 0.0
    
    ci_lower = mean_val - margin_err
    ci_upper = mean_val + margin_err
    
    return {
        "mean": mean_val,
        "sd": sd_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "formatted_sd": f"{mean_val:.3f} ± {sd_val:.3f}",
        "formatted_ci": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
        "formatted_full": f"{mean_val:.3f} ± {sd_val:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})"
    }

def standardize_patient_id(series):
    """Standardizes patient identifier formats across datasets."""
    def parse_val(v):
        s = str(v).strip().upper()
        s = s.replace("PATIENT", "P").replace("PACIENTE", "P").replace(" ", "")
        if s.startswith("P"):
            m = re.match(r"P0*(\d+.*)", s)
            if m:
                return "P" + m.group(1)
        return s
    return series.apply(parse_val)

def load_excel_robust(file_path):
    """Loads Excel files with native fallback support for XML inconsistencies."""
    try:
        return pd.read_excel(file_path)
    except Exception:
        with zipfile.ZipFile(file_path, 'r') as z:
            sheet_xml = z.read('xl/worksheets/sheet1.xml')
            shared_strings_xml = z.read('xl/sharedStrings.xml') if 'xl/sharedStrings.xml' in z.namelist() else None

        strings = []
        if shared_strings_xml:
            tree_ss = ET.fromstring(shared_strings_xml)
            for elem in tree_ss.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                text = "".join([t.text for t in elem.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if t.text])
                strings.append(text)

        tree_sheet = ET.fromstring(sheet_xml)
        rows_data = []
        ns = {'sm': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

        for row in tree_sheet.findall('.//sm:row', ns):
            row_dict = {}
            for cell in row.findall('sm:c', ns):
                r = cell.attrib.get('r')
                col_letter = ''.join([c for c in r if c.isalpha()])
                t = cell.attrib.get('t')
                v_elem = cell.find('sm:v', ns)
                val = v_elem.text if v_elem is not None else None
                
                if t == 's' and val is not None:
                    val = strings[int(val)]
                elif val is not None:
                    try:
                        val = float(val) if '.' in val else int(val)
                    except ValueError:
                        pass
                row_dict[col_letter] = val
            rows_data.append(row_dict)

        df = pd.DataFrame(rows_data)
        def col_to_num(col_str):
            num = 0
            for c in col_str:
                num = num * 26 + (ord(c) - ord('A') + 1)
            return num

        cols = sorted(df.columns, key=col_to_num)
        df = df[cols]
        df.columns = df.iloc[0]
        return df[1:].reset_index(drop=True)

def evaluate_and_plot_pipeline(
    X_data,
    y_true,
    label,
    color,
    outer_cv,
    results_summary,
    linestyle="-",
    show_sd_shade=False,
    filter_nonzero=False,
):
    acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []
    tprs = []
    base_fpr = np.linspace(0, 1, 100)

    model_id = label.split(":")[0].strip()

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(X_data, y_true)
    ):
        X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
        y_train, y_test = y_true[train_idx], y_true[test_idx]

        if filter_nonzero:
            nonzero_rois = X_train.columns[(X_train != 0).all(axis=0)]
            X_train = X_train[nonzero_rois]
            X_test = X_test[nonzero_rois]

        n_cols = X_train.shape[1]

        # M1, M2, and M3: No Feature Selection
        if model_id in ["M1", "M2", "M3"]:
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(random_state=BASE_SEED, class_weight="balanced", n_jobs=1))
            ])
            param_grid = {
                "rf__n_estimators": [100, 200],
                "rf__max_depth": [3, 5],
                "rf__min_samples_split": [2, 4],
                "rf__max_features": ["sqrt", None]
            }
        else:
            # Models with Connectomics (M4, M5, M6)
            max_features_sfs = min(10, n_cols)
            candidate_n_features = [3, 5, 8, 10]
            candidate_n_features = [n for n in candidate_n_features if n <= max_features_sfs]
            if not candidate_n_features:
                candidate_n_features = [max_features_sfs]

            candidate_k_anova = [15, 25, 50]
            candidate_k_anova = [k for k in candidate_k_anova if k <= n_cols]
            if not candidate_k_anova:
                candidate_k_anova = [n_cols]

            anova_selector = SelectKBest(score_func=f_classif)

            sfs_selector = SequentialFeatureSelector(
                estimator=RandomForestClassifier(
                    n_estimators=30,
                    max_depth=4,
                    random_state=BASE_SEED,
                    class_weight="balanced",
                    n_jobs=1,
                ),
                direction="forward",
                scoring="f1_macro",
                cv=5,
                n_jobs=1,
            )

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("prefilter", anova_selector),
                ("selector", sfs_selector),
                ("rf", RandomForestClassifier(random_state=BASE_SEED, class_weight="balanced", n_jobs=1)),
            ])

            param_grid = {
                "prefilter__k": candidate_k_anova,
                "selector__n_features_to_select": candidate_n_features,
                "rf__n_estimators": [100, 200],
                "rf__max_depth": [3, 5],
                "rf__min_samples_split": [4, 8],
                "rf__max_features": ["sqrt"],
            }

        inner_cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=BASE_SEED + fold_idx
        )

        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_probs = best_model.predict_proba(X_test)[:, 1]

        acc_list.append(accuracy_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_list.append(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_list.append(f1_score(y_test, y_pred, average="macro", zero_division=0))

        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc_val = auc(fpr, tpr)
        auc_list.append(auc_val)

        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    mean_tprs = np.mean(tprs, axis=0)
    mean_tprs[0] = 0.0
    std_tprs = np.std(tprs, axis=0)

    # Statistical Calculation for AUC (Mean, SD, and 95% CI)
    auc_stats = calculate_stats(auc_list)
    acc_stats = calculate_stats(acc_list)
    prec_stats = calculate_stats(prec_list)
    rec_stats = calculate_stats(rec_list)
    f1_stats = calculate_stats(f1_list)

    # Plot ROC curve including AUC Mean ± SD and 95% CI in the legend label
    plt.plot(
        base_fpr,
        mean_tprs,
        color=color,
        linestyle=linestyle,
        label=rf"{label} ($\text{{AUC}} = {auc_stats['mean']:.2f} \pm {auc_stats['sd']:.2f}, \text{{95\% CI: [{auc_stats['ci_lower']:.2f}-{auc_stats['ci_upper']:.2f}]}}$)",
        lw=2.0,
    )

    if show_sd_shade:
        tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        tprs_lower = np.maximum(mean_tprs - std_tprs, 0)
        tprs_lower[0] = 0.0
        plt.fill_between(
            base_fpr, tprs_lower, tprs_upper, color=color, alpha=0.10
        )

    # Enriched record for saving to CSV
    results_summary.append(
        {
            "Model ID": model_id,
            "Model Name": label,
            
            # Formatted text for reports
            "Accuracy (Mean ± SD [95% CI])": f"{acc_stats['formatted_sd']} {acc_stats['formatted_ci']}",
            "Precision Macro (Mean ± SD [95% CI])": f"{prec_stats['formatted_sd']} {prec_stats['formatted_ci']}",
            "Recall Macro (Mean ± SD [95% CI])": f"{rec_stats['formatted_sd']} {rec_stats['formatted_ci']}",
            "F1-Score Macro (Mean ± SD [95% CI])": f"{f1_stats['formatted_sd']} {f1_stats['formatted_ci']}",
            "AUC (Mean ± SD [95% CI])": f"{auc_stats['formatted_sd']} {auc_stats['formatted_ci']}",
            
            # Numeric values breakdown
            "AUC Mean": round(auc_stats["mean"], 4),
            "AUC SD": round(auc_stats["sd"], 4),
            "AUC CI Lower 95%": round(auc_stats["ci_lower"], 4),
            "AUC CI Upper 95%": round(auc_stats["ci_upper"], 4),
            
            "F1 Mean": round(f1_stats["mean"], 4),
            "F1 SD": round(f1_stats["sd"], 4),
            "F1 CI Lower 95%": round(f1_stats["ci_lower"], 4),
            "F1 CI Upper 95%": round(f1_stats["ci_upper"], 4),
            
            "Accuracy Mean": round(acc_stats["mean"], 4),
            "Accuracy SD": round(acc_stats["sd"], 4),
            "Accuracy CI Lower 95%": round(acc_stats["ci_lower"], 4),
            "Accuracy CI Upper 95%": round(acc_stats["ci_upper"], 4),
        }
    )

# =============================================================================
# EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    working_dir = r"C:\Users\pfran\Desktop\Connectomics\Connectomics Github\Dataset"
    os.makedirs(working_dir, exist_ok=True)

    combined_path = os.path.join(working_dir, "combined_patient_data.csv")
    patients_excel_path = os.path.join(working_dir, "Patients_Project - Copy.xlsx")
    radiomics_path = os.path.join(working_dir, "radiomics_with_classes_cleaned.csv")

    df_features = pd.read_csv(combined_path)
    df_patients = load_excel_robust(patients_excel_path)
    df_radiomics = pd.read_csv(radiomics_path)

    df_patients.columns = df_patients.columns.str.lower()

    df_features["std_id"] = standardize_patient_id(df_features["Patient_ID"])
    df_patients["std_id"] = standardize_patient_id(df_patients["id"])
    df_radiomics["std_id"] = standardize_patient_id(df_radiomics["Subjects"])

    lobar_cols = ["frontal_lobulo", "parietal_lobulo", "temporal_lobulo", "insular_lobulo"]
    for col in lobar_cols:
        if col in df_patients.columns:
            df_patients[col] = pd.to_numeric(df_patients[col], errors="coerce").fillna(0).astype(int)
        else:
            df_patients[col] = 0

    df_patients_clean = df_patients.drop_duplicates(subset=["std_id"]).dropna(subset=["std_id"])
    df_radiomics_clean = df_radiomics.drop_duplicates(subset=["std_id"]).dropna(subset=["std_id"])

    volume_cols = [
        col for col in df_radiomics_clean.columns if "volume" in col.lower() or "mesh" in col.lower()
    ]
    selected_vol_col = volume_cols[0]
    df_vol = df_radiomics_clean[["std_id", selected_vol_col]].rename(
        columns={selected_vol_col: "volume"}
    )

    cols_to_drop_pre_merge = [
        c for c in ["edad", "sexo", "hemisferio", "grado", "histologia", "age", "sex", "grade"] + lobar_cols if c in df_features.columns
    ]
    df_features_base = df_features.drop(columns=cols_to_drop_pre_merge, errors="ignore")

    df = pd.merge(df_features_base, df_patients_clean, on="std_id", how="inner")
    df = pd.merge(df, df_vol, on="std_id", how="inner")

    grade_col = "grade" if "grade" in df.columns else "grado"
    if grade_col in df.columns:
        df[grade_col] = pd.to_numeric(df[grade_col], errors="coerce")
        df = df.dropna(subset=[grade_col])

    print(f"Total Synchronized Subjects/Records: N = {len(df)}")

    y = df[grade_col].apply(lambda x: 1 if x > 2 else 0).values

    age_col = [c for c in ["edad", "age"] if c in df.columns][0]
    sex_col = [c for c in ["sexo", "sex"] if c in df.columns][0]
    hemis_col = [c for c in ["hemisferio", "hemisphere"] if c in df.columns]

    metadata_cols = [
        "id", "Patient_ID", "clean_id", "std_id", "grado", "grade",
        "histologia", "Subjects", "labels", "Unnamed: 0"
    ]

    all_clinical_possible = [age_col, sex_col, "volume"] + hemis_col + lobar_cols
    cat_cols_m3 = [c for c in [sex_col] + hemis_col if c in df.columns]

    df_encoded = pd.get_dummies(df, columns=cat_cols_m3, drop_first=True)

    m1_cols = [c for c in df_encoded.columns if c.startswith(sex_col) or c == age_col]
    X_M1 = df_encoded[m1_cols].astype(float)

    m2_cols = m1_cols + ["volume"]
    X_M2 = df_encoded[m2_cols].astype(float)

    m3_extra_cols = [c for c in df_encoded.columns if any(c.startswith(h) for h in hemis_col)] + lobar_cols
    m3_cols = m2_cols + [c for c in m3_extra_cols if c in df_encoded.columns]
    X_M3 = df_encoded[m3_cols].astype(float)

    non_connectomic_cols = set(metadata_cols + all_clinical_possible + list(df_patients.columns) + list(df_radiomics.columns))
    connectomic_cols = [c for c in df.columns if c not in non_connectomic_cols and np.issubdtype(df[c].dtype, np.number)]
    
    # -------------------------------------------------------------------------
    # MODIFICACIÓN SOLICITADA EN M4:
    # Se eliminan las variables de conectómica cuya media sea igual a 0.
    # -------------------------------------------------------------------------
    X_M4 = df[connectomic_cols].astype(float)
    X_M4 = X_M4.loc[:, X_M4.mean(axis=0) != 0]

    X_M5 = X_M4.copy()

    X_M6 = pd.concat([X_M3, X_M4], axis=1)

    print(f"\nFeature dimensions for each model:")
    print(f"M1: {X_M1.shape[1]} features | M2: {X_M2.shape[1]} features | M3: {X_M3.shape[1]} features")
    print(f"M4: {X_M4.shape[1]} features | M5: {X_M5.shape[1]} features | M6: {X_M6.shape[1]} features\n")

    outer_cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=BASE_SEED
    )

    results_summary = []
    plt.figure(figsize=(9, 7.5))

    print("Evaluating M1: Demographic...")
    evaluate_and_plot_pipeline(
        X_M1, y, "M1: Demographic", "#E63946", outer_cv, results_summary, linestyle="--"
    )

    print("Evaluating M2: Demographic + Tumor Volume...")
    evaluate_and_plot_pipeline(
        X_M2, y, "M2: Demographic + Tumor Volume", "#F4A261", outer_cv, results_summary, linestyle="--"
    )

    print("Evaluating M3: Clinical + Volume + Location...")
    evaluate_and_plot_pipeline(
        X_M3, y, "M3: Clinical + Volume + Location", "#E76F51", outer_cv, results_summary
    )

    print("Evaluating M4: Pure Connectomics...")
    evaluate_and_plot_pipeline(
        X_M4, y, "M4: Pure Connectomics", "#2A9D8F", outer_cv, results_summary
    )

    print("Evaluating M5: Non-zero Connectomics...")
    evaluate_and_plot_pipeline(
        X_M5, y, "M5: Non-zero Connectomics", "#9B59B6", outer_cv, results_summary, linestyle=":", filter_nonzero=True
    )

    print("Evaluating M6: Clinical + Connectomics...")
    evaluate_and_plot_pipeline(
        X_M6, y, "M6: Clinical + Connectomics", "#264653", outer_cv, results_summary
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#8D99AE",
        label=r"Random Guessing ($\text{AUC} = 0.50$)",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title(
        rf"\textbf{{Model Hierarchy Comparison (ANOVA Pre-selection) (N = {len(df)})}}",
        pad=12,
    )
    plt.legend(loc="lower right", fontsize=7.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_fig_path = os.path.join(
        working_dir, "anova_model_hierarchy_roc_comparison.png"
    )
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure successfully saved to: {output_fig_path}")

    plt.show()
    plt.close()

    # Save metrics table to CSV
    results_df = pd.DataFrame(results_summary)
    output_csv_path = os.path.join(
        working_dir, "anova_model_hierarchy_performance_metrics.csv"
    )
    results_df.to_csv(output_csv_path, index=False)
    print(f"Metrics with SD & 95% CI successfully saved to: {output_csv_path}\n")
    print("Performance Metrics Summary:")
    print(results_df[["Model ID", "Model Name", "AUC (Mean ± SD [95% CI])", "F1-Score Macro (Mean ± SD [95% CI])"]].to_string(index=False))
