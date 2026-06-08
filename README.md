# Quantifying White Matter Disruption in Gliomas via Tumor-Masked Structural Connectomics and Explainable Machine Learning: A Pilot Study 

> **Note for Reviewers:** This repository hosts the official computational framework and reproducible workflows corresponding to the abstract submitted to the **Nueroradiology**

![Tractografía del Paciente 1](Images/Paciente1wm_dti.fib.gif)

> **Note:** This visualization is a video made in **DSI Studio** to show the global tractography data.

This repository contains the official **explainable machine learning (ML) pipeline** for classifying glioma malignancy grades (Low-Grade Glioma (LGG) vs. High-Grade Glioma (HGG)) and stratifying tract-specific white matter (WM) disruption, using personalized, tumor-masked structural connectivity networks (connectograms).

> ### **Authors**
> **Pamela Franco** [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--7629--3653-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0001-7629-3653)¹
> **Cristian Montalba** [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--3370--0233-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0003-3370-0233)²'³'⁴
> **Ignacio Espinoza** [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--2400--4498-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0003-2400-4498)⁵
> **M. Daniela Cornejo** [![ORCID](https://img.shields.io/badge/ORCID-0009--0003--0425--5721-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0003-0425-5721)⁵'⁶
> **Francisco Torres** [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--0003--2446-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0002-0003-2446)²'⁷
> **Carlos Bennett** [![ORCID](https://img.shields.io/badge/ORCID-0009--0007--1434--273X-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0007-1434-273X)⁸
> **Steren Chabert** [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--2890--5077-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0002-2890-5077)²'⁹'¹⁰
> **Rodrigo Salas** [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--0350--6811-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0002-0350-6811)²'⁸'⁹

>(¹Faculty of Engineering / Universidad Andrés Bello, ²Biomedical Imaging Center / School of Medicine / Pontificia Universidad Católica de Chile, ³Millennium Institute for Intelligent Healthcare Engineering - iHEALTH, ⁴Radiology Department / School of Medicine / Pontificia Universidad Católica de Chile, ⁵Institute of Physics / Pontificia Universidad Católica de Chile, ⁶Department of Psychiatry / School of Medicine / Pontificia Universidad Católica de Chile, ⁷Radiology Department / Hospital Carlos Van Buren, ⁸Neurosurgery Department / Hospital Carlos Van Buren, ⁹Biomedical Engineering School / Faculty of Engineering/ Universidad de Valparaíso, ¹⁰Center of Interdisciplinary Biomedical and Engineering Research for Health - MEDING / Universidad de Valparaíso)*

The present proof-of-concept study aims to demonstrate an explainable ML pipeline utilizing personalized structural connectivity networks to stratify glioma malignancy grades (LGG, WHO 2 vs. HGG, WHO: 3-4) and map associated WM disruption. By dynamically modifying standardized anatomical templates to strictly exclude tumor-encroached regions, we construct individualized structural connectivity matrices, or connectograms, via advanced probabilistic tractography. Through this approach, we seek to isolate specific macrostructural and graph-theoretic biomarkers capable of discriminating between tumor grades, ultimately providing a highly precise, individualized computational framework for presurgical planning and therapeutic monitoring.

---

## Pipeline Overview

The pipeline implements an advanced neuroimaging and ML workflow:
1. **Diffusion Preprocessing**: Geometry/motion correction via Gaussian Processes and brain masking.
2. **Tumor-Masked Atlas Adaptation**: Voxel-wise intersection and automatic exclusion of tumor-encroached regions from the JHU ICBM-DTI-81 atlas to avoid confounding signals.
3. **Probabilistic Tractography**: Fiber orientation modeling via BEDPOSTX (up to 2 crossing fibers per voxel) and network streamline propagation via ProbtrackX2.
4. **Connectomic Graph Construction & Multi-Level Profiling:** Structural connectivity matrices are mapped using fiber streamline counts between regions of interest (ROIs) using the standard JHU White-Matter Tractography Atlas ($50$ predefined anatomical tracts). These raw adjacency matrices are programmatically analyzed via a specialized automated MATLAB framework to construct individual brain graphs and compute a comprehensive, multi-level topological profile. For each patient, the pipeline extracts and structures parameters across three independent levels: Construction descriptors (Matrix Size, Max/Mean Weights), Global topological network metrics (Edge counts, Density, Total Weight, Global Efficiency), and Local nodal metrics (Nodal Strength, Nodal Degree, PageRank, Closeness Centrality, Betweenness Centrality, and Clustering Coefficient) calculated individually for each of the $50$ ROIs. The resulting high-dimensional topological fingerprints are consolidated and exported into an integrated master dataset ($35$ Patients $\times$ $307$ Columns) for downstream machine learning pipelines.
5. **Two-Stage Feature Selection & Stratified Repeated Nested CV:** The high-dimensional structural connectivity vectors ($p = 307$ features) are processed through an exploratory statistical screening and an optimization framework. First, zero-variance constant features are programmatically pruned. In **Stage 1**, a univariate ANOVA $F$-test (`SelectKBest` via `f_classif`) evaluates the statistical significance of mean differences across target classes on the training split, truncating the search space down to the top $100$ features. In **Stage 2**, a wrapper-based Forward Sequential Feature Selection (SFS) loop driven by an optimized `RandomForestClassifier` isolates the absolute minimal, non-redundant optimal subset of graph metrics. Finally, the generalizability of the optimized subspace is evaluated using a robust **Repeated Nested 5-Fold Stratified CV** framework ($10$ independent random repetitions, $50$ distinct outer evaluation folds) with hyperparameter optimization (`GridSearchCV`) isolated in the inner loops. Predictive insights are complemented by game-theoretic tree-based interpretability through SHAP summary mapping and multi-dimensional t-SNE space analysis.
---

## Repository Contents

Given the tractography and structural network focus of this manuscript, the repository structure is organized as follows:


```text
├── Codes/
│   ├── ML_Pipeline.ipynb                      # Explainable ML pipeline for connectome classification via SFS, Random Forest, and SHAP.
│   └── requirements.txt                       # Required Python packages and dependencies.
├── Dataset/
└── dataset_conectomica_with_labels.csv        # Multi-level connectomic graph features from JHU atlas (307 features/patient).
├── Results/                                   # Automatically generated pipeline outputs and diagnostics.
│   ├── Data Leakage.png                       # Comparative analysis showing the effect of data leakage on validation metrics.
│   ├── HGG_Connectogram.png                   # Average connectogram profile for High-Grade Glioma cohort.
│   ├── LGG_Connectogram.png                   # Average connectogram profile for Low-Grade Glioma cohort.
│   ├── accuracy_vs_features.png               # SFS trajectory optimization curve mapping Accuracy vs. feature subsets.
│   ├── clustermap.png                         # Dual-axis hierarchical correlation matrix with average-linkage cluster boundaries.
│   ├── dendogram.png                          # Hierarchical clustering tree mapping structural feature redundancies (Threshold = 6.5).
│   ├── feature_importance_classification.png  # Gini feature importance weights from the SFS-selected minimal subset.
│   ├── logistic_feature_importance.png        # Initial screening weights computed using absolute coefficients from L2-Logistic Regression.
│   ├── logistic_feature_importance_pareto_chart.png # Pareto distribution ranking of the Top 20 features against the 80% threshold.
│   ├── rf_3d_feature_space_topology.png       # Three-dimensional topological mapping of the isolated optimal feature subspace.
│   ├── rf_feature_space_interaction.png       # Scatter plot mapping the interaction space of the top 2 selected features.
│   ├── shap_summary_plot_classification.png   # SHAP summary plot mapping local tree-based feature attributions.
│   ├── shap_waterfall_classification.png      # SHAP waterfall plot illustrating local additive attribution for clinical cases.
│   ├── tsne_after_selection.png               # Low-dimensional optimized t-SNE embedding of the SFS selected subspace.
│   └── tsne_before_selection.png              # High-dimensional t-SNE embedding of the entire filtered feature space.
└── README.md                                  # Project documentation and laboratory guidelines.

```

---

## Methods Summary

- **Modalities**: T1-weighted MRI and Diffusion Tensor Imaging (DTI: 25 non-collinear directions, $b = 1000\text{ s/mm}^2$).
- **Subjects**: 35 glioma patients from a single Chilean tertiary center (LGG $58.3$\% and HGG $41.7$\% confirmed via histopathology).
- **Hardware Setup**: Executed via CPU-based parallel processing on a 13th-generation Intel Core i7 architecture (24 GB RAM) and accelerated using an NVIDIA GPU (6 GB VRAM).

## Connectome Construction & Graph Theory

* **Node Definition ($V$):** Remaining non-invaded white matter (WM) structures ($50$ target regions of interest) are derived from the *JHU ICBM-DTI-81 White-Matter Labels Atlas* after applying dynamic, patient-specific tumor masking to exclude tumor-encroached regions.
* **Streamline Propagation & Edge Definition ($E$):** 500 probabilistic samples per voxel are generated via FSL's `ProbtrackX2` to construct customized streamline count-based structural connectivity adjacency matrices.
* **Network Metrics & Feature Extraction:** The structural brain graphs are processed using `NetworkX` and MATLAB to extract a comprehensive multi-level profile ($p = 307$ total topological features per patient after removing constant identifiers):
  * *Raw Edge Descriptors:* Symmetrized connectivity matrix coefficients.
  * *Global Network Topology:* Global efficiency, network density, transitivity, and average clustering coefficient.
  * *Local Nodal Characterization:* ROI-specific metrics including node degree, betweenness centrality, closeness centrality, nodal strength, clustering coefficient, and PageRank scores to map precise spatial WM integrity.

---

## Machine Learning Pipeline Architecture

The machine learning core is engineered to handle the high-dimensional structural connectivity vectors derived from personalized, tumor-masked brain graphs.

### 1. Two-Stage Feature Selection and Dimensionality Reduction
* **Global Standardization & Data Pruning:** Features derived from the structural connectivity matrices are globally scaled using a standard Z-score transformation (`StandardScaler`) prior to data splitting. To prevent downstream mathematical instability during correlation matrix generation, zero-variance constant features are automatically identified and removed.
* **Hierarchical Collinearity Mapping:** Features are systematically analyzed using average linkage and Euclidean distance. The pipeline maps and visualizes topographical collinearity and structural redundancy among network edges using a predefined cophenetic distance threshold of $6.5$.
* **Stage 1 (Univariate ANOVA Filtering):** A univariate ANOVA $F$-test (`SelectKBest` via `f_classif`) is applied across the aggregated training partition ($80\%$ of the dataset). This stage filters out noisy, non-informative variables by evaluating the significance of mean differences across discrete classes, truncating the feature search space down to the top $100$ most discriminative metrics.
* **Stage 2 (Forward Sequential Feature Selection):** A wrapper-based stepwise feature selection loop is executed on the $100$ pre-filtered features over a 5-fold cross-validation (CV) split on the training data. Driven by an optimized base Random Forest (RF) Classifier, features are added sequentially up to 15 combinations to isolate the absolute minimal, non-redundant optimal subset of structural connectivity metrics that maximizes classification accuracy. This extracts a single optimized subspace ($\text{Argmax}[\text{Accuracy}]$) which is frozen for downstream evaluation.

### 2. Models Evaluated
* **Linear Exploratory Screening:** `LogisticRegression` with $L_2$ regularization is executed during the initial global screening phase to extract absolute log-odds coefficient weights and build a Pareto chart of the top 20 features against an 80% cumulative threshold.
* **Primary Predictive Ensemble:** `RandomForestClassifier` serves as the core classification model optimized and validated within the CV framework.

### 3. Model Selection, Hyperparameter Tuning & Validation
* **Baseline Grid Optimization:** Prior to the Sequential Feature Selection (SFS) loop, `GridSearchCV` is implemented over a 5-fold stratified CV split on the training data to systematically evaluate combinations of tree-based hyperparameters—including the number of estimators (`n_estimators`), maximum tree depth (`max_depth`), minimum splitting criteria (`min_samples_split`), and maximum feature sub-sampling strategies (`max_features`).
* **Repeated Nested 5-Fold Stratified CV:** Implemented with $10$ independent random repetitions (generating $50$ distinct outer evaluation folds) to ensure strict class-balance control. In this framework, the outer loops evaluate stable model generalizability over the optimal feature subspace, while the inner loops (`GridSearchCV`) isolate hyperparameter tuning **strictly over the pre-selected static feature subspace** obtained from the SFS phase, rather than re-running the feature selection wrapper within each individual outer fold split.
* **Classification Performance Metrics:** Diagnostic robustness is tracked across all $50$ folds using Accuracy, Precision, Sensitivity (Recall), and F1-Score (calculated via macro-averaging for robust class-balance control).

### 4. Post-Hoc Data Leakage Analysis
To guarantee the scientific integrity and reproducibility of the results, the pipeline includes a strict post-hoc diagnostic test designed to quantify and detect potential selection bias (*data leakage*).

The test isolates and compares two computational setups across a 5-fold Stratified Cross-Validation scheme to compute the Discrepancy Delta ($\Delta_{\text{leakage}}$) on the Macro F1-Score:

* **Potential Leakage Setup (Biased Framework):** Evaluates the performance of the Random Forest model using the features pre-selected by the global workflow. Since the feature selection was frozen on a static partition before the outer loops, it maps how much the model could benefit from knowing properties of the underlying dataset distribution beforehand.
* **Strict Isolation Setup (Unbiased Framework):** Executes a fully contained pipeline completely from scratch inside each individual cross-validation fold. The univariate ANOVA filter and the Forward Sequential Feature Selection wrapper are executed strictly on the training partition of the active fold, completely hiding the respective validation split.

The pipeline automatically prints out the performance discrepancies. A delta ($\Delta_{\text{leakage}} \le 0.05$) mathematically demonstrates that the feature selection space captures stable, objective structural connectivity biomarkers rather than artificial statistical dependencies, confirming a valid low-dimensional optimized subspace

### 5. Interpretability and Visual Analytics
* **Explainable AI (SHAP Analysis):** Implemented using a specialized tree-based explainer (`TreeExplainer`) applied directly to the test partitions ($20\%$ of the dataset). It quantifies game-theoretic feature contributions, extracting additive feature attribution values (SHAP values) to map how specific structural graph edges drive predictions toward targeted pathological profiles.
* **Feature Space Projections:** High-dimensional transformations are mapped before and after feature selection via t-SNE embeddings to visually confirm class segregation within the optimized low-dimensional subspace. Additionally, scatter plots track the empirical interaction space of the top selected features driving the RF decisions.

---

![Connectomica](Images/Figure1.png)

> **Methodological workflow for glioma stratification and white matter (WM) vulnerability mapping:** Stage 1: Input Data and preprocessing. Integration of multimodal MRI data. (a) Anatomical T1-weighted imaging and (b) DWI data were acquired, co-registered to perform brain extraction, and utilized to compute DTI maps. (c) Personalized tumor masks are manually segmented to delineate neoplastic boundaries. (d) All structural and diffusion datasets are aligned with the Johns Hopkins University (JHU) WM atlas in MNI space to establish anatomical standardization across the cohort. (e) Spatial normalization to atlas space. Subject-specific images were co-registered to a JHU WM atlas. Stage 2: Tumor-masked tractography. Network node definition and structural connectivity matrix construction. (a) Precise mapping and exclusion of tumor-encroached regions are performed to eliminate confounding infiltrative artifacts. (b) Advanced probabilistics diffusion modeling (BEDPOSTX) evaluates fiber tract configurations, generating a 3D structural connectome. (c) Personalized adjacency matrices are built based on streamline connection probabilities. (d) Multi-level feature extraction yields a high-dimensional profile composed of m = 307 continuous graph-theoretical metrics. Stage 3: Explainable ML strategy. (a) Feature selection using a forward sequential feature selection (SFS). (b) Hyperparameter optimization and classification using a random forest ensemble validate via repeated nested 5-fold stratified cross-validation. (c) Model transparency and localized feature attributions are computed utilizing tree SHAP values to quantify tract-specific WM disruptions.


## Empirical Results & Performance Summary

* **Demographic & Clinical Baseline:** Demographic analysis demonstrated a median cohort age of 43 years, with no significant differences in age or sex between groups. Glioma grade was significantly associated with hemispheric tumor location ($p = 0.038$), showing left-hemisphere predominance in LGG and right-hemisphere predominance in HGG.
* **Feature Redundancy & Selection Trajectory:** Hierarchical clustering of the unreduced feature space ($p = 307$) confirmed substantial information redundancy, partitioning the variables into three highly collinear modules. During Forward Sequential Feature Selection (SFS) expansion, cross-validated training accuracy peaked at exactly three features, achieving an optimal cross-validated accuracy of $96.00 \pm 8.00$ %. Expanding beyond this optimal subset caused a strict performance plateau followed by sudden variance instability, defining the onset of high-dimensional noise degradation.
* **The Optimal Connectomic Subspace:** The final isolated 3-feature subset capable of discriminating malignancy grades comprised:
  1. Node strength of the right retrolenticular part of the internal capsule.
  2. Clustering coefficient of the right fornix/crescent stria terminalis.
  3. Clustering coefficient of the middle cerebellar peduncle.
* **Post-Hoc Data Leakage Validation:** The robust data leakage analysis proved the mathematical necessity of the isolated nested framework. Standard non-nested validation severely overfitted the glioma cohort, artificially overestimating the model's predictive capacity by a Macro F1-score delta of $0.1144$ ($0.7537$ in the biased setup vs. the true $0.6392$ under strict isolation).
* **Game-Theoretic Interpretability:** The implementation of tree-based SHAP values dismantled the algorithmic "black box" by establishing a stable, monotonic alignment between white matter integrity metrics, led primarily by the right fornix crescent stria terminalis, and the model's clinical classifications, ensuring explicit, tract-specific neuroradiological interpretability.
---

## Reproducibility & Data Availability

- **Code**: The ML pipelines script is hosted in this repository.
- **Data Privacy**: MRI datasets and structural connectivity matrices generated during this study are not publicly available due to patient data privacy restrictions imposed by the Ethics Committee of the Servicio de Salud Valparaíso San Antonio (ORD.001413).
- **Clinical Inquiries**: Anonymized data may be made available upon reasonable request and subject to institutional approval by contacting **Steren Chabert (steren.chabert@uv.cl)**.

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/pamelaFranco/glioma-ml-tractography.git](https://github.com/pamelaFranco/glioma-ml-tractography.git)
   cd glioma-ml-tractography
   ```

2. **Set up neuroimaging environment**: Ensure you have FSL (v6.0) installed and configured in your environment path.

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. Execute the connectomics pipeline:  
   ```bash
   # Run machine learning pipeline
   python ML_pipeline.py
   ```
  
---

## Acknowledgements
This work was supported by the National Agency for Research and Development (ANID) of Chile through:

* FONDECYT N°1221938 ("An Explainable Deep Neuro-Fuzzy Inference System for the segmentation of BT in multi-contrast magnetic resonance imaging").

* ANID Millennium Science Initiative Program ICN2021_004 (Millennium Institute for Intelligent Healthcare Engineering - iHEALTH).

* Additionally, this work was funded by the Endowment I+D in Health Competition of the Universidad Andrés Bello (UNAB) 2025, project No. DI-07-25/ICS.

---
## Citation

If you find this pipeline useful for your research, please cite our preliminary work prepared for the ESMRMB 2026 conference:

```bibtex
@inproceedings{montalba2026quantifying,
  title={Feasibility of Tumor-Masked Structural Connectomics and Explainable Machine Learning for Assessing White Matter Disruption in Gliomas: A Pilot Study},
  author={Montalba, Cristian and Espinoza, Ignacio and Cornejo, M. Daniela and Torres, Francisco and Bennett, Carlos and Chabert, Steren trade and Salas, Rodrigo and Franco, Pamela},
  booktitle={Submitted to the 43rd Annual Scientific Meeting of the European Society for Magnetic Resonance in Medicine and Biology (ESMRMB 2026)},
  year={2026},
  note={Abstract under review}
}
```

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)