# Tumor-Masked Structural Connectomics for Assessing Tract-Specific White Matter Disruption Associated with Glioma Aggressiveness: An Interpretable Machine Learning Pilot Study 

> **Note for Reviewers:** This repository hosts the official computational framework and reproducible workflows corresponding to the abstract submitted to the **Nueroradiology**

![Tractografía del Paciente 1](Images/Paciente1wm_dti.fib.gif)

> **Note:** This visualization is a video made in **DSI Studio** to show the global tractography data.

This repository contains the official **interpretable machine learning (ML) pipeline** for classifying glioma malignancy grades (Low-Grade Glioma (LGG) vs. High-Grade Glioma (HGG)) and stratifying tract-specific white matter (WM) disruption, using personalized, tumor-masked structural connectivity networks (connectograms).

### Authors & Affiliations

<p align="left">
  <strong>Pamela Franco</strong> <a href="https://orcid.org/0000-0001-7629-3653"><img src="https://img.shields.io/badge/ORCID-0000--0001--7629--3653-A6CE39?logo=orcid&logoColor=white&style=flat-square" height="16"></a><br>
  <small>• Faculty of Engineering, Universidad Andrés Bello, Santiago, Chile.</small>
</p>

<p align="left">
  <strong>Cristian Montalba</strong> <a href="https://orcid.org/0000-0003-3370-0233"><img src="https://img.shields.io/badge/ORCID-0000--0003--3370--0233-A6CE39?logo=orcid&logoColor=white&style=flat-square" height="16"></a><br>
  <small>• Biomedical Imaging Center / Radiology Department, Pontificia Universidad Católica de Chile<br>•Millennium Institute for Intelligent Healthcare Engineering (iHEALTH)</small>
</p>

<p align="left">
  <strong>Ignacio Espinoza</strong> <a href="https://orcid.org/0000-0003-2400-4498"><img src="https://img.shields.io/badge/ORCID-0000--0003--2400--4498-A6CE39?logo=orcid&logoColor=white&style=flat-square" height="16"></a><br>
  <small>• Institute of Physics, Pontificia Universidad Católica de Chile.</small>
</p>

<p align="left">
  <strong>M. Daniela Cornejo</strong> <a href="https://orcid.org/0009-0003-0425-5721"><img src="https://img.shields.io/badge/ORCID-0009--0003--0425--5721-A6CE39?logo=orcid&logoColor=white&style=flat-square" height="16"></a><br>
  <small>• Institute of Physics / Department of Psychiatry, Pontificia Universidad Católica de Chile.</small>
</p>

<p align="left">
  <strong>Francisco Torres</strong> <a href="https://orcid.org/0000-0002-0003-2446"><img src="https://img.shields.io/badge/ORCID-0000--0002--0003--2446-A6CE39?logo=orcid&logoColor=white&style=flat-square" height="16"></a><br>
  <small>• Radiology Department, Hospital Carlos Van Buren</small>
</p>

<p align="left">
  <strong>Carlos Bennett</strong> <a href="https://orcid.org/0009-0007-1434-273X"><img src="https://img.shields.io/badge/ORCID-0009--0007--1434--273X-A6CE39?logo=orcid&logoColor=white&style=flat-square" height="16"></a><br>
  <small>• Neurosurgery Department, Hospital Carlos Van Buren</small>
</p>

<p align="left">
  <strong>Steren Chabert</strong> <a href="https://orcid.org/0000-0002-2890-5077"><img src="https://img.shields.io/badge/ORCID-0000--0002--2890--5077-A6CE39?logo=orcid&logoColor=white&style=flat-square" height="16"></a><br>
  <small>• Biomedical Engineering School / Center MEDING, Universidad de Valparaíso<br>• Millennium Institute for Intelligent Healthcare Engineering (iHEALTH)</small>
</p>

<p align="left">
  <strong>Rodrigo Salas</strong> <a href="https://orcid.org/0000-0002-0350-6811"><img src="https://img.shields.io/badge/ORCID-0000--0002--0350--6811-A6CE39?logo=orcid&logoColor=white&style=flat-square" height="16"></a><br>
  <small>• Biomedical Engineering School / Center MEDING, Universidad de Valparaíso<br>• Millennium Institute for Intelligent Healthcare Engineering (iHEALTH)</small>
</p>

---

To assess the feasibility of personalized tumor-masked structural connectomics combined with interpretable ML for differentiating LGG from HGG, and to identify candidate tract-specific connectomic biomarkers of glioma aggressiveness.


---

## Pipeline Overview

The proposed workflow integrates advanced neuroimaging processing with a rigorous, leakage-free machine learning architecture:

1. **Multimodal MRI Acquisition & Preprocessing**:
   - High-resolution T1-weighted and Diffusion Tensor Imaging (DTI) data ($b = 1000\text{ s/mm}^2$, 25 directions) undergo brain extraction, geometry/motion artifact correction via Gaussian Process modeling, and spatial normalization to standard MNI space.
2. **Tumor-Masked Atlas Adaptation**:
   - Manual 3D tumor segmentation maps in subject space are co-registered and mapped onto the JHU ICBM-DTI-81 White Matter Labels Atlas ($50$ ROIs). Voxel-wise intersection programmatically excludes tumor-invaded regions, preventing false streamline propagation through pathological tissue.
3. **Probabilistic Diffusion Tractography**:
   - Intra-voxel fiber orientation distributions are estimated using **BEDPOSTX** (modeling up to 2 crossing fibers per voxel).
   - **ProbtrackX2** propagates $5,000$ streamline samples per seed voxel to generate personalized 3D structural connectomes and weighted adjacency matrices.
4. **Multi-Level Connectomic Profiling**:
   - Extraction of a comprehensive $296$-feature matrix per subject, encompassing three topological levels:
     - *Matrix Construction & Adjacency Descriptors* (density, edge count, network total weight).
     - *Macro-Topological Global Network Metrics* (global efficiency, modularity, transitivity, characteristic path length).
     - *Nodal & ROI-Specific Local Metrics* (degree, strength, closeness, betweenness, clustering coefficient, PageRank centrality) calculated per JHU tract.
5. **Leakage-Free Feature Selection & Machine Learning**:
   - Features undergo zero-variance pruning and agglomerative hierarchical clustering to evaluate multicollinearity.
   - To eliminate data leakage, feature selection (ANOVA filter combined with Sequential Feature Selection using Random Forest) is embedded strictly **inside** each outer fold of a 5-Fold Stratified Nested Cross-Validation scheme.
6. **Explainable AI (XAI) & Interpretability**:
   - Tree-based SHAP (SHapley Additive exPlanations) values are computed on out-of-sample test folds to quantify the local and global contributions of specific white matter tracts, providing game-theoretic interpretation of structural disruption.
---

## Repository Contents

Given the tractography and structural network focus of this manuscript, the repository structure is organized as follows:


```text
├── Codes/
│   ├── ML_Pipeline.py
│   │   # Main explainable ML pipeline for tumor-masked structural connectomics,
│   │   # including preprocessing, ANOVA filtering, sequential feature selection
│   │   # (SFS), Random Forest classification, nested cross-validation,
│   │   # hyperparameter optimization, sensitivity analysis, and Tree SHAP.
│   │
│   ├── sensitivity_analysis.py
│   │   # Comparative clinical-radiological baseline and integrated-model analysis,
│   │   # evaluating demographic, tumor-volume, anatomical-location, connectomic,
│   │   # non-zero connectomic, and combined feature spaces.
│   │
│   └── requirements.txt
│       # Required Python packages and computational dependencies.
│
├── Dataset/
│   ├── dataset_conectomica_with_labels.csv
│   │   # Tumor-masked structural connectomic dataset containing graph-theoretical
│   │   # features and glioma-grade labels.
│   │
│   ├── radiomics_with_classes_cleaned.csv
│   │   # Cleaned radiomics feature dataset with glioma classification labels.
│   │
│   └── dataset_conectomica_with_patient_details.csv
│       # Structural connectomic features combined with patient-level
│       # clinical and demographic information.
│
├── Results/
│   │
│   ├── Figure_1.png
│   │   # Methodological workflow for glioma stratification and white matter
│   │   # vulnerability mapping. The figure summarizes multimodal MRI
│   │   # preprocessing, tumor-masked tractography, structural connectome
│   │   # construction, feature extraction, SFS, Random Forest classification,
│   │   # nested 5-fold cross-validation, and Tree SHAP interpretation.
│   │   # [Main manuscript Figure 1]
│   │
│   ├── Figure_2.png
│   │   #  Group-level structural connectograms derived from tumor-masked diffusion MRI connectomes. 
│   │   # Circular network representations illustrating the mean structural connectivity patterns of 
│   │   # (a) low-grade gliomas (LGG) and (b) high-grade gliomas (HGG). 
│   │   # [Main manuscript Figure 2]
│   │
│   ├── Figure_3.png
│   │   # Agglomerative hierarchical clustering matrix of multi-level
│   │   # connectomic features. The symmetric heatmap represents Pearson
│   │   # correlations across the 296-feature connectomic profile, with
│   │   # average-linkage/Euclidean dendrograms and three macro-structural
│   │   # feature clusters identified at a cophenetic distance threshold of 6.5.
│   │   # [Main manuscript Figure 3]
│   │
│   ├── Figure_4.png
│   │   # Sequential Feature Selection (SFS) performance curve showing
│   │   # internal cross-validation accuracy as a function of the number
│   │   # of graph-theoretical features. Peak performance was achieved
│   │   # at N = 9 features (accuracy = 0.887).
│   │   # [Main manuscript Figure 4]
│   │
│   ├── Figure_5.png
│   │   # Receiver operating characteristic (ROC) curves comparing the
│   │   # biased pipeline with the strictly isolated pipeline.
│   │   # [Main manuscript Figure 5]
│   │
│   ├── Figure_6.png
│   │   # Global network interpretability using Shapley Additive
│   │   # exPlanations (SHAP), including global feature-importance
│   │   # ranking and SHAP summary scatter visualization.
│   │   # [Main manuscript Figure 6]
│   │
│   ├── Figure_7.png
│   │   # Local interpretability, patient trajectories, and non-linear
│   │   # tract dependence using SHAP decision trajectories.
│   │   # [Main manuscript Figure 7]
│   │
│   ├── Figure_8.png
│   │   # SHAP dependence plots illustrating non-linear relationships
│   │   # for the clustering coefficient of the left medial lemniscus
│   │   # and connectivity strength of the left sagittal stratum.
│   │   # [Main manuscript Figure 8]
│   │
│   ├── Figure_9.png
│   │   # t-Distributed Stochastic Neighbor Embedding (t-SNE) visualization
│   │   # comparing feature-space class segregation between biased and
│   │   # unbiased workflows.
│   │   # [Main manuscript Figure 9]
│   │
│   ├── Supplementary_Figure_1.png
│   │   # Out-of-fold confusion matrices comparing the biased pipeline
│   │   # (global feature selection) with the isolated nested pipeline.
│   │   # [Supplementary Figure 1]
│   │
│   ├── Supplementary_Figure_2.png
│   │   # Non-linear Random Forest decision-boundary surface in a
│   │   # representative two-dimensional feature space defined by
│   │   # Betweenness Centrality of the right posterior thalamic
│   │   # radiation and Strength Centrality of the left sagittal stratum.
│   │   # [Supplementary Figure 2]
│   │
│   ├── Supplementary_Figure_3.png
│   │   # Decision-rule architecture of a representative Random Forest
│   │   # tree, showing hierarchical splitting based on candidate
│   │   # connectomic biomarkers and terminal subgroup allocations.
│   │   # [Supplementary Figure 3]
│   │
│   ├── Supplementary_Figure_4.png
│   │   # Receiver operating characteristic (ROC) curves comparing
│   │   # hierarchical clinical-radiological, structural connectomic,
│   │   # non-zero connectomic, and integrated models (M1–M6).
│   │   # [Supplementary Figure 4]
│   │
│   ├── Table_1.docx
│   │   # Demographic and clinical characteristics of the glioma cohort,
│   │   # comparing Low-Grade Glioma (LGG) and High-Grade Glioma (HGG)
│   │   # patients.
│   │   # [Main manuscript Table 1]
│   │
│   ├── Supplementary_Table_1.docx
│   │   # Stability of graph-theoretical feature selection across the
│   │   # outer cross-validation folds, reporting selection frequency
│   │   # and percentage for each retained metric.
│   │   # [Supplementary Table 1]
│   │
│   ├── Supplementary_Table_2.docx
│   │   # Comparative performance metrics between biased and unbiased
│   │   # evaluation pipelines, including 95% confidence intervals
│   │   # and the absolute performance difference (Δ).
│   │   # [Supplementary Table 2]
│   │
│   ├── Supplementary_Table_3.docx
│   │   # Fold-wise performance metrics, selected graph features, and
│   │   # optimized Random Forest hyperparameters for the leak-free
│   │   # isolated nested pipeline.
│   │   # [Supplementary Table 3]
│   │
│   ├── Supplementary_Table_4.docx
│   │   # Classification performance across the evaluated pipelines,
│   │   # including Accuracy, Macro Precision, Macro Recall, Macro F1,
│   │   # and ROC-AUC from repeated stratified 5-fold cross-validation.
│   │   # [Supplementary Table 4]
│   │
│   ├── Supplementary_Table_5.docx
│   │   # Top-ranking structural connectomic features according to
│   │   # Tree SHAP attributions, including mean absolute SHAP,
│   │   # mean signed SHAP, selection frequency, and selection percentage.
│   │   # [Supplementary Table 5]
│   │
│   └── Supplementary_Table_6.docx
│       # Quantitative diffusion MRI data-quality assessment and
│       # outlier-slice distribution across LGG and HGG cohorts.
│       # [Supplementary Table 6]
│
└── README.md
    # Project documentation, computational workflow, dataset description,
    # reproducibility instructions, and laboratory guidelines.

```

---

## Methods Summary

- **Modalities**: T1-weighted MRI and Diffusion Tensor Imaging (DTI: 25 non-collinear directions, $b = 1000\text{ s/mm}^2$).
- **Subjects**: 35 glioma patients from a single Chilean tertiary center (LGG $58.3$\% and HGG $41.7$\% confirmed via histopathology).
- **Hardware Setup**: Executed via CPU-based parallel processing on a 13th-generation Intel Core i7 architecture (24 GB RAM) and accelerated using an NVIDIA GPU (6 GB VRAM).

## Connectome Construction & Graph Theory

* **Node Definition ($V$):** Remaining non-invaded WM structures ($50$ target regions of interest) are derived from the *JHU ICBM-DTI-81 WM Labels Atlas* after applying dynamic, patient-specific tumor masking to strictly exclude tumor-encroached regions.
* **Streamline Propagation & Edge Definition ($E$):** 5000 probabilistic samples per voxel are generated via FSL's `ProbtrackX2` to construct customized streamline count-based structural connectivity adjacency matrices.
* **Network Metrics & Feature Extraction:** The structural brain graphs are processed via a specialized MATLAB computational framework using native graph functions to extract a comprehensive multi-level profile ($p = 307$ total topological features per patient after removing constant identifiers):
  * *Raw Matrix Construction Descriptors:* Base characteristics including Matrix Size, Max Weights, and Mean Weights.
  * *Global Network Topology:* Total Edge counts, Network Density, Total Weight, and Global Efficiency.
  * *Local Nodal Characterization:* ROI-specific topological metrics including Nodal Strength, Nodal Degree, PageRank, Closeness Centrality, Betweenness Centrality, and Clustering Coefficient calculated individually for each of the $50$ ROIs to map precise spatial WM integrity.

---

## Machine Learning Pipeline Architecture


An exploratory proof-of-concept framework designed to address high-dimensional, highly collinear structural connectomic feature spaces derived from personalized tumor-masked brain networks.

---

## Overview

* **Design Philosophy:** Leakage-controlled, nested cross-validation pipeline engineered specifically for high feature-to-sample ratios and multicollinear graph-theoretical variables.
* **Primary Classifier:** Random Forest (RF)
* **Interpretability:** Out-of-sample Tree SHAP attributions

---

## 1. Feature Preprocessing & Exploratory Characterization

* **Initial Feature Space (307 Features):**
  * **3** matrix-derived measures
  * **4** global network descriptors
  * **300** ROI-specific nodal features across 50 JHU white-matter atlas regions (*strength, degree, PageRank, closeness, betweenness, local clustering coefficient*)
* **Data Curation & Cleaning:**
  * Feature name standardization
  * Feature-wise mean imputation for missing and infinite values
  * Zero-variance feature removal (eliminated 11 constant variables $\rightarrow$ **296 informative features**)
* **Exploratory Analysis (Non-predictive):**
  * **Global Standardization:** $Z$-score transformation via `StandardScaler`.
  * **Hierarchical Collinearity Mapping:** Agglomerative hierarchical clustering (average linkage, Euclidean distance, cophenetic threshold = 6.5) to map feature redundancy.
  * **Exploratory Linear Baseline:** $L_2$-regularized `LogisticRegression` (`liblinear` solver) to assess baseline feature associations.

---

## 2. Leakage-Controlled Feature Selection

A two-stage feature selection wrapper embedded strictly inside the cross-validation loops to prevent data leakage:

1. **Stage 1 — Univariate ANOVA Filtering:** One-way ANOVA applied to candidate graph-theoretical features, retaining variables with $p < 0.05$.
2. **Stage 2 — Sequential Feature Selection (SFS):** Forward SFS driven by `RandomForestClassifier` with internal 5-fold stratified cross-validation, optimized dynamically via the F1-score trajectory.

* **Dynamic Selection Across Folds:** Feature subsets were determined dynamically per outer fold (fold counts: 8, 10, 3, 5, and 8 features; mean: $6.8 \pm 2.8$). A total of 24 distinct metrics were selected across all outer folds.
* **Recurrently Selected Markers:**
  * Betweenness centrality: Right posterior thalamic radiation
  * PageRank & Strength: Left sagittal stratum
  * Clustering coefficient: Left cerebral white matter
  * Centrality measures: Genu of corpus callosum, cerebellar peduncles

---

## 3. Evaluated Models & Multi-Modal Configurations

### 3.1 Model Architectures
1. **Random Forest Classifier (Primary):** Captures non-linear interactions among distributed features with robust performance against collinearity in small-sample regimes.
2. **$L_2$-Regularized Logistic Regression (Baseline):** Serves as a linear reference model.

### 3.2 Sensitivity Feature Configurations
To test whether structural connectomics offer incremental predictive power over traditional measures, four feature sets were benchmarked:

| Configuration | Description / Included Features |
| :--- | :--- |
| **Clinical-Radiological Baseline** | Age, sex, tumor volume (`MeshVolume` via PyRadiomics), lateralization, anatomical location, binary indicators for lobe involvement (frontal, parietal, temporal, insular). |
| **Unconstrained Structural Connectome** | Full available connectomic feature space (296 variables). |
| **Non-Zero Restricted Connectome** | Connectomic features filtered to retain only variables non-zero across all subjects in the outer training fold. |
| **Integrated Multimodal Model** | Combined Clinical-Radiological baseline + complete structural connectomic feature space. |

---

## 4. Optimization & Dual-Pathway Validation

### 4.1 Hyperparameter Tuning
Hyperparameters were optimized via `GridSearchCV` inside the inner CV loop:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
```
### 4.2 Validation Architectures

* **Biased Validation Pathway:** Stratified 5-fold CV using a static feature set derived from global dataset-level feature selection (used solely for leakage quantification).
* **Strictly Isolated Nested Pathway (Primary):**
  * **Outer Loop:** Stratified 5-fold CV for out-of-sample performance evaluation.
  * **Inner Loop:** Stratified 5-fold CV for feature selection and hyperparameter optimization.
  * All transformers and scalers fitted strictly on training splits within each fold.
  * Fixed random state ($\text{seed} = 42$).

---

### 4.3 Out-of-Sample Performance

Across the outer folds of the strictly isolated pathway, the connectomics model achieved:

* **Accuracy:** $0.829 \pm 0.186$
* **Precision:** $0.774 \pm 0.298$
* **Sensitivity / Recall:** $0.817 \pm 0.214$
* **Macro-F1 Score:** $0.786 \pm 0.265$
* **ROC-AUC:** $0.814 \pm 0.327$

---

## 5. Leakage Assessment

Data leakage impact was quantified using the differential macro-F1 score:

$$\Delta_{\mathrm{Leakage}} = F1_{\mathrm{biased}} - F1_{\mathrm{unbiased}}$$

* **Biased Pathway Macro-F1:** $0.737$
* **Strictly Isolated Pathway Macro-F1:** $0.786$
* **$\Delta_{\mathrm{Leakage}}$:** $-0.049$

---

## 6. Interpretability & Visualization

### 6.1 Out-of-Sample Tree SHAP (`shap v0.46`)

* Calculated strictly on unseen outer test folds per CV iteration.
* Unselected features assigned zero attribution before aggregation across patients.
* **Top Informative Tracts:** Left sagittal stratum, medial lemniscus, superior fronto-occipital fasciculus.
* **Non-linear Dependencies:** Observed notably for left medial lemniscus clustering coefficient and left sagittal stratum strength.

### 6.2 Visual Analysis Tools

* **t-SNE Projections:** Contrasted artificial class segregation in biased pipelines against realistic overlapping representations under nested isolation.
* **Decision Surface Mapping:** Visualized 2D non-linear decision boundaries using high-ranking features (e.g., right posterior thalamic radiation betweenness vs. left sagittal stratum strength).
* **Tree Inspection:** Hierarchical rule analysis of individual decision trees within the RF ensemble.

---

![Connectomica](Images/Figure1.png)

> **Methodological workflow for glioma stratification and WM vulnerability mapping:** Stage 1: Input Data and preprocessing. Integration of multimodal MRI data. (a) Anatomical T1-weighted imaging and (b) DWI data were acquired, co-registered to perform brain extraction, and utilized to compute DTI maps. (c) Personalized tumor masks are manually segmented to delineate neoplastic boundaries. (d) All structural and diffusion datasets are aligned with the Johns Hopkins University (JHU) WM atlas in MNI space to establish anatomical standardization across the cohort. (e) Spatial normalization to atlas space. Subject-specific images were co-registered to a JHU WM atlas. Stage 2: Tumor-masked tractography. Network node definition and structural connectivity matrix construction. (a) Precise mapping and exclusion of tumor-encroached regions are performed to eliminate confounding infiltrative artifacts. (b) Advanced probabilistics diffusion modeling (BEDPOSTX) evaluates fiber tract configurations, generating a 3D structural connectome. (c) Personalized adjacency matrices are built based on streamline connection probabilities. (d) Multi-level feature extraction yields a high-dimensional profile composed of m = 297 continuous graph-theoretical metrics. Stage 3: Explainable ML strategy. (a) Feature selection using a forward sequential feature selection (SFS). (b) Hyperparameter optimization and classification using a random forest ensemble validate via repeated nested 5-fold stratified cross-validation. (c) Model transparency and localized feature attributions are computed utilizing tree SHAP values to quantify tract-specific WM disruptions.



## Empirical Results & Performance Summary

**Demographic & Clinical Characteristics**
The study population comprised 35 patients (20 LGG [57.1%], 15 HGG [42.9%]) with a median cohort age of 43 years (range: 23–71). No significant differences were observed between LGG and HGG regarding age ($p = 0.442$), sex ($p = 0.310$), or lobar extension ($p = 0.535$). High-grade lesions exhibited significantly higher tumor volume ($159,248.7\text{ mm}^3$ vs. $64,227.7\text{ mm}^3$, $p = 0.012$). Hemispheric location differed significantly ($p = 0.024$), with left-hemisphere predominance in LGG (70.0%) and right-hemisphere predominance in HGG (60.0%).

**Feature Redundancy & Unsupervised Clustering**
Agglomerative hierarchical clustering of the $296$ non-constant features identified three distinct intercorrelated feature modules at a cophenetic distance threshold of 6.5 (Cluster 1: 107 features; Cluster 2: 90 features; Cluster 3: 99 features), confirming extensive multicollinearity ($R \approx 1.0$).

**Classification Performance & Leakage Prevention**
Under a strict, leakage-free nested cross-validation scheme (Isolated Pipeline), the pure structural connectomics model achieved a mean ROC-AUC of $0.814 \pm 0.327$ ($95\%$ CI: $0.500\text{--}1.000$), an accuracy of $82.9 \pm 18.6\%$, and a macro F1-score of $0.786 \pm 0.265$. Comparing this against the biased pipeline ($0.798 \pm 0.178$ AUC, $74.3\%$ accuracy) confirmed that dynamic per-fold feature selection prevents artificial data-leakage distortions.

**Model Sensitivity & Clinical Baselines**

* **Clinical-Radiological Baseline (M3):** The comprehensive clinical-radiological model, incorporating demographic characteristics, tumor volume, and anatomical location, demonstrated favorable discriminative capacity, with a mean AUC of $0.817 \pm 0.149$ and a macro F1-score of $0.809 \pm 0.136$. These findings support the foundational predictive utility of conventional clinical and anatomical markers for glioma characterization.

* **Non-Zero Connectomics Model (M5):** As an exploratory sensitivity analysis, restricting the structural connectomic feature space to variables that were non-zero across subjects yielded a mean AUC of $0.833 \pm 0.167$ and a macro F1-score of $0.727 \pm 0.256$. This configuration showed similar discriminative behavior to the unconstrained structural connectomics model while noticeably narrowing the variance across cross-validation folds.

* **Combined Multimodal Model (M6):** Integrating clinical-radiological variables with structural connectomic features yielded the highest point estimate for discrimination, with a mean AUC of $0.933 \pm 0.109$ ($95\%$ CI: $0.798\text{--}1.068$) and a macro F1-score of $0.774 \pm 0.271$ ($95\%$ CI: $0.438\text{--}1.110$). Given the overlapping confidence intervals and the exploratory proof-of-concept nature of the study, these results should be interpreted cautiously. Nevertheless, they suggest that tumor-masked structural connectomics may capture complementary pathophysiological signatures that potentially synergize with conventional demographic and anatomical markers.

**Game-Theoretic Interpretability (SHAP Analysis)**

Out-of-sample Tree SHAP attributions identified topological metrics within projection, association, and commissural pathways as important contributors to glioma-grade discrimination. PageRank Centrality in the left sagittal stratum showed the highest mean absolute SHAP value ($|\mathrm{SHAP}| = 0.066$) and was selected in 60.0% of the outer cross-validation folds. Strength Centrality in the left sagittal stratum was the second-ranked feature ($|\mathrm{SHAP}| = 0.033$), also selected in 60.0% of folds. Additional recurrent features included the Clustering Coefficient in the left cerebral white matter ($|\mathrm{SHAP}| = 0.032$, selected in 40.0% of folds), Closeness Centrality in the middle cerebellar peduncle ($|\mathrm{SHAP}| = 0.032$, selected in 20.0% of folds), and Betweenness Centrality in the right posterior thalamic radiation ($|\mathrm{SHAP}| = 0.019$, selected in 60.0% of folds). These findings identify tract-specific network metrics as candidate contributors to glioma-grade discrimination, while the variability in selection frequency across folds supports their interpretation as exploratory imaging biomarkers rather than definitive biomarker signatures.
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

If you find this pipeline useful for your research, please cite our preliminary work prepared for **Neuroradiology**:

```bibtex
@article{franco2026quantifying,
  title={Tumor-Masked Structural Connectomics Reveals Tract-Specific White Matter Disruption Associated with Glioma Aggressiveness: An Explainable Machine Learning Pilot Study},
  author={Franco, Pamela and Montalba, Cristian and Espinoza, Ignacio and Cornejo, M. Daniela and Torres, Francisco and Bennett, Carlos and Chabert, Steren and Salas, Rodrigo},
  journal={Neuroradiology},
  year={2026},
  note={Submitted for publication / Under review}
}
```

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)