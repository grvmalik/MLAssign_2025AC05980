# Machine Learning Classification Models - Assignment

- **Assignment:** Machine Learning - Assignment 2
- **Name:** Gaurav Malik
- **BITS Id:** 2025AC05980

## Problem Statement

The objective of this assignment is to implement and evaluate multiple machine learning classification models on a real-world medical dataset. The models will be trained on the "Breast Cancer Classification" dataset to predict whether a tumor is malignant (cancerous) or benign (non-cancerous) based on various diagnostic measurements.

This assignment demonstrates the complete machine learning pipeline: data loading, preprocessing, model training, evaluation, and deployment through an interactive web application.

## Dataset Description

**Dataset Name:** Breast Cancer Classification (UCI Machine Learning Repository)

**Dataset Source:** sklearn.datasets.load_breast_cancer()

**Dataset Characteristics:**
- **Total Instances:** 569 samples
- **Total Features:** 30 features (exceeds minimum requirement of 12)
- **Feature Type:** Continuous (real-valued)
- **Target Variable:** Binary (Malignant=0, Benign=1)
- **Class Distribution:** 
  - Malignant: 212 instances (37.3%)
  - Benign: 357 instances (62.7%)

**Features Included:**
The dataset contains 30 features representing diagnostic measurements such as:
- radius, texture, perimeter, area (mean, standard error, worst)
- smoothness, compactness, concavity, concave points (mean, standard error, worst)
- symmetry, fractal dimension (mean, standard error, worst)

**Dataset Suitability:**
- ✓ Exceeds minimum feature requirement (30 > 12)
- ✓ Exceeds minimum instance requirement (569 > 500)
- ✓ Binary classification problem (suitable for all assigned models)
- ✓ No missing values, clean and preprocessed
- ✓ Well-balanced target classes

## GitHub Repository Link

**Repository:** [grvmalik/MLAssign_2025AC05980](https://github.com/grvmalik/MLAssign_2025AC05980)

**Repository Contents:**
```
ML-Classification-Assignment/
├── app.py                                      # Streamlit web application
├── requirements.txt                            # Project dependencies
├── README.md                                   # This file
├── test_data.csv                               # Test dataset with predictions
└── model/                                      # Directory for saved models
    ├── ML_Assignment_2025AC05980.ipynb         # notebook file for all five models
```

## Models Used

### 5 Classification Models Implemented:

#### 1. **Logistic Regression**
- Linear model for binary classification
- Uses sigmoid function for probability estimation
- Fast training, easily interpretable
- Good baseline model

#### 2. **Decision Tree Classifier**
- Tree-based model that makes hierarchical splits
- Non-parametric approach
- Can capture non-linear relationships
- Prone to overfitting if depth not controlled

#### 3. **K-Nearest Neighbor (KNN)**
- Instance-based learning algorithm
- Classifies based on k nearest neighbors
- No training phase (lazy learner)
- Sensitive to feature scaling

#### 4. **Naive Bayes Classifier**
- Probabilistic model based on Bayes' theorem
- Assumes feature independence
- Fast and efficient
- Works well with small datasets

#### 5. **Random Forest (Ensemble)**
- Ensemble of multiple decision trees
- Reduces overfitting through bagging
- More robust than single decision tree
- Provides feature importance scores

---

### Evaluation Metrics (6 per model):

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness of predictions |
| **AUC Score** | Area under ROC curve | Ability to distinguish between classes (0-1) |
| **Precision** | TP / (TP + FP) | Reliability of positive predictions |
| **Recall** | TP / (TP + FN) | Coverage of actual positive cases |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of Precision and Recall |
| **MCC Score** | (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Correlation between actual and predicted |

---

### Model Performance Comparison Table:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.9649 | 0.9925 | 0.9583 | 0.9688 | 0.9636 | 0.9211 |
| Decision Tree | 0.9298 | 0.9296 | 0.9121 | 0.9479 | 0.9297 | 0.8536 |
| K-Nearest Neighbor | 0.9474 | 0.9747 | 0.9463 | 0.9479 | 0.9471 | 0.8903 |
| Naive Bayes | 0.9385 | 0.9768 | 0.9167 | 0.9583 | 0.9372 | 0.8717 |
| Random Forest (Ensemble) | 0.9561 | 0.9941 | 0.9538 | 0.9583 | 0.9560 | 0.9087 |

---

### Observations on Model Performance:

| ML Model Name | Observation about model performance |
|---------------|-----------------------------------|
| Logistic Regression | **Excellent overall performance.** Achieved highest accuracy (96.49%) and strong AUC (0.9925). The linear decision boundary works well for this dataset. Shows good generalization with high precision (95.83%) and recall (96.88%). Best for interpretability and production deployment. |
| Decision Tree | **Good but shows signs of complexity.** Despite controlled depth (max_depth=10), slightly lower accuracy (92.98%). The model may still be capturing dataset-specific patterns. Shows good recall (94.79%) but lower precision (91.21%), indicating some false positives. Consider pruning for better generalization. |
| K-Nearest Neighbor | **Strong performance with stable metrics.** Achieved 94.74% accuracy with balanced precision and recall. Effective with feature scaling applied. The non-parametric approach captures local patterns well. Performance depends heavily on hyperparameter k=5. Consider k-fold validation for k selection. |
| Naive Bayes | **Solid performance despite independence assumption.** Achieved 93.85% accuracy and highest recall (95.83%), making it excellent for minimizing false negatives. The independence assumption doesn't severely hurt performance. AUC score (0.9768) indicates strong discriminative ability. Lower precision suggests more false positives. |
| Random Forest (Ensemble) | **Best overall model.** Achieved highest AUC score (0.9941) and strong accuracy (95.61%). Ensemble method effectively reduces overfitting compared to single decision tree. Balanced performance across all metrics (F1: 0.9560, MCC: 0.9087). Provides feature importance rankings and most robust for production use. |

**Overall Winner for this dataset:** **Random Forest (Ensemble)** - Best AUC (0.9941), balanced metrics, robust to overfitting, and most suitable for real-world deployment.

---
## Model Training Results

All models were trained on the training set (455 samples) and evaluated on test set (114 samples).

### Key Findings:
- All models achieved >92% accuracy
- High AUC scores (>0.92) indicate excellent class separation
- Random Forest provides best balance of metrics
- Feature scaling was crucial for KNN and Logistic Regression
- No significant overfitting observed

## Files Description

# 1. **app.py**
Streamlit web application with:
- Dataset upload functionality
- Model selection dropdown
- Evaluation metrics display
- Confusion matrices visualization
- Classification reports
- Interactive predictions

# 2. **requirements.txt**
Python package dependencies:
```
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
```

# 3. **test_data.csv**
Test dataset containing:
- 114 test samples
- 30 feature columns
- Actual target labels
- Predictions from all 5 models

# 4. **model/ML_Assignment_2025AC05980.ipynb**
Main Jupyter notebook containing:
- Complete data loading and preprocessing
- Training of all 5 models
- Metric calculations and comparisons
- Visualizations (performance charts, confusion matrices)
- Test data generation
- Ready to run on BITS Virtual Lab

