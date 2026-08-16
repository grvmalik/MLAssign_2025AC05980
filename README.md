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
- Exceeds minimum feature requirement (30 > 12)
- Exceeds minimum instance requirement (569 > 500)
- Binary classification problem (suitable for all assigned models)
- No missing values, clean and preprocessed
- Well-balanced target classes

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

| Model               | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree       | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286 | 0.8174 |
| K-Nearest Neighbor  | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes         | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest       | 0.9561   | 0.9939 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
---

### Observations on Model Performance:

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | **Best overall model.** Achieved the highest accuracy (98.25%) and the highest AUC (0.9954) among all models. The linear decision boundary fits this dataset very well, with precision, recall, and F1 all tied at 98.61%. The highest MCC (0.9623) confirms strong, balanced performance across both classes. Best choice for interpretability and production deployment. |
| Decision Tree | **Weakest performer of the five.** Lowest accuracy (91.23%) and lowest AUC (0.9157), despite the controlled depth (max_depth=10). Precision (95.59%) is noticeably higher than recall (90.28%), indicating the model misses more actual malignant/benign cases than it misclassifies incorrectly as positive. The lowest MCC (0.8174) reflects the weakest overall correlation between predictions and true labels. Likely still overfitting to training-specific splits despite depth control. |
| K-Nearest Neighbor | **Strong, balanced performance.** Accuracy of 95.61% with well-balanced precision (95.89%) and recall (97.22%), giving a solid F1 of 96.55%. Performance is close to Random Forest across every metric, showing that distance-based classification works well once features are scaled. Results are sensitive to the choice of k (k=5 used here); k-fold cross-validation could help confirm this is optimal. |
| Naive Bayes | **Solid but the second-weakest model.** Accuracy of 92.98%, the second-lowest among the five. However, it achieves a strong AUC (0.9868), the second-best after Logistic Regression, showing good class-separation ability despite the feature-independence assumption. Precision and recall are equal (94.44%), meaning errors are evenly split between false positives and false negatives rather than skewed toward one type. |
| Random Forest (Ensemble) | **Second-best overall model.** Matches KNN exactly on accuracy (95.61%), precision (95.89%), recall (97.22%), F1 (96.55%), and MCC (0.9054), but edges ahead with a stronger AUC (0.9939) — the second-highest of all models. The ensemble's bagging approach clearly reduces the overfitting seen in the single Decision Tree, and it remains a robust, production-ready alternative to Logistic Regression, with the added benefit of feature importance rankings. |

**Overall Winner for this dataset:** **Logistic Regression** — highest across all six metrics: Accuracy (98.25%), AUC (0.9954), Precision (98.61%), Recall (98.61%), F1 (98.61%), and MCC (0.9623). It is also the simplest and most interpretable of the five models, making it the best choice for both performance and real-world deployment.

---
## Model Training Results

All models were trained on the training set (455 samples) and evaluated on test set (114 samples).

### Key Findings:
- All models achieved >91% accuracy
- High AUC scores (>0.91) indicate excellent class separation
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


## Streamlit Deployment Link
**Streamlit Link:** https://ml-2025ac05980.streamlit.app/
 

 
 

