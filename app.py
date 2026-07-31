"""
Streamlit Web Application for ML Classification Models
Dataset: Breast Cancer Classification
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
import json
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("Machine Learning Classification Models")
st.markdown("### Breast Cancer Classification - ML Assignment")
st.markdown("---")

# Sidebar
st.topbar.title("Navigation")
page = st.topbar.radio(
    "Select Page:",
    ["Home", "Dataset", "Model Training", "Results Comparison", "Predictions", "About"]
)

# Load and prepare data
@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y, data

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    models = {}
    metrics = {}
    predictions = {}
    probabilities = {}
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=10000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    predictions['Logistic Regression'] = lr_pred
    probabilities['Logistic Regression'] = lr_proba
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt
    dt_pred = dt.predict(X_test)
    dt_proba = dt.predict_proba(X_test)[:, 1]
    predictions['Decision Tree'] = dt_pred
    probabilities['Decision Tree'] = dt_proba
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    models['K-Nearest Neighbor'] = knn
    knn_pred = knn.predict(X_test)
    knn_proba = knn.predict_proba(X_test)[:, 1]
    predictions['K-Nearest Neighbor'] = knn_pred
    probabilities['K-Nearest Neighbor'] = knn_proba
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    models['Naive Bayes'] = nb
    nb_pred = nb.predict(X_test)
    nb_proba = nb.predict_proba(X_test)[:, 1]
    predictions['Naive Bayes'] = nb_pred
    probabilities['Naive Bayes'] = nb_proba
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    predictions['Random Forest'] = rf_pred
    probabilities['Random Forest'] = rf_proba
    
    # Calculate metrics
    for model_name in models.keys():
        y_pred = predictions[model_name]
        y_proba = probabilities[model_name]
        
        metrics[model_name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_proba),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_test, y_pred)
        }
    
    return models, metrics, predictions, probabilities

# Page: Home
if page == "Home":
    st.markdown("## Welcome to ML Classification Assignment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Dataset:** Breast Cancer Classification\n- Instances: 569\n- Features: 30")
    
    with col2:
        st.warning("**Models:** 5 Classification Algorithms\n- Logistic Regression\n- Decision Tree")
    
    with col3:
        st.success("**Metrics:** 6 Evaluation Metrics\n- Accuracy, AUC, Precision\n- Recall, F1, MCC")
    
    st.markdown("---")
    
    st.markdown("""
    ### About This Application
    
    This Streamlit application demonstrates a complete machine learning workflow:
    
    1. **Dataset Exploration** - Analyze the Breast Cancer classification dataset
    2. **Model Training** - Train 5 different classification models
    3. **Results Comparison** - Compare model performance across 6 metrics
    4. **Interactive Predictions** - Make predictions on new data
    5. **Detailed Analysis** - View confusion matrices and classification reports
    
    ### Features
    1. Upload and visualize CSV data
    2. Train multiple ML models with one click
    3. Compare model performance in detail
    4. Make predictions on test data
    5. View comprehensive metrics and reports
    
    Use the sidebar to navigate through different sections!
    """)

# Page: Dataset
elif page == "Dataset":
    st.markdown("##Dataset Overview")
    
    X, y, data = load_data()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Instances", X.shape[0])
    with col2:
        st.metric("Total Features", X.shape[1])
    with col3:
        st.metric("Malignant Cases", sum(y == 0))
    with col4:
        st.metric("Benign Cases", sum(y == 1))
    
    st.markdown("---")
    
    st.subheader("Target Distribution")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Bar chart
    y.value_counts().plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
    ax1.set_title('Target Distribution')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(['Malignant (0)', 'Benign (1)'], rotation=0)
    
    # Pie chart
    y.value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%', 
                          colors=['#FF6B6B', '#4ECDC4'])
    ax2.set_title('Target Distribution (%)')
    ax2.set_ylabel('')
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("Dataset Sample")
    st.dataframe(X.head(10), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Feature Statistics")
    st.dataframe(X.describe(), use_container_width=True)
    
    # Upload CSV option
    st.markdown("---")
    st.subheader("Upload Test Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write(f"Shape: {test_df.shape}")
        st.dataframe(test_df.head(), use_container_width=True)

# Page: Model Training
elif page == "Model Training":
    st.markdown("## Model Training")
    
    X, y, _ = load_data()
    
    st.info("Training models with 80-20 train-test split...")
    
    # Data preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train models
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    models, metrics, predictions, probabilities = train_models(X_train, X_test, y_train, y_test)
    
    progress_bar.progress(100)
    status_text.success("All models trained successfully!")
    
    st.markdown("---")
    
    st.subheader("Training Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Test Samples", len(X_test))
    with col3:
        st.metric("Models Trained", len(models))
    
    st.markdown("---")
    
    st.subheader("Models Overview")
    model_info = {
        "Logistic Regression": "Linear model for binary classification",
        "Decision Tree": "Tree-based model with interpretable splits",
        "K-Nearest Neighbor": "Instance-based learning with k=5",
        "Naive Bayes": "Probabilistic model based on Bayes' theorem",
        "Random Forest": "Ensemble of 100 decision trees"
    }
    
    cols = st.columns(len(models))
    for i, (model_name, model) in enumerate(models.items()):
        with cols[i]:
            st.markdown(f"### {model_name}")
            st.write(model_info[model_name])

# Page: Results Comparison
elif page == "Results Comparison":
    st.markdown("## Model Comparison")
    
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    models, metrics, predictions, _ = train_models(X_train, X_test, y_train, y_test)
    
    # Metrics Table
    st.subheader("Metrics Comparison Table")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
    
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'), 
                 use_container_width=True)
    
    st.markdown("---")
    
    # Charts
    st.subheader("Performance Charts")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Model Performance Comparison\n', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        values = metrics_df[metric].values
        models_list = metrics_df.index
        
        bars = ax.bar(models_list, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
        ax.set_ylabel(metric, fontsize=10, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Confusion Matrices
    st.subheader("Confusion Matrices")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Confusion Matrices for All Models\n', fontsize=16, fontweight='bold')
    
    model_list = list(models.keys())
    for idx, model_name in enumerate(model_list):
        ax = axes[idx // 3, idx % 3]
        y_pred = predictions[model_name]
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Malignant', 'Benign'],
                    yticklabels=['Malignant', 'Benign'])
        ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
    
    # Hide extra subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)

# Page: Predictions
elif page == "Predictions":
    st.markdown("## Make Predictions")
    
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    models, metrics, predictions, probabilities = train_models(X_train, X_test, y_train, y_test)
    
    # Model selection
    selected_model = st.selectbox("Select a Model:", list(models.keys()))
    
    st.markdown("---")
    
    # Display metrics for selected model
    st.subheader(f"{selected_model} - Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{metrics[selected_model]['Accuracy']:.4f}")
    with col2:
        st.metric("AUC Score", f"{metrics[selected_model]['AUC']:.4f}")
    with col3:
        st.metric("F1 Score", f"{metrics[selected_model]['F1']:.4f}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precision", f"{metrics[selected_model]['Precision']:.4f}")
    with col2:
        st.metric("Recall", f"{metrics[selected_model]['Recall']:.4f}")
    with col3:
        st.metric("MCC", f"{metrics[selected_model]['MCC']:.4f}")
    
    st.markdown("---")
    
    # Classification Report
    st.subheader("Classification Report")
    y_pred = predictions[selected_model]
    report = classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'])
    st.text(report)
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'],
                cbar_kws={'label': 'Count'})
    ax.set_title(f'{selected_model} - Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    st.pyplot(fig)

# Page: About
elif page == "About":
    st.markdown("## About This Application")
    
    st.markdown("""
    ### Machine Learning Assignment
    
    ---
    
    ### Assignment Details
    - **Name:** Gaurav Malik
    - **BITS Id:** 2025AC05980
    - **Subject:** Machine Learning
    - **Assignment:** 2

    
    ---
    
    ### Implemented Models
    
    1. **Logistic Regression**
       - Linear classifier using sigmoid function
       - Best for: Interpretable binary classification
    
    2. **Decision Tree Classifier**
       - Hierarchical splitting of features
       - Best for: Non-linear relationships, feature importance
    
    3. **K-Nearest Neighbor (KNN)**
       - Instance-based lazy learning (k=5)
       - Best for: Local pattern recognition
    
    4. **Naive Bayes Classifier**
       - Probabilistic model assuming feature independence
       - Best for: Fast inference, small datasets
    
    5. **Random Forest (Ensemble)**
       - Ensemble of 100 decision trees
       - Best for: Robust predictions, avoiding overfitting
    
    ---
    
    ### Evaluation Metrics
    
    - **Accuracy:** Overall correctness of predictions
    - **AUC Score:** Ability to distinguish between classes
    - **Precision:** Reliability of positive predictions
    - **Recall:** Coverage of actual positive cases
    - **F1 Score:** Harmonic mean of precision and recall
    - **MCC:** Matthews Correlation Coefficient
    
    ---
    
    ### Dataset Information
    
    **Breast Cancer Classification Dataset (UCI)**
    
    - **Instances:** 569
    - **Features:** 30
    - **Target:** Binary (Malignant=0, Benign=1)
    - **Distribution:** 37.3% Malignant, 62.7% Benign
    
    ---
    
    ### Features
    
    Dataset exploration and visualization
    Multi-model training with one click
    Comprehensive metrics comparison
    Interactive predictions
    Detailed confusion matrices
    Classification reports
    CSV data upload support
    
    ---
    
    ### Repository
    
    All code, models, and data are available on GitHub.
    
    **Files Included:**
    - ML_Assignment_2.ipynb (Training notebook)
    - app.py (Streamlit application)
    - requirements.txt (Python dependencies)
    - README.md (Complete documentation)
    - test_data.csv (Test dataset)
    
    """)

