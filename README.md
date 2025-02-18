# Stroke Prediction: A Data-Driven Approach to Healthcare

## Overview
Stroke remains one of the leading causes of death and disability globally. This project uses machine learning techniques to predict stroke occurrences based on demographic and health-related features. By leveraging predictive models, it aims to offer early detection and valuable insights for stroke prevention.

## Key Highlights
- **94% Accuracy** achieved with Random Forest model
- **Feature Importance Analysis** using SHAP
- **Balanced Data** using SMOTE to handle class imbalance
- **Cross-Validation** for robust model evaluation
- **Hyperparameter Tuning** using GridSearchCV for performance optimization

## Dataset
- **Source**: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets)
- **Total Records**: 5110 entries
- **Features**: 
  - `age`, `bmi`, `avg_glucose_level`, `hypertension`, `heart_disease`, `smoking_status`, etc.

## Tech Stack
- **Programming Language**: Python
- **Environment**: Jupyter Notebook
- **Libraries Used**:
  - `scikit-learn` for machine learning algorithms
  - `imbalanced-learn` for handling imbalanced data
  - `shap` for model interpretability and feature importance analysis
  - `pandas`, `numpy`, `seaborn`, `matplotlib` for data manipulation and visualization

## Methodology
### 1. Exploratory Data Analysis (EDA)
- **Data Visualization**: Histograms, Violin Plots, Count Plots to understand feature distributions
- **Outlier Detection**: Using IQR and Percentile methods
- **Missing Value Handling**: Applied KNN Imputation for missing data

### 2. Feature Engineering
- **Encoding**: Categorical variables were encoded using suitable techniques
- **Scaling**: Standardization of features using `StandardScaler`

### 3. Model Training & Evaluation
- **Algorithms Tested**: Random Forest, XGBoost, SVM, KNN, Naïve Bayes
- **Cross-Validation**: Stratified 10-Fold Cross-Validation
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC
- **Best Model**: Random Forest

### 4. Model Interpretability
- **SHAP**: Analyzed feature importance and provided local and global interpretability
  - Global Feature Importance
  - Local Instance-Based Explanations (Waterfall, Beeswarm, Force Plots)

### 5. Hyperparameter Tuning
- Optimized model performance using `GridSearchCV`

## Results
| Model           | Accuracy | Precision | Recall | F1 Score |
|-----------------|----------|-----------|--------|----------|
| **Random Forest** | 94%      | 0.91      | 0.89   | 0.90     |
| **XGBoost**      | 92%      | 0.88      | 0.86   | 0.87     |
| **SVM**          | 87%      | 0.83      | 0.81   | 0.82     |

- **Confusion Matrix**, **ROC Curve**, and **Feature Importance (SHAP)** visualizations are included for detailed analysis.

## Future Scope
- **Deep Learning**: Implementation of advanced models like LSTMs and CNNs for time-series or sequential data prediction.
- **Real-Time Stroke Prediction**: Integration with IoT devices to predict strokes in real-time using sensors.
- **Electronic Health Records (EHR) Integration**: Incorporating patient data from EHR systems for automated stroke prediction.


