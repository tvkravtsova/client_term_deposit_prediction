# Predicting Client Subscription to Term Deposits

## Project Overview
This project aims to predict whether a client will subscribe to a term deposit based on various attributes. This is a binary classification task. The primary audience for this project includes data analysts, data scientists, and marketing professionals looking to understand and predict customer behavior in the banking sector. 
*   **Models Explored:** Logistic Regression, k-Nearest Neighbors (kNN), Decision Tree, and XGBoost.
*   **Primary Metric:** ROC AUC, suitable for imbalanced classes and additionally F1 for class 1
*   **Secondary Metric:**  F1 Score for class 1 (useful when class 1 is the minority)
*   **Interpretation:** Focus on understanding model decisions (e.g., via SHAP, feature importance, error analysis).

## Project Structure

```
client_term_deposit_prediction/
├── data/
│   └── bank-additional-full.csv  # Main dataset
├── models/
│   └── xgb_model.joblib          # Pre-trained XGBoost model
├── notebooks/
│   ├── bank_term_deposit_EDA.ipynb # Exploratory Data Analysis
│   └── bank_term_deposit_main.ipynb # Main notebook for model training, evaluation, and interpretation
├── src/
│   └── bank_preprocessing.py     # Script for data preprocessing
├── requirements.txt              # Project dependencies
└── README.md                     # # Describes project overview, setup, and usage
```

## Data

*   **Source:** [Bank Marketing Kaggle Dataset](https://www.kaggle.com/datasets/sahistapatel96/bankadditionalfullcsv) 
*   **Content:** Client demographic data, campaign interaction details, and socio-economic indicators.
*   **Target Variable:** `y` (yes/no for term deposit subscription).
*   **Important Note on `duration`:** The `duration` feature (last contact duration) is excluded from the final model features to prevent data leakage, as its value is typically known only after the call outcome.

## Methodology

### 1. Feature Engineering & Preprocessing

Key steps performed (details in `notebooks/bank_term_deposit_main.ipynb` and `src/bank_preprocessing.py`):
*   **New Features Created:** Including `contacted_before`, `many_contacts`, `contact_ratio`, `previous_outcome_success`, and ratios of economic indicators (e.g., `euribor3m_to_emp.var.rate`).
*   **Encoding:** Ordinal encoding for `education` (simplified categories), One-Hot Encoding for other categoricals.
*   **Outliers:** `campaign` feature clipped at the 99th percentile.
*   **Scaling:** MinMaxScaler applied to numerical features for relevant models (e.g., Logistic Regression, kNN).
*   **Data Split:** Stratified train-validation split.

### 2. Modeling & Evaluation

*   **Models Trained:** Logistic Regression, kNN, Decision Tree, XGBoost.
*   **Tuning:** GridSearchCV (for kNN, Decision Tree), RandomizedSearchCV and Hyperopt (for XGBoost).
*   **Evaluation:** Primarily ROC AUC; F1-score (class 1) also considered.

## Results Summary

Below are key ROC AUC scores. For detailed metrics, hyperparameters, and F1-scores, please refer to `notebooks/bank_term_deposit_main.ipynb`.

| Model                        | Train AUROC | Val AUROC   | Key Hyperparameters / Tuning Notes                 |
|:-----------------------------|:------------|:------------|:---------------------------------------------------|
| Logistic Regression          | 0.7937      | 0.8067      | `class_weight='balanced'`, L1 penalty            |
| kNN                          | 0.8601      | 0.7743      | Tuned `n_neighbors` via GridSearchCV (19)          |
| Decision Tree (Tuned)        | 0.7840      | 0.7952      | Tuned: `max_depth=7, max_leaf_nodes=10`          |
| **XGBoost (Hyperopt Tuned)** | 0.8333      | **0.8188**  | Tuned via Hyperopt (see notebook for full params)  |


## Model Interpretation (Best Model: XGBoost Tuned)
Detailed analysis is in `notebooks/bank_term_deposit_main.ipynb`.

## How to Use Pre-trained Model

The best performing model (`xgb_hyperopt_best`) is saved in `models/xgb_model.joblib`. Example usage:

```python
import joblib
# model = joblib.load('models/xgb_model.joblib')
# new_data_processed = ... # Ensure data is preprocessed identically to training
# predictions = model.predict(new_data_processed)
# probabilities = model.predict_proba(new_data_processed)[:, 1]
```
Refer to `notebooks/bank_term_deposit_main.ipynb` for full preprocessing steps.

## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/tvkravtsova/client_term_deposit_prediction.git
    cd client_term_deposit_prediction
    ```
2.  **Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Launch Jupyter:** `jupyter notebook` and navigate to the `notebooks` directory.

## Ideas for Improvement

*   **Optimize Classification Threshold** 
*   **Enhance Class Balancing Strategies**
*   **Refine Feature Set**
*   **Explore Alternative Modeling Approaches** 


## Author 

*   Tetiana Kravtsova
*   LinkedIn: https://www.linkedin.com/in/tetiana--kravtsova/
