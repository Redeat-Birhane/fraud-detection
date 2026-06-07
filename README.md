# Fraud Detection Project

## Overview

This project focuses on building a machine learning pipeline to detect fraudulent transactions using multiple datasets. The workflow includes data cleaning, exploratory data analysis (EDA), feature engineering, geolocation integration, and handling class imbalance.

The goal of Task 1 is to prepare a clean, feature-rich dataset suitable for modeling.

---

## Project Structure

fraud-detection/
│
├── data/
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned and feature-engineered data
│
├── notebooks/
│ ├── eda-fraud-data.ipynb
│ ├── eda-creditcard.ipynb
│ ├── feature-engineering.ipynb
│ └── shap-explainability.ipynb
│
├── src/ # Reusable scripts and utilities
├── tests/ # Unit tests
├── models/ # Saved ML models
├── scripts/ # Helper scripts
│
├── requirements.txt
├── README.md
└── .gitignore


---

## Task 1: Data Analysis and Preprocessing

### Objective

Prepare clean, structured, and enriched datasets ready for machine learning models.

---

## Datasets

1. **Fraud_Data.csv**
   - User transaction data with behavioral features

2. **IpAddress_to_Country.csv**
   - Mapping of IP address ranges to countries

3. **creditcard.csv**
   - Credit card transactions labeled for fraud detection

---

## Key Steps

### 1. Data Cleaning
- Handling missing values (imputation or removal with justification)
- Removing duplicate records
- Correcting incorrect data types (e.g., datetime conversion)

---

### 2. Exploratory Data Analysis (EDA)
- Univariate analysis of key features
- Bivariate analysis with target variable (fraud vs non-fraud)
- Distribution of numerical and categorical variables
- Class imbalance analysis

---

### 3. Geolocation Integration
- Convert IP addresses to integer format
- Map IP ranges to countries using range-based lookup
- Analyze fraud patterns by country

---

### 4. Feature Engineering (Fraud_Data.csv)
- `hour_of_day` → time-based behavior
- `day_of_week` → weekly patterns
- `time_since_signup` → time difference between signup and purchase
- `transaction_count` → frequency per user
- `transaction_velocity` → time between consecutive transactions

---

### 5. Data Transformation
- One-hot encoding for categorical variables
- Feature scaling using StandardScaler or MinMaxScaler

---

### 6. Class Imbalance Handling
- Applied SMOTE (Synthetic Minority Oversampling Technique)
- Justification:
  - Fraud cases are rare
  - Undersampling risks losing important information
  - SMOTE improves model learning on minority class
- Class distribution documented before and after resampling

---

## Tools & Libraries

- Python 3.10+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
- Jupyter Notebook

---

## Results (Task 1 Output)

After completing Task 1:

- Cleaned datasets are stored in `data/processed/`
- New engineered features added
- Categorical variables encoded
- Numerical features scaled
- Class imbalance handled
- Ready-to-train dataset prepared for modeling

---

## Next Steps

- Task 2: Model Training (Logistic Regression, Random Forest, XGBoost)
- Task 3: Model Explainability (SHAP analysis)
- Performance evaluation and optimization

---


