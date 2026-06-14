# Fraud Detection — Adey Innovations Inc.
## Business Context

Adey Innovations Inc. serves FinTech clients who need reliable, real-time fraud
detection across two transaction streams:

- **False Positives** — legitimate transactions wrongly flagged, frustrating customers and eroding trust.
- **False Negatives** — actual fraud missed, causing direct financial loss.

The system must balance both. Overall accuracy is not used as an evaluation metric
because it is misleading on highly imbalanced datasets.

---

## Project Structure

```
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI — runs pytest on push/PR
├── data/                          # Gitignored
│   ├── raw/                       # Original datasets
│   └── processed/                 # Cleaned, engineered, split data
├── notebooks/
│   ├── eda-fraud-data.ipynb       # EDA + cleaning + geolocation (Fraud_Data)
│   ├── eda-creditcard.ipynb       # EDA + cleaning + feature engineering (creditcard)
│   ├── feature-engineering.ipynb  # Encoding, scaling, SMOTE
│   ├── modeling.ipynb             # Model training, evaluation, comparison
│   └── shap-explainability.ipynb  # SHAP global + local explanations
├── src/
│   └── __init__.py
├── tests/
│   └── __init__.py
├── models/                        # Saved model artifacts (.pkl)
├── scripts/
│   └── README.md
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Datasets

| Dataset | Records | Features | Fraud Rate |
|---------|---------|----------|------------|
| `Fraud_Data.csv` | ~151,112 | 11 | ~9.4% |
| `IpAddress_to_Country.csv` | ~138,846 | 3 | — |
| `creditcard.csv` | ~284,807 | 30 | ~0.17% |

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Redeat-Birhane/fraud-detection.git
cd fraud-detection

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place raw datasets
#    data/raw/Fraud_Data.csv
#    data/raw/IpAddress_to_Country.csv
#    data/raw/creditcard.csv
```

---

## Pipeline

Run notebooks in this order:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `eda-fraud-data.ipynb` | Data cleaning, EDA, geolocation enrichment for Fraud_Data |
| 2 | `eda-creditcard.ipynb` | Data cleaning, EDA, feature engineering for creditcard |
| 3 | `feature-engineering.ipynb` | Encoding, scaling, SMOTE for both datasets |
| 4 | `modeling.ipynb` | Model training, evaluation, and comparison |
| 5 | `shap-explainability.ipynb` | SHAP-based interpretability and recommendations |

---

## Task 1 — Data Analysis and Preprocessing ✅

### Data Cleaning
- No missing values found in either dataset
- Duplicate rows removed
- `signup_time` and `purchase_time` parsed from string to datetime
- Target variable (`class` / `Class`) cast to integer

### Exploratory Data Analysis
- Univariate distributions for `purchase_value`, `age`, `source`, `browser`, `sex`
- Bivariate analysis: feature distributions overlaid by fraud class
- Fraud rate computed per category (source, browser, sex, country)
- Class imbalance quantified and visualized for both datasets

### Geolocation Integration
- IP addresses converted from dotted-decimal to 64-bit integer
- Range-based country lookup using `numpy.searchsorted` — O(log n) per record
- Country column attached to Fraud_Data; fraud rates analyzed by country
- Luxembourg (~39%), Ecuador (~26%), Tunisia (~26%) show highest fraud rates

### Feature Engineering
| Feature | Description | Dataset |
|---------|-------------|---------|
| `hour_of_day` | Hour of purchase (0–23) | Fraud_Data |
| `day_of_week` | Day of week (0=Mon, 6=Sun) | Fraud_Data |
| `time_since_signup` | Hours between signup and purchase | Fraud_Data |
| `user_tx_count` | Total transactions per user | Fraud_Data |
| `device_tx_count` | Total transactions per device | Fraud_Data |
| `Amount_log` | Log-transformed transaction amount | creditcard |
| `hour_of_day` | Hour derived from Time (seconds elapsed) | creditcard |

### Data Transformation
- One-hot encoding applied to `source`, `browser`, `sex`, `country`
- `StandardScaler` fitted on training split only — applied to both splits
- Original `Amount` and `Time` dropped after feature engineering in creditcard

### Class Imbalance Handling
| Dataset | Legitimate | Fraud | Ratio | Strategy |
|---------|------------|-------|-------|----------|
| Fraud_Data | 136,961 (90.6%) | 14,151 (9.4%) | ~10:1 | SMOTE |
| creditcard | 284,315 (99.83%) | 492 (0.17%) | ~578:1 | SMOTE |

SMOTE applied on the **training split only** to prevent data leakage.
Undersampling was rejected to avoid discarding real majority-class signal.

---

## Task 2 — Model Building and Training ✅

### Models Trained
- **Logistic Regression** — interpretable baseline with `class_weight='balanced'`
- **Random Forest** — ensemble model with `class_weight='balanced_subsample'`

### Evaluation Metrics
Accuracy was excluded. Models evaluated on:
- **AUC-PR** — primary metric; rewards high precision at high recall
- **F1-Score** — harmonic mean of precision and recall
- **Confusion Matrix** — raw TP, FP, TN, FN for cost analysis

### Cross-Validation
- Stratified K-Fold (k=5) on pre-SMOTE full dataset
- Mean ± standard deviation reported for F1 and AUC-PR

### Model Selection
**Random Forest selected** as the production model based on:
- Higher AUC-PR and F1 than Logistic Regression on both datasets
- Captures non-linear feature interactions (velocity, time-since-signup, device reuse)
- No feature scaling required — robust to new raw features in production
- Low cross-validation standard deviation confirms stable generalization
- Native feature importance scores feed directly into SHAP analysis

### Saved Artifacts
| File | Description |
|------|-------------|
| `models/lr_fraud.pkl` | Logistic Regression — Fraud_Data |
| `models/lr_cc.pkl` | Logistic Regression — creditcard |
| `models/rf_fraud.pkl` | Random Forest — Fraud_Data |
| `models/rf_cc.pkl` | Random Forest — creditcard |
| `models/fraud_scaler.pkl` | Fitted StandardScaler — Fraud_Data |
| `models/cc_scaler.pkl` | Fitted StandardScaler — creditcard |

---

## Task 3 — Model Explainability 🔜

- SHAP global summary plots (feature importance across all predictions)
- SHAP force plots for individual predictions (TP, FP, FN)
- Top 5 fraud drivers identified and connected to business recommendations

---

## Evaluation Philosophy

| Metric | Why it matters |
|--------|---------------|
| AUC-PR | Imbalanced datasets make AUC-ROC optimistic — AUC-PR does not |
| F1-Score | Penalizes both missed fraud and false customer alerts equally |
| Confusion Matrix | Translates model errors into business cost estimates |

## Tools & Libraries

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Models, metrics, preprocessing, CV |
| `imbalanced-learn` | SMOTE resampling |
| `shap` | Model explainability (Task 3) |
| `joblib` | Model serialization |

---
